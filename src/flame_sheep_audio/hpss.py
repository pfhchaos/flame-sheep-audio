"""Spectrum transforms — composable SpectrumFrame → SpectrumFrame.

A SpectrumTransform takes a SpectrumFrame and returns a SpectrumFrame.
Same shape, composable. Each transform owns its own state.

HPSS is two independent transforms:
  - HarmonicTransform: keeps stable content, suppresses transients
  - PercussiveTransform: keeps transient content, suppresses stable

They may use different estimators, different thresholds, or different
mask application curves. They don't have to sum to the original.

Transforms that need external state (like a shared stability mask)
receive it as a parameter to __call__, not via internal update.

F0 collapsing will follow the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ._spectrum import SpectrumFrame


class SpectrumTransform(ABC):
    """Interface: SpectrumFrame → SpectrumFrame.

    Implementations must:
      - Preserve array shapes (n_bins in = n_bins out)
      - Handle silence without crashing
      - Be composable: transform(transform(frame)) must work
      - Support reset() for engine hotswapping

    Transforms may be stateful (own estimator) or stateless (receive
    mask via apply()). The ABC supports both patterns.
    """

    @abstractmethod
    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        """Transform a frame. Stateful transforms update internally."""
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class LogMagnitudeTransform(SpectrumTransform):
    """Convert magnitude to log scale: ln(1 + gain * magnitude).

    Placed right after the spectrum engine, before HPSS/detection/energy.
    Models human loudness perception (Weber-Fechner law).

    Flux is recomputed as the half-wave rectified difference of
    log-magnitudes — measures relative spectral change instead of
    absolute. A low-band hit at -20dB and -40dB produce similar flux.
    """

    def __init__(self, gain: float = 1000.0) -> None:
        self._gain = gain
        self._norm = 1.0 / np.log1p(gain)  # normalize to ~0-1 range
        self._prev_log_mag: np.ndarray | None = None

    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        log_mag = (np.log1p(self._gain * frame.magnitude) * self._norm).astype(np.float32)

        # Recompute flux on log magnitudes
        if self._prev_log_mag is not None and len(log_mag) == len(self._prev_log_mag):
            log_flux = np.maximum(log_mag - self._prev_log_mag, 0.0).astype(np.float32)
        else:
            log_flux = np.zeros_like(log_mag)
        self._prev_log_mag = log_mag.copy()

        return SpectrumFrame(
            magnitude=log_mag,
            flux=log_flux,
            waveform=frame.waveform,
            phase=frame.phase,
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        self._prev_log_mag = None


class HarmonicTransform(SpectrumTransform):
    """Extract harmonic content — suppress transients.

    Can be used standalone (owns estimator) or with apply() to use
    an externally-computed mask.
    """

    def __init__(self, estimator=None) -> None:
        self._estimator = estimator

    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        if self._estimator is None:
            return frame
        self._estimator.update(frame.magnitude)
        return self.apply(frame, self._estimator.stability_per_bin())

    def apply(self, frame: SpectrumFrame, mask: np.ndarray) -> SpectrumFrame:
        """Apply a precomputed harmonic mask."""
        return SpectrumFrame(
            magnitude=(frame.magnitude * mask).astype(np.float32),
            flux=(frame.flux * mask).astype(np.float32),
            waveform=frame.waveform,
            phase=frame.phase,
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        if self._estimator is not None:
            self._estimator.reset()


class ComplexSpectralDiffTransform(SpectrumTransform):
    """Replace flux with complex spectral difference.

    Uses phase acceleration (2nd derivative) to detect onsets even in
    wall-of-sound tracks where magnitude flux is flat. Sustained tones
    have linear phase evolution (predictable 2nd derivative ≈ 0); onsets
    create phase discontinuities.

    Algorithm (Bello et al. / Gist OnsetDetectionFunction):
      1. phase_dev = phase - 2*prev_phase + prev_prev_phase
      2. princarg(phase_dev) → wrap to [-π, π]
      3. mag_diff = |current| - |previous|  (half-wave rectified)
      4. phase_diff = -|current| * sin(phase_dev)  (magnitude-weighted)
      5. complex_flux = mag_diff + phase_weight * max(phase_diff, 0)

    The raw CSD has a higher floor than magnitude flux (phase noise in
    sustained tones), so we subtract a running mean per bin to normalize.
    """

    def __init__(self) -> None:
        self._prev_phase: np.ndarray | None = None
        self._prev_prev_phase: np.ndarray | None = None
        self._prev_magnitude: np.ndarray | None = None
        self._running_mean: np.ndarray | None = None
        self._alpha = 0.05  # EMA smoothing for running mean

    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        if frame.phase is None:
            return frame  # no phase available, pass through unchanged

        phase = frame.phase
        mag = frame.magnitude

        # Auto-reset on spectrum size change
        if self._prev_phase is not None and len(mag) != len(self._prev_phase):
            self.reset()

        if self._prev_phase is not None and self._prev_prev_phase is not None:
            # Phase deviation (2nd derivative)
            phase_dev = phase - 2.0 * self._prev_phase + self._prev_prev_phase

            # Principal argument: wrap to [-π, π]
            phase_dev = (phase_dev + np.pi) % (2.0 * np.pi) - np.pi

            # Magnitude difference (half-wave rectified)
            mag_diff = np.maximum(mag - self._prev_magnitude, 0.0)

            # Phase difference (magnitude-weighted)
            phase_diff = np.abs(mag * np.sin(phase_dev))

            # Complex spectral difference
            raw_csd = np.sqrt(mag_diff ** 2 + phase_diff ** 2)

            # Subtract running mean to remove phase noise floor,
            # then half-wave rectify so only spikes remain
            if self._running_mean is None:
                self._running_mean = raw_csd.copy()
            else:
                self._running_mean += self._alpha * (raw_csd - self._running_mean)

            complex_flux = np.maximum(
                raw_csd - self._running_mean, 0.0
            ).astype(np.float32)
        else:
            complex_flux = np.zeros_like(mag)

        self._prev_prev_phase = self._prev_phase
        self._prev_phase = phase.copy()
        self._prev_magnitude = mag.copy()

        return SpectrumFrame(
            magnitude=frame.magnitude,
            flux=complex_flux,
            waveform=frame.waveform,
            phase=frame.phase,
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        self._prev_phase = None
        self._prev_prev_phase = None
        self._prev_magnitude = None
        self._running_mean = None


class PercussiveTransform(SpectrumTransform):
    """Extract percussive content — suppress harmonics.

    Uses sqrt(1 - mask) for flux to preserve signal in ambiguous bins.
    """

    def __init__(self, estimator=None) -> None:
        self._estimator = estimator

    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        if self._estimator is None:
            return frame
        self._estimator.update(frame.magnitude)
        return self.apply(frame, self._estimator.stability_per_bin())

    def apply(self, frame: SpectrumFrame, mask: np.ndarray) -> SpectrumFrame:
        """Apply a precomputed harmonic mask (inverted for percussive)."""
        pmask = 1.0 - mask
        return SpectrumFrame(
            magnitude=(frame.magnitude * pmask).astype(np.float32),
            flux=(frame.flux * np.sqrt(pmask)).astype(np.float32),
            waveform=frame.waveform,
            phase=frame.phase,
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        if self._estimator is not None:
            self._estimator.reset()
