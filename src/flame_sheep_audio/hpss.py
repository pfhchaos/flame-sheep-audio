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
    absolute. A kick at -20dB and -40dB produce similar flux.
    """

    def __init__(self, gain: float = 1000.0) -> None:
        self._gain = gain
        self._norm = 1.0 / np.log1p(gain)  # normalize to ~0-1 range
        self._prev_log_mag: np.ndarray | None = None

    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        log_mag = (np.log1p(self._gain * frame.magnitude) * self._norm).astype(np.float32)

        # Recompute flux on log magnitudes
        if self._prev_log_mag is not None:
            log_flux = np.maximum(log_mag - self._prev_log_mag, 0.0).astype(np.float32)
        else:
            log_flux = np.zeros_like(log_mag)
        self._prev_log_mag = log_mag.copy()

        return SpectrumFrame(
            magnitude=log_mag,
            flux=log_flux,
            waveform=frame.waveform,
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
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        if self._estimator is not None:
            self._estimator.reset()


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
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        if self._estimator is not None:
            self._estimator.reset()
