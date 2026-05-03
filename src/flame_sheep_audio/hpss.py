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
