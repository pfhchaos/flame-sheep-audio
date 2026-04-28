"""Per-bin magnitude variance tracking for harmonic/percussive separation.

Tracks rolling variance of FFT magnitude per frequency bin at two
timescales:
  - Fast (~200ms) — beat-level separation, used for detection thresholds
  - Slow (~2s) — section-level separation, tracks longer-term character

Bins with low temporal variance contain sustained harmonic content
(vocals, pads). Bins with high variance contain transient percussive
content (kicks, snares).

This is a lightweight alternative to full HPSS — no spectrogram
buffering, O(N_BINS) per frame, zero latency.
"""

from __future__ import annotations

import numpy as np

from ._constants import N_BINS
from .config import cfg


class _StabilityEMA:
    """Single-timescale per-bin EMA variance tracker."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._mag_ema = np.zeros(N_BINS, dtype=np.float32)
        self._mag_var = np.zeros(N_BINS, dtype=np.float32)

    def update(self, magnitude: np.ndarray) -> None:
        diff = magnitude - self._mag_ema
        self._mag_ema = self._alpha * self._mag_ema + (1 - self._alpha) * magnitude
        self._mag_var = self._alpha * self._mag_var + (1 - self._alpha) * diff * diff

    def band_stability(self, mask: np.ndarray) -> float:
        """0.0 = transient, 1.0 = stable/harmonic."""
        if not mask.any():
            return 1.0
        band_var = self._mag_var[mask].mean()
        band_mag = self._mag_ema[mask].mean()
        if band_mag < 1e-10:
            return 1.0
        cv = float(np.sqrt(band_var) / (band_mag + 1e-10))
        return max(0.0, min(1.0, 1.0 - cv))

    def stability_per_bin(self) -> np.ndarray:
        """Per-bin stability scores, 0..1. 1=harmonic, 0=transient."""
        cv = np.sqrt(self._mag_var) / (self._mag_ema + 1e-10)
        return np.clip(1.0 - cv, 0.0, 1.0).astype(np.float32)

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """RMS weighted by stability — only sustained content contributes."""
        if not mask.any():
            return 0.0
        stability = self.stability_per_bin()
        weighted = magnitude * stability
        band = weighted[mask]
        return float(np.sqrt(np.mean(band ** 2)))

    def reset(self) -> None:
        self._mag_ema[:] = 0.0
        self._mag_var[:] = 0.0


class MagnitudeStability:
    """Multi-timescale per-bin magnitude stability.

    Fast (~200ms window, alpha=0.95): beat-level, used for detection
    threshold scaling and per-frame harmonic RMS.

    Slow (~2s window, alpha=0.995): section-level, tracks whether a
    band has been consistently harmonic or percussive over a longer
    period. Useful for section change detection and mode blending.
    """

    def __init__(self, fast_alpha: float | None = None, slow_alpha: float | None = None) -> None:
        if fast_alpha is None:
            fast_alpha = cfg.stability.fast_alpha
        if slow_alpha is None:
            slow_alpha = cfg.stability.slow_alpha
        self._fast = _StabilityEMA(fast_alpha)
        self._slow = _StabilityEMA(slow_alpha)

    def update(self, magnitude: np.ndarray) -> None:
        self._fast.update(magnitude)
        self._slow.update(magnitude)

    def band_stability(self, mask: np.ndarray) -> float:
        """Fast stability — beat-level (for detection thresholds)."""
        return self._fast.band_stability(mask)

    def band_stability_slow(self, mask: np.ndarray) -> float:
        """Slow stability — section-level (for mode/character tracking)."""
        return self._slow.band_stability(mask)

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """Harmonic RMS using fast stability weighting."""
        return self._fast.harmonic_rms(magnitude, mask)

    def reset(self) -> None:
        self._fast.reset()
        self._slow.reset()
