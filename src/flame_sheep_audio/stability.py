"""Per-bin magnitude stability for harmonic/percussive separation.

Two implementations:
  - _StabilityEMA: lightweight EMA variance tracker (O(N_BINS) per frame)
  - _StabilityMedian: causal median filter HPSS (O(N_BINS * K) per frame)

The median approach matches librosa's HPSS algorithm but runs causally
(only past frames). Uses a circular buffer of recent magnitude frames
and computes per-bin median along time (harmonic) and per-frame median
along frequency (percussive).

MagnitudeStability selects the implementation via config.
"""

from __future__ import annotations

import numpy as np

from ._constants import N_BINS
from .config import cfg


# ----------------------------------------------------------------
# EMA-based stability (original, lightweight)
# ----------------------------------------------------------------

class _StabilityEMA:
    """Single-timescale per-bin EMA variance tracker."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._mag_ema = np.zeros(N_BINS, dtype=np.float32)
        self._mag_var = np.zeros(N_BINS, dtype=np.float32)

    def update(self, magnitude: np.ndarray) -> None:
        diff = magnitude - self._mag_ema
        self._mag_ema = self._alpha * self._mag_ema + (1 - self._alpha) * magnitude
        diff2 = magnitude - self._mag_ema  # post-update residual (Welford)
        self._mag_var = self._alpha * self._mag_var + (1 - self._alpha) * diff * diff2

    def band_stability(self, mask: np.ndarray) -> float:
        """0.0 = transient, 1.0 = stable/harmonic."""
        if not mask.any():
            return 1.0
        band_var = self._mag_var[mask].mean()
        band_mag = self._mag_ema[mask].mean()
        if band_mag < 1e-10:
            return 1.0
        cv = float(np.sqrt(band_var) / (band_mag + 1e-10))
        return 1.0 / (1.0 + cv)

    def stability_per_bin(self) -> np.ndarray:
        """Per-bin stability scores, 0..1. 1=harmonic, 0=transient."""
        cv = np.sqrt(self._mag_var) / (self._mag_ema + 1e-10)
        return (1.0 / (1.0 + cv)).astype(np.float32)

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


# ----------------------------------------------------------------
# Causal median filter HPSS
# ----------------------------------------------------------------

class _StabilityMedian:
    """Causal median-filter harmonic/percussive separation.

    Maintains a circular buffer of recent magnitude frames. Each frame:
    - Time median (per-bin median across buffer) → harmonic estimate
    - Frequency median (per-frame median across bins) → percussive estimate
    - Soft mask: H / (H + P + eps) → stability score

    This matches librosa's HPSS algorithm but uses only past frames
    (causal) instead of a symmetric window.
    """

    def __init__(self, kernel_time: int = 31, kernel_freq: int = 31) -> None:
        self._kt = kernel_time
        self._kf = kernel_freq
        self._buf = np.zeros((kernel_time, N_BINS), dtype=np.float32)
        self._pos = 0
        self._filled = 0
        self._harmonic_mask = np.full(N_BINS, 0.5, dtype=np.float32)

    def update(self, magnitude: np.ndarray) -> None:
        # Write new frame to circular buffer
        self._buf[self._pos] = magnitude
        self._pos = (self._pos + 1) % self._kt
        self._filled = min(self._filled + 1, self._kt)

        # Time median: per-bin median across buffer → harmonic
        active = self._buf[:self._filled] if self._filled < self._kt else self._buf
        h_median = np.median(active, axis=0)

        # Frequency median: sliding median along frequency for current frame
        # Use a simple approach: pad and compute with stride tricks
        half_k = self._kf // 2
        padded = np.pad(magnitude, half_k, mode='reflect')
        # Vectorized sliding window median
        windows = np.lib.stride_tricks.sliding_window_view(padded, self._kf)
        p_median = np.median(windows, axis=1).astype(np.float32)

        # Soft mask: harmonic / (harmonic + percussive + eps)
        total = h_median + p_median
        # Silence: both medians near zero → default to 1.0 (stable)
        self._harmonic_mask = np.where(
            total > 1e-10,
            h_median / (total + 1e-10),
            1.0,
        ).astype(np.float32)

    def band_stability(self, mask: np.ndarray) -> float:
        """Mean harmonic mask value in band."""
        if not mask.any():
            return 1.0
        return float(self._harmonic_mask[mask].mean())

    def stability_per_bin(self) -> np.ndarray:
        """Per-bin harmonic mask, 0..1. 1=harmonic, 0=percussive."""
        return self._harmonic_mask.copy()

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """RMS weighted by harmonic mask."""
        if not mask.any():
            return 0.0
        weighted = magnitude * self._harmonic_mask
        band = weighted[mask]
        return float(np.sqrt(np.mean(band ** 2)))

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos = 0
        self._filled = 0
        self._harmonic_mask[:] = 0.5


# ----------------------------------------------------------------
# Public interface (selects implementation)
# ----------------------------------------------------------------

class MagnitudeStability:
    """Multi-timescale per-bin magnitude stability.

    Fast (~200ms EMA or ~330ms median): beat-level, used for detection
    threshold scaling and per-frame harmonic RMS.

    Slow (~2s EMA): section-level, tracks whether a band has been
    consistently harmonic or percussive over a longer period.

    Set stability.method = "median" in audio.toml to use causal
    median-filter HPSS instead of EMA variance.
    """

    def __init__(self, fast_alpha: float | None = None,
                 slow_alpha: float | None = None,
                 method: str | None = None) -> None:
        if fast_alpha is None:
            fast_alpha = cfg.stability.fast_alpha
        if slow_alpha is None:
            slow_alpha = cfg.stability.slow_alpha
        if method is None:
            method = getattr(cfg.stability, 'method', 'ema')

        self._method = method
        if method == 'median':
            kernel_time = getattr(cfg.stability, 'median_kernel_time', 31)
            kernel_freq = getattr(cfg.stability, 'median_kernel_freq', 31)
            self._fast = _StabilityMedian(kernel_time=kernel_time,
                                           kernel_freq=kernel_freq)
        else:
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

    def stability_per_bin(self) -> np.ndarray:
        """Per-bin stability scores (fast timescale), 0..1. 1=harmonic."""
        return self._fast.stability_per_bin()

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """Harmonic RMS using fast stability weighting."""
        return self._fast.harmonic_rms(magnitude, mask)

    def reset(self) -> None:
        self._fast.reset()
        self._slow.reset()
