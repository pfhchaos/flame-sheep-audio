"""Per-bin magnitude stability for harmonic/percussive separation.

Three implementations:
  - _StabilityEMA: lightweight EMA variance tracker (O(N_BINS) per frame)
  - _StabilityMedian: causal median filter HPSS (O(N_BINS * K) per frame)
  - _StabilityShape: local cosine similarity HPSS (vibrato-tolerant)

The median approach matches librosa's HPSS algorithm but runs causally
(only past frames). Uses a circular buffer of recent magnitude frames
and computes per-bin median along time (harmonic) and per-frame median
along frequency (percussive).

MagnitudeStability selects the implementation via config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ._constants import N_BINS
from .config import cfg


class StabilityMethod(ABC):
    """Interface for per-bin stability / HPSS methods.

    Implementations must:
      - Accept any bin count (no hardcoded N_BINS)
      - Handle silence (all-zero magnitude) on any frame including first
      - Return arrays matching the input magnitude size after update()
      - Support reset() for engine hotswapping (new bin count after reset)
    """

    @abstractmethod
    def update(self, magnitude: np.ndarray) -> None:
        """Update internal state from a new magnitude frame."""
        ...

    @abstractmethod
    def stability_per_bin(self) -> np.ndarray:
        """Per-bin stability scores, shape matching last update input.

        Returns float32 array, values in [0, 1].
        1 = harmonic/stable, 0 = percussive/transient.
        """
        ...

    @abstractmethod
    def band_stability(self, mask: np.ndarray) -> float:
        """Mean stability over masked bins. 0=transient, 1=stable."""
        ...

    @abstractmethod
    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """RMS weighted by stability — only sustained content contributes."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all state. Must accept different bin count after reset."""
        ...


# ----------------------------------------------------------------
# EMA-based stability (original, lightweight)
# ----------------------------------------------------------------

class _StabilityEMA(StabilityMethod):
    """Single-timescale per-bin EMA variance tracker."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._n_bins: int = 0
        self._mag_ema: np.ndarray | None = None
        self._mag_var: np.ndarray | None = None
        self._cached_stability: np.ndarray | None = None

    def update(self, magnitude: np.ndarray) -> None:
        if self._mag_ema is None:
            self._n_bins = len(magnitude)
            self._mag_ema = np.zeros_like(magnitude)
            self._mag_var = np.zeros_like(magnitude)
        diff = magnitude - self._mag_ema
        self._mag_ema = self._alpha * self._mag_ema + (1 - self._alpha) * magnitude
        diff2 = magnitude - self._mag_ema  # post-update residual (Welford)
        self._mag_var = self._alpha * self._mag_var + (1 - self._alpha) * diff * diff2
        self._cached_stability = None  # invalidate cache

    def band_stability(self, mask: np.ndarray) -> float:
        """0.0 = transient, 1.0 = stable/harmonic."""
        if not mask.any() or self._mag_var is None:
            return 1.0
        band_var = self._mag_var[mask].mean()
        band_mag = self._mag_ema[mask].mean()
        if band_mag < 1e-10:
            return 1.0
        cv = float(np.sqrt(band_var) / (band_mag + 1e-10))
        return 1.0 / (1.0 + cv)

    def stability_per_bin(self) -> np.ndarray:
        """Per-bin stability scores, 0..1. 1=harmonic, 0=transient."""
        if self._cached_stability is not None:
            return self._cached_stability
        if self._mag_var is None:
            return np.full(self._n_bins or N_BINS, 0.5, dtype=np.float32)
        cv = np.sqrt(self._mag_var) / (self._mag_ema + 1e-10)
        self._cached_stability = (1.0 / (1.0 + cv)).astype(np.float32)
        return self._cached_stability

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """RMS weighted by stability — only sustained content contributes."""
        if not mask.any():
            return 0.0
        stability = self.stability_per_bin()
        weighted = magnitude * stability
        band = weighted[mask]
        return float(np.sqrt(np.mean(band ** 2)))

    def reset(self) -> None:
        self._n_bins = 0
        self._mag_ema = None
        self._mag_var = None
        self._cached_stability = None


# ----------------------------------------------------------------
# Causal median filter HPSS
# ----------------------------------------------------------------

class _StabilityMedian(StabilityMethod):
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
        self._buf: np.ndarray | None = None
        self._pos = 0
        self._filled = 0
        self._n_bins: int = 0
        self._harmonic_mask: np.ndarray | None = None
        self._sustained: np.ndarray | None = None  # time-median magnitude
        self._cached_stability: np.ndarray | None = None

    def update(self, magnitude: np.ndarray) -> None:
        n_bins = len(magnitude)
        if self._buf is None:
            self._n_bins = n_bins
            self._buf = np.zeros((self._kt, n_bins), dtype=np.float32)
            self._harmonic_mask = np.full(n_bins, 0.5, dtype=np.float32)
            self._sustained = np.zeros(n_bins, dtype=np.float32)
        # Write new frame to circular buffer
        self._buf[self._pos] = magnitude
        self._pos = (self._pos + 1) % self._kt
        self._filled = min(self._filled + 1, self._kt)

        # Time median: per-bin median across buffer → sustained magnitude
        active = self._buf[:self._filled] if self._filled < self._kt else self._buf
        h_median = np.median(active, axis=0)
        self._sustained = h_median.astype(np.float32)

        # Frequency median: sliding median along frequency for current frame
        half_k = self._kf // 2
        padded = np.pad(magnitude, half_k, mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, self._kf)
        p_median = np.median(windows, axis=1).astype(np.float32)

        # Soft mask: harmonic / (harmonic + percussive + eps)
        total = h_median + p_median
        self._harmonic_mask = np.where(
            total > 1e-10,
            h_median / (total + 1e-10),
            1.0,
        ).astype(np.float32)
        self._cached_stability = None

    def band_stability(self, mask: np.ndarray) -> float:
        """Mean harmonic mask value in band."""
        if not mask.any() or self._harmonic_mask is None:
            return 1.0
        return float(self._harmonic_mask[mask].mean())

    def stability_per_bin(self) -> np.ndarray:
        """Per-bin harmonic mask, 0..1. 1=harmonic, 0=percussive."""
        if self._cached_stability is not None:
            return self._cached_stability
        if self._harmonic_mask is None:
            return np.full(self._n_bins or N_BINS, 0.5, dtype=np.float32)
        self._cached_stability = self._harmonic_mask.copy()
        return self._cached_stability

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        """RMS of sustained (time-median) magnitude in band."""
        if not mask.any() or self._sustained is None:
            return 0.0
        band = self._sustained[mask]
        return float(np.sqrt(np.mean(band ** 2)))

    def sustained_magnitude(self) -> np.ndarray | None:
        """Per-bin sustained magnitude (time median). None before first update."""
        return self._sustained

    def reset(self) -> None:
        self._n_bins = 0
        self._buf = None
        self._pos = 0
        self._filled = 0
        self._harmonic_mask = None
        self._cached_stability = None


# ----------------------------------------------------------------
# Local shape projection HPSS
# ----------------------------------------------------------------

class _StabilityShape(StabilityMethod):
    """Local spectral shape projection for harmonic/percussive separation.

    Maintains an EMA of the normalized spectrum. For each frame, slides
    a window across both the current and EMA spectrum, computing cosine
    distance per position. Low local distance = harmonic (matches
    recent shape), high = percussive (new/different).

    Handles vibrato naturally — energy wobbling between adjacent bins
    still matches the local patch shape.
    """

    def __init__(self, alpha: float = 0.95, kernel: int = 7) -> None:
        self._alpha = alpha
        self._kernel = kernel
        self._shape_ema: np.ndarray | None = None
        self._harmonic_mask: np.ndarray | None = None
        self._n_bins: int = 0

    def update(self, magnitude: np.ndarray) -> None:
        if self._n_bins == 0:
            self._n_bins = len(magnitude)

        if self._shape_ema is None:
            self._shape_ema = magnitude.copy()
            self._harmonic_mask = np.full(self._n_bins, 0.5, dtype=np.float32)
            return

        # Sliding window cosine distance between current and EMA
        # Uses raw magnitudes so magnitude changes (attacks) are detected
        half_k = self._kernel // 2
        cur_pad = np.pad(magnitude, half_k, mode='reflect')
        ema_pad = np.pad(self._shape_ema, half_k, mode='reflect')

        cur_windows = np.lib.stride_tricks.sliding_window_view(cur_pad, self._kernel)
        ema_windows = np.lib.stride_tricks.sliding_window_view(ema_pad, self._kernel)

        # Per-position cosine similarity
        dots = np.sum(cur_windows * ema_windows, axis=1)
        cur_norms = np.sqrt(np.sum(cur_windows ** 2, axis=1))
        ema_norms = np.sqrt(np.sum(ema_windows ** 2, axis=1))
        cos_sim = dots / (cur_norms * ema_norms + 1e-10)

        # Convert similarity to stability mask (0 = percussive, 1 = harmonic)
        # Sigmoid sharpening: small drops in similarity → large drops in mask
        # Centers at 0.95 so anything below ~0.9 reads as fully percussive
        cos_clipped = np.clip(cos_sim, 0.0, 1.0)
        sharpness = 40.0
        center = 0.95
        self._harmonic_mask = (1.0 / (1.0 + np.exp(-sharpness * (cos_clipped - center)))).astype(np.float32)

        # Silence: if both patches are near-zero, default to harmonic
        n = len(magnitude)
        silent_bins = (cur_norms[:n] < 1e-8) & (ema_norms[:n] < 1e-8)
        self._harmonic_mask[silent_bins] = 1.0

        # Update EMA (no normalization — track raw magnitudes)
        self._shape_ema = self._alpha * self._shape_ema + (1 - self._alpha) * magnitude

    def band_stability(self, mask: np.ndarray) -> float:
        if not mask.any() or self._harmonic_mask is None:
            return 1.0
        return float(self._harmonic_mask[mask].mean())

    def stability_per_bin(self) -> np.ndarray:
        if self._harmonic_mask is None:
            return np.full(self._n_bins or N_BINS, 0.5, dtype=np.float32)
        return self._harmonic_mask.copy()

    def harmonic_rms(self, magnitude: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any() or self._harmonic_mask is None:
            return 0.0
        weighted = magnitude * self._harmonic_mask
        band = weighted[mask]
        return float(np.sqrt(np.mean(band ** 2)))

    def reset(self) -> None:
        self._n_bins = 0
        self._shape_ema = None
        self._harmonic_mask = None


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
            # Convert Beats to alpha at default 120 BPM
            from .tempo_scaler import TempoScaler
            _ts = TempoScaler()
            fast_alpha = _ts.alpha_for_beats(120.0, cfg.stability.hpss_ema_window)
        if slow_alpha is None:
            from .tempo_scaler import TempoScaler
            _ts = TempoScaler()
            slow_alpha = _ts.seconds_to_alpha(cfg.stability.slow_window)
        if method is None:
            method = getattr(cfg.stability, 'method', 'ema')

        self._method = method
        if method == 'median':
            # hpss_time_window is in Beats — convert to frames at default 120 BPM
            # Uses slowest expected tempo (60 BPM) for buffer sizing so there's
            # room to adapt. Actual median window adapts via filled count.
            from .tempo_scaler import TempoScaler
            _ts = TempoScaler()
            beats = getattr(cfg.stability, 'hpss_time_window', 1.0)
            kernel_time = _ts.beats_to_frames(60.0, float(beats))  # size for slow tempo
            kernel_freq = int(getattr(cfg.stability, 'hpss_freq_kernel', 15))
            self._fast = _StabilityMedian(kernel_time=kernel_time,
                                           kernel_freq=kernel_freq)
        elif method == 'shape':
            shape_kernel = getattr(cfg.stability, 'shape_kernel', 7)
            self._fast = _StabilityShape(alpha=fast_alpha, kernel=shape_kernel)
            # Note: shape method scored 0.832 HPSS sim (vs EMA 0.839, median 0.859).
            # Local cosine similarity is too permissive on smooth spectra.
            # Kept as option for experimentation but not recommended.
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
