"""Onset density tracker — per-band onset rate as a continuous feature.

Tracks how many onsets per second are arriving in each band.
At normal tempos (120 BPM), low-band density is ~2/s.  At speedcore
(270+ BPM), it's 4-5/s.  Axes can use this to smoothly blend
behavior rather than counting individual onsets.

Maintains two smoothing rates:
  - fast: for visualization output (Seconds(0.1), responsive)
  - slow: for tempo tracker octave disambiguation (Seconds(2.0), stable)

Also provides per-band density_delta — the accelerando signal.
Positive = speeding up, negative = slowing down.
"""

from __future__ import annotations

from collections import deque

from .config import cfg
from ._band_config import BandConfig, default_band_config


class OnsetDensityTracker:
    """Multi-band onset density tracker with dual smoothing.

    Call process_onset() for each detected onset, update() each frame.
    Read density properties for current values.
    """

    def __init__(self, band_config: BandConfig | None = None) -> None:
        if band_config is None:
            band_config = default_band_config()
        self._band_names = band_config.detection_band_names

        # Convert Seconds config to EMA alphas
        from .tempo_scaler import TempoScaler
        _ts = TempoScaler()
        self._fast_alpha = _ts.seconds_to_alpha(cfg.density.fast_smoothing)
        self._slow_alpha = _ts.seconds_to_alpha(cfg.density.slow_smoothing)
        self._window = cfg.density.window
        self._delta_window = cfg.density.delta_window

        self._times: dict[str, deque[float]] = {b: deque() for b in self._band_names}
        self._density_fast: dict[str, float] = {b: 0.0 for b in self._band_names}
        self._density_slow: dict[str, float] = {b: 0.0 for b in self._band_names}
        # Per-band history for density delta (accelerando signal)
        self._delta_history: dict[str, deque[tuple[float, float]]] = {
            b: deque() for b in self._band_names
        }

    def reset(self) -> None:
        """Clear all state — call on song change."""
        for b in self._band_names:
            self._times[b].clear()
            self._density_fast[b] = 0.0
            self._density_slow[b] = 0.0
            self._delta_history[b].clear()

    def process_onset(self, kind: str, timestamp: float) -> None:
        """Record an onset event."""
        if kind in self._times:
            self._times[kind].append(timestamp)

    def update(self, now: float) -> None:
        """Recompute densities. Call once per audio frame."""
        cutoff = now - self._window
        for band in self._band_names:
            times = self._times[band]
            while times and times[0] < cutoff:
                times.popleft()
            raw = len(times) / self._window

            # Dual smoothing
            self._density_fast[band] = (self._fast_alpha * self._density_fast[band]
                                        + (1 - self._fast_alpha) * raw)
            self._density_slow[band] = (self._slow_alpha * self._density_slow[band]
                                        + (1 - self._slow_alpha) * raw)

            # Track density history for delta
            history = self._delta_history[band]
            history.append((now, self._density_fast[band]))
            delta_cutoff = now - self._delta_window
            while history and history[0][0] < delta_cutoff:
                history.popleft()

    @property
    def densities(self) -> dict[str, float]:
        """Per-band onset densities (fast smoothing, for visualization)."""
        return dict(self._density_fast)

    @property
    def densities_slow(self) -> dict[str, float]:
        """Per-band onset densities (slow smoothing, for tempo tracker)."""
        return dict(self._density_slow)

    @property
    def density_deltas(self) -> dict[str, float]:
        """Per-band rate of change of onset density (accelerando signal)."""
        result = {}
        for band in self._band_names:
            history = self._delta_history[band]
            if len(history) < 2:
                result[band] = 0.0
            else:
                result[band] = self._density_fast[band] - history[0][1]
        return result
