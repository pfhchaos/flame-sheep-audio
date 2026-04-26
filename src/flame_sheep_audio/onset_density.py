"""Onset density tracker — per-band onset rate as a continuous feature.

Tracks how many onsets per second are arriving in each band.
At normal tempos (120 BPM), kick density is ~2/s.  At speedcore
(270+ BPM), it's 4-5/s.  Axes can use this to smoothly blend
behavior rather than counting individual kicks.

Also provides per-band density_delta — the accelerando signal.
Positive = speeding up, negative = slowing down.
"""

from collections import deque

from .config import cfg
from ._band_config import BandConfig, default_band_config


class OnsetDensityTracker:
    """Multi-band onset density tracker.

    Call process_onset() for each detected onset, update() each frame.
    Read density properties for current values.
    """

    @property
    def WINDOW(self): return cfg.density.window
    @property
    def DELTA_WINDOW(self): return cfg.density.delta_window
    @property
    def ALPHA(self): return cfg.density.alpha

    def __init__(self, band_config: BandConfig | None = None):
        if band_config is None:
            band_config = default_band_config()
        self._band_names = band_config.detection_band_names
        self._times: dict[str, deque[float]] = {b: deque() for b in self._band_names}
        self._density: dict[str, float] = {b: 0.0 for b in self._band_names}
        # Per-band history for density delta (accelerando signal)
        self._delta_history: dict[str, deque[tuple[float, float]]] = {
            b: deque() for b in self._band_names
        }

    def reset(self):
        """Clear all state — call on song change."""
        for b in self._band_names:
            self._times[b].clear()
            self._density[b] = 0.0
            self._delta_history[b].clear()

    def process_onset(self, kind: str, timestamp: float):
        """Record an onset event."""
        if kind in self._times:
            self._times[kind].append(timestamp)

    def update(self, now: float):
        """Recompute densities. Call once per audio frame."""
        cutoff = now - self.WINDOW
        for band in self._band_names:
            times = self._times[band]
            while times and times[0] < cutoff:
                times.popleft()
            raw = len(times) / self.WINDOW
            self._density[band] = (self.ALPHA * self._density[band]
                                   + (1 - self.ALPHA) * raw)

            # Track density history for delta
            history = self._delta_history[band]
            history.append((now, self._density[band]))
            delta_cutoff = now - self.DELTA_WINDOW
            while history and history[0][0] < delta_cutoff:
                history.popleft()

    @property
    def densities(self) -> dict[str, float]:
        """Per-band onset densities (onsets/second)."""
        return dict(self._density)

    @property
    def density_deltas(self) -> dict[str, float]:
        """Per-band rate of change of onset density (accelerando signal)."""
        result = {}
        for band in self._band_names:
            history = self._delta_history[band]
            if len(history) < 2:
                result[band] = 0.0
            else:
                result[band] = self._density[band] - history[0][1]
        return result
