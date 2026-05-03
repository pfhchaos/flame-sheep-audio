"""BTrack tempo tracker — wraps the BTrack real-time beat tracking library.

BTrack (Adam Stark) is a causal beat tracking algorithm designed for
real-time use. It handles onset detection, tempo estimation, and beat
prediction internally.

Requires the _btrack native module (pybind11 wrapper around libBTrack).

See: https://github.com/adamstark/BTrack
"""

from __future__ import annotations

from .tempo import TempoTrackerBase
from ._constants import SAMPLE_RATE, HOP_SIZE
from .config import cfg

try:
    from ._btrack import BTrack
    _HAS_BTRACK = True
except ImportError:
    _HAS_BTRACK = False


class BTrackTempoTracker(TempoTrackerBase):
    """Real-time tempo tracking via BTrack.

    Accepts either raw audio frames (process_audio_frame) or
    onset detection function samples (feed). Using raw audio lets
    BTrack compute its own onset detection function, which it was
    tuned for.
    """

    def __init__(self, hop_size: int = HOP_SIZE,
                 sample_rate: int = SAMPLE_RATE) -> None:
        if not _HAS_BTRACK:
            raise ImportError("_btrack not available — build BTrack Python bindings")
        self._bt = BTrack(hop_size, hop_size * 2, sample_rate)
        self._hop_size = hop_size
        self._sample_rate = sample_rate
        self._beat_count = 0
        self._frame_count = 0

    def feed(self, onset_strength: float, onset_density: float = 0.0) -> None:
        """Feed one onset detection function sample."""
        self._bt.process_onset(onset_strength)
        self._frame_count += 1
        if self._bt.beat_due_in_current_frame():
            self._beat_count += 1

    def feed_audio(self, hop: 'np.ndarray') -> bool:
        """Feed one hop of raw audio. Returns True if a beat was detected.

        Preferred over feed() — lets BTrack use its own onset detection.
        """
        import numpy as np
        self._bt.process_audio_frame(hop.astype(np.float64))
        self._frame_count += 1
        beat = self._bt.beat_due_in_current_frame()
        if beat:
            self._beat_count += 1
        return beat

    @property
    def bpm(self) -> float:
        return self._bt.get_current_tempo_estimate()

    @property
    def effective_bpm(self) -> float:
        # BTrack's estimate is already smoothed internally
        return self._bt.get_current_tempo_estimate()

    @property
    def confidence(self) -> float:
        # BTrack doesn't expose confidence directly.
        # Use cumulative score as a proxy — higher = more confident.
        score = self._bt.get_latest_cumulative_score_value()
        return min(1.0, score / 100.0)

    @property
    def phase(self) -> float:
        # BTrack tracks beat timing internally via beat_due_in_current_frame.
        # We don't have a continuous phase estimate, so return 0.
        # TODO: compute from beat timing history
        return 0.0

    @property
    def locked(self) -> bool:
        """BTrack doesn't have a lock concept."""
        return False

    @property
    def saturated(self) -> bool:
        """BTrack doesn't saturate."""
        return False

    @property
    def bpm_delta(self) -> float:
        """Rate of tempo change — not tracked by BTrack."""
        return 0.0

    def reset(self) -> None:
        self._bt = BTrack(self._hop_size, self._hop_size * 2, self._sample_rate)
        self._beat_count = 0
        self._frame_count = 0
