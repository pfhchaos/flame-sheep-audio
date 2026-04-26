"""Bass break detector — detects sub-bass energy dropout.

Complements the EDM DropDetector which tracks full centroid_rms silence.
This detector catches pop/rock-style breaks where kick and bass disappear
but vocals and mids continue — the sub-bass vanishes while centroid_rms
stays high.

Exposes a continuous `breaking` state that drives exponential morph
slowdown in the visual axes.
"""

import logging

from ._types import BeatEvent
from .config import cfg

log = logging.getLogger(__name__)


class BassDropDetector:
    """Detect breaks from sub-bass energy dropout.

    Call detect() each frame. Read .breaking for current state.
    """

    @property
    def QUIET_THRESHOLD_FRAMES(self): return cfg.breaks.bass_quiet_threshold_frames
    @property
    def MIN_KICKS_BEFORE_DROP(self): return cfg.breaks.min_kicks_before_break
    @property
    def BREAK_COOLDOWN(self): return cfg.breaks.cooldown_seconds
    @property
    def SUBBASS_DROP_RATIO(self): return cfg.breaks.subbass_drop_ratio

    def __init__(self):
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._subbass_avg = 0.0
        self._warmup_frames = 0
        self.breaking = False
        self._break_start_frame = 0

    def reset(self):
        """Reset state — call on song change."""
        self._quiet_frames = 0
        self._total_kicks = 0
        self._cooldown = 0.0
        self._subbass_avg = 0.0
        self.breaking = False
        self._break_start_frame = 0

    def detect(self, events: list[BeatEvent], subbass_rms: float,
               bpm: float, drifting: bool, dt: float) -> None:
        """Update break state. Read .breaking for current state.

        Args:
            events: beat events this frame
            subbass_rms: current 20-200Hz RMS energy (snap.rms)
            bpm: current tempo estimate (0 if unknown)
            drifting: True if drift mode is active
            dt: frame time delta
        """
        self._subbass_avg = 0.97 * self._subbass_avg + 0.03 * subbass_rms
        self._warmup_frames += 1

        if self._cooldown > 0:
            self._cooldown -= dt

        if self._warmup_frames < 300:
            return

        is_bass_quiet = (subbass_rms < self._subbass_avg * self.SUBBASS_DROP_RATIO
                         and self._subbass_avg > 1e-6)

        has_kick = any(e.kind == 'kick' for e in events)

        if has_kick:
            self._total_kicks += 1
            if self.breaking:
                log.info(f'[BASS BREAK END] after {self._quiet_frames} quiet frames')
                self._cooldown = self.BREAK_COOLDOWN
                self.breaking = False
            self._quiet_frames = 0
        elif is_bass_quiet:
            self._quiet_frames += 1
        else:
            if self.breaking:
                self._cooldown = self.BREAK_COOLDOWN
                self.breaking = False
            self._quiet_frames = 0

        if (not self.breaking
                and self._quiet_frames >= self.QUIET_THRESHOLD_FRAMES
                and self._total_kicks > self.MIN_KICKS_BEFORE_DROP
                and self._cooldown <= 0
                and not drifting):
            self.breaking = True
            self._break_start_frame = self._quiet_frames
            log.info(f'[BASS BREAK] subbass_rms={subbass_rms:.4f} '
                     f'avg={self._subbass_avg:.4f}')

    @property
    def break_intensity(self) -> float:
        """0.0 = not breaking, ramps to 1.0 over ~1s after activation."""
        if not self.breaking:
            return 0.0
        frames_since = self._quiet_frames - self._break_start_frame
        return min(1.0, frames_since / 60.0)
