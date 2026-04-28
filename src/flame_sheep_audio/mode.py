"""Three-state mode machine: beat / energy / idle.

Determines how the visualization responds based on what the audio
engine is detecting:

  beat:   Rhythmic music — all axes driven by onsets and density.
  energy: Audio present but no rhythm (speech, ambient, drones) —
          brightness/detail driven by slow envelope, genome drifts,
          palette/zoom suppressed.
  idle:   Silence — slow autonomous genome drift.

State transitions use percussiveness as the music/not-music
discriminator and broadband RMS for silence detection. Transitions
have asymmetric timing: fast entry to beat mode, slow exit.
"""

from __future__ import annotations

import logging
from enum import Enum

from ._types import AudioState

log = logging.getLogger(__name__)


class Mode(Enum):
    IDLE = 'idle'
    ENERGY = 'energy'
    BEAT = 'beat'


# Transition timing (in frames at ~60fps)
BEAT_ENTER_FRAMES = 30       # ~0.5s of high percussiveness -> enter beat
BEAT_EXIT_FRAMES = 900       # ~15s of low percussiveness -> exit beat
ENERGY_TO_IDLE_FRAMES = 1800 # ~30s of silence -> enter idle
IDLE_EXIT_FRAMES = 1         # instant exit from idle on any audio

# Thresholds (hysteresis: harder to enter beat than to stay)
PERC_ENTER = 0.5             # percussiveness to enter beat mode
PERC_EXIT = 0.35             # percussiveness to exit beat mode
ACF_CONFIDENCE_ENTER = 0.7   # ACF confidence to enter beat (rhythmic non-percussive music)
RMS_THRESHOLD = 0.0001       # broadband RMS below this = silence


class ModeDetector:
    """Determines the current audio mode from continuous features.

    Does NOT own any genome/morph state — that stays in DriftMode and
    GenomeAxis. This class only decides which mode is active.
    """

    def __init__(self) -> None:
        self.mode = Mode.IDLE
        self._perc_ema = 0.0    # smoothed percussiveness (start low = unknown)
        self._perc_alpha = 0.95 # EMA smoothing for percussiveness

        # Energy mode disabled pending better speech/music discrimination.
        # All non-silent audio goes to beat mode for now.
        self._energy_mode_enabled = False

        # Frame counters for hysteresis
        self._beat_frames = 0   # frames of high percussiveness
        self._quiet_frames = 0  # frames of low percussiveness (in beat mode)
        self._silence_frames = 0  # frames of silence

    def tick(self, audio: AudioState) -> Mode:
        """Update mode from current audio state. Returns the new mode."""
        old_mode = self.mode

        # Smooth percussiveness
        self._perc_ema = (self._perc_alpha * self._perc_ema
                          + (1 - self._perc_alpha) * audio.percussiveness)

        # Broadband RMS
        max_rms = max(b.rms for b in audio.bands.values())

        is_silent = max_rms < RMS_THRESHOLD
        # Hysteresis: harder to enter beat than to stay
        # ACF confidence as secondary: rhythmic non-percussive music
        # (acoustic guitar, piano) has strong periodicity but low percussiveness
        if self.mode == Mode.BEAT:
            # Stay in beat: percussiveness alone (drums), OR
            # lower percussiveness AND ACF confidence (rhythmic content).
            # Speech (~0.3 perc, ~0.4 acf) fails both conditions.
            PERC_STAY = 0.25
            is_percussive = (self._perc_ema > PERC_EXIT
                             or (self._perc_ema > PERC_STAY
                                 and audio.tempo_confidence > ACF_CONFIDENCE_ENTER))
        else:
            # Enter beat: percussiveness alone (drums), OR
            # both moderate percussiveness AND ACF confidence (rhythmic non-percussive)
            is_percussive = (self._perc_ema > PERC_ENTER
                             or (self._perc_ema > PERC_EXIT
                                 and audio.tempo_confidence > ACF_CONFIDENCE_ENTER))

        # Update counters
        if is_silent:
            self._silence_frames += 1
            self._beat_frames = 0
            self._quiet_frames += 1
        else:
            self._silence_frames = 0
            if is_percussive:
                self._beat_frames += 1
                self._quiet_frames = 0
            else:
                self._beat_frames = 0
                self._quiet_frames += 1

        # State transitions
        if self.mode == Mode.IDLE:
            if not is_silent:
                if self._energy_mode_enabled:
                    if audio.percussiveness > PERC_ENTER:
                        self.mode = Mode.BEAT
                    else:
                        self.mode = Mode.ENERGY
                else:
                    self.mode = Mode.BEAT

        elif self.mode == Mode.ENERGY:
            if self._beat_frames >= BEAT_ENTER_FRAMES:
                self.mode = Mode.BEAT
            elif self._silence_frames >= ENERGY_TO_IDLE_FRAMES:
                self.mode = Mode.IDLE

        elif self.mode == Mode.BEAT:
            if is_silent and self._silence_frames >= ENERGY_TO_IDLE_FRAMES:
                self.mode = Mode.IDLE
            elif self._energy_mode_enabled and self._quiet_frames >= BEAT_EXIT_FRAMES:
                if is_silent:
                    self.mode = Mode.IDLE
                else:
                    self.mode = Mode.ENERGY

        if self.mode != old_mode:
            log.info(f'[mode] {old_mode.value} -> {self.mode.value}  '
                     f'perc={self._perc_ema:.3f} acf={audio.tempo_confidence:.3f} '
                     f'rms={max_rms:.6f}')

        return self.mode

    def reset(self) -> None:
        """Reset to idle — call on song change."""
        self.mode = Mode.IDLE
        self._perc_ema = 0.0
        self._beat_frames = 0
        self._quiet_frames = 0
        self._silence_frames = 0
