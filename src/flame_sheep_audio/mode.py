"""Three-state mode machine: beat / energy / idle.

Determines how the visualization responds based on what the audio
engine is detecting:

  beat:   Music — all axes driven by onsets and density.
  energy: Non-music audio (speech, ambient, drones) —
          brightness/detail driven by slow envelope, genome drifts,
          palette/zoom suppressed.
  idle:   Silence — slow autonomous genome drift.

Uses spectral novelty (cosine distance from EMA spectral shape) as the
primary music/speech discriminator. Music is spectrally predictable
(low novelty), speech constantly shifts formants (high novelty).
ACF tempo confidence as secondary confirmation of rhythmic content.
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


# Transition timing (in frames at ~93fps HOP cadence)
BEAT_ENTER_FRAMES = 90       # ~1s of music detected -> enter beat
BEAT_EXIT_FRAMES = 900       # ~10s of non-music -> exit beat
ENERGY_TO_IDLE_FRAMES = 1800 # ~20s of silence -> enter idle

# Spectral novelty thresholds (from corpus analysis 2026-04-30):
#   music:  mean 0.09-0.20
#   speech: mean 0.22-0.29
#   noise:  mean 0.11-0.15
NOVELTY_SPEECH = 0.20        # above this = likely speech -> energy mode
NOVELTY_MUSIC = 0.16         # below this = likely music -> beat mode (hysteresis)
ACF_CONFIDENCE_ENTER = 0.7   # ACF confidence to enter beat (rhythmic non-percussive music)
RMS_THRESHOLD = 0.0001       # broadband RMS below this = silence


class ModeDetector:
    """Determines the current audio mode from continuous features.

    Does NOT own any genome/morph state — that stays in DriftMode and
    GenomeAxis. This class only decides which mode is active.
    """

    def __init__(self) -> None:
        self.mode = Mode.IDLE
        self._novelty_ema = 0.0    # smoothed spectral novelty
        self._novelty_alpha = 0.95 # EMA smoothing

        # Frame counters for hysteresis
        self._beat_frames = 0      # frames of music-like audio
        self._quiet_frames = 0     # frames of non-music (in beat mode)
        self._silence_frames = 0   # frames of silence

    def tick(self, audio: AudioState) -> Mode:
        """Update mode from current audio state. Returns the new mode."""
        old_mode = self.mode

        # Smooth spectral novelty
        self._novelty_ema = (self._novelty_alpha * self._novelty_ema
                             + (1 - self._novelty_alpha) * audio.spectral_novelty)

        # Broadband RMS
        max_rms = max(b.rms for b in audio.bands.values())

        is_silent = max_rms < RMS_THRESHOLD

        # Music detection: low spectral novelty (predictable spectrum)
        # OR strong tempo confidence (rhythmic content even if novelty is moderate)
        if self.mode == Mode.BEAT:
            # Stay in beat: wider threshold (hysteresis)
            is_music = (self._novelty_ema < NOVELTY_SPEECH
                        or audio.tempo_confidence > ACF_CONFIDENCE_ENTER)
        else:
            # Enter beat: stricter threshold
            is_music = (self._novelty_ema < NOVELTY_MUSIC
                        or (self._novelty_ema < NOVELTY_SPEECH
                            and audio.tempo_confidence > ACF_CONFIDENCE_ENTER))

        # Update counters
        if is_silent:
            self._silence_frames += 1
            self._beat_frames = 0
            self._quiet_frames += 1
        else:
            self._silence_frames = 0
            if is_music:
                self._beat_frames += 1
                self._quiet_frames = 0
            else:
                self._beat_frames = 0
                self._quiet_frames += 1

        # State transitions
        if self.mode == Mode.IDLE:
            if not is_silent:
                if is_music:
                    self.mode = Mode.BEAT
                else:
                    self.mode = Mode.ENERGY

        elif self.mode == Mode.ENERGY:
            if self._beat_frames >= BEAT_ENTER_FRAMES:
                self.mode = Mode.BEAT
            elif self._silence_frames >= ENERGY_TO_IDLE_FRAMES:
                self.mode = Mode.IDLE

        elif self.mode == Mode.BEAT:
            if is_silent and self._silence_frames >= ENERGY_TO_IDLE_FRAMES:
                self.mode = Mode.IDLE
            elif self._quiet_frames >= BEAT_EXIT_FRAMES:
                if is_silent:
                    self.mode = Mode.IDLE
                else:
                    self.mode = Mode.ENERGY

        if self.mode != old_mode:
            log.info(f'[mode] {old_mode.value} -> {self.mode.value}  '
                     f'novelty={self._novelty_ema:.3f} acf={audio.tempo_confidence:.3f} '
                     f'rms={max_rms:.6f}')

        return self.mode

    def reset(self) -> None:
        """Reset to idle — call on song change."""
        self.mode = Mode.IDLE
        self._novelty_ema = 0.0
        self._beat_frames = 0
        self._quiet_frames = 0
        self._silence_frames = 0
