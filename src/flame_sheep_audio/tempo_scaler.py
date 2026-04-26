"""Tempo-aware constant scaling via sigmoid mapping.

Converts fixed constants into tempo-dependent values. Slow music gets
slow values, fast music gets fast values, with a smooth S-curve
transition centered at a configurable midpoint (default 120 BPM).

Each constant family can have its own steepness parameter controlling
how sharp the slow→fast transition is.

Usage:
    scaler = TempoScaler()
    # At 120 BPM: returns midpoint between slow and fast
    # At 60 BPM: returns ~slow_val
    # At 240 BPM: returns ~fast_val
    cooldown = scaler.scale(effective_bpm, slow_val=20, fast_val=4)

    # Convert "N beats" to frames at current tempo
    frames = scaler.beats_to_frames(effective_bpm, beats=2.0)

    # Get an EMA alpha for "N beats of memory" at current tempo
    alpha = scaler.alpha_for_beats(effective_bpm, beats=2.0)
"""

import math

from ._constants import HOP_SIZE, SAMPLE_RATE

# Precompute for beats_to_frames
_FRAMES_PER_SECOND = SAMPLE_RATE / HOP_SIZE


class TempoScaler:
    """Maps constants between slow/fast values via sigmoid of BPM."""

    def __init__(self, midpoint: float = 120.0, steepness: float = 0.03):
        self.midpoint = midpoint
        self.steepness = steepness

    def sigmoid(self, bpm: float) -> float:
        """Raw sigmoid: 0 at slow tempos, 1 at fast tempos."""
        return 1.0 / (1.0 + math.exp(-self.steepness * (bpm - self.midpoint)))

    def scale(self, bpm: float, slow_val: float, fast_val: float,
              steepness: float = None) -> float:
        """Blend between slow_val and fast_val based on tempo.

        Args:
            bpm: effective BPM
            slow_val: value at very slow tempos
            fast_val: value at very fast tempos
            steepness: override instance steepness for this call
        """
        k = steepness if steepness is not None else self.steepness
        t = 1.0 / (1.0 + math.exp(-k * (bpm - self.midpoint)))
        return slow_val + (fast_val - slow_val) * t

    def beats_to_frames(self, bpm: float, beats: float) -> int:
        """Convert a duration in beats to frames at current tempo."""
        if bpm <= 0:
            bpm = self.midpoint
        seconds = beats * 60.0 / bpm
        return max(1, int(seconds * _FRAMES_PER_SECOND))

    def alpha_for_beats(self, bpm: float, beats: float) -> float:
        """EMA alpha that gives ~N beats of memory at current tempo.

        alpha = 1 - 1/(beats * frames_per_beat)
        """
        frames = self.beats_to_frames(bpm, beats)
        return max(0.5, min(0.999, 1.0 - 1.0 / frames))
