"""Tempo-adaptive constant scaling.

Converts musical durations (in beats) to frame counts and EMA alphas
based on the current tempo. Constants expressed in beats adapt
automatically when tempo changes.

Two unit types prevent mixing up beats, frames, and alphas:
  Beats(0.4)     — 40% of one beat (musical time)
  Percentile(90) — 90th percentile of recent signal

TempoScaler converts Beats to frames or alphas each tick, smoothing
tempo changes to avoid snapping.

Usage:
    scaler = TempoScaler()

    # Each frame (in audio loop):
    scaler.update(effective_bpm)

    # Convert beat-relative constants:
    cooldown = scaler.frames(Beats(0.4))    # frames for 40% of a beat
    alpha = scaler.alpha(Beats(1.0))        # EMA alpha for 1-beat 95% decay
"""

from __future__ import annotations

import numpy as np

from ._constants import SAMPLE_RATE, HOP_SIZE


# Frames per hop at our sample rate
_FRAMES_PER_SECOND = SAMPLE_RATE / HOP_SIZE


class Beats(float):
    """A duration measured in beats, not seconds or frames.

    Used in config to express tempo-relative constants:
        cooldown = Beats(0.4)       # 40% of one beat
        stability_window = Beats(1) # one beat until 95% decay
    """
    pass


class Percentile(float):
    """A threshold expressed as a percentile of recent signal distribution.

    Used in config to express headroom-adaptive thresholds:
        threshold = Percentile(90)  # fire at 90th percentile of recent flux
    """
    pass


class TempoScaler:
    """Converts beat-relative constants to frame counts and EMA alphas.

    Updates each frame with the current effective_bpm. Outputs
    change smoothly — no snapping on tempo changes.
    """

    def __init__(self, smooth_alpha: float = 0.95) -> None:
        self._bpm: float = 120.0
        self._beat_frames: float = _FRAMES_PER_SECOND * 60.0 / 120.0
        self._smooth_alpha = smooth_alpha

    def update(self, effective_bpm: float) -> None:
        """Update with current tempo. Call once per frame."""
        if effective_bpm <= 0:
            return
        target = _FRAMES_PER_SECOND * 60.0 / effective_bpm
        self._beat_frames = (self._smooth_alpha * self._beat_frames
                             + (1 - self._smooth_alpha) * target)
        self._bpm = effective_bpm

    @property
    def beat_frames(self) -> float:
        """Current frames per beat (smoothed)."""
        return self._beat_frames

    @property
    def bpm(self) -> float:
        """Current BPM being used for scaling."""
        return self._bpm

    def frames(self, beats: Beats | float) -> int:
        """Convert a duration in beats to frame count.

        Beats(1.0) at 120 BPM ≈ 47 frames
        Beats(0.4) at 120 BPM ≈ 19 frames
        """
        return max(1, int(round(float(beats) * self._beat_frames)))

    def alpha(self, beats: Beats | float) -> float:
        """Convert a decay window in beats to an EMA alpha.

        The alpha is computed so that after `beats` beats worth of
        frames, 95% of the original value has decayed (5% remains).

        Beats(1.0) at 120 BPM → alpha ≈ 0.936 (47 frames to 95% decay)
        Beats(1.0) at 60 BPM  → alpha ≈ 0.968 (93 frames to 95% decay)
        """
        n_frames = max(1.0, float(beats) * self._beat_frames)
        # alpha^n = 0.05 → alpha = 0.05^(1/n)
        return float(np.power(0.05, 1.0 / n_frames))

    def blend(self, slow_val: float, fast_val: float,
              steepness: float = 0.03, midpoint: float = 120.0) -> float:
        """Sigmoid blend between slow and fast values based on current tempo.

        Smooth S-curve: returns ~slow_val at low BPM, ~fast_val at high BPM,
        midpoint between them at `midpoint` BPM. Continuous and differentiable.

        Args:
            slow_val: value at very slow tempos
            fast_val: value at very fast tempos
            steepness: how sharp the transition is (0.03 = gentle)
            midpoint: BPM where the blend is 50/50 (default 120)
        """
        import math
        t = 1.0 / (1.0 + math.exp(-steepness * (self._bpm - midpoint)))
        return slow_val + (fast_val - slow_val) * t

    # --- Convenience for non-Beats usage ---

    def beats_to_frames(self, bpm: float, beats: float) -> int:
        """Static conversion without internal state. For eval harnesses."""
        if bpm <= 0:
            bpm = 120.0
        beat_frames = _FRAMES_PER_SECOND * 60.0 / bpm
        return max(1, int(round(beats * beat_frames)))

    def alpha_for_beats(self, bpm: float, beats: float) -> float:
        """Static conversion without internal state. For eval harnesses."""
        frames = self.beats_to_frames(bpm, beats)
        return float(np.power(0.05, 1.0 / max(1, frames)))
