"""Tests for the tempo scaler — beat-relative constant conversion.

Covers Beats/Percentile types, frames conversion, alpha (95% decay),
sigmoid blending, and smooth tempo tracking.
"""

from __future__ import annotations

import pytest

from flame_sheep_audio.tempo_scaler import (
    TempoScaler, Beats, Percentile, _FRAMES_PER_SECOND,
)


def _scaler_at(bpm: float) -> TempoScaler:
    """Create a TempoScaler converged to a specific BPM."""
    s = TempoScaler()
    for _ in range(300):
        s.update(bpm)
    return s


class TestBeatsType:

    def test_is_float(self):
        b = Beats(1.5)
        assert isinstance(b, float)
        assert float(b) == 1.5

    def test_arithmetic(self):
        assert Beats(0.5) * 2 == 1.0


class TestPercentileType:

    def test_is_float(self):
        p = Percentile(90)
        assert isinstance(p, float)
        assert float(p) == 90


class TestFrames:

    def test_120bpm_one_beat(self):
        s = _scaler_at(120)
        frames = s.frames(Beats(1.0))
        expected = round(0.5 * _FRAMES_PER_SECOND)
        assert frames == expected

    def test_60bpm_one_beat(self):
        s = _scaler_at(60)
        frames = s.frames(Beats(1.0))
        expected = round(1.0 * _FRAMES_PER_SECOND)
        assert frames == expected

    def test_faster_tempo_fewer_frames(self):
        slow = _scaler_at(60).frames(Beats(1.0))
        fast = _scaler_at(240).frames(Beats(1.0))
        assert fast < slow

    def test_fractional_beats(self):
        s = _scaler_at(120)
        full = s.frames(Beats(1.0))
        half = s.frames(Beats(0.5))
        assert abs(half - full / 2) <= 1  # within 1 frame of half

    def test_minimum_one_frame(self):
        s = _scaler_at(999)
        assert s.frames(Beats(0.001)) >= 1

    def test_accepts_plain_float(self):
        s = _scaler_at(120)
        assert s.frames(1.0) == s.frames(Beats(1.0))


class TestAlpha:

    def test_95_percent_decay(self):
        """After N frames, 95% should have decayed (5% remains)."""
        s = _scaler_at(120)
        a = s.alpha(Beats(1.0))
        n = s.frames(Beats(1.0))
        remaining = a ** n
        assert abs(remaining - 0.05) < 0.01

    def test_faster_tempo_lower_alpha(self):
        slow = _scaler_at(60).alpha(Beats(1.0))
        fast = _scaler_at(240).alpha(Beats(1.0))
        assert fast < slow

    def test_more_beats_higher_alpha(self):
        s = _scaler_at(120)
        short = s.alpha(Beats(0.5))
        long = s.alpha(Beats(2.0))
        assert long > short

    def test_reasonable_range(self):
        for bpm in [60, 120, 240]:
            s = _scaler_at(bpm)
            a = s.alpha(Beats(1.0))
            assert 0.5 < a < 0.999


class TestBlend:

    def test_midpoint_is_average(self):
        s = _scaler_at(120)
        result = s.blend(10.0, 20.0, midpoint=120.0)
        assert abs(result - 15.0) < 0.1

    def test_slow_tempo_returns_slow_val(self):
        s = _scaler_at(40)
        result = s.blend(10.0, 20.0, steepness=0.05, midpoint=120.0)
        assert abs(result - 10.0) < 0.5

    def test_fast_tempo_returns_fast_val(self):
        s = _scaler_at(300)
        result = s.blend(10.0, 20.0, steepness=0.05, midpoint=120.0)
        assert abs(result - 20.0) < 0.5

    def test_inverted_range(self):
        s = _scaler_at(120)
        result = s.blend(20.0, 4.0, midpoint=120.0)
        assert abs(result - 12.0) < 0.1


class TestSmoothing:

    def test_gradual_convergence(self):
        """Tempo changes should converge smoothly, not snap."""
        s = _scaler_at(120)
        frames_before = s.frames(Beats(1.0))
        # Sudden tempo change
        s.update(60.0)
        frames_after_one = s.frames(Beats(1.0))
        # Should not have fully converged after one update
        frames_target = _scaler_at(60).frames(Beats(1.0))
        assert frames_before < frames_after_one < frames_target

    def test_default_bpm(self):
        """Fresh scaler should default to 120 BPM."""
        s = TempoScaler()
        assert s.bpm == 120.0


class TestStaticConvenience:

    def test_beats_to_frames(self):
        s = TempoScaler()
        assert s.beats_to_frames(120, 1.0) == round(0.5 * _FRAMES_PER_SECOND)

    def test_alpha_for_beats(self):
        s = TempoScaler()
        a = s.alpha_for_beats(120, 1.0)
        n = round(0.5 * _FRAMES_PER_SECOND)
        remaining = a ** n
        assert abs(remaining - 0.05) < 0.01
