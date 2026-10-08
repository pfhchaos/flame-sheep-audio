"""Tests for beat-rate octave correction of the tempo tracker.

Covers the BeatRateEstimator + reconcile_tempo design that fixes the
classic half/double/triplet octave-error failure mode in periodicity-
based tempo trackers. See tempo_reconcile.py for the rationale.
"""
from __future__ import annotations

import math

import pytest

from flame_sheep_audio.tempo_reconcile import (
    BeatRateEstimator,
    ReconciliationResult,
    reconcile_tempo,                 # alias for octave_shift
    reconcile_tempo_override,        # strategy A
    reconcile_tempo_octave_shift,    # strategy B
    reconcile_tempo_octave_only,     # strategy C
)


# ---------------------------------------------------------------------------
# BeatRateEstimator
# ---------------------------------------------------------------------------

class TestBeatRateEstimator:

    def test_no_estimate_before_two_events(self):
        e = BeatRateEstimator()
        assert not e.has_estimate
        assert e.beat_rate_bpm is None
        e.feed('mid', 0.0)
        assert not e.has_estimate  # 1 event = 0 IBIs

    def test_estimate_from_two_events(self):
        e = BeatRateEstimator()
        e.feed('mid', 0.0)
        e.feed('mid', 0.5)  # IBI = 0.5s = 120 BPM
        assert e.has_estimate
        assert e.beat_rate_bpm == pytest.approx(120.0)

    def test_median_filters_one_outlier(self):
        """A single late beat (e.g. dropped pulse) shouldn't move the
        median much — that's the whole point of using median over mean."""
        e = BeatRateEstimator(window_size=6)
        # 5 beats at 120 BPM (0.5s spacing), then one missed beat
        # (skip a slot, so the gap is 1.0s)
        ts = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0]
        # IBIs: 0.5, 0.5, 0.5, 1.0, 0.5  → median = 0.5
        for t in ts:
            e.feed('mid', t)
        assert e.beat_rate_bpm == pytest.approx(120.0)

    def test_onset_events_ignored(self):
        """Onset (high) events shouldn't pollute the beat-rate estimate;
        they're subdivisions, not beats."""
        e = BeatRateEstimator()
        e.feed('mid', 0.0)
        # Onset firing in the middle would corrupt the IBI if counted
        e.feed('high', 0.1)
        e.feed('high', 0.2)
        e.feed('high', 0.3)
        e.feed('mid', 0.5)
        assert e.beat_rate_bpm == pytest.approx(120.0)  # not 600 from onset spam

    def test_downbeat_events_count(self):
        """Both downbeat (low) and beat (mid) are beat-level events."""
        e = BeatRateEstimator()
        e.feed('low', 0.0)
        e.feed('mid', 0.5)
        e.feed('mid', 1.0)
        e.feed('low', 1.5)
        # All IBIs = 0.5 → 120 BPM
        assert e.beat_rate_bpm == pytest.approx(120.0)

    def test_monotonicity_violation_dropped(self):
        """A backward timestamp (clock jitter) shouldn't produce a
        negative IBI."""
        e = BeatRateEstimator()
        e.feed('mid', 0.0)
        e.feed('mid', 1.0)
        e.feed('mid', 0.5)  # backward — drop
        e.feed('mid', 1.5)
        # Effective sequence: 0.0, 1.0, 1.5 → IBIs 1.0, 0.5 → median 0.75
        assert e.beat_rate_bpm == pytest.approx(60.0 / 0.75)

    def test_stable_when_ibis_consistent(self):
        e = BeatRateEstimator()
        for i in range(6):
            e.feed('mid', i * 0.5)
        assert e.stable

    def test_unstable_under_rubato(self):
        """Wildly varying IBIs → high CoV → unstable."""
        e = BeatRateEstimator(stability_cov=0.20)
        # Alternating fast/slow beats — CoV will be ~0.33
        ts = [0.0, 0.3, 0.6, 1.2, 1.5, 2.1]
        for t in ts:
            e.feed('mid', t)
        assert not e.stable

    def test_window_aging(self):
        """When window fills, oldest events drop — recent tempo wins."""
        e = BeatRateEstimator(window_size=4)
        # Start at 60 BPM (1s spacing)
        for t in [0.0, 1.0, 2.0]:
            e.feed('mid', t)
        # Switch to 120 BPM (0.5s spacing) for many beats
        for t in [2.5, 3.0, 3.5, 4.0, 4.5]:
            e.feed('mid', t)
        # Window now holds the most recent 4 timestamps: [3.0, 3.5, 4.0, 4.5]
        # IBIs: 0.5, 0.5, 0.5 → 120 BPM
        assert e.beat_rate_bpm == pytest.approx(120.0)

    def test_reset_clears_state(self):
        e = BeatRateEstimator()
        e.feed('mid', 0.0)
        e.feed('mid', 0.5)
        assert e.has_estimate
        e.reset()
        assert not e.has_estimate
        assert e.beat_rate_bpm is None

    def test_window_size_validation(self):
        with pytest.raises(ValueError):
            BeatRateEstimator(window_size=2)


# ---------------------------------------------------------------------------
# reconcile_tempo
# ---------------------------------------------------------------------------

def _stable_estimator(bpm: float, n: int = 6) -> BeatRateEstimator:
    """Build an estimator pre-fed with `n` perfectly-spaced events at `bpm`."""
    e = BeatRateEstimator(window_size=n)
    interval = 60.0 / bpm
    for i in range(n):
        e.feed('mid', i * interval)
    assert e.stable
    assert e.beat_rate_bpm == pytest.approx(bpm)
    return e


class TestSharedBehavior:
    """Behavior shared by both strategies — fallback, dataclass shape, etc."""

    def test_unstable_estimator_falls_back_to_tracker(self):
        e = BeatRateEstimator()
        e.feed('mid', 0.0)  # only 1 event, no estimate
        for fn in (reconcile_tempo_override, reconcile_tempo_octave_shift):
            result = fn(tracker_bpm=140.0, beat_estimator=e)
            assert result.bpm == 140.0
            assert not result.trustable

    def test_returns_dataclass(self):
        e = _stable_estimator(120.0)
        for fn in (reconcile_tempo_override, reconcile_tempo_octave_shift):
            result = fn(tracker_bpm=120.0, beat_estimator=e)
            assert isinstance(result, ReconciliationResult)


class TestOverride:
    """Strategy A: snap to beat_rate when off; use tracker only on agreement."""

    def test_tracker_in_tolerance_keeps_precision(self):
        e = _stable_estimator(120.0)
        result = reconcile_tempo_override(tracker_bpm=121.5, beat_estimator=e)
        assert result.bpm == pytest.approx(121.5)  # tracker preserved
        assert result.tracker_agreed

    def test_half_tempo_tracker_overridden(self):
        """tracker=70, truth=140. Strategy A returns 140 (truth)."""
        e = _stable_estimator(140.0)
        result = reconcile_tempo_override(tracker_bpm=70.0, beat_estimator=e)
        assert result.bpm == pytest.approx(140.0)
        assert not result.tracker_agreed
        # No precision preserved — pure beat_rate value
        assert result.octave_factor == pytest.approx(1.0)

    def test_double_tempo_tracker_overridden(self):
        e = _stable_estimator(120.0)
        result = reconcile_tempo_override(tracker_bpm=240.0, beat_estimator=e)
        assert result.bpm == pytest.approx(120.0)
        assert not result.tracker_agreed


class TestOctaveShift:
    """Strategy B: shift tracker by best ratio so it matches beat_rate octave."""

    def test_tracker_in_tolerance_keeps_precision(self):
        e = _stable_estimator(120.0)
        result = reconcile_tempo_octave_shift(tracker_bpm=121.5, beat_estimator=e)
        # ratio=1.0, so tracker pass-through with full precision
        assert result.bpm == pytest.approx(121.5)
        assert result.octave_factor == pytest.approx(1.0)
        assert result.tracker_agreed

    def test_half_tempo_tracker_shifted_up(self):
        """tracker=70, truth=140. Strategy B shifts: 70 × 2 = 140."""
        e = _stable_estimator(140.0)
        result = reconcile_tempo_octave_shift(tracker_bpm=70.0, beat_estimator=e)
        assert result.bpm == pytest.approx(140.0)
        assert result.octave_factor == pytest.approx(2.0)
        assert not result.tracker_agreed

    def test_half_tempo_with_noise_preserves_precision(self):
        """Tracker at 70.3 (close to half of 140), beats at 140.
        Strategy B shifts by 2 → 140.6. Tracker's 0.3 jitter preserved
        (this is the key win over strategy A, which would return 140)."""
        e = _stable_estimator(140.0)
        result = reconcile_tempo_octave_shift(tracker_bpm=70.3, beat_estimator=e)
        assert result.bpm == pytest.approx(140.6)
        assert result.octave_factor == pytest.approx(2.0)

    def test_double_tempo_tracker_shifted_down(self):
        e = _stable_estimator(120.0)
        result = reconcile_tempo_octave_shift(tracker_bpm=240.0, beat_estimator=e)
        assert result.bpm == pytest.approx(120.0)
        assert result.octave_factor == pytest.approx(0.5)

    def test_triplet_octave_handled(self):
        """tracker=50, truth=150 (musical 3:1 e.g. shuffle).
        Strategy B shifts by 3."""
        e = _stable_estimator(150.0)
        result = reconcile_tempo_octave_shift(tracker_bpm=50.0, beat_estimator=e)
        assert result.bpm == pytest.approx(150.0)
        assert result.octave_factor == pytest.approx(3.0)

    def test_zero_tracker_bpm_handled(self):
        """Degenerate: tracker uninitialized. Default to no shift."""
        e = _stable_estimator(120.0)
        result = reconcile_tempo_octave_shift(tracker_bpm=0.0, beat_estimator=e)
        # tracker × 1.0 = 0 — degenerate output but at least no crash
        assert result.bpm == 0.0
        assert result.octave_factor == pytest.approx(1.0)


class TestOctaveOnly:
    """Strategy C: only act on ×2 / ×0.5; pass through otherwise."""

    def test_half_tempo_corrected(self):
        e = _stable_estimator(140.0)
        result = reconcile_tempo_octave_only(tracker_bpm=70.0, beat_estimator=e)
        assert result.bpm == pytest.approx(140.0)
        assert result.octave_factor == pytest.approx(2.0)

    def test_double_tempo_corrected(self):
        e = _stable_estimator(120.0)
        result = reconcile_tempo_octave_only(tracker_bpm=240.0, beat_estimator=e)
        assert result.bpm == pytest.approx(120.0)
        assert result.octave_factor == pytest.approx(0.5)

    def test_triplet_NOT_corrected(self):
        """Strategy C ignores triplet (×3) ratios — beats probably wrong."""
        e = _stable_estimator(150.0)
        # Tracker=50, beats=150 → strategy B would shift ×3 to 150.
        # Strategy C: shift=3.0 isn't in {0.5, 2.0} → pass through tracker.
        result = reconcile_tempo_octave_only(tracker_bpm=50.0, beat_estimator=e)
        assert result.bpm == pytest.approx(50.0)  # tracker preserved
        assert result.octave_factor == pytest.approx(1.0)

    def test_non_octave_NOT_corrected(self):
        """When beats are nonsensical (~1.5× tracker), don't act.
        This is the eval-observed failure mode that motivates strategy C."""
        e = _stable_estimator(216.0)  # beats over-fire on subdivisions
        # Tracker says 140 (correct). Best octave shift would be 2.0
        # (giving 280, off from 216 by 30%). C's tolerance check rejects.
        result = reconcile_tempo_octave_only(tracker_bpm=140.0, beat_estimator=e)
        # The shift selected is 2.0 (closest of {0.5,2.0,3.0,...} to ratio 216/140≈1.54),
        # but 140×2=280 is 30% off from 216 → tolerance check fails → pass through.
        assert result.bpm == pytest.approx(140.0)  # tracker preserved
        assert result.octave_factor == pytest.approx(1.0)

    def test_tracker_already_correct_passthrough(self):
        e = _stable_estimator(120.0)
        result = reconcile_tempo_octave_only(tracker_bpm=121.5, beat_estimator=e)
        assert result.bpm == pytest.approx(121.5)
        assert result.tracker_agreed


class TestStrategyComparison:
    """Side-by-side: same inputs, document tradeoffs."""

    def test_octave_correct_jitter_preserved_by_B_lost_by_A(self):
        """When tracker is at half of truth with sub-BPM jitter, strategy B
        preserves the jitter (scaled), strategy A discards it."""
        e = _stable_estimator(140.0)
        a = reconcile_tempo_override(tracker_bpm=70.5, beat_estimator=e)
        b = reconcile_tempo_octave_shift(tracker_bpm=70.5, beat_estimator=e)
        assert a.bpm == pytest.approx(140.0)   # exact beat_rate, no jitter
        assert b.bpm == pytest.approx(141.0)   # tracker's 0.5 jitter shifted to 1.0

    def test_bad_beat_rate_C_passes_through_AB_corrupt(self):
        """Subdivision-driven over-fire: beats fire at ~1.5× true tempo.
        A snaps to (wrong) beat_rate; B shifts (wrongly); C does nothing."""
        e = _stable_estimator(216.0)  # spurious ~1.5× rate
        a = reconcile_tempo_override(tracker_bpm=140.0, beat_estimator=e)
        b = reconcile_tempo_octave_shift(tracker_bpm=140.0, beat_estimator=e)
        c = reconcile_tempo_octave_only(tracker_bpm=140.0, beat_estimator=e)
        assert a.bpm == pytest.approx(216.0)   # A corrupts to false beat_rate
        # B picks ratio 2.0 → 280, off from 216 but it's the "closest"
        assert b.bpm == pytest.approx(280.0)
        assert c.bpm == pytest.approx(140.0)   # C correctly does nothing

    def test_default_alias_is_octave_shift(self):
        """The bare `reconcile_tempo` name uses strategy B."""
        assert reconcile_tempo is reconcile_tempo_octave_shift
