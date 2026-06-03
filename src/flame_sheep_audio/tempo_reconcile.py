"""Beat-rate octave correction for the tempo tracker.

The periodicity-based tempo tracker (BTrack / autocorrelation) gives
sub-BPM precision but suffers the classic octave-error failure mode:
locking to half-tempo (60 instead of 120) or double-tempo (240
instead of 120), and occasionally triplet-relationships (180/120 ≈
1.5x). The May 2026 baseline measured 51% strict accuracy / 72% with
half-tempo tolerance — that 21-point gap is largely the octave-error
rate.

Beat detection events are an independent measurement of the beat
*rate*. They don't have the precision of an autocorrelator, but they
do know what counts as a beat. Combining them resolves the octave
ambiguity:

  1. BeatRateEstimator watches the event stream and reports the
     median inter-beat-interval (IBI), giving us beat_rate_bpm.
  2. reconcile_tempo() picks the octave of beat_rate that best
     matches the tracker's estimate, returning the corrected BPM.

The tracker keeps doing what it's good at (precise periodicity); the
beats keep doing what they're good at (knowing what counts as a
beat); they reconcile on the only thing they disagree about (octave).

Per `feedback_listening_tests.md`: response calibration is the main
issue, not detection. This is the calibration layer for tempo.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import statistics


# Beat-rate-aware reconciliation considers these octave ratios. Covers
# the standard half/double + triplet relationships that the periodicity
# tracker most commonly confuses.
_OCTAVE_RATIOS: tuple[float, ...] = (1 / 3, 1 / 2, 1.0, 2.0, 3.0)

# Default window size for the median-IBI estimator. 6 events at 120 BPM
# spans ~3 seconds — long enough to filter noise, short enough to react
# to a real tempo change (a DJ set BPM ramp) within a few seconds.
_DEFAULT_IBI_WINDOW = 6

# Maximum coefficient-of-variation of inter-beat-intervals to consider
# the beat rate "stable". If CoV > this, beats are arriving irregularly
# (e.g. rubato, broken rhythm, false-positive storm) and the estimator
# reports unstable — reconciliation falls back to the raw tracker.
_DEFAULT_STABILITY_COV = 0.20

# Tolerance for "tracker is in the right octave already". If the chosen
# beat-rate octave is within this fraction of the tracker's BPM, the
# tracker is presumed correctly locked and we return its (more precise)
# value rather than the (coarser) beat-rate-derived one.
_DEFAULT_OCTAVE_TOLERANCE = 0.05


@dataclass(frozen=True)
class ReconciliationResult:
    """Output of reconcile_tempo.

    octave_factor is the multiplier applied to the tracker to arrive
    at bpm. 1.0 = tracker was correct as-is; 2.0 = tracker was at
    half-tempo; 0.5 = tracker was at double-tempo; etc.
    """
    bpm: float
    trustable: bool       # True if beat-rate was stable enough to have an opinion
    octave_factor: float
    tracker_agreed: bool  # True if octave_factor == 1.0 (within tolerance)


class BeatRateEstimator:
    """Watches beat / downbeat events and reports median inter-beat-interval.

    Onset events are excluded — they're subdivision-level and would
    pollute the beat rate. Only "beat-level" events (kind='low' or
    kind='mid' under the role_mapper convention) contribute.

    The estimator is monotonic-clock-driven; callers feed it event
    timestamps in seconds. The internal window is fixed-size so old
    events naturally age out.
    """

    # Event kinds that count as beat-level. Downbeat ('low') and beat
    # ('mid') both contribute; onset ('high') is treated as subdivision
    # and excluded.
    _BEAT_KINDS = frozenset({'low', 'mid'})

    def __init__(self,
                 window_size: int = _DEFAULT_IBI_WINDOW,
                 stability_cov: float = _DEFAULT_STABILITY_COV) -> None:
        if window_size < 3:
            raise ValueError(
                f'window_size must be >= 3 for median sanity '
                f'(got {window_size}); below 3 the median collapses '
                f'to picking one of the values rather than smoothing.')
        self._window = window_size
        self._stability_cov = stability_cov
        # Just timestamps — we don't need to keep the event objects.
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def feed(self, kind: str, timestamp: float) -> None:
        """Record a beat-level event timestamp; onsets are ignored."""
        if kind not in self._BEAT_KINDS:
            return
        # Drop monotonicity violations (clock went backward) — would
        # produce negative IBIs that destroy the median.
        if self._timestamps and timestamp < self._timestamps[-1]:
            return
        self._timestamps.append(timestamp)

    def reset(self) -> None:
        """Clear the window. Call on song change / tempo discontinuity."""
        self._timestamps.clear()

    @property
    def has_estimate(self) -> bool:
        """At least 2 events seen → at least 1 IBI computable."""
        return len(self._timestamps) >= 2

    @property
    def beat_rate_bpm(self) -> float | None:
        """Median IBI converted to BPM, or None if no estimate yet."""
        if not self.has_estimate:
            return None
        ibis = [b - a for a, b in zip(self._timestamps, list(self._timestamps)[1:])]
        # All IBIs are positive by feed() invariant.
        median_ibi = statistics.median(ibis)
        if median_ibi <= 0:
            return None
        return 60.0 / median_ibi

    @property
    def stable(self) -> bool:
        """Coefficient of variation of IBIs is below threshold.

        Unstable beat rate (rubato, broken rhythm, false-positive storm)
        means the median is misleading and reconciliation should fall
        back to the raw tracker.
        """
        if len(self._timestamps) < 3:
            return False  # need ≥ 2 IBIs for a meaningful CoV
        ibis = [b - a for a, b in zip(self._timestamps, list(self._timestamps)[1:])]
        m = statistics.mean(ibis)
        if m <= 0:
            return False
        sd = statistics.stdev(ibis)
        return (sd / m) <= self._stability_cov


def _best_octave_shift(tracker_bpm: float, beat_rate: float) -> float:
    """Pick the multiplier s ∈ {1/3, 1/2, 1, 2, 3} such that
    tracker_bpm × s is closest to beat_rate (in relative gap).

    Interpretation: tracker is at the (1/s)× octave of the true tempo,
    so we shift its output by × s to land at the right octave.
    """
    if tracker_bpm <= 0:
        return 1.0  # degenerate; no shift makes sense
    return min(
        _OCTAVE_RATIOS,
        key=lambda s: abs(tracker_bpm * s - beat_rate) / max(beat_rate, 1e-6))


def reconcile_tempo_override(tracker_bpm: float,
                              beat_estimator: BeatRateEstimator,
                              tolerance: float = _DEFAULT_OCTAVE_TOLERANCE
                              ) -> ReconciliationResult:
    """Strategy A: beats are truth; tracker only used when it agrees.

    If `|tracker - beat_rate| / beat_rate <= tolerance`, tracker is
    locked at the right octave and we use its (higher-precision) value.
    Otherwise output `beat_rate` directly — losing sub-BPM precision
    in exchange for an instant octave correction.

    Fast convergence (one reconciliation = corrected), lower steady-
    state precision than the tracker alone when tracker is right.
    """
    beat_rate = beat_estimator.beat_rate_bpm
    if beat_rate is None or not beat_estimator.stable:
        return ReconciliationResult(
            bpm=tracker_bpm, trustable=False,
            octave_factor=1.0, tracker_agreed=False)

    rel_gap = abs(tracker_bpm - beat_rate) / max(beat_rate, 1e-6)
    if rel_gap <= tolerance:
        return ReconciliationResult(
            bpm=tracker_bpm, trustable=True,
            octave_factor=1.0, tracker_agreed=True)
    return ReconciliationResult(
        bpm=beat_rate, trustable=True,
        octave_factor=1.0, tracker_agreed=False)


def reconcile_tempo_octave_shift(tracker_bpm: float,
                                  beat_estimator: BeatRateEstimator,
                                  tolerance: float = _DEFAULT_OCTAVE_TOLERANCE
                                  ) -> ReconciliationResult:
    """Strategy B: shift tracker to the octave the beats suggest.

    Pick s ∈ {1/3, 1/2, 1, 2, 3} minimizing |tracker×s - beat_rate|,
    return `tracker × s`. Preserves the tracker's sub-BPM precision
    while resolving the octave error in one shot.

    Higher steady-state precision than strategy A. Same instant
    correction (no convergence delay). The only place A could
    out-perform B is if the tracker's value is meaningfully wrong
    even after octave correction — in which case strategy A's beat-
    rate-direct value is the better fallback.
    """
    beat_rate = beat_estimator.beat_rate_bpm
    if beat_rate is None or not beat_estimator.stable:
        return ReconciliationResult(
            bpm=tracker_bpm, trustable=False,
            octave_factor=1.0, tracker_agreed=False)

    shift = _best_octave_shift(tracker_bpm, beat_rate)
    out = tracker_bpm * shift
    return ReconciliationResult(
        bpm=out, trustable=True,
        octave_factor=shift,
        tracker_agreed=(shift == 1.0))


def reconcile_tempo_octave_only(tracker_bpm: float,
                                  beat_estimator: BeatRateEstimator,
                                  tolerance: float = _DEFAULT_OCTAVE_TOLERANCE
                                  ) -> ReconciliationResult:
    """Strategy C: octave-shift but ONLY for strict ×2 / ×0.5 errors.

    Conservative variant of strategy B. If beat_rate suggests tracker
    is at half-tempo (shift=2.0) or double-tempo (shift=0.5), apply
    the correction. For any other ratio (1/3, 3, 1.5, etc.) — assume
    beat_rate is wrong (over-firing on subdivisions, swing/triplet
    confusion, etc.) and leave tracker alone.

    This guards against the eval-observed failure mode where the beat
    head fires on subdivisions, producing a beat_rate that's
    arbitrarily higher than the true tempo. Under that condition,
    strategies A and B both make things worse; strategy C correctly
    does nothing.

    Trade: misses triplet-rate errors that strategy B would catch.
    Net effect depends on which failure mode dominates in real music.
    """
    beat_rate = beat_estimator.beat_rate_bpm
    if beat_rate is None or not beat_estimator.stable:
        return ReconciliationResult(
            bpm=tracker_bpm, trustable=False,
            octave_factor=1.0, tracker_agreed=False)

    shift = _best_octave_shift(tracker_bpm, beat_rate)
    # Only act on ×2 or ×0.5; pass through for shift=1 (tracker correct
    # at this octave) or 1/3, 3 (non-octave ratio = beats suspicious).
    if shift in (0.5, 2.0):
        # Confirm the shifted tracker actually matches beat_rate within
        # tolerance — sanity check against degenerate cases.
        shifted = tracker_bpm * shift
        rel_gap = abs(shifted - beat_rate) / max(beat_rate, 1e-6)
        if rel_gap <= tolerance:
            return ReconciliationResult(
                bpm=shifted, trustable=True,
                octave_factor=shift, tracker_agreed=False)
    return ReconciliationResult(
        bpm=tracker_bpm, trustable=True,
        octave_factor=1.0, tracker_agreed=True)


# Backward-compat alias — default to strategy B (octave-shift) since
# it preserves precision. Pick a strategy explicitly when you care.
reconcile_tempo = reconcile_tempo_octave_shift
