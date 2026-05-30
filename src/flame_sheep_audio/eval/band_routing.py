"""Band-routing evaluation — does the daemon classify events to the right band?

For each synthetic stimulus we know which band(s) the algorithm SHOULD
fire on. We feed it through AudioProcessor, count emitted events per
band, and roll up per-band recall and dominance metrics.

What this catches:
  - 80Hz sine no longer routes to 'low' (e.g. band-mask regression)
  - hihat patterns no longer dominate the 'high' band (high-band death)
  - 808 + hihat mixes no longer separate cleanly (band-isolation regression)

What this does NOT catch:
  - Drift in absolute event counts: per-stimulus counts vary widely with
    detector tuning; the eval surfaces the aggregates that survive that
    jitter (per-band recall, band dominance), not raw counts. Per-
    stimulus counts go into the per_stimulus details bag for debugging.

Stimuli were extracted from the 7 failing tests in test_pipeline.py +
test_beat_engine.py that this eval supersedes — same intent, real
ground truth, comparison against a checked-in baseline rather than
arbitrary thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE
from flame_sheep_audio.eval.synths import (
    synth_kick, synth_hihat, synth_808_kick, synth_vocal,
)


# ---------------------------------------------------------------------------
# Stimuli — synthetic audio patterns + ground truth about which band(s)
# the algorithm should fire on
# ---------------------------------------------------------------------------

@dataclass
class BandRoutingStimulus:
    name: str
    description: str
    pcm_factory: Callable[[], np.ndarray]
    # Which band(s) the algorithm SHOULD fire on. A stimulus's "success"
    # is firing at least one event in any expected band.
    expected_bands: frozenset[str]
    # Which bands MUST NOT dominate the output. Used for adversarial mixes
    # where the wrong band could plausibly fire but shouldn't out-count
    # the right one. Default empty = no anti-routing constraint.
    must_not_dominate: frozenset[str] = field(default_factory=frozenset)


def _make_sine(freq: float, n_samples: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _place_hits(duration_s: float, interval_s: float,
                synth_fn: Callable[..., np.ndarray],
                **kwargs) -> np.ndarray:
    """Place synthesized hits at regular intervals in a buffer.
    Mirrors the helper in test_pipeline.py — kept here so the eval is
    self-contained and doesn't pull from tests/."""
    n = int(SAMPLE_RATE * duration_s)
    signal = np.zeros(n, dtype=np.float32)
    hit = synth_fn(**kwargs)
    t = 0.0
    while t < duration_s:
        pos = int(t * SAMPLE_RATE)
        end = min(pos + len(hit), n)
        signal[pos:end] += hit[:end - pos]
        t += interval_s
    return signal


def _stim_low_sine_80hz() -> np.ndarray:
    """80Hz tone preceded by silence — onset should classify as low."""
    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    tone = _make_sine(80, FFT_SIZE * 4, amplitude=0.9)
    return np.concatenate([silence] * 5 + [tone])


def _stim_high_hihat_pattern() -> np.ndarray:
    """4s of hihat hits at 0.25s — pure high-band stimulus."""
    return _place_hits(4.0, 0.25, synth_hihat, amplitude=0.4)


def _stim_808_plus_hihat() -> np.ndarray:
    """Sustained 808 bass + hihat hits. Bass should NOT dominate; the
    transient hihat should drive the event stream."""
    bass = _make_sine(40, int(SAMPLE_RATE * 4.0), amplitude=0.7)
    high_hits = _place_hits(4.0, 0.25, synth_hihat, amplitude=0.4)
    return bass + high_hits


def _stim_kick_over_vocal() -> np.ndarray:
    """Kick under sustained vocal — the kick's transient should fire
    'low' events despite the vocal's interfering low-frequency energy."""
    vocal = synth_vocal(4.0, pitch=80, amplitude=0.3)
    kicks = _place_hits(4.0, 0.5, synth_kick, amplitude=0.8)
    return vocal + kicks


def _stim_kick_over_sustained_bass() -> np.ndarray:
    """Kicks at 0.5s under a sustained 60Hz bass pad with second harmonic.
    Tests that transient kicks fire 'low' despite continuous low-band
    interference (separate from kick_over_vocal which uses formant-rich
    vocal interference)."""
    n = int(SAMPLE_RATE * 4.0)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    bass = (0.6 * np.sin(2 * np.pi * 60 * t)
            + 0.18 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    kicks = _place_hits(4.0, 0.5, synth_kick, amplitude=0.9)
    return bass + kicks


def _stim_fast_kicks() -> np.ndarray:
    """4s of kicks at 0.15s intervals — dense low-band stimulus."""
    return _place_hits(4.0, 0.15, synth_kick, amplitude=0.8)


def _stim_slow_kicks() -> np.ndarray:
    """4s of kicks at 1.0s intervals — sparse low-band stimulus."""
    return _place_hits(4.0, 1.0, synth_kick, amplitude=0.8)


STIMULI: list[BandRoutingStimulus] = [
    BandRoutingStimulus(
        name='low_sine_80hz',
        description='80 Hz pure sine onset; should fire low, not high.',
        pcm_factory=_stim_low_sine_80hz,
        expected_bands=frozenset({'low'}),
        must_not_dominate=frozenset({'high'}),
    ),
    BandRoutingStimulus(
        name='high_hihat_pattern',
        description='4s of hihat hits at 0.25s; pure high-band stimulus.',
        pcm_factory=_stim_high_hihat_pattern,
        expected_bands=frozenset({'high'}),
        must_not_dominate=frozenset({'low'}),
    ),
    BandRoutingStimulus(
        name='808_plus_hihat',
        description='Sustained 40Hz 808 + hihat pattern; high should '
                    'dominate, low must not (bass-isolation test).',
        pcm_factory=_stim_808_plus_hihat,
        expected_bands=frozenset({'high'}),
        must_not_dominate=frozenset({'low'}),
    ),
    BandRoutingStimulus(
        name='kick_over_vocal',
        description='Kicks at 0.5s under sustained 80Hz vocal; low should '
                    'still fire despite vocal interference.',
        pcm_factory=_stim_kick_over_vocal,
        expected_bands=frozenset({'low'}),
    ),
    BandRoutingStimulus(
        name='kick_over_sustained_bass',
        description='Kicks at 0.5s under sustained 60Hz bass + harmonic; '
                    'low should fire despite continuous low-band '
                    'interference (Wall of Bass scenario).',
        pcm_factory=_stim_kick_over_sustained_bass,
        expected_bands=frozenset({'low'}),
    ),
    BandRoutingStimulus(
        name='fast_kicks',
        description='4s of kicks at 0.15s; dense low-band stimulus '
                    '(compare with slow_kicks for density-ranking).',
        pcm_factory=_stim_fast_kicks,
        expected_bands=frozenset({'low'}),
    ),
    BandRoutingStimulus(
        name='slow_kicks',
        description='4s of kicks at 1.0s; sparse low-band stimulus '
                    '(compare with fast_kicks for density-ranking).',
        pcm_factory=_stim_slow_kicks,
        expected_bands=frozenset({'low'}),
    ),
]


# ---------------------------------------------------------------------------
# Pipeline + per-stimulus measurement
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 20  # silence frames fed before the stimulus, matches tests


def _run_pipeline(pcm: np.ndarray) -> list:
    """Feed `pcm` through a fresh AudioProcessor and return all emitted
    BeatEvents in encounter order."""
    # Local imports — the eval module shouldn't drag AudioProcessor into
    # the module-load path for callers that only want the registry.
    from flame_sheep_audio import AudioProcessor
    from flame_sheep_audio.source import FeedSource
    from flame_sheep_audio.config import cfg

    # Reset cfg so user audio.toml overrides don't perturb eval runs.
    cfg.reset_to_defaults()
    proc = AudioProcessor(source=FeedSource())

    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(WARMUP_FRAMES):
        proc.feed(silence)
        proc.process()

    events = []
    pos = 0
    while pos < len(pcm):
        chunk = pcm[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        proc.feed(chunk)
        events.extend(proc.process())
        pos += FFT_SIZE
    return events


def _per_stimulus_breakdown(stim: BandRoutingStimulus, events: list) -> dict:
    by_band: dict[str, int] = {'low': 0, 'mid': 0, 'high': 0}
    for e in events:
        if e.kind in by_band:
            by_band[e.kind] += 1
    expected_count = sum(by_band[b] for b in stim.expected_bands
                         if b in by_band)
    non_expected_count = sum(by_band[b] for b in by_band
                              if b not in stim.expected_bands)
    return {
        'by_band': by_band,
        'expected_count': expected_count,
        'non_expected_count': non_expected_count,
        'total_events': sum(by_band.values()),
    }


# ---------------------------------------------------------------------------
# Headline metrics — chosen to be robust to per-stimulus count jitter
# ---------------------------------------------------------------------------

def _aggregate(per_stim: dict[str, dict]) -> dict[str, float]:
    """Build the headline metric dict.

    Choices and why:
      recall_{band}: fraction of stimuli expecting `band` that fired at
        least one event in that band. Range [0, 1]. Robust to count
        jitter; sensitive to actual routing breakage.

      dominance_correct_rate: fraction of stimuli where the expected
        band(s) had MORE events than any must_not_dominate band(s).
        Range [0, 1]. Captures adversarial-mix tests (808+hihat).

      stimuli_with_any_event: fraction of stimuli that produced at
        least one event of any kind. Drift here is a sensitivity
        indicator (whole-detector died vs over-firing).

      density_ranking_fast_vs_slow: ratio of fast_kicks events to
        slow_kicks events; should be > 1 (more dense → more events).
        Real property assertion, baseline captures current ratio.
    """
    metrics: dict[str, float] = {}

    # Per-band recall
    for band in ('low', 'mid', 'high'):
        applicable = [s for s in STIMULI if band in s.expected_bands]
        if not applicable:
            metrics[f'recall_{band}'] = 0.0
            continue
        hits = sum(1 for s in applicable
                    if per_stim[s.name]['by_band'].get(band, 0) > 0)
        metrics[f'recall_{band}'] = hits / len(applicable)

    # Dominance: expected-band count > any must_not_dominate band count
    domiance_applicable = [s for s in STIMULI if s.must_not_dominate]
    if domiance_applicable:
        correct = 0
        for s in domiance_applicable:
            ps = per_stim[s.name]
            expected_count = max(
                ps['by_band'].get(b, 0) for b in s.expected_bands)
            forbidden_max = max(
                ps['by_band'].get(b, 0) for b in s.must_not_dominate)
            if expected_count > forbidden_max:
                correct += 1
        metrics['dominance_correct_rate'] = correct / len(domiance_applicable)
    else:
        metrics['dominance_correct_rate'] = 0.0

    # Sensitivity indicator: fraction of stimuli with any event
    any_event = sum(1 for s in STIMULI
                     if per_stim[s.name]['total_events'] > 0)
    metrics['stimuli_with_any_event'] = any_event / len(STIMULI)

    # Density ranking: fast_kicks should produce more events than slow_kicks
    fast = per_stim.get('fast_kicks', {}).get('by_band', {}).get('low', 0)
    slow = per_stim.get('slow_kicks', {}).get('by_band', {}).get('low', 0)
    # +1 in denom so slow=0 doesn't blow up. baseline will capture the
    # current ratio; drift signals a change in density tracking.
    metrics['density_ranking_fast_vs_slow'] = float(fast) / float(slow + 1)

    return metrics


# ---------------------------------------------------------------------------
# Public API for the framework
# ---------------------------------------------------------------------------

NAME = 'band_routing'

# Per-metric relative-drift tolerances. Tight on band-recall because
# those should be near 1.0 and a single-stimulus failure is meaningful;
# looser on density ranking because it's a ratio that can swing on
# minor cooldown/threshold changes.
TOLERANCES: dict[str, float] = {
    '_default': 0.15,
    'recall_low': 0.10,
    'recall_mid': 0.10,
    'recall_high': 0.10,
    'dominance_correct_rate': 0.10,
    'stimuli_with_any_event': 0.10,
    'density_ranking_fast_vs_slow': 0.30,
}


def run() -> tuple[dict[str, float], dict[str, Any]]:
    """Run all band-routing stimuli, return (headline_metrics, per_stim).

    Used by both:
      - tools/refresh_eval_baselines.py — to snapshot current behavior
      - tests/eval/test_audio_regressions.py — to check drift vs baseline

    Both call paths use the same numbers, so a refresh-then-test cycle
    is always clean.
    """
    per_stim: dict[str, dict[str, Any]] = {}
    for stim in STIMULI:
        pcm = stim.pcm_factory()
        events = _run_pipeline(pcm)
        per_stim[stim.name] = _per_stimulus_breakdown(stim, events)
    return _aggregate(per_stim), per_stim
