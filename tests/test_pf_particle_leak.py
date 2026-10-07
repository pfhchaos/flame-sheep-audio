"""Regression test for the particle-filter particle-count leak.

Background
----------
BeatNet's `particle_filter_cascade.process()` (driven in production via the
vectorized `_pf_fast.install_fast_process`) has two correction blocks — one
for beat particles, one for downbeat particles — that each APPEND reseed
particles on a strong-onset frame and then called `universal_resample`, which
is LENGTH-PRESERVING (returns `len(particles)` samples, NOT resample-to-N).
The compensating deletes never removed what was appended:

  * beat block:     `np.delete(...)` result was discarded -> a no-op.
  * downbeat block: deleted ~1 of the ~3 just appended.

So the populations grew without bound across strong-onset frames: per-hop
cost O(N), memory proportional to N, and float drift in the oversized cumsum
eventually threw IndexError from `np.searchsorted`. On the continuous-audio
daemon (never reset) this pinned a core (pre-fix, ~1000 strong frames grew
the beat population to ~8500 and the downbeat to ~488).

The fix replaces the length-preserving resample + dead delete with
`_resample_to_n(..., particle_size / down_particle_size)` (systematic
resampling), keeping the reseed append, so the population size is invariant
by construction.

What this pins
--------------
Both populations stay EXACTLY particle_size (1500) and down_particle_size
(250) after a heavy strong-onset load — leak dead, count invariant. The PF
is driven directly (no LSTM / audio frontend) with synthetic all-strong
activation frames, which exercise BOTH correction blocks. Construction
mirrors `beat_detector_beatnet.BeatNetLiveDetector._make_particle_filter`
(particle_size=1500, down_particle_size=250, fps=50), so this is the
production particle filter, not a reimplementation.
"""
from __future__ import annotations

import numpy as np
import pytest

# Production population sizes (see _make_particle_filter).
PARTICLE_SIZE = 1500
DOWN_PARTICLE_SIZE = 250

FPS = 50
FAST_FRAMES = 4000           # ~80 s equivalent; grew unbounded pre-fix
SLOW_FRAMES = 15 * 60 * FPS  # 45000 == 15 min @ 50 fps
SEED = 20260101


def _build_pf():
    """Construct the production particle filter deterministically."""
    # numpy<2.0 compat shim the production adapter also applies.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]
    from BeatNet.particle_filtering_cascade import particle_filter_cascade
    from flame_sheep_audio._pf_fast import install_fast_process

    np.random.seed(SEED)
    pf = particle_filter_cascade(
        beats_per_bar=[],
        particle_size=PARTICLE_SIZE,
        down_particle_size=DOWN_PARTICLE_SIZE,
        min_bpm=55.0, max_bpm=215.0,
        fps=FPS, plot=[], mode=None)
    install_fast_process(pf, rng=np.random.default_rng(SEED))
    return pf


def _strong_onset_activations(n_frames: int) -> np.ndarray:
    """`n_frames` of (beat, downbeat) activations, all strongly onset.

    Both columns 0.9 -> every frame trips the >0.8 beat reseed and the >0.7
    downbeat reseed, the worst case for the leak.
    """
    return np.full((n_frames, 2), 0.9, dtype=np.float64)


def test_fix_keeps_population_size():
    """The fix keeps both populations EXACTLY at their fixed sizes after a
    heavy strong-onset load — the leak is dead and the count is invariant by
    construction (resample-to-N, no post-hoc delete). Pre-fix the same load
    grew the populations without bound."""
    pf = _build_pf()
    pf.process(_strong_onset_activations(FAST_FRAMES))
    assert len(pf.particles) == PARTICLE_SIZE
    assert len(pf.down_particles) == DOWN_PARTICLE_SIZE


@pytest.mark.slow
def test_fix_invariant_over_15_minutes():
    """Full 15-minute-equivalent (45000 frames @ 50 fps) all-strong load:
    the population stays pinned to its fixed size the whole way, at the scale
    that pinned a core pre-fix."""
    pf = _build_pf()
    pf.process(_strong_onset_activations(SLOW_FRAMES))
    assert len(pf.particles) == PARTICLE_SIZE
    assert len(pf.down_particles) == DOWN_PARTICLE_SIZE
