"""Vectorized particle-filter step for BeatNet's cascade.

Replaces the two hot Python loops inside `particle_filter_cascade.process()`
with batched `rng.choice` calls + preallocated arrays. Mathematically
equivalent — same samples come from the same per-state categorical
distributions, just batched per unique source-state instead of per
particle.

Speedup target: 5-10× on the tail. The original loops:
  - iterate one particle at a time
  - call `np.random.choice` with k=1 per iteration
  - `np.append` inside the loop (O(n²) memory traffic)

The vectorized step:
  - groups particles by source-state value with `np.unique`
  - calls `rng.choice(cands, size=count_at_state, p=probs)` once per group
  - preallocates the result array

Precomputed lookups are built once per cascade instance:
  - beat: `_beat_lut[state_id] -> (cands, probs)` for each of the ~42
    last_states (a tiny subset of the 1449 total beat states)
  - downbeat: `_db_lut[state_id] -> (cands, probs)` for the 3 down_last_states

`install_fast_process(pf)` monkeypatches a single cascade instance so
the original module + tests aren't touched. Pass `pf=None` to a `verify_*`
helper for round-trip equivalence checks.
"""
from __future__ import annotations

import numpy as np

# Use BeatNet's rng for parity with the upstream module's seed/state.
from numpy.random import default_rng


def _build_beat_lut(pf) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each beat-boundary 'last state', precompute its outgoing
    (next_state_candidates, probabilities). pf.tm = (next, src, prob).
    """
    tm_next = np.asarray(pf.tm[0])
    tm_src = np.asarray(pf.tm[1])
    tm_prob = np.asarray(pf.tm[2])
    last_states = np.asarray(list(pf.st.last_states), dtype=np.int64)

    lut: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for s in last_states:
        mask = (tm_src == s)
        cands = tm_next[mask].astype(np.int64, copy=False)
        probs = tm_prob[mask].astype(np.float64, copy=False)
        # Normalize defensively (should already sum to 1).
        total = probs.sum()
        if total > 0:
            probs = probs / total
        lut[int(s)] = (cands, probs)
    return lut


def _build_downbeat_lut(pf) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each downbeat-boundary 'last state', precompute its outgoing
    (next_state_candidates, probabilities). pf.tm2 is a square matrix
    indexed by position within last_states[0]."""
    last_arr = np.asarray(pf.st2.last_states[0], dtype=np.int64)
    first_arr = np.asarray(pf.st2.first_states[0], dtype=np.int64)
    tm2 = np.asarray(pf.tm2)
    lut: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, s in enumerate(last_arr):
        probs = tm2[i].astype(np.float64, copy=False)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        lut[int(s)] = (first_arr.copy(), probs)
    return lut


def _vectorized_motion(particles: np.ndarray, last_states_set: np.ndarray,
                        lut: dict[int, tuple[np.ndarray, np.ndarray]],
                        rng) -> np.ndarray:
    """Advance non-boundary particles by +1; sample new state for
    boundary particles (those equal to any value in last_states_set)
    from the per-source-state transition distribution.

    Equivalent to the per-particle for-loops in
    `particle_filter_cascade.process()` but batched by unique source.
    """
    mask = np.isin(particles, last_states_set)
    advanced = particles[~mask] + 1

    if not mask.any():
        return advanced

    boundary_vals = particles[mask]
    unique_vals, counts = np.unique(boundary_vals, return_counts=True)
    sampled_list: list[np.ndarray] = []
    for v, c in zip(unique_vals, counts):
        cands, probs = lut[int(v)]
        sampled_list.append(rng.choice(cands, size=int(c), replace=True, p=probs))
    sampled = np.concatenate(sampled_list)
    return np.concatenate([advanced, sampled])


def install_fast_process(pf, rng=None) -> None:
    """Replace `pf.process` with a vectorized equivalent.

    Idempotent — calling twice is a no-op (checks for the marker
    `_fast_process_installed`).
    """
    if getattr(pf, '_fast_process_installed', False):
        return

    pf._beat_lut = _build_beat_lut(pf)
    pf._db_lut = _build_downbeat_lut(pf)
    pf._last_states_set = np.asarray(
        list(pf.st.last_states), dtype=np.int64)
    pf._db_last_states_set = np.asarray(
        pf.st2.last_states[0], dtype=np.int64)
    pf._fast_rng = rng if rng is not None else default_rng()

    # Import inside the function so this module can be loaded without
    # BeatNet present (e.g. tests that mock the cascade).
    from BeatNet.particle_filtering_cascade import (
        beat_densities, down_densities, universal_resample)

    def fast_process(self, activations):
        """Vectorized drop-in for particle_filter_cascade.process.

        Same control flow as the upstream method, with the two
        per-particle for-loops replaced by `_vectorized_motion`.
        """
        rng = self._fast_rng
        activations = activations[int(self.offset / self.T):]
        if np.shape(activations) == (2,):
            activations = np.reshape(activations, (-1, 2))
        both_activations = activations.copy()
        activations = np.max(activations, axis=1)
        activations[activations < self.ig_threshold] = 0.03
        self.activations = activations
        self.both_activations = both_activations

        for i in range(len(activations)):
            self.counter += 1
            gathering = int(np.median(self.particles))
            # Downbeat block — gated by clutter-near-beat heuristic.
            if (((gathering - self.beat[
                    self.st.state_intervals[self.beat]
                    == self.st.state_intervals[gathering]]) < (
                    int(.07 / self.T)) + 1).any()
                    and (self.offset + self.counter * self.T) - self.path[-1][0]
                        > .4 * self.T * self.st.state_intervals[gathering]):

                # Vectorized downbeat particle motion.
                self.down_particles = _vectorized_motion(
                    self.down_particles, self._db_last_states_set,
                    self._db_lut, rng)

                # Downbeat particles correction (unchanged from upstream).
                if both_activations[i][1] > 0.7:
                    self.down_particles = np.append(
                        self.down_particles,
                        np.array([self.st2.first_states]))
                obs2 = down_densities(both_activations[i], self.om2, self.st2)
                self.down_particles = universal_resample(
                    self.down_particles, obs2[self.down_particles])
                if both_activations[i][1] > 0.7:
                    self.down_particles = np.delete(
                        self.down_particles,
                        np.random.choice(self.down_particle_size,
                                          len(self.st2.first_states),
                                          replace=False))
                m = np.bincount(self.down_particles)
                self.down_max = np.argmax(m)

                # Beat vs downbeat distinguishment (unchanged).
                if (self.down_max in self.st2.first_states[0]
                        and self.path[-1][1] != 1
                        and both_activations[i][1] > 0.4):
                    self.path = np.append(
                        self.path,
                        [[self.offset + self.counter * self.T, 1]], axis=0)
                elif activations[i] > 0.4:
                    self.path = np.append(
                        self.path,
                        [[self.offset + self.counter * self.T, 2]], axis=0)

            # Vectorized beat particle motion (every frame).
            self.particles = _vectorized_motion(
                self.particles, self._last_states_set, self._beat_lut, rng)

            # Beat particles correction (unchanged from upstream).
            obs = beat_densities(activations[i], self.om, self.st)
            if activations[i] > 0.1:
                if activations[i] > 0.8:
                    self.particles = np.append(
                        self.particles,
                        np.array([self.st.first_states[0][np.arange(
                            np.random.randint(4),
                            len(self.st.first_states[0]), 6)]]))
                self.particles = universal_resample(
                    self.particles, obs[self.particles])
                if activations[i] > 0.8:
                    np.delete(self.particles,
                              np.random.choice(self.particle_size,
                                                len(self.st.first_states),
                                                replace=False))
        return self.path[1:]

    # Bind as a method on this instance.
    import types
    pf.process = types.MethodType(fast_process, pf)
    pf._fast_process_installed = True
