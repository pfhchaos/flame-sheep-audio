"""Tests for BeatRNNDetector's architecture-discovery loader.

The loader has to handle two file formats:

  - **Legacy** files (pre-2026-05-29): no explicit dim metadata in the
    .npz. The runtime falls back to a hardcoded lookup table keyed by
    flat-weight-array length. v1 (18801) and v3 (30451) are baked in.

  - **New** files (post-2026-05-29): explicit input_size, proj_size,
    hidden_size, n_classes fields in the .npz. Discovery prefers these
    over the legacy table — they're authoritative.

Real model files in flame_sheep/data/ are exercised when present so
the lookup-table entries actually match the layout the trainer
produced. Synthetic tests cover the new-format path and the explicit
failure mode (unknown legacy layout, no dim metadata).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# flame_sheep_audio package layout: tests live alongside src/. The
# flame_sheep_audio package is installed via the repo's editable install,
# so direct import should work.
from flame_sheep_audio.beat_rnn import (
    BeatRNNDetector,
    MultiDepthBeatRNNDetector,
    _LEGACY_ARCH_BY_FLAT_LEN,
    _discover_arch,
    _discover_arch_multidepth,
    _multidepth_param_count,
    _unpack_weights,
    load_beat_rnn,
)


# Path to the project root for loading actual deployed model files.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_V1_PATH = _PROJECT_ROOT / 'flame_sheep' / 'data' / 'beat_rnn_continuous.npz'
_V3_PATH = _PROJECT_ROOT / 'flame_sheep' / 'data' / 'beat_rnn_3head.npz'


def _expected_flat_len(I: int, P: int, H: int, C: int) -> int:
    """Layout: W_in(I*P) + b_in(P) + W_gru(3*P*H) + U_gru(3*H*H)
              + bias_gru(6*H) + W_out(H*C) + b_out(C)"""
    return I * P + P + 3 * P * H + 3 * H * H + 6 * H + H * C + C


def _make_synthetic_weights(I: int, P: int, H: int, C: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(
        _expected_flat_len(I, P, H, C)).astype(np.float32)


# ============================================================================
# Real deployed model files — verify the legacy lookup-table entries
# actually match what the trainer wrote.
# ============================================================================

class TestRealCheckpoints:

    @pytest.mark.skipif(not _V1_PATH.exists(), reason='v1 weights file not present')
    def test_load_v1_continuous(self):
        """v1 single-channel — uses legacy fallback (18801 → 32/48/1)."""
        det = BeatRNNDetector(_V1_PATH)
        assert det._input_size == 216
        assert det._proj_size == 32
        assert det._hidden_size == 48
        assert det._n_classes == 1
        # Weight shapes derived correctly
        assert det._W_in.shape == (216, 32)
        assert det._W_gru.shape == (3, 32, 48)
        assert det._W_out.shape == (48, 1)

    @pytest.mark.skipif(not _V3_PATH.exists(), reason='v3 weights file not present')
    def test_load_v3_hierarchical(self):
        """v3 3-head — uses legacy fallback (30451 → 64/48/3)."""
        det = BeatRNNDetector(_V3_PATH)
        assert det._input_size == 216
        assert det._proj_size == 64
        assert det._hidden_size == 48
        assert det._n_classes == 3
        assert det._W_in.shape == (216, 64)
        assert det._W_gru.shape == (3, 64, 48)
        assert det._W_out.shape == (48, 3)


# ============================================================================
# Synthetic tests — control the file contents to verify each resolution path
# ============================================================================

class TestArchDiscovery:
    """_discover_arch picks dims via: explicit fields → legacy table →
    raise. These tests pin each path."""

    def test_explicit_dims_take_priority(self, tmp_path):
        """When all four dim fields are present, use them — even if the
        flat length doesn't match any legacy entry."""
        weights_path = tmp_path / 'explicit.npz'
        I, P, H, C = 216, 96, 192, 5  # bespoke config, no entry in legacy table
        np.savez(weights_path,
                 weights=_make_synthetic_weights(I, P, H, C),
                 next_epoch=0, best_val_loss=0.0,
                 input_size=I, proj_size=P,
                 hidden_size=H, n_classes=C)
        data = np.load(weights_path)
        flat = data['weights']
        I_d, P_d, H_d, C_d = _discover_arch(data, len(flat))
        assert (I_d, P_d, H_d, C_d) == (I, P, H, C)

    def test_legacy_fallback_v1_layout(self, tmp_path):
        """No dim metadata, but flat length matches v1's known layout."""
        weights_path = tmp_path / 'legacy_v1.npz'
        np.savez(weights_path,
                 weights=_make_synthetic_weights(216, 32, 48, 1),
                 next_epoch=0, best_val_loss=0.0)
        data = np.load(weights_path)
        flat = data['weights']
        I, P, H, C = _discover_arch(data, len(flat))
        assert (I, P, H, C) == (216, 32, 48, 1)

    def test_legacy_fallback_v3_layout(self, tmp_path):
        """No dim metadata, flat length matches v3."""
        weights_path = tmp_path / 'legacy_v3.npz'
        np.savez(weights_path,
                 weights=_make_synthetic_weights(216, 64, 48, 3),
                 next_epoch=0, best_val_loss=0.0)
        data = np.load(weights_path)
        I, P, H, C = _discover_arch(data, len(data['weights']))
        assert (I, P, H, C) == (216, 64, 48, 3)

    def test_unknown_layout_raises(self, tmp_path):
        """No metadata AND length doesn't match any known legacy
        configuration. Must raise — silent fallback to defaults would
        produce silent-garbage inference on a misconfigured file."""
        weights_path = tmp_path / 'unknown.npz'
        # Some random length not in _LEGACY_ARCH_BY_FLAT_LEN
        unknown_len = 12345
        np.savez(weights_path,
                 weights=np.zeros(unknown_len, dtype=np.float32),
                 next_epoch=0, best_val_loss=0.0)
        data = np.load(weights_path)
        with pytest.raises(ValueError, match='Cannot determine.*architecture'):
            _discover_arch(data, len(data['weights']))

    def test_legacy_table_contains_known_layouts(self):
        """Sanity: the lookup table's entries pass the param formula.
        If trainer-side bias_gru layout changes (e.g. flat vs gated
        biases), this catches the table going stale."""
        for flat_len, (P, H, C) in _LEGACY_ARCH_BY_FLAT_LEN.items():
            assert _expected_flat_len(216, P, H, C) == flat_len, \
                f'legacy entry {flat_len} → ({P},{H},{C}) does not match formula'


# ============================================================================
# Full detector construction — explicit and legacy paths both produce
# a usable detector instance (not just dims, the unpack must succeed).
# ============================================================================

class TestDetectorConstruction:

    def test_explicit_format_constructs_detector(self, tmp_path):
        I, P, H, C = 216, 32, 48, 1  # use v1-shape so per-band/threshold
                                       # config doesn't need 3 heads
        weights_path = tmp_path / 'explicit_v1.npz'
        np.savez(weights_path,
                 weights=_make_synthetic_weights(I, P, H, C),
                 next_epoch=0, best_val_loss=0.0,
                 input_size=I, proj_size=P,
                 hidden_size=H, n_classes=C)
        det = BeatRNNDetector(weights_path)
        assert det._proj_size == 32
        assert det._hidden_size == 48
        assert det._n_classes == 1

    def test_per_head_thresholds_default_to_shared(self, tmp_path):
        """When no per-head threshold is set, all three default to the
        shared `threshold` value. Pinning current behavior so a future
        change to the default-fallback logic doesn't silently shift it."""
        weights_path = tmp_path / 'v3.npz'
        np.savez(weights_path,
                 weights=_make_synthetic_weights(216, 64, 48, 3),
                 next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64,
                 hidden_size=48, n_classes=3)
        det = BeatRNNDetector(weights_path, threshold=0.42)
        assert det._th_downbeat == 0.42
        assert det._th_beat == 0.42
        assert det._th_onset == 0.42

    def test_per_head_thresholds_override(self, tmp_path):
        weights_path = tmp_path / 'v3.npz'
        np.savez(weights_path,
                 weights=_make_synthetic_weights(216, 64, 48, 3),
                 next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64,
                 hidden_size=48, n_classes=3)
        det = BeatRNNDetector(
            weights_path, threshold=0.3,
            downbeat_threshold=0.15, beat_threshold=0.30,
            onset_threshold=0.30)
        assert det._th_downbeat == 0.15
        assert det._th_beat == 0.30
        assert det._th_onset == 0.30

    def test_hidden_state_sized_by_discovered_hidden(self, tmp_path):
        """Catches the bug pattern where a module-level _HIDDEN_SIZE
        constant gets out of sync with the loaded file's actual H
        (the issue that motivated this refactor)."""
        weights_path = tmp_path / 'v3.npz'
        np.savez(weights_path,
                 weights=_make_synthetic_weights(216, 64, 48, 3),
                 next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64,
                 hidden_size=48, n_classes=3)
        det = BeatRNNDetector(weights_path)
        assert det._h.shape == (48,)


# ============================================================================
# Unpack — wrong dims → loud error rather than silent slicing
# ============================================================================

class TestUnpackValidation:

    def test_offset_mismatch_raises(self):
        """If the caller passes wrong dims, the parsed offset won't
        equal the flat length and we should error rather than return
        garbage views into the array."""
        flat = _make_synthetic_weights(216, 64, 48, 3)  # actually 30451 elements
        # Pass dims for a different model (v1 shape)
        with pytest.raises(ValueError, match='Weight length mismatch'):
            _unpack_weights(flat, 216, 32, 48, 1)


# ============================================================================
# Multi-depth loader — separate code path from single-arch (stacked GRUs,
# one head per layer; explicit dim fields required, no legacy fallback).
# ============================================================================

def _make_synthetic_multidepth_weights(I: int, P: int, H: int, L: int,
                                       n_heads: int | None = None
                                       ) -> np.ndarray:
    """Generate a flat weight vector sized for the multi-depth layout."""
    n = _multidepth_param_count(I, P, H, L, n_heads)
    return np.random.default_rng(0).standard_normal(n).astype(np.float32)


def _save_multidepth_npz(path: Path, I: int, P: int, H: int, L: int,
                         flat: np.ndarray | None = None,
                         architecture: str | None = 'multidepth',
                         n_heads: int | None = None) -> None:
    """Write a synthetic multi-depth .npz with full dim metadata. The
    `architecture` field gets stored as a 0-d unicode array (same way
    save_checkpoint does it)."""
    if flat is None:
        flat = _make_synthetic_multidepth_weights(I, P, H, L, n_heads)
    extras = dict(input_size=I, proj_size=P, hidden_size=H, n_gru_layers=L)
    if architecture is not None:
        extras['architecture'] = np.array(architecture)
    if n_heads is not None:
        extras['n_heads'] = n_heads
    np.savez(path, weights=flat, next_epoch=0, best_val_loss=0.0, **extras)


class TestMultiDepthArchDiscovery:
    """_discover_arch_multidepth requires explicit dim fields (no legacy
    fallback) and validates the file length against expected layout."""

    def test_explicit_dims_resolve(self, tmp_path):
        """The canonical case: all four dim fields present, length matches."""
        weights_path = tmp_path / 'md.npz'
        _save_multidepth_npz(weights_path, I=216, P=64, H=48, L=3)
        data = np.load(weights_path)
        I, P, H, L, n_heads = _discover_arch_multidepth(
            data, len(data['weights']))
        assert (I, P, H, L, n_heads) == (216, 64, 48, 3, 3)

    def test_missing_dim_field_raises(self, tmp_path):
        """If any required dim field is missing, refuse to load —
        there's no legacy table for multi-depth so silent fallback
        isn't an option."""
        weights_path = tmp_path / 'md_incomplete.npz'
        # Save without n_gru_layers
        flat = _make_synthetic_multidepth_weights(216, 64, 48, 3)
        np.savez(weights_path, weights=flat, next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64, hidden_size=48,
                 architecture=np.array('multidepth'))
        data = np.load(weights_path)
        with pytest.raises(ValueError, match='missing required dim'):
            _discover_arch_multidepth(data, len(data['weights']))

    def test_length_mismatch_raises(self, tmp_path):
        """If the dim metadata says one shape but the flat weights are
        sized for another, loud failure — silently using wrong dims
        produces garbage."""
        weights_path = tmp_path / 'md_bad_len.npz'
        wrong_len_flat = np.zeros(12345, dtype=np.float32)
        np.savez(weights_path, weights=wrong_len_flat, next_epoch=0,
                 best_val_loss=0.0,
                 input_size=216, proj_size=64, hidden_size=48, n_gru_layers=3,
                 architecture=np.array('multidepth'))
        data = np.load(weights_path)
        with pytest.raises(ValueError, match='length mismatch'):
            _discover_arch_multidepth(data, len(data['weights']))


class TestMultiDepthDetectorConstruction:

    def test_constructs_from_synthetic_file(self, tmp_path):
        weights_path = tmp_path / 'md.npz'
        _save_multidepth_npz(weights_path, I=216, P=64, H=48, L=3)
        det = MultiDepthBeatRNNDetector(weights_path)
        assert det._n_gru_layers == 3
        assert det._hidden_size == 48
        assert det._proj_size == 64
        assert len(det._hiddens) == 3
        assert all(h.shape == (48,) for h in det._hiddens)
        # Per-layer GRU weight shapes: layer 0 takes proj_size, rest H
        assert det._grus[0][0].shape == (3, 64, 48)
        assert det._grus[1][0].shape == (3, 48, 48)
        assert det._grus[2][0].shape == (3, 48, 48)
        # Each head: (H → 1)
        for W_h, b_h in det._heads:
            assert W_h.shape == (48, 1)
            assert b_h.shape == (1,)

    def test_per_head_thresholds_stored(self, tmp_path):
        weights_path = tmp_path / 'md.npz'
        _save_multidepth_npz(weights_path, I=216, P=64, H=48, L=3)
        det = MultiDepthBeatRNNDetector(
            weights_path, threshold=0.3,
            downbeat_threshold=0.15, beat_threshold=0.30,
            onset_threshold=0.30)
        assert det._th_downbeat == 0.15
        assert det._th_beat == 0.30
        assert det._th_onset == 0.30

    def test_reset_bands_clears_all_hiddens(self, tmp_path):
        weights_path = tmp_path / 'md.npz'
        _save_multidepth_npz(weights_path, I=216, P=64, H=48, L=3)
        det = MultiDepthBeatRNNDetector(weights_path)
        # Dirty all hidden states
        for h in det._hiddens:
            h[:] = 1.0
        det.reset_bands()
        for h in det._hiddens:
            assert (h == 0).all()


class TestLoadBeatRNNFactory:
    """The factory reads the architecture field and dispatches to the
    right detector class. Default (no architecture field) = single-arch
    for back-compat with pre-2026-05-29 checkpoints."""

    def test_dispatches_to_multidepth(self, tmp_path):
        weights_path = tmp_path / 'md.npz'
        _save_multidepth_npz(weights_path, I=216, P=64, H=48, L=3,
                              architecture='multidepth')
        det = load_beat_rnn(weights_path)
        assert isinstance(det, MultiDepthBeatRNNDetector)

    def test_explicit_single_arch_dispatches_to_base(self, tmp_path):
        """A file with architecture='single' explicitly marked routes
        to BeatRNNDetector (the base, not the multidepth subclass).
        Synthesized as v3 shape so the base loader has dims to discover."""
        weights_path = tmp_path / 'single.npz'
        flat = _make_synthetic_weights(216, 64, 48, 3)
        np.savez(weights_path, weights=flat, next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64, hidden_size=48, n_classes=3,
                 architecture=np.array('single'))
        det = load_beat_rnn(weights_path)
        # NOT MultiDepthBeatRNNDetector — the factory must distinguish them
        assert type(det) is BeatRNNDetector

    def test_no_architecture_field_defaults_to_single(self, tmp_path):
        """Legacy files (no architecture field) load as single-arch.
        Critical for not breaking the v3 deployment."""
        weights_path = tmp_path / 'legacy.npz'
        flat = _make_synthetic_weights(216, 64, 48, 3)
        # No architecture field — legacy style
        np.savez(weights_path, weights=flat, next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64, hidden_size=48, n_classes=3)
        det = load_beat_rnn(weights_path)
        assert type(det) is BeatRNNDetector

    def test_unknown_architecture_raises(self, tmp_path):
        weights_path = tmp_path / 'bad.npz'
        flat = _make_synthetic_weights(216, 64, 48, 3)
        np.savez(weights_path, weights=flat, next_epoch=0, best_val_loss=0.0,
                 input_size=216, proj_size=64, hidden_size=48, n_classes=3,
                 architecture=np.array('totally-new-architecture'))
        with pytest.raises(ValueError, match='unknown architecture'):
            load_beat_rnn(weights_path)

    def test_kwargs_forward_to_detector(self, tmp_path):
        """Per-head thresholds passed through the factory reach the
        detector — guards against the factory dropping kwargs."""
        weights_path = tmp_path / 'md.npz'
        _save_multidepth_npz(weights_path, I=216, P=64, H=48, L=3)
        det = load_beat_rnn(weights_path, downbeat_threshold=0.11,
                             beat_threshold=0.22, onset_threshold=0.33)
        assert det._th_downbeat == 0.11
        assert det._th_beat == 0.22
        assert det._th_onset == 0.33


class TestMultiDepthHeadRouting:
    """Verify the head-index → (downbeat/beat/onset) mapping in
    _forward_step matches the training convention. If the mapping
    silently swaps, the detector would misclassify every event without
    raising — high-priority test."""

    def test_head_routing_matches_training_convention(self, tmp_path):
        """Construct a synthetic file where each head's output linear
        is set so that — given identical hidden states — head[0]
        produces logit 100, head[1] produces logit 0, head[2] produces
        logit -100. After sigmoid: head[0]≈1.0, head[1]=0.5, head[2]≈0.

        Multi-depth maps: head 0 → onset, head 1 → beat, head 2 →
        downbeat. So _forward_step should return (downbeat≈0,
        beat=0.5, onset≈1.0).
        """
        I, P, H, L = 216, 64, 48, 3
        n_heads = L
        # Start with random weights for everything BUT the heads
        rng = np.random.default_rng(42)
        flat = rng.standard_normal(
            _multidepth_param_count(I, P, H, L, n_heads)).astype(np.float32)
        # Compute the head section offset — everything up to but not
        # including the first head's W:
        head_start = I * P + P  # input linear
        for k in range(L):
            in_k = P if k == 0 else H
            head_start += 3 * in_k * H + 3 * H * H + 6 * H
        # Heads in order: head_0 (H, 1) + b_0 (1), head_1 (...), head_2 (...)
        # We want each head's bias to dominate, regardless of hidden state.
        # Zero the weights (so hidden doesn't matter) and set the biases.
        for k in range(n_heads):
            off = head_start + k * (H + 1)
            flat[off:off + H] = 0.0  # W = 0
            # Bias: pick a value the sigmoid maps to a known activation
            flat[off + H] = (100.0, 0.0, -100.0)[k]

        weights_path = tmp_path / 'routing.npz'
        _save_multidepth_npz(weights_path, I=I, P=P, H=H, L=L, flat=flat)
        det = MultiDepthBeatRNNDetector(weights_path)

        # Run one frame — input doesn't matter (heads zero out hidden)
        x = np.zeros(I, dtype=np.float32)
        downbeat, beat, onset = det._forward_step(x)
        assert onset > 0.99,    f'onset should ≈ sigmoid(100) ≈ 1; got {onset}'
        assert abs(beat - 0.5) < 0.01, f'beat should = sigmoid(0) = 0.5; got {beat}'
        assert downbeat < 0.01, f'downbeat should ≈ sigmoid(-100) ≈ 0; got {downbeat}'
