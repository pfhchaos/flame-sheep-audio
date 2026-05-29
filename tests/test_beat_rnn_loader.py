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
    _LEGACY_ARCH_BY_FLAT_LEN,
    _discover_arch,
    _unpack_weights,
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
