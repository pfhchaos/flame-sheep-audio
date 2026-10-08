"""Tests for AudioProcessor's detector dispatch based on cfg.detector.kind.

Stage 3 of docs/beat_rnn_deploy_plan.md: the daemon must select between
PercentileBeatDetector, FluxBeatDetector, and BeatRNNDetector via the
config field, with PercentileBeatDetector as the safe default.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parents[2]

from flame_sheep_audio.config import cfg
from flame_sheep_audio.source import FeedSource


WEIGHTS = _PROJECT / 'flame_sheep' / 'data' / 'beat_rnn_continuous.npz'


def _make_processor():
    """Build an AudioProcessor with FeedSource so no real audio device
    is required for the test. Returns the processor."""
    from flame_sheep_audio.processor import AudioProcessor
    return AudioProcessor(source=FeedSource())


def test_default_is_percentile():
    """When detector.kind is set to 'percentile' the dispatch picks
    PercentileBeatDetector. (cfg.reload() may apply a user audio.toml
    that overrides the default, so we force 'percentile' explicitly
    here rather than relying on the as-shipped default.)"""
    cfg.reload()
    cfg.detector.kind = 'percentile'
    try:
        proc = _make_processor()
        from flame_sheep_audio.beat_detector import PercentileBeatDetector
        assert isinstance(proc._detector, PercentileBeatDetector)
    finally:
        cfg.reload()


def test_explicit_flux_selects_flux():
    cfg.reload()
    cfg.detector.kind = 'flux'
    try:
        proc = _make_processor()
        from flame_sheep_audio.beat_detector import FluxBeatDetector
        assert isinstance(proc._detector, FluxBeatDetector)
    finally:
        cfg.reload()


def test_rnn_requires_weights_path():
    """detector.kind='rnn' with empty weights path must raise; the
    daemon should fail loud rather than silently fall back."""
    cfg.reload()
    cfg.detector.kind = 'rnn'
    cfg.detector.rnn_weights_path = ''   # force empty even if user toml sets one
    try:
        with pytest.raises(ValueError, match='rnn_weights_path'):
            _make_processor()
    finally:
        cfg.reload()


def test_rnn_with_weights_path_constructs():
    if not WEIGHTS.exists():
        pytest.skip(f'weights not present at {WEIGHTS}')
    cfg.reload()
    cfg.detector.kind = 'rnn'
    cfg.detector.rnn_weights_path = str(WEIGHTS)
    try:
        proc = _make_processor()
        from flame_sheep_audio.beat_rnn import BeatRNNDetector
        assert isinstance(proc._detector, BeatRNNDetector)
    finally:
        cfg.reload()


def test_unknown_kind_falls_back_to_percentile():
    """A misconfigured detector.kind should warn and fall back to
    percentile, not crash. Keeps the daemon resilient to typos."""
    cfg.reload()
    cfg.detector.kind = 'notarealthing'
    try:
        proc = _make_processor()
        from flame_sheep_audio.beat_detector import PercentileBeatDetector
        assert isinstance(proc._detector, PercentileBeatDetector)
    finally:
        cfg.reload()
