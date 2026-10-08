"""Dedicated conformance coverage for BeatNetLiveDetector.

It is exempt from the bin-count-parametrized suite in
tests/conformance/test_beat_detector.py because it consumes the raw waveform
(not the magnitude bins, which it ignores) and requires an on-disk BeatNet-lite
weights file — so that suite can't drive it. This mirrors the skip-if-no-weights
pattern used for the RNN detectors in tests/audio/test_beat_rnn_streaming.py.
The particle-filter path it can run is covered separately by
test_pf_particle_leak.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from flame_sheep_audio._constants import HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumFrame
from flame_sheep_audio.beat_detector import BeatDetectorBase
from flame_sheep_audio.beat_detector_beatnet import BeatNetLiveDetector


def _detector_or_skip() -> BeatNetLiveDetector:
    """Construct from the packaged BeatNet-lite weights, or skip if absent."""
    try:
        return BeatNetLiveDetector()  # model_index=1, particle filter off
    except FileNotFoundError as e:
        pytest.skip(f'BeatNet-lite weights not present: {e}')


def _frame(waveform: np.ndarray) -> SpectrumFrame:
    # The detector reads the raw waveform; magnitude/flux bins are ignored, so
    # their shape is irrelevant here (108 just matches the daemon's CQT width).
    return SpectrumFrame(
        magnitude=np.zeros(108, dtype=np.float32),
        flux=np.zeros(108, dtype=np.float32),
        waveform=waveform.astype(np.float32),
    )


def test_is_beatdetector_subclass():
    """It's a real BeatDetectorBase subclass (the thing the registration
    conformance test enumerates)."""
    assert issubclass(BeatNetLiveDetector, BeatDetectorBase)


def test_construction_loads_weights():
    det = _detector_or_skip()
    assert det is not None


def test_detect_returns_list():
    """detect() honors the BeatDetectorBase contract (returns a list) across
    enough hops to cross a BeatNet feature-hop boundary."""
    det = _detector_or_skip()
    rng = np.random.default_rng(20260101)
    for _ in range(64):
        events = det.detect(_frame(rng.standard_normal(HOP_SIZE)))
        assert isinstance(events, list)


def test_silence_does_not_crash():
    """Silence must not crash and must still return a list."""
    det = _detector_or_skip()
    for _ in range(64):
        events = det.detect(_frame(np.zeros(HOP_SIZE)))
        assert isinstance(events, list)
