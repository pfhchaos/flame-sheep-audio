"""Beat detector conformance tests."""

import numpy as np
import pytest

from flame_sheep_audio._constants import HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumFrame
from flame_sheep_audio.beat_detector import BeatDetectorBase, FluxBeatDetector, PercentileBeatDetector

BIN_COUNTS = [108, 1025, 64]

# All concrete BeatDetectorBase subclasses with bin-count-agnostic factories.
# Used for the parametrized conformance suite below (varies n_bins from 64 to
# 1025 and verifies the detector handles each shape).
DETECTOR_CLASSES = [
    ("flux", lambda freqs: FluxBeatDetector(freqs=freqs)),
    ("percentile", lambda freqs: PercentileBeatDetector(freqs=freqs)),
]

# Subclasses that opt out of the bin-count-parametrized conformance tests
# because they have hard input-shape requirements that those tests can't
# satisfy. BeatRNNDetector + MultiDepthBeatRNNDetector both:
#   - require a 108-bin CQT spectrum (model trained on librosa CQT)
#   - require an on-disk weights file
# Their conformance is covered by ../test_beat_rnn_streaming.py
# (real weights file, real spectrum shape).
#
# BeatNetLiveDetector is exempt for the same reason: it consumes the raw
# waveform (ignoring the magnitude bins entirely) and needs an on-disk
# BeatNet-lite weights file, so the bin-count-parametrized suite can't drive
# it. Covered by tests/test_beatnet_live_detector.py (real weights,
# skip-if-absent), with the particle-filter path covered by
# tests/test_pf_particle_leak.py.
_BIN_COUNT_AGNOSTIC_EXEMPT = {'BeatRNNDetector', 'MultiDepthBeatRNNDetector',
                              'BeatNetLiveDetector'}


def test_all_beat_detectors_registered():
    """Ensure every BeatDetectorBase subclass either has a test entry in
    DETECTOR_CLASSES (bin-count-parametrized suite) or is explicitly
    exempt with its own dedicated tests elsewhere."""
    concrete = {cls.__name__ for cls in BeatDetectorBase.__subclasses__()}
    tested = {factory(np.linspace(20, 20000, 108).astype(np.float32)).__class__.__name__
              for _, factory in DETECTOR_CLASSES}
    accounted_for = tested | _BIN_COUNT_AGNOSTIC_EXEMPT
    missing = concrete - accounted_for
    assert not missing, (
        f"BeatDetectorBase subclasses without test entries: {missing}. "
        f"Either add to DETECTOR_CLASSES (must accept arbitrary n_bins) or "
        f"add to _BIN_COUNT_AGNOSTIC_EXEMPT with a dedicated test elsewhere.")


class TestBeatDetectorConformance:
    """Every beat detector must handle arbitrary bin counts."""

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_detect_returns_list(self, name, factory, n_bins):
        """detect() must return a list of BeatEvents."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        det = factory(freqs)
        frame = SpectrumFrame(
            magnitude=np.abs(np.random.randn(n_bins).astype(np.float32)),
            flux=np.abs(np.random.randn(n_bins).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        events = det.detect(frame)
        assert isinstance(events, list)

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_detect_silent_input(self, name, factory, n_bins):
        """Silence must not crash and should produce no events."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        det = factory(freqs)
        frame = SpectrumFrame(
            magnitude=np.zeros(n_bins, dtype=np.float32),
            flux=np.zeros(n_bins, dtype=np.float32),
            waveform=np.zeros(HOP_SIZE, dtype=np.float32),
        )
        events = det.detect(frame)
        assert isinstance(events, list)
        assert len(events) == 0

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_reset_then_detect(self, name, factory, n_bins):
        """reset_bands() then detect() must not crash."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        det = factory(freqs)
        # Feed some frames
        for _ in range(20):
            frame = SpectrumFrame(
                magnitude=np.abs(np.random.randn(n_bins).astype(np.float32)),
                flux=np.abs(np.random.randn(n_bins).astype(np.float32)),
                waveform=np.random.randn(HOP_SIZE).astype(np.float32),
            )
            det.detect(frame)
        det.reset_bands()
        # Should work after reset
        events = det.detect(frame)
        assert isinstance(events, list)

    @pytest.mark.parametrize("name,factory", DETECTOR_CLASSES)
    def test_events_have_valid_kinds(self, name, factory):
        """BeatEvent.kind must be a string."""
        freqs = np.linspace(20, 20000, 108).astype(np.float32)
        det = factory(freqs)
        # Feed enough frames with signal to trigger events
        for _ in range(50):
            frame = SpectrumFrame(
                magnitude=np.abs(np.random.randn(108).astype(np.float32)) * 0.5,
                flux=np.abs(np.random.randn(108).astype(np.float32)) * 0.5,
                waveform=np.random.randn(HOP_SIZE).astype(np.float32),
            )
            events = det.detect(frame)
            for e in events:
                assert isinstance(e.kind, str)
                assert isinstance(e.energy, (int, float))
