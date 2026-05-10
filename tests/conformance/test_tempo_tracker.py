"""Tempo tracker conformance tests."""

import numpy as np
import pytest

from flame_sheep_audio.tempo import TempoTrackerBase, PercivalTempoTracker

TEMPO_CLASSES = [
    ("percival", lambda: PercivalTempoTracker(log_compress=False)),
]

try:
    from flame_sheep_audio.tempo_btrack import BTrackTempoTracker
    TEMPO_CLASSES.append(("btrack", lambda: BTrackTempoTracker()))
except ImportError:
    pass

try:
    from flame_sheep_audio.tempo_acf import AutocorrelationTempoTracker
    TEMPO_CLASSES.append(("acf", lambda: AutocorrelationTempoTracker()))
except ImportError:
    pass


class TestTempoTrackerConformance:
    """Every tempo tracker must implement the full interface."""

    @pytest.mark.parametrize("name,factory", TEMPO_CLASSES)
    def test_feed_and_properties(self, name, factory):
        """feed() + read all properties without crashing."""
        t = factory()
        for _ in range(50):
            t.feed(0.1)
        assert isinstance(t.bpm, (int, float))
        assert isinstance(t.effective_bpm, (int, float))
        assert isinstance(t.confidence, (int, float))
        assert isinstance(t.phase, (int, float))
        assert isinstance(t.locked, bool)
        assert isinstance(t.saturated, bool)
        assert isinstance(t.bpm_delta, (int, float))

    @pytest.mark.parametrize("name,factory", TEMPO_CLASSES)
    def test_reset(self, name, factory):
        t = factory()
        for _ in range(50):
            t.feed(0.5)
        t.reset()
        # Should not crash after reset
        t.feed(0.1)
        assert isinstance(t.bpm, (int, float))

    @pytest.mark.parametrize("name,factory", TEMPO_CLASSES)
    def test_song_started(self, name, factory):
        t = factory()
        for _ in range(50):
            t.feed(0.5)
        t.song_started()
        t.feed(0.1)
        assert isinstance(t.bpm, (int, float))

    @pytest.mark.parametrize("name,factory", TEMPO_CLASSES)
    def test_hint_tempo(self, name, factory):
        t = factory()
        t.hint_tempo(120.0)
        t.feed(0.1)
        # Should not crash

    @pytest.mark.parametrize("name,factory", TEMPO_CLASSES)
    def test_silence(self, name, factory):
        """Feeding silence should not crash or produce NaN."""
        t = factory()
        for _ in range(100):
            t.feed(0.0)
        assert np.isfinite(t.bpm)
        assert np.isfinite(t.effective_bpm)
