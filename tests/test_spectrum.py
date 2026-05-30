"""
Tests for audio processing and beat detection.
Uses synthetic audio — no hardware or PipeWire required.
"""

import numpy as np
import pytest
from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE
from flame_sheep_audio._cqt_engine import CqtEngine

from audio_helpers import make_processor, make_sine, make_silence, feed_audio

# AudioProcessor now uses CqtEngine, which produces 108 log-spaced bins
# (~C1 to ~C9). The old FFT_SIZE-based N_BINS=1025 no longer matches the
# spectrum shape. Tests below derive their bin masks from the engine's
# bin_centers so they stay correct as long as CqtEngine's range covers
# the test frequencies.
CQT_BIN_CENTERS = CqtEngine().bin_centers
CQT_N_BINS = len(CQT_BIN_CENTERS)


# ----------------------------------------------------------------
# FFT / spectrum
# ----------------------------------------------------------------

class TestSpectrum:

    def test_silence_produces_low_spectrum(self):
        proc = make_processor()
        feed_audio(proc, make_silence(FFT_SIZE))
        proc.process()
        assert proc.spectrum.max() < 1.0

    def test_sine_peaks_at_correct_bin(self):
        """A 100Hz sine should produce a peak in the low band.

        Mid mask starts at 400Hz (not 150Hz like the old FFT-based test)
        because CQT filters have broad Q at low frequencies — a 100Hz
        tone smears across 110-200Hz bins, putting non-trivial energy
        in any mask that starts at 150Hz. 400Hz is well clear of the
        100Hz tone's first-octave bleed."""
        proc = make_processor()
        feed_audio(proc, make_sine(100, FFT_SIZE * 4))
        proc.process()
        low_mask = (CQT_BIN_CENTERS >= 20) & (CQT_BIN_CENTERS < 150)
        mid_mask = (CQT_BIN_CENTERS >= 400) & (CQT_BIN_CENTERS < 2000)
        low_energy = proc.spectrum[low_mask].max()
        mid_energy = proc.spectrum[mid_mask].max()
        assert low_energy > mid_energy * 5, \
            f"100Hz sine should dominate low band: low={low_energy:.1f} mid={mid_energy:.1f}"

    def test_high_freq_sine_peaks_in_high(self):
        """A 10kHz sine should produce a peak in the high band."""
        proc = make_processor()
        feed_audio(proc, make_sine(10000, FFT_SIZE * 4))
        proc.process()
        high_mask = CQT_BIN_CENTERS >= 8000
        low_mask = (CQT_BIN_CENTERS >= 20) & (CQT_BIN_CENTERS < 150)
        high_energy = proc.spectrum[high_mask].max()
        low_energy = proc.spectrum[low_mask].max()
        assert high_energy > low_energy * 5, \
            f"10kHz sine should dominate high band"

    def test_spectrum_length(self):
        proc = make_processor()
        feed_audio(proc, make_silence(FFT_SIZE))
        proc.process()
        assert len(proc.spectrum) == CQT_N_BINS


# ----------------------------------------------------------------
# Beat detection
# ----------------------------------------------------------------

class TestBeatDetection:

    def _warm_up(self, proc, signal):
        """Feed enough history that onset detection has a baseline."""
        for _ in range(50):
            feed_audio(proc, signal)
            proc.process()

    def test_silence_no_beats(self):
        proc = make_processor()
        self._warm_up(proc, make_silence(FFT_SIZE))
        events = proc.process()
        assert len(events) == 0, f"silence should produce no beats, got {events}"

    def test_sustained_tone_no_repeated_beats(self):
        """Sustained tone fires once then cooldown prevents re-firing."""
        proc  = make_processor()
        tone  = make_sine(80, FFT_SIZE, amplitude=0.8)
        self._warm_up(proc, make_silence(FFT_SIZE))
        feed_audio(proc, tone)
        first = proc.process()
        subsequent_hits = 0
        for _ in range(10):
            feed_audio(proc, tone)
            events = proc.process()
            subsequent_hits += len(events)
        assert subsequent_hits == 0, \
            f"sustained tone should not re-trigger after cooldown, got {subsequent_hits} hits"

    def test_kick_frequency_triggers_kick_band(self):
        """Strong 80Hz onset should trigger low, not mid or high."""
        proc      = make_processor()
        silence   = make_silence(FFT_SIZE)
        kick_tone = make_sine(80, FFT_SIZE, amplitude=0.9)
        self._warm_up(proc, silence)
        feed_audio(proc, kick_tone)
        events = proc.process()
        kinds  = [e.kind for e in events]
        assert 'low' in kinds, f"80Hz onset should trigger low, got {kinds}"
        assert 'high' not in kinds, f"80Hz should not trigger high, got {kinds}"

    def test_beat_energy_normalized(self):
        """Beat energy should be in 0..1 range."""
        proc      = make_processor()
        silence   = make_silence(FFT_SIZE)
        kick_tone = make_sine(80, FFT_SIZE, amplitude=0.9)
        self._warm_up(proc, silence)
        feed_audio(proc, kick_tone)
        events = proc.process()
        for e in events:
            assert 0.0 <= e.energy <= 1.0, f"energy {e.energy} out of range"

    def test_silence_no_flux(self):
        """Constant signal (zero flux) should never trigger beats."""
        proc = make_processor()
        tone = make_sine(80, FFT_SIZE, amplitude=0.5)
        self._warm_up(proc, tone)
        for _ in range(10):
            feed_audio(proc, tone)
            events = proc.process()
            assert len(events) == 0, \
                f"constant tone should have zero flux, got {events}"
