"""Unit tests for the response library (flame_sheep_audio.response)."""

from __future__ import annotations

import numpy as np
import pytest

from flame_sheep_audio.response import (
    EMA, AsymmetricEnvelope, OnsetDensity, Delta, MelCentroid, Normalize,
    _alpha_from_time, _beat_to_seconds,
)

HOP_TIME = 512 / 48000  # ~0.01067s — standard frame duration


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_converges_to_constant(self):
        ema = EMA(time_constant=0.1, hop_time=HOP_TIME)
        for _ in range(1000):
            ema.update(1.0)
        assert abs(ema.value - 1.0) < 1e-6

    def test_first_frame_initializes(self):
        ema = EMA(time_constant=1.0, hop_time=HOP_TIME)
        result = ema.update(5.0)
        assert result == 5.0

    def test_time_constant_decay(self):
        """After one time constant, should reach ~63% of a step."""
        tc = 0.5  # 500ms
        ema = EMA(time_constant=tc, hop_time=HOP_TIME)
        ema.update(0.0)  # initialize at 0

        # Step to 1.0 for tc seconds worth of frames
        n_frames = int(tc / HOP_TIME)
        for _ in range(n_frames):
            ema.update(1.0)

        # Should be near 1 - 1/e ≈ 0.632
        assert 0.55 < ema.value < 0.72

    def test_instant_when_tc_zero(self):
        ema = EMA(time_constant=0.0, hop_time=HOP_TIME)
        ema.update(0.0)
        result = ema.update(1.0)
        assert result == 1.0

    def test_beat_unit(self):
        """At 120 BPM, 1 beat = 0.5s. At 60 BPM, 1 beat = 1.0s."""
        ema_fast = EMA(time_constant=1.0, hop_time=HOP_TIME, unit='beats')
        ema_slow = EMA(time_constant=1.0, hop_time=HOP_TIME, unit='beats')

        ema_fast.update(0.0)
        ema_slow.update(0.0)

        # Run 50 frames at different BPMs
        for _ in range(50):
            ema_fast.update(1.0, bpm=120.0)  # 1 beat = 0.5s
            ema_slow.update(1.0, bpm=60.0)   # 1 beat = 1.0s

        # Faster BPM → shorter time constant → converges faster
        assert ema_fast.value > ema_slow.value

    def test_reset(self):
        ema = EMA(time_constant=0.5, hop_time=HOP_TIME)
        ema.update(10.0)
        ema.reset(0.0)
        assert ema.value == 0.0


# ---------------------------------------------------------------------------
# AsymmetricEnvelope
# ---------------------------------------------------------------------------

class TestAsymmetricEnvelope:
    def test_fast_attack_slow_release(self):
        env = AsymmetricEnvelope(attack=0.01, release=1.0, hop_time=HOP_TIME)
        env.update(0.0)

        # Fast attack — should track rising signal quickly
        for _ in range(10):
            env.update(1.0)
        after_attack = env.value
        assert after_attack > 0.9  # fast attack tracks quickly

        # Slow release — should stay high for a while
        for _ in range(10):
            env.update(0.0)
        after_short_release = env.value
        assert after_short_release > 0.8  # slow release holds

    def test_slow_attack_fast_release(self):
        env = AsymmetricEnvelope(attack=1.0, release=0.01, hop_time=HOP_TIME)
        env.update(0.0)

        # Slow attack — won't track rising signal quickly
        for _ in range(10):
            env.update(1.0)
        after_attack = env.value
        assert after_attack < 0.3  # slow attack lags

        # Fast release — drops quickly
        env._value = 1.0  # force high
        for _ in range(10):
            env.update(0.0)
        assert env.value < 0.1  # fast release

    def test_beat_unit(self):
        env = AsymmetricEnvelope(attack=0.5, release=2.0, hop_time=HOP_TIME, unit='beats')
        env.update(0.0, bpm=120.0)
        env.update(1.0, bpm=120.0)
        # Should not crash, should produce a value between 0 and 1
        assert 0.0 < env.value < 1.0


# ---------------------------------------------------------------------------
# OnsetDensity
# ---------------------------------------------------------------------------

class TestOnsetDensity:
    def test_counts_within_window(self):
        od = OnsetDensity(window=1.0)
        # 4 events in 1 second
        for t in [0.1, 0.3, 0.5, 0.7]:
            od.push(t)
        assert od.density(now=0.9) == pytest.approx(4.0, abs=0.01)

    def test_expires_old_events(self):
        od = OnsetDensity(window=1.0)
        od.push(0.0)
        od.push(0.5)
        od.push(1.5)
        # At t=2.0, only the event at 1.5 is within [1.0, 2.0]
        assert od.density(now=2.0) == pytest.approx(1.0, abs=0.01)

    def test_empty_returns_zero(self):
        od = OnsetDensity(window=1.0)
        assert od.density(now=5.0) == 0.0

    def test_beat_window(self):
        od = OnsetDensity(window=4.0, unit='beats')
        # At 120 BPM, 4 beats = 2 seconds
        for t in [0.0, 0.5, 1.0, 1.5]:
            od.push(t)
        density = od.density(now=1.9, bpm=120.0)
        assert density == pytest.approx(4.0 / 2.0, abs=0.01)  # 4 events / 2s window

    def test_reset(self):
        od = OnsetDensity(window=1.0)
        od.push(0.5)
        od.reset()
        assert od.density(now=0.9) == 0.0


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------

class TestDelta:
    def test_first_frame_returns_zero(self):
        d = Delta()
        assert d.update(5.0) == 0.0

    def test_returns_abs_difference(self):
        d = Delta()
        d.update(1.0)
        assert d.update(3.0) == 2.0
        assert d.update(1.0) == 2.0  # abs(1-3) = 2

    def test_negative_handled(self):
        d = Delta()
        d.update(0.0)
        assert d.update(-5.0) == 5.0

    def test_reset(self):
        d = Delta()
        d.update(10.0)
        d.reset()
        assert d.update(3.0) == 0.0  # first frame after reset


# ---------------------------------------------------------------------------
# MelCentroid
# ---------------------------------------------------------------------------

class TestMelCentroid:
    @pytest.fixture
    def freqs(self):
        """Standard FFT frequencies for 48kHz, 2048-point."""
        return np.linspace(0, 24000, 1025)

    def test_low_content_gives_low_value(self, freqs):
        mc = MelCentroid(freqs)
        # Energy concentrated in low bins (< 200 Hz)
        mag = np.zeros(1025)
        mag[1:10] = 1.0  # ~23-210 Hz
        result = mc.compute(mag)
        # Should be in low mel range
        assert result < 500  # mel value for ~300 Hz

    def test_high_content_gives_high_value(self, freqs):
        mc = MelCentroid(freqs)
        # Energy concentrated in high bins (> 5 kHz)
        mag = np.zeros(1025)
        mag[200:400] = 1.0  # ~4700-9400 Hz
        result = mc.compute(mag)
        assert result > 2000  # mel value for high freq

    def test_reduces_hf_bias_vs_hz_centroid(self, freqs):
        """Mel centroid should weight octaves equally, unlike Hz centroid."""
        mc = MelCentroid(freqs)
        # Two spectra: energy at 100 Hz vs energy at 10 kHz
        mag_low = np.zeros(1025)
        mag_high = np.zeros(1025)

        # Put equal energy at 100 Hz and 10 kHz
        low_bin = int(100 / (24000 / 1025))
        high_bin = int(10000 / (24000 / 1025))
        mag_low[low_bin] = 1.0
        mag_high[high_bin] = 1.0

        mel_low = mc.compute(mag_low)
        mel_high = mc.compute(mag_high)

        # Hz centroid: 10000/100 = 100x ratio
        # Mel centroid: compressed to ~20x (mel is log-ish, not linear)
        ratio = mel_high / max(mel_low, 1.0)
        assert ratio < 25  # much less than the 100x Hz ratio

    def test_silent_returns_zero(self, freqs):
        mc = MelCentroid(freqs)
        assert mc.compute(np.zeros(1025)) == 0.0

    def test_mel_to_hz_roundtrip(self):
        hz_orig = 440.0
        mel_val = MelCentroid.hz_to_mel(hz_orig)
        hz_back = MelCentroid.mel_to_hz(mel_val)
        assert abs(hz_back - hz_orig) < 0.01


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_log_perceptual_range(self):
        assert Normalize.log_perceptual(0.0) == 0.0
        val = Normalize.log_perceptual(1.0)
        assert val == pytest.approx(1.0)  # ln(1+1000)/ln(1001) = 1.0 exactly
        # Values > 1.0 map above 1 (unbounded input)
        assert Normalize.log_perceptual(2.0) > 1.0

    def test_log_perceptual_monotonic(self):
        vals = [Normalize.log_perceptual(x) for x in [0.0, 0.01, 0.1, 0.5, 1.0]]
        assert vals == sorted(vals)

    def test_log_perceptual_array(self):
        arr = np.array([0.0, 0.5, 1.0])
        result = Normalize.log_perceptual(arr)
        assert result.shape == (3,)
        assert result[0] == 0.0
        assert result[2] > result[1] > result[0]

    def test_linear_clamps(self):
        assert Normalize.linear(-1.0, 0.0, 1.0) == 0.0
        assert Normalize.linear(2.0, 0.0, 1.0) == 1.0
        assert Normalize.linear(0.5, 0.0, 1.0) == 0.5

    def test_linear_degenerate(self):
        assert Normalize.linear(5.0, 1.0, 1.0) == 0.0  # hi == lo

    def test_mel_monotonic(self):
        vals = [Normalize.mel(hz) for hz in [20, 100, 440, 1000, 5000, 20000]]
        assert vals == sorted(vals)
        assert vals[0] == pytest.approx(0.0, abs=0.01)
        assert vals[-1] == pytest.approx(1.0, abs=0.01)

    def test_db_floor(self):
        assert Normalize.db(0.0) == 0.0
        assert Normalize.db(1.0) == 1.0  # 0 dB = full scale

    def test_db_at_floor(self):
        # -60 dB = 0.001 amplitude
        val = Normalize.db(0.001, floor=-60.0)
        assert abs(val) < 0.01  # at the floor → ~0

    def test_db_mid(self):
        # -30 dB = 0.0316... amplitude → should map to ~0.5
        val = Normalize.db(10**(-30/20), floor=-60.0)
        assert abs(val - 0.5) < 0.01
