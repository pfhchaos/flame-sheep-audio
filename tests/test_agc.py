"""Tests for the source-level audio AGC."""
from __future__ import annotations

import json

import numpy as np
import pytest

from flame_sheep_audio._agc import AudioLevelAgc


SR = 48000


def _sine(amp: float, n_seconds: float = 1.0, freq: float = 440.0,
          sr: int = SR) -> np.ndarray:
    """Pure tone at target amplitude."""
    t = np.arange(int(sr * n_seconds)) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_silence_passes_through_zeroed(tmp_path):
    """Sub-noise-floor input becomes zero; EMA is not updated."""
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        target_rms=0.1, noise_floor=0.01)
    # Construct samples that are *all* in [-0.005, 0.005] so they
    # cleanly fall below noise_floor=0.01. Using uniform instead of
    # gaussian to avoid tail samples that exceed the floor.
    rng = np.random.default_rng(0)
    pcm = (rng.uniform(-0.005, 0.005, size=1024)).astype(np.float32)
    assert np.all(np.abs(pcm) < 0.01)  # sanity: actually below floor
    initial_rms = agc.slow_rms
    out = agc.process(pcm)
    assert np.all(out == 0.0)
    assert agc.slow_rms == initial_rms


def test_quiet_audio_gets_amplified(tmp_path):
    """A signal well above noise floor but below target gets gain > 1."""
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        target_rms=0.1, noise_floor=0.001,
                        time_constant_sec=1.0)  # fast for the test
    # Feed 5 seconds of audio in 1024-sample blocks so the EMA settles.
    quiet = _sine(amp=0.01, n_seconds=5.0)
    block_size = 1024
    for i in range(0, len(quiet), block_size):
        agc.process(quiet[i:i + block_size])
    # Quiet input → low slow_rms → high gain
    assert agc.current_gain > 5.0
    # And the EMA should have settled near the input RMS (sine RMS = amp/sqrt(2))
    expected_rms = 0.01 / np.sqrt(2)
    assert abs(agc.slow_rms - expected_rms) < 0.005


def test_loud_audio_gets_attenuated(tmp_path):
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        target_rms=0.1, noise_floor=0.001,
                        time_constant_sec=1.0)
    loud = _sine(amp=0.5, n_seconds=5.0)
    block_size = 1024
    for i in range(0, len(loud), block_size):
        agc.process(loud[i:i + block_size])
    # Loud input → gain < 1
    assert agc.current_gain < 1.0
    # Sine RMS = 0.5/sqrt(2) ≈ 0.354
    assert abs(agc.slow_rms - 0.5 / np.sqrt(2)) < 0.02


def test_silence_holds_baseline(tmp_path):
    """After a song's level is learned, a silent passage doesn't drop
    the EMA. The gain stays where it was."""
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        target_rms=0.1, noise_floor=0.001,
                        time_constant_sec=1.0)
    # Settle on music
    music = _sine(amp=0.05, n_seconds=3.0)
    for i in range(0, len(music), 1024):
        agc.process(music[i:i + 1024])
    settled = agc.slow_rms

    # Now feed 5 seconds of silence (below noise floor)
    silence = np.zeros(SR * 5, dtype=np.float32)
    for i in range(0, len(silence), 1024):
        agc.process(silence[i:i + 1024])

    # slow_rms unchanged
    assert agc.slow_rms == settled


def test_dynamic_range_passes_through(tmp_path):
    """Within-block dynamics are preserved up to the gain multiply.
    Specifically: if a block has both quiet and loud samples, their
    relative ratio is preserved on output."""
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        target_rms=0.1, noise_floor=0.0001,
                        time_constant_sec=1.0)
    # Quiet portion then loud portion, same block
    pcm = np.concatenate([
        _sine(amp=0.01, n_seconds=0.1),
        _sine(amp=0.10, n_seconds=0.1),
    ])
    out = agc.process(pcm)
    # Ratio of peak between halves should be preserved
    quiet_peak = float(np.abs(out[:SR // 10]).max())
    loud_peak = float(np.abs(out[SR // 10:]).max())
    # Input ratio is ~10x; output ratio should also be ~10x.
    assert 8.0 < loud_peak / max(quiet_peak, 1e-9) < 12.0


def test_persistence_roundtrip(tmp_path):
    """slow_rms saved on one instance is loaded by the next."""
    state_path = tmp_path / 'agc_state.json'
    agc1 = AudioLevelAgc(sample_rate=SR, persist_path=state_path,
                         target_rms=0.1, noise_floor=0.001,
                         time_constant_sec=0.5,
                         save_interval_sec=0.0)  # save every block
    music = _sine(amp=0.05, n_seconds=2.0)
    for i in range(0, len(music), 1024):
        agc1.process(music[i:i + 1024])
    settled = agc1.slow_rms
    # The file should exist now
    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert abs(payload['slow_rms'] - settled) < 1e-6

    # Second instance loads from disk
    agc2 = AudioLevelAgc(sample_rate=SR, persist_path=state_path)
    assert abs(agc2.slow_rms - settled) < 1e-6


def test_invalid_persisted_value_bootstraps(tmp_path):
    """If the saved state is garbage, fall back to target_rms."""
    state_path = tmp_path / 'agc_state.json'
    state_path.write_text(json.dumps({'slow_rms': -42.0}))
    agc = AudioLevelAgc(sample_rate=SR, persist_path=state_path,
                        target_rms=0.1)
    assert agc.slow_rms == 0.1


def test_missing_persist_file_bootstraps(tmp_path):
    """No saved state file → init at target_rms (gain = 1)."""
    state_path = tmp_path / 'does_not_exist.json'
    agc = AudioLevelAgc(sample_rate=SR, persist_path=state_path,
                        target_rms=0.1)
    assert agc.slow_rms == 0.1
    # No file accidentally created
    assert not state_path.exists()


def test_persist_path_none_disables_save(tmp_path):
    """persist_path=None: never reads or writes state. Useful for
    tests / deterministic runs."""
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        save_interval_sec=0.0)
    music = _sine(amp=0.05, n_seconds=1.0)
    for i in range(0, len(music), 1024):
        agc.process(music[i:i + 1024])
    # No file in cwd or tmp dir
    assert not (tmp_path / 'agc_state.json').exists()


def test_short_time_constant_adapts_fast():
    """Sanity: a 0.1s time constant tracks within a fraction of a second."""
    agc = AudioLevelAgc(sample_rate=SR, persist_path=None,
                        target_rms=0.1, noise_floor=0.001,
                        time_constant_sec=0.1)
    # Feed loud audio for 0.5 seconds; expect slow_rms to settle near input.
    loud = _sine(amp=0.2, n_seconds=0.5)
    for i in range(0, len(loud), 1024):
        agc.process(loud[i:i + 1024])
    expected_rms = 0.2 / np.sqrt(2)
    # After 5× the time constant, EMA is >99% of step → very close
    assert abs(agc.slow_rms - expected_rms) < 0.01
