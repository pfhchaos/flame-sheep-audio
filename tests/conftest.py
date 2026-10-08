"""
Pytest conftest for flame_sheep_audio tests.

Re-exports audio helpers so they're discoverable as imports.

Also resets the global audio cfg to DEFAULTS at the start of every test.
Without this, the test suite reads ~/.config/flame-sheep/audio.toml and
inherits the user's runtime overrides (detector.kind, custom thresholds,
HPSS method, etc.) — making behavior non-reproducible across machines
and breaking tests when a user pins an experimental config.
"""

import sys
from pathlib import Path

# Ensure audio_helpers (sibling file) is importable when this conftest
# loads before the project-root tests/conftest.py — which is the case
# under --import-mode=importlib + subset test runs (e.g.
# `pytest flame_sheep_audio/tests/test_foo.py`).
_audio_tests_dir = str(Path(__file__).resolve().parent)
if _audio_tests_dir not in sys.path:
    sys.path.insert(0, _audio_tests_dir)

import pytest

from audio_helpers import make_processor, make_sine, make_silence, make_impulse, feed_audio
from flame_sheep_audio.config import cfg


@pytest.fixture(autouse=True)
def _hermetic_audio_config():
    """Reset the global audio cfg to DEFAULTS before each test, undoing
    any user overrides loaded at import time. Per-test scope so a test
    that mutates cfg can't leak that mutation into the next test."""
    cfg.reset_to_defaults()
    yield
    cfg.reset_to_defaults()


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    """Point the XDG data/config dirs at a fresh per-test temp location so
    no test reads or writes the real ~/.local/share/flame-sheep.

    Critically this isolates the AGC's persisted gain state. Every
    AudioProcessor builds an AudioLevelAgc that, on construction, LOADS
    slow_rms from data_dir()/agc_state.json and periodically SAVES it back
    (processor.py — "Not optional: signal conditioning"). That on-disk file
    is shared across tests AND across pytest runs, so without isolation the
    gain calibration leaks between them — shifting the detection thresholds
    and making the threshold-sensitive beat-detection tests
    (test_beat_engine, test_pipeline) flaky and order-dependent (it was the
    cross-run 153/153-vs-145/153 non-determinism). A fresh empty dir per
    test means the AGC finds no saved state and bootstraps at target every
    time => deterministic, order-independent. Production persistence is
    intentional and untouched; this only sandboxes the tests."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg-data'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg-config'))
    yield


__all__ = ['make_processor', 'make_sine', 'make_silence', 'make_impulse', 'feed_audio']
