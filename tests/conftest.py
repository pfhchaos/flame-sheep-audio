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


__all__ = ['make_processor', 'make_sine', 'make_silence', 'make_impulse', 'feed_audio']
