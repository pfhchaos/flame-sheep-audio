"""
Pytest conftest for flame_sheep_audio tests.

Re-exports audio helpers so they're discoverable as imports.
"""

from audio_helpers import make_processor, make_sine, make_silence, make_impulse, feed_audio

__all__ = ['make_processor', 'make_sine', 'make_silence', 'make_impulse', 'feed_audio']
