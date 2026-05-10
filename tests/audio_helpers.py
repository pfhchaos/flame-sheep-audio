"""
Shared test helpers for flame_sheep_audio tests.

Plain functions (not pytest fixtures) so they can be imported by name
from any test file that has this directory on sys.path.
"""

import numpy as np

from flame_sheep_audio import AudioProcessor, SAMPLE_RATE
from flame_sheep_audio.source import FeedSource


def make_processor(adaptive: bool = False, sharpness: bool = True) -> AudioProcessor:
    """Create an AudioProcessor with a FeedSource (no audio hardware)."""
    return AudioProcessor(source=FeedSource(), adaptive=adaptive, sharpness=sharpness)


def make_sine(freq: float, duration_samples: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave at given frequency."""
    t = np.arange(duration_samples) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration_samples: int) -> np.ndarray:
    return np.zeros(duration_samples, dtype=np.float32)


def make_impulse(duration_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Single-sample impulse -- broadband onset."""
    sig = np.zeros(duration_samples, dtype=np.float32)
    sig[duration_samples // 2] = amplitude
    return sig


def feed_audio(processor: AudioProcessor, signal: np.ndarray):
    """Feed signal into processor buffer."""
    processor.feed(signal)
