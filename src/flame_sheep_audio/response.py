"""Response library — viz-side signal processing toolkit.

Provides building blocks for visualization authors to shape raw audio engine
data into reactive visuals. All classes are stateful, lightweight, and designed
for per-frame calling at ~93 fps (HOP_SIZE/SAMPLE_RATE cadence).

Time constants can be specified in seconds or beats. Beat-relative times adapt
dynamically to tempo changes — "smooth over 2 beats" is faster at 180 BPM
than at 60 BPM.
"""

from __future__ import annotations

from collections import deque
from math import exp, log, log2, sqrt

import numpy as np


def _beat_to_seconds(beats: float, bpm: float) -> float:
    """Convert a duration in beats to seconds at the given tempo."""
    return beats * 60.0 / max(bpm, 1.0)


def _alpha_from_time(time_constant: float, hop_time: float) -> float:
    """Compute EMA alpha from time constant and hop duration.

    Alpha represents how much weight goes to the NEW sample.
    time_constant is the time (seconds) to reach ~63% of a step change.
    """
    if time_constant <= 0.0:
        return 1.0  # instant
    return 1.0 - exp(-hop_time / time_constant)


class EMA:
    """Exponential moving average with configurable time constant.

    Args:
        time_constant: Smoothing time — seconds or beats depending on unit.
        hop_time: Duration of one frame in seconds (HOP_SIZE / SAMPLE_RATE).
        unit: 'seconds' or 'beats'. If 'beats', pass bpm to update().
    """

    __slots__ = ('_tc', '_hop_time', '_unit', '_value', '_initialized')

    def __init__(self, time_constant: float, hop_time: float, unit: str = 'seconds') -> None:
        self._tc = time_constant
        self._hop_time = hop_time
        self._unit = unit
        self._value = 0.0
        self._initialized = False

    def update(self, value: float, bpm: float = 120.0) -> float:
        """Feed a new sample, return smoothed value."""
        if not self._initialized:
            self._value = value
            self._initialized = True
            return value

        tc = self._tc if self._unit == 'seconds' else _beat_to_seconds(self._tc, bpm)
        alpha = _alpha_from_time(tc, self._hop_time)
        self._value += alpha * (value - self._value)
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self, value: float = 0.0) -> None:
        self._value = value
        self._initialized = value != 0.0


class AsymmetricEnvelope:
    """Attack/release envelope — different rates for rising vs falling.

    Args:
        attack: Time constant for rising signal (seconds or beats).
        release: Time constant for falling signal (seconds or beats).
        hop_time: Frame duration in seconds.
        unit: 'seconds' or 'beats'.
    """

    __slots__ = ('_attack', '_release', '_hop_time', '_unit', '_value', '_initialized')

    def __init__(self, attack: float, release: float, hop_time: float,
                 unit: str = 'seconds') -> None:
        self._attack = attack
        self._release = release
        self._hop_time = hop_time
        self._unit = unit
        self._value = 0.0
        self._initialized = False

    def update(self, value: float, bpm: float = 120.0) -> float:
        """Feed a new sample, return envelope value."""
        if not self._initialized:
            self._value = value
            self._initialized = True
            return value

        if value > self._value:
            tc = self._attack
        else:
            tc = self._release

        if self._unit == 'beats':
            tc = _beat_to_seconds(tc, bpm)

        alpha = _alpha_from_time(tc, self._hop_time)
        self._value += alpha * (value - self._value)
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self, value: float = 0.0) -> None:
        self._value = value
        self._initialized = value != 0.0


class OnsetDensity:
    """Windowed event counter — events per second.

    Args:
        window: Window duration in seconds or beats.
        unit: 'seconds' or 'beats'.
    """

    __slots__ = ('_window', '_unit', '_events')

    def __init__(self, window: float = 2.0, unit: str = 'seconds') -> None:
        self._window = window
        self._unit = unit
        self._events: deque[float] = deque()

    def push(self, timestamp: float) -> None:
        """Record an onset event at the given time."""
        self._events.append(timestamp)

    def density(self, now: float, bpm: float = 120.0) -> float:
        """Return current event density (events per second)."""
        window_sec = (self._window if self._unit == 'seconds'
                      else _beat_to_seconds(self._window, bpm))
        cutoff = now - window_sec
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        if window_sec <= 0.0:
            return 0.0
        return len(self._events) / window_sec

    def reset(self) -> None:
        self._events.clear()


class Delta:
    """Frame-to-frame absolute difference (first derivative magnitude).

    Returns 0.0 on the first frame.
    """

    __slots__ = ('_prev', '_initialized')

    def __init__(self) -> None:
        self._prev = 0.0
        self._initialized = False

    def update(self, value: float) -> float:
        """Feed a new sample, return |current - previous|."""
        if not self._initialized:
            self._prev = value
            self._initialized = True
            return 0.0
        delta = abs(value - self._prev)
        self._prev = value
        return delta

    def reset(self) -> None:
        self._prev = 0.0
        self._initialized = False


class MelCentroid:
    """Spectral centroid in mel space — perceptually uniform pitch weighting.

    Unlike Hz-space centroid, a one-octave shift at 100 Hz has the same
    weight as a one-octave shift at 5000 Hz.

    Args:
        freqs: Bin center frequencies in Hz (from engine).
    """

    __slots__ = ('_mel_freqs', '_valid_mask', '_n_bins')

    def __init__(self, freqs: np.ndarray) -> None:
        self._n_bins = len(freqs)
        # Precompute mel frequencies, masking DC bin (0 Hz)
        self._valid_mask = freqs > 0
        safe_freqs = np.where(self._valid_mask, freqs, 1.0)
        self._mel_freqs = 2595.0 * np.log10(1.0 + safe_freqs / 700.0)
        self._mel_freqs[~self._valid_mask] = 0.0

    def compute(self, magnitude: np.ndarray) -> float:
        """Compute mel-weighted spectral centroid.

        Returns value in mel scale (~0-3500 for audible range).
        """
        mag = magnitude[self._valid_mask]
        mel = self._mel_freqs[self._valid_mask]
        mag_sum = mag.sum()
        if mag_sum < 1e-10:
            return 0.0
        return float(np.dot(mel, mag) / mag_sum)

    @property
    def n_bins(self) -> int:
        """Number of frequency bins this centroid was built for."""
        return self._n_bins

    @staticmethod
    def mel_to_hz(mel_value: float) -> float:
        """Convert mel value back to Hz."""
        return 700.0 * (10.0 ** (mel_value / 2595.0) - 1.0)

    @staticmethod
    def hz_to_mel(hz: float) -> float:
        """Convert Hz to mel."""
        return 2595.0 * log10_safe(1.0 + hz / 700.0)


class Normalize:
    """Static normalization functions for mapping raw values to viz-friendly ranges."""

    @staticmethod
    def log_perceptual(x: float | np.ndarray, gain: float = 1000.0) -> float | np.ndarray:
        """Weber-Fechner log curve, maps [0, ∞) to [0, 1).

        Models human loudness perception. Default gain=1000 matches
        the engine's existing LogMagnitudeTransform.
        """
        norm = 1.0 / log(1.0 + gain)
        if isinstance(x, np.ndarray):
            return np.log(1.0 + gain * np.clip(x, 0, None)) * norm
        return log(1.0 + gain * max(x, 0.0)) * norm

    @staticmethod
    def linear(x: float, lo: float, hi: float) -> float:
        """Linear map to [0, 1], clamped."""
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    @staticmethod
    def mel(hz: float) -> float:
        """Hz to normalized mel [0, 1] across audible range (20 Hz - 20 kHz)."""
        mel_20 = 2595.0 * log10_safe(1.0 + 20.0 / 700.0)
        mel_20k = 2595.0 * log10_safe(1.0 + 20000.0 / 700.0)
        mel_val = 2595.0 * log10_safe(1.0 + max(hz, 0.0) / 700.0)
        return max(0.0, min(1.0, (mel_val - mel_20) / (mel_20k - mel_20)))

    @staticmethod
    def db(x: float, floor: float = -60.0) -> float:
        """Linear amplitude to [0, 1] dB scale with noise floor.

        0 dB (x=1.0) maps to 1.0. Floor (default -60 dB) maps to 0.0.
        """
        if x <= 0.0:
            return 0.0
        db_val = 20.0 * log10_safe(x)
        return max(0.0, min(1.0, (db_val - floor) / (0.0 - floor)))


def log10_safe(x: float) -> float:
    """log10 that handles x <= 0 gracefully."""
    if x <= 0.0:
        return -100.0  # ~ -inf but finite
    return log(x) / log(10.0)
