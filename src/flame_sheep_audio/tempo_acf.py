"""
Autocorrelation-based tempo tracker.

Estimates tempo from the continuous onset strength signal (summed
spectral flux) rather than discrete detected events. This decouples
tempo estimation from beat detection — the tempo estimate is derived
independently from the raw signal.

Algorithm:
  1. Accumulate onset strength in a ring buffer (~8s window)
  2. Every ~0.5s, autocorrelate the buffer over lag range [60-400 BPM]
  3. Weight by Rayleigh tempo prior (preference for ~120 BPM perception)
  4. Pick the dominant peak as BPM estimate
  5. Smooth the estimate with hysteresis to avoid jitter

References:
  - Ellis 2007, "Beat Tracking by Dynamic Programming"
  - Scheirer 1998, "Tempo and Beat Analysis of Acoustic Musical Signals"
"""

import numpy as np
from collections import deque

from .config import cfg


# Tempo range (matches old tracker)
MIN_BPM = 60
MAX_BPM = 400

# Analysis parameters
WINDOW_SECONDS = 8.0        # autocorrelation window length
UPDATE_INTERVAL = 0.5       # recompute every N seconds
PRIOR_CENTER = 110.0        # Rayleigh prior center (perceptual preference)
PRIOR_WIDTH = 1.4           # Rayleigh prior width (std dev in log-BPM space)

# Smoothing
BPM_SMOOTH_ALPHA = 0.8      # EMA for BPM estimate (0=instant, 1=frozen)
CONFIDENCE_THRESHOLD = 0.15 # autocorrelation peak must exceed this
LOCK_THRESHOLD = 0.5
UNLOCK_THRESHOLD = 0.2


class AutocorrelationTempoTracker:
    """Tempo estimation via autocorrelation of onset strength.

    Call feed() with a scalar onset strength value each audio frame.
    Read bpm/confidence/effective_bpm properties for current estimate.

    Does NOT process discrete beat events — it works on the continuous
    onset strength signal upstream of any thresholding.
    """

    def __init__(self, hop_duration: float = 0.01067):
        """
        Args:
            hop_duration: seconds per frame (HOP_SIZE / SAMPLE_RATE).
        """
        self._hop_duration = hop_duration

        # Ring buffer for onset strength
        self._buffer_size = int(WINDOW_SECONDS / hop_duration)
        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._frames_fed = 0

        # Precompute lag range (in frames) for BPM range
        self._min_lag = int(60.0 / MAX_BPM / hop_duration)
        self._max_lag = int(60.0 / MIN_BPM / hop_duration)
        self._max_lag = min(self._max_lag, self._buffer_size // 2)
        self._lags = np.arange(self._min_lag, self._max_lag + 1)
        self._lag_bpms = 60.0 / (self._lags * hop_duration)

        # Tempo prior: Rayleigh distribution in log-BPM space
        # Peaks around PRIOR_CENTER, falls off for very slow/fast tempos
        log_bpm = np.log2(self._lag_bpms / PRIOR_CENTER)
        self._prior = np.exp(-0.5 * (log_bpm / PRIOR_WIDTH) ** 2)

        # Update cadence
        self._update_every = max(1, int(UPDATE_INTERVAL / hop_duration))
        self._frames_since_update = 0

        # State
        self._bpm = 0.0
        self._raw_bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._has_estimate = False

        # External hint
        self._hint_bpm: float | None = None

    def feed(self, onset_strength: float):
        """Feed one frame of onset strength (scalar, e.g. summed flux)."""
        self._buffer[self._write_pos] = onset_strength
        self._write_pos = (self._write_pos + 1) % self._buffer_size
        self._frames_fed += 1
        self._frames_since_update += 1

        if self._frames_since_update >= self._update_every:
            self._frames_since_update = 0
            if self._frames_fed >= self._buffer_size // 2:
                self._update()

    def _update(self):
        """Recompute tempo estimate from autocorrelation."""
        # Unroll ring buffer into contiguous array
        buf = np.roll(self._buffer, -self._write_pos)

        # Normalize (zero-mean, unit-variance)
        mean = buf.mean()
        std = buf.std()
        if std < 1e-10:
            return  # silence — no estimate
        normed = (buf - mean) / std

        # Autocorrelation via FFT (much faster than direct for large windows)
        n = len(normed)
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        fft = np.fft.rfft(normed, n=fft_size)
        acf_full = np.fft.irfft(fft * np.conj(fft), n=fft_size)[:n]
        acf_full /= acf_full[0] + 1e-10  # normalize so lag-0 = 1.0

        # Extract lag range
        acf = acf_full[self._min_lag:self._max_lag + 1]

        # Weight by tempo prior
        weighted = acf * self._prior

        # Sub-harmonic reinforcement: if lag 2L and 3L also have peaks,
        # lag L is more likely the true period (not a harmonic artifact)
        for multiplier in [2, 3]:
            for i, lag in enumerate(self._lags):
                sub_lag = lag * multiplier
                if sub_lag < len(acf_full):
                    weighted[i] += acf_full[sub_lag] * self._prior[i] * (0.5 / multiplier)

        # Find peak
        if len(weighted) == 0:
            return
        peak_idx = np.argmax(weighted)
        peak_val = weighted[peak_idx]
        raw_bpm = float(self._lag_bpms[peak_idx])

        # Confidence from peak prominence relative to the ACF floor
        # Use the raw (unweighted) ACF peak value — it's already normalized
        # so lag-0 = 1.0, and a strong periodicity gives peak > 0.1
        raw_peak = acf[peak_idx]
        # Also check sharpness: peak should stand out from neighbors
        local_mean = float(np.mean(np.sort(weighted)[-len(weighted)//4:]))
        if local_mean > 0:
            sharpness = peak_val / local_mean
        else:
            sharpness = 1.0
        # Combined confidence: raw ACF strength + peak sharpness
        confidence = min(1.0, raw_peak * 2.0 * min(sharpness, 3.0))

        self._raw_bpm = raw_bpm

        if confidence < CONFIDENCE_THRESHOLD and not self._has_estimate:
            return  # not confident enough for first estimate

        # Smooth BPM estimate
        if not self._has_estimate:
            self._bpm = raw_bpm
            self._has_estimate = True
        else:
            # Check for octave jump — if new estimate is ~2x or ~0.5x,
            # and confidence is similar, prefer staying at current octave
            ratio = raw_bpm / self._bpm if self._bpm > 0 else 1.0
            if 0.45 < ratio < 0.55 or 1.9 < ratio < 2.1:
                # Octave jump — only follow if much more confident
                if confidence > self._confidence * 1.5:
                    self._bpm = raw_bpm
                # else: keep current octave
            else:
                # Normal update — EMA smooth
                self._bpm = (BPM_SMOOTH_ALPHA * self._bpm
                             + (1 - BPM_SMOOTH_ALPHA) * raw_bpm)

        self._confidence = confidence

        # Lock/unlock
        if self._confidence >= LOCK_THRESHOLD:
            self._locked = True
        elif self._confidence < UNLOCK_THRESHOLD:
            self._locked = False

    def hint_tempo(self, bpm: float):
        """Provide external tempo hint."""
        if MIN_BPM <= bpm <= MAX_BPM:
            self._hint_bpm = bpm
            self._bpm = bpm
            self._confidence = 0.5
            self._has_estimate = True
            self._locked = False

    def song_started(self):
        """Reset for new song."""
        self.reset()

    def reset(self):
        """Clear all state."""
        self._buffer[:] = 0
        self._write_pos = 0
        self._frames_fed = 0
        self._frames_since_update = 0
        self._bpm = 0.0
        self._raw_bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._has_estimate = False
        self._hint_bpm = None

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def effective_bpm(self) -> float:
        """BPM blended with default based on confidence."""
        default = cfg.tempo.default_bpm
        if not self._has_estimate:
            return float(default)
        return default * (1 - self._confidence) + self._bpm * self._confidence

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def saturated(self) -> bool:
        """Not applicable for ACF tracker — always False."""
        return False

    @property
    def raw_bpm(self) -> float:
        """Unsmoothed BPM from latest autocorrelation."""
        return self._raw_bpm
