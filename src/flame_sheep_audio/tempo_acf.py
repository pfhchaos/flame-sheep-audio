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

from __future__ import annotations

import numpy as np
from collections import deque

from .config import cfg


# Tempo range (matches old tracker)
# Defaults (overridden by config if present)
MIN_BPM = 60
MAX_BPM = 400


from .tempo import TempoTrackerBase


class AutocorrelationTempoTracker(TempoTrackerBase):
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

        # Read all constants from config
        window_seconds = cfg.tempo.window_seconds
        update_interval = cfg.tempo.update_interval
        prior_center = cfg.tempo.prior_center
        prior_width = cfg.tempo.prior_width
        self._SMOOTH_ALPHA = cfg.tempo.smooth_alpha
        self._CONFIDENCE_THRESHOLD = cfg.tempo.confidence_threshold
        self._LOCK_THRESHOLD = cfg.tempo.lock_threshold
        self._UNLOCK_THRESHOLD = cfg.tempo.unlock_threshold

        # Ring buffer for onset strength
        self._buffer_size = int(window_seconds / hop_duration)
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
        log_bpm = np.log2(self._lag_bpms / prior_center)
        self._prior = np.exp(-0.5 * (log_bpm / prior_width) ** 2)

        # Update cadence
        self._update_every = max(1, int(update_interval / hop_duration))
        self._frames_since_update = 0

        # State
        self._bpm = 0.0
        self._raw_bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._has_estimate = False

        # Temporal confidence: ring buffer of recent BPM estimates
        self._recent_bpms: list[float] = []
        self._temporal_confidence = 0.0
        self._TEMPORAL_WINDOW = 10  # ~5 seconds of ACF updates

        # Last confident BPM: replaces default in effective_bpm blend
        self._last_confident_bpm = float(cfg.tempo.default_bpm)

        # BPM delta: dual-EMA (fast ~0.5s, slow ~3s)
        self._bpm_fast_ema = 0.0
        self._bpm_slow_ema = 0.0
        self._BPM_FAST_ALPHA = 0.3   # fast EMA: ~1 ACF update to converge
        self._BPM_SLOW_ALPHA = 0.85  # slow EMA: ~6 ACF updates (~3s)

        # Onset density (updated each frame, used for octave disambiguation)
        self._onset_density = 0.0

        # External hint
        self._hint_bpm: float | None = None

    def feed(self, onset_strength: float, onset_density: float = 0.0) -> None:
        """Feed one frame of onset strength and total onset density."""
        self._onset_density = onset_density
        self._buffer[self._write_pos] = onset_strength
        self._write_pos = (self._write_pos + 1) % self._buffer_size
        self._frames_fed += 1
        self._frames_since_update += 1

        if self._frames_since_update >= self._update_every:
            self._frames_since_update = 0
            if self._frames_fed >= self._buffer_size // 2:
                self._update()

    def _update(self) -> None:
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

        # Octave disambiguation: if onset density is high relative to the
        # detected BPM, prefer the faster octave. At 87 BPM you'd expect
        # ~6 onsets/s with a full kit; density >> 6 suggests 174 BPM.
        # Threshold: onsets/s that would be expected at the detected BPM
        # with a typical kit (~3-4 instruments hitting per beat).
        half_lag_idx = self._lags[peak_idx] // 2 - self._min_lag
        expected_density = raw_bpm / 60.0 * 3.5  # ~3.5 onsets per beat
        if (0 <= half_lag_idx < len(weighted)
                and self._onset_density > expected_density * 1.5):
            half_peak = weighted[half_lag_idx]
            if half_peak > peak_val * 0.3:
                peak_idx = half_lag_idx
                peak_val = half_peak
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

        if confidence < self._CONFIDENCE_THRESHOLD and not self._has_estimate:
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
                self._bpm = (self._SMOOTH_ALPHA * self._bpm
                             + (1 - self._SMOOTH_ALPHA) * raw_bpm)

        # Temporal confidence: how stable have recent estimates been?
        # Only count estimates where spatial confidence was meaningful —
        # consistent weak measurements shouldn't produce high confidence
        SPATIAL_FLOOR = 0.3
        if confidence >= SPATIAL_FLOOR:
            self._recent_bpms.append(raw_bpm)
        if len(self._recent_bpms) > self._TEMPORAL_WINDOW:
            self._recent_bpms = self._recent_bpms[-self._TEMPORAL_WINDOW:]
        if len(self._recent_bpms) >= 3:
            arr = np.array(self._recent_bpms)
            mean = arr.mean()
            if mean > 0:
                cv = arr.std() / mean  # coefficient of variation
                self._temporal_confidence = min(1.0, max(0.0, 1.0 - cv * 10))
            else:
                self._temporal_confidence = 0.0

        # Combined confidence: temporal can boost spatial but not override it.
        # Cap temporal at 2x spatial so weak-but-consistent signals stay low.
        capped_temporal = min(self._temporal_confidence, confidence * 2.5)
        self._confidence = max(confidence, capped_temporal)

        # Update last confident BPM
        if self._confidence >= self._LOCK_THRESHOLD:
            self._last_confident_bpm = self._bpm
            self._locked = True
        elif self._confidence < self._UNLOCK_THRESHOLD:
            self._locked = False

        # BPM delta: dual-EMA difference (fast - slow = trend)
        if self._has_estimate:
            fa = self._BPM_FAST_ALPHA
            sa = self._BPM_SLOW_ALPHA
            self._bpm_fast_ema = fa * self._bpm_fast_ema + (1 - fa) * self._bpm
            self._bpm_slow_ema = sa * self._bpm_slow_ema + (1 - sa) * self._bpm

    def hint_tempo(self, bpm: float) -> None:
        """Provide external tempo hint."""
        if MIN_BPM <= bpm <= MAX_BPM:
            self._hint_bpm = bpm
            self._bpm = bpm
            self._last_confident_bpm = bpm
            self._confidence = 0.5
            self._has_estimate = True
            self._locked = False
            self._bpm_fast_ema = bpm
            self._bpm_slow_ema = bpm

    def song_started(self) -> None:
        """Reset for new song."""
        self.reset()

    def reset(self) -> None:
        """Clear all state."""
        self._buffer[:] = 0
        self._write_pos = 0
        self._frames_fed = 0
        self._frames_since_update = 0
        self._bpm = 0.0
        self._raw_bpm = 0.0
        self._confidence = 0.0
        self._temporal_confidence = 0.0
        self._locked = False
        self._has_estimate = False
        self._hint_bpm = None
        self._recent_bpms.clear()
        self._last_confident_bpm = float(cfg.tempo.default_bpm)
        self._bpm_fast_ema = 0.0
        self._bpm_slow_ema = 0.0

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def effective_bpm(self) -> float:
        """BPM blended with last confident estimate based on confidence.

        When confident, returns the current estimate. When unsure, blends
        toward the last value we were confident about (not an arbitrary
        default). Initializes to default_bpm, updated by hints and locks.
        """
        if not self._has_estimate:
            return self._last_confident_bpm
        return (self._last_confident_bpm * (1 - self._confidence)
                + self._bpm * self._confidence)

    @property
    def bpm_delta(self) -> float:
        """Rate of change of tempo (BPM/s). Positive = accelerando."""
        return self._bpm_fast_ema - self._bpm_slow_ema

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
    def phase(self) -> float:
        """ACF tracker doesn't estimate phase."""
        return 0.0

    @property
    def raw_bpm(self) -> float:
        """Unsmoothed BPM from latest autocorrelation."""
        return self._raw_bpm
