"""Tempo tracker interface and Percival-style implementation.

TempoTrackerBase defines the interface. Implementations:
  - AutocorrelationTempoTracker (tempo_acf.py): original, minimal ACF
  - PercivalTempoTracker: log OSS, LP filter, enhanced ACF, pulse train scoring

Based on:
  Percival & Tzanetakis (2014) "Streamlined Tempo Estimation Based on
  Autocorrelation and Cross-correlation With Pulses"
  IEEE/ACM Trans. Audio, Speech, Language Processing
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import firwin, lfilter

from .config import cfg


class TempoTrackerBase(ABC):
    """Interface for tempo estimation.

    Implementations must:
      - Accept scalar onset strength via feed()
      - Provide bpm, effective_bpm, confidence properties
      - Support reset() for song changes
      - Work in real-time (causal, no future lookahead)
    """

    @abstractmethod
    def feed(self, onset_strength: float, onset_density: float = 0.0) -> None:
        """Feed one frame of onset strength."""
        ...

    @property
    @abstractmethod
    def bpm(self) -> float:
        """Current raw BPM estimate."""
        ...

    @property
    @abstractmethod
    def effective_bpm(self) -> float:
        """BPM blended with confidence — stable for driving visualization."""
        ...

    @property
    @abstractmethod
    def confidence(self) -> float:
        """Confidence in current estimate, 0.0-1.0."""
        ...

    @property
    @abstractmethod
    def phase(self) -> float:
        """Current beat phase, 0.0-1.0. 0=on beat, 0.5=between beats."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all state for a song change."""
        ...


class PercivalTempoTracker(TempoTrackerBase):
    """Tempo estimation following Percival & Tzanetakis (2014).

    Pipeline:
      1. Log-compress onset strength (reduces dynamic range)
      2. Low-pass filter at max_bpm/60 Hz (removes jitter)
      3. Generalized autocorrelation (exponent c=0.5) on windowed OSS
      4. Enhance harmonics: EAC(t) = A(t) + A(2t) + A(4t)
      5. Pick top N peaks as tempo candidates
      6. Score candidates via cross-correlation with pulse trains
      7. Accumulate scores in Gaussian histogram
    """

    def __init__(self, hop_duration: float = 0.01067,
                 min_bpm: float = 50.0, max_bpm: float = 210.0,
                 acf_exponent: float = 0.5,
                 window_seconds: float = 6.0,
                 update_seconds: float = 0.37,
                 n_candidates: int = 10,
                 log_compress: bool = True) -> None:
        self._hop = hop_duration
        self._log_compress = log_compress
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._c = acf_exponent
        self._n_candidates = n_candidates

        # OSS sample rate (1 / hop_duration)
        self._oss_sr = 1.0 / hop_duration

        # Buffer for onset strength signal
        self._window_frames = int(window_seconds / hop_duration)
        self._update_every = max(1, int(update_seconds / hop_duration))
        self._buffer = np.zeros(self._window_frames, dtype=np.float64)
        self._write_pos = 0
        self._frames_fed = 0
        self._frames_since_update = 0

        # Low-pass filter: cutoff at max_bpm / 60 Hz
        # (max_bpm=210 → cutoff=3.5 Hz)
        lp_cutoff = max_bpm / 60.0
        nyquist = self._oss_sr / 2.0
        if lp_cutoff < nyquist:
            n_taps = 15  # 14th order = 15 taps
            self._lp_b = firwin(n_taps, lp_cutoff / nyquist).astype(np.float64)
        else:
            self._lp_b = np.array([1.0])  # passthrough if SR too low
        self._lp_state = np.zeros(len(self._lp_b) - 1, dtype=np.float64)

        # Lag range for ACF (in OSS samples)
        self._min_lag = int(60.0 * self._oss_sr / max_bpm)
        self._max_lag = int(60.0 * self._oss_sr / min_bpm) + 1

        # Gaussian accumulator histogram (in lag space)
        self._histogram = np.zeros(self._max_lag + 1, dtype=np.float64)
        self._hist_decay = 0.99  # per-update decay (~25s half-life)

        # Output state
        self._bpm = float(cfg.tempo.default_bpm)
        self._confidence = 0.0
        self._has_estimate = False
        self._last_confident_bpm = float(cfg.tempo.default_bpm)

        # Phase tracking
        self._best_period = 0       # lag in frames
        self._best_phase = 0        # offset in frames from last update
        self._phase_counter = 0     # frames since last beat

    def feed(self, onset_strength: float, onset_density: float = 0.0) -> None:
        self._phase_counter += 1

        # If upstream already provides log-compressed flux (via LogMagnitudeTransform),
        # skip the log compression here. Otherwise apply it.
        if self._log_compress:
            compressed = np.log1p(1000.0 * max(0.0, onset_strength))
        else:
            compressed = max(0.0, onset_strength)

        # Low-pass filter (causal, sample-by-sample)
        x = np.array([compressed])
        filtered, self._lp_state = lfilter(self._lp_b, 1.0, x, zi=self._lp_state)

        # Store in ring buffer
        self._buffer[self._write_pos] = filtered[0]
        self._write_pos = (self._write_pos + 1) % self._window_frames
        self._frames_fed += 1
        self._frames_since_update += 1

        if self._frames_since_update >= self._update_every:
            self._frames_since_update = 0
            if self._frames_fed >= self._window_frames // 2:
                self._update()

    def _update(self) -> None:
        # Unroll ring buffer
        buf = np.roll(self._buffer, -self._write_pos)
        n = len(buf)

        # Zero-mean
        mean = buf.mean()
        std = buf.std()
        if std < 1e-10:
            return  # silence
        normed = (buf - mean) / std

        # Generalized autocorrelation via FFT
        # A(t) = IFFT(|FFT(x)|^c)
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        X = np.fft.rfft(normed, n=fft_size)
        power = np.abs(X) ** self._c
        acf = np.fft.irfft(power, n=fft_size)[:n]

        # Enhanced autocorrelation: EAC(t) = A(t) + A(2t) + A(4t)
        eac = np.zeros(n, dtype=np.float64)
        for t in range(self._min_lag, min(self._max_lag, n)):
            val = acf[t]
            if 2 * t < n:
                val += acf[2 * t]
            if 4 * t < n:
                val += acf[4 * t]
            eac[t] = val

        # Pick top N peaks in the valid lag range
        candidates = []
        for t in range(self._min_lag, min(self._max_lag, n - 1)):
            if eac[t] > eac[t - 1] and eac[t] > eac[t + 1]:
                candidates.append((eac[t], t))
        candidates.sort(reverse=True)
        candidates = candidates[:self._n_candidates]

        if not candidates:
            return

        # Use the top EAC peak directly
        best_lag = candidates[0][1]
        best_eac = candidates[0][0]

        # Get phase from pulse train cross-correlation at the chosen lag
        _, _, best_phase = self._score_pulse_train(buf, best_lag)

        self._best_period = best_lag
        self._best_phase = best_phase
        self._phase_counter = (self._window_frames - best_phase) % best_lag if best_lag > 0 else 0

        # Convert lag to BPM
        raw_bpm = 60.0 * self._oss_sr / best_lag

        # Confidence from EAC peak strength relative to noise floor
        # Strong peak = confident, weak peak = uncertain
        eac_range = max(eac[self._min_lag:min(self._max_lag, n)]) - np.median(eac[self._min_lag:min(self._max_lag, n)])
        if eac_range > 0:
            self._confidence = min(1.0, best_eac / (eac_range + 1e-10) * 0.5)
        else:
            self._confidence = 0.0

        # Smooth BPM estimate: jump instantly if confident, blend if unsure
        if not self._has_estimate:
            self._bpm = raw_bpm
            self._has_estimate = True
        else:
            # Blend toward new estimate, weighted by confidence
            alpha = 0.3 * self._confidence + 0.05  # 0.05-0.35
            self._bpm = (1 - alpha) * self._bpm + alpha * raw_bpm

        if self._confidence > 0.3:
            self._last_confident_bpm = self._bpm

    def _score_pulse_train(self, oss: np.ndarray, period: int) -> tuple[float, float, int]:
        """Score a tempo candidate per Percival & Tzanetakis eq. 8-11.

        Builds a combined pulse train with three metrical levels:
          - Every beat (period P), weight 1.0
          - Every 1.5 beats (period 1.5*P), weight 0.5
          - Every 2 beats (period 2*P), weight 0.5
        Cross-correlates with OSS at all phases.
        Returns (SC_x, SC_v, best_phase).
        SC_x = max cross-correlation across phases.
        SC_v = variance of cross-correlation across phases.
        """
        n = len(oss)
        if period < 2:
            return 0.0, 0.0, 0

        # Cross-correlate pulse train with OSS at each phase
        xcorr = np.zeros(period, dtype=np.float64)

        for phase in range(period):
            total = 0.0
            n_pulses = 0
            # Train 1: pulses every P samples, weight 1.0
            b = 0
            while True:
                idx = phase + b * period
                if idx >= n:
                    break
                total += oss[idx] * 1.0
                n_pulses += 1
                b += 1

            # Train 2: pulses every 1.5*P samples, weight 0.5
            b = 0
            step_15 = period * 1.5
            while True:
                idx = int(phase + b * step_15)
                if idx >= n:
                    break
                total += oss[idx] * 0.5
                n_pulses += 1
                b += 1

            # Train 3: pulses every 2*P samples, weight 0.5
            b = 0
            step_2 = period * 2
            while True:
                idx = phase + b * step_2
                if idx >= n:
                    break
                total += oss[idx] * 0.5
                n_pulses += 1
                b += 1

            xcorr[phase] = total

        # SC_x: max cross-correlation (best phase alignment)
        sc_x = float(xcorr.max())
        # SC_v: variance across phases (rhythmic clarity)
        sc_v = float(xcorr.var())
        best_phase = int(np.argmax(xcorr))

        return sc_x, sc_v, best_phase

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def effective_bpm(self) -> float:
        if not self._has_estimate:
            return self._last_confident_bpm
        return (self._last_confident_bpm * (1 - self._confidence)
                + self._bpm * self._confidence)

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def phase(self) -> float:
        """Current beat phase, 0.0-1.0. 0.0 = on beat, 0.5 = between beats."""
        if self._best_period <= 0:
            return 0.0
        return (self._phase_counter % self._best_period) / self._best_period

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._write_pos = 0
        self._frames_fed = 0
        self._frames_since_update = 0
        self._histogram[:] = 0.0
        self._lp_state[:] = 0.0
        self._bpm = float(cfg.tempo.default_bpm)
        self._confidence = 0.0
        self._has_estimate = False
        self._last_confident_bpm = float(cfg.tempo.default_bpm)
        self._best_period = 0
        self._best_phase = 0
        self._phase_counter = 0
