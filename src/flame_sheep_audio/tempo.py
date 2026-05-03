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
                 n_candidates: int = 10) -> None:
        self._hop = hop_duration
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
        self._hist_decay = 0.95  # per-update decay

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

        # Log-compress: ln(1 + 1000 * x)
        compressed = np.log1p(1000.0 * max(0.0, onset_strength))

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

        # Score candidates via cross-correlation with pulse trains
        best_score = -1.0
        best_lag = candidates[0][1]
        best_phase = 0

        for _, lag in candidates:
            score, phase = self._score_pulse_train(buf, lag)
            if score > best_score:
                best_score = score
                best_lag = lag
                best_phase = phase

        self._best_period = best_lag
        # Phase offset: how many frames from "now" to next predicted beat
        # The best_phase is relative to the OSS window start; convert to
        # frames-until-next-beat from current position
        self._best_phase = best_phase
        self._phase_counter = (self._window_frames - best_phase) % best_lag if best_lag > 0 else 0

        # Accumulate into Gaussian histogram
        self._histogram *= self._hist_decay
        sigma = 10.0  # spread in lag samples
        lags = np.arange(len(self._histogram))
        gaussian = np.exp(-0.5 * ((lags - best_lag) / sigma) ** 2)
        self._histogram += gaussian

        # Read peak of histogram as final estimate
        peak_lag = np.argmax(self._histogram[self._min_lag:self._max_lag]) + self._min_lag
        if peak_lag > 0:
            self._bpm = 60.0 * self._oss_sr / peak_lag
            self._has_estimate = True

        # Confidence from histogram sharpness
        hist_norm = self._histogram[self._min_lag:self._max_lag]
        if hist_norm.max() > 0:
            hist_norm = hist_norm / hist_norm.max()
            # Entropy-based: sharp peak = low entropy = high confidence
            h = hist_norm + 1e-10
            h = h / h.sum()
            entropy = -np.sum(h * np.log(h))
            max_entropy = np.log(len(h))
            self._confidence = max(0.0, 1.0 - entropy / max_entropy)
        else:
            self._confidence = 0.0

        if self._confidence > 0.5:
            self._last_confident_bpm = self._bpm

    def _score_pulse_train(self, oss: np.ndarray, period: int) -> tuple[float, int]:
        """Score a tempo candidate by cross-correlating OSS with pulse trains.

        Tests all phases. Returns (variance_score, best_phase).
        High variance = clear beats at this period.
        Best phase = the phase offset with highest cross-correlation.
        """
        n = len(oss)
        scores = []
        for phase in range(period):
            total = 0.0
            count = 0
            for multiplier in [1.0, 1.5, 2.0]:
                weight = 1.0 if multiplier == 1.0 else 0.5
                pos = phase
                while pos < n:
                    total += oss[int(pos)] * weight
                    count += 1
                    pos += period * multiplier
            if count > 0:
                scores.append(total / count)
            else:
                scores.append(0.0)

        if len(scores) < 2:
            return 0.0, 0
        best_phase = int(np.argmax(scores))
        return float(np.var(scores)), best_phase

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
