"""Spectrum computation engine — FFT, windowing, spectral flux."""

import numpy as np
from dataclasses import dataclass
from scipy.signal import windows

from ._constants import FFT_SIZE, N_BINS, SAMPLE_RATE, FREQS
from ._bands import a_weight_curve


# Precomputed A-weighting for onset strength
_A_WEIGHTS = a_weight_curve(FREQS)


@dataclass
class SpectrumFrame:
    """Output of SpectrumEngine — computed once, consumed by all analyzers."""
    magnitude: np.ndarray    # (N_BINS,) float32, FFT magnitude spectrum
    flux: np.ndarray         # (N_BINS,) float32, half-wave rectified spectral flux
    waveform: np.ndarray     # (FFT_SIZE,) float32, raw PCM window
    onset_strength: float = 0.0  # A-weighted flux sum — scalar for tempo tracking


class SpectrumEngine:
    """Computes FFT magnitude spectrum and spectral flux from raw PCM.

    Call compute() once per audio frame. The resulting SpectrumFrame is
    shared by all downstream analyzers (beat detector, energy analyzer, etc.)
    so the FFT is only computed once.
    """

    def __init__(self):
        self._window = windows.hann(FFT_SIZE, sym=False).astype(np.float32)
        self._prev_spectrum: np.ndarray | None = None
        self._buffer = np.zeros(FFT_SIZE, dtype=np.float32)  # sliding window for push_hop

    def compute(self, pcm: np.ndarray) -> SpectrumFrame:
        """Compute spectrum and flux from a PCM window.

        Args:
            pcm: float32 array of FFT_SIZE samples.

        Returns:
            SpectrumFrame with magnitude, flux, and waveform.
        """
        windowed = pcm * self._window
        magnitude = (np.abs(np.fft.rfft(windowed)) / FFT_SIZE).astype(np.float32)

        if self._prev_spectrum is not None:
            flux = np.maximum(magnitude - self._prev_spectrum, 0.0).astype(np.float32)
        else:
            flux = np.zeros(N_BINS, dtype=np.float32)

        self._prev_spectrum = magnitude.copy()

        return SpectrumFrame(
            magnitude=magnitude,
            flux=flux,
            waveform=pcm.copy(),
            onset_strength=float(np.dot(flux, _A_WEIGHTS)),
        )

    def push_hop(self, hop: np.ndarray) -> SpectrumFrame:
        """Slide the internal window by len(hop) samples and compute spectrum.

        Used by the audio thread for overlapping analysis (e.g., 512-sample
        hops with 2048-sample FFT = 75% overlap).
        """
        n = len(hop)
        self._buffer[:FFT_SIZE - n] = self._buffer[n:]
        self._buffer[FFT_SIZE - n:] = hop
        return self.compute(self._buffer)

    def reset(self):
        """Clear previous spectrum state."""
        self._prev_spectrum = None
        self._buffer[:] = 0.0
