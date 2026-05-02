"""Multi-resolution spectrum engine — large FFT with log-spaced rebinning.

Uses an 8192-sample FFT for high frequency resolution at low frequencies
(5.9 Hz/bin vs 23.4 Hz/bin with 2048), then rebins into ~84 log-spaced
bins. Low frequencies keep the fine resolution; high frequencies are
collapsed to save downstream computation.

Produces the same SpectrumFrame interface as SpectrumEngine — drop-in
replacement. Downstream analyzers (stability, beat detector, energy)
don't know the difference.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows

from ._spectrum import SpectrumFrame
from ._constants import SAMPLE_RATE, HOP_SIZE
from ._bands import a_weight_curve


# Large FFT for better low-frequency resolution
MULTIRES_FFT_SIZE = 8192
MULTIRES_N_BINS_RAW = MULTIRES_FFT_SIZE // 2 + 1
MULTIRES_FREQS = np.fft.rfftfreq(MULTIRES_FFT_SIZE, 1.0 / SAMPLE_RATE)


def _build_log_bins(f_min: float = 30.0, f_max: float = 20000.0,
                    bins_per_octave: int = 12) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build log-spaced frequency bins and their FFT bin masks.

    Returns:
        bin_centers: (n_bins,) center frequency of each log bin
        bin_masks: list of boolean masks into the raw FFT bins
    """
    n_octaves = np.log2(f_max / f_min)
    n_bins = int(n_octaves * bins_per_octave)
    edges = f_min * 2.0 ** (np.arange(n_bins + 1) / bins_per_octave)

    bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    bin_masks = []
    for i in range(n_bins):
        mask = (MULTIRES_FREQS >= edges[i]) & (MULTIRES_FREQS < edges[i + 1])
        bin_masks.append(mask)

    return bin_centers.astype(np.float32), bin_masks


class MultiResSpectrumEngine:
    """Multi-resolution spectrum engine with log-spaced output bins.

    Uses 8192-sample FFT internally, rebins to ~84 log-spaced bins.
    Same push_hop() / compute() interface as SpectrumEngine.
    """

    def __init__(self, f_min: float = 30.0, f_max: float = 20000.0,
                 bins_per_octave: int = 12) -> None:
        self._fft_size = MULTIRES_FFT_SIZE
        self._window = windows.hann(self._fft_size, sym=False).astype(np.float32)
        self._buffer = np.zeros(self._fft_size, dtype=np.float32)
        self._prev_spectrum: np.ndarray | None = None

        # Build log binning
        self.bin_centers, self._bin_masks = _build_log_bins(f_min, f_max, bins_per_octave)
        self.n_bins = len(self.bin_centers)

        # Precompute A-weighting for the log bins
        self._a_weights = a_weight_curve(self.bin_centers)

        # Build a sparse matrix for fast rebinning: (n_log_bins, n_raw_bins)
        # Each row sums the raw bins belonging to that log bin, normalized by count
        self._rebin_matrix = np.zeros((self.n_bins, MULTIRES_N_BINS_RAW), dtype=np.float32)
        for i, mask in enumerate(self._bin_masks):
            count = mask.sum()
            if count > 0:
                self._rebin_matrix[i, mask] = 1.0 / count

    def _rebin(self, raw: np.ndarray) -> np.ndarray:
        """Rebin raw FFT magnitude into log-spaced bins via matrix multiply."""
        # Mean magnitude per log bin (matrix multiply)
        return np.sqrt(self._rebin_matrix @ (raw ** 2)).astype(np.float32)

    def compute(self, pcm: np.ndarray) -> SpectrumFrame:
        """Compute spectrum from a PCM window of fft_size samples."""
        windowed = pcm[-self._fft_size:] * self._window
        raw_mag = (np.abs(np.fft.rfft(windowed)) / self._fft_size).astype(np.float32)

        magnitude = self._rebin(raw_mag)

        if self._prev_spectrum is not None:
            flux = np.maximum(magnitude - self._prev_spectrum, 0.0).astype(np.float32)
        else:
            flux = np.zeros(self.n_bins, dtype=np.float32)

        self._prev_spectrum = magnitude.copy()

        # ZCR on the most recent HOP_SIZE samples
        recent = pcm[-HOP_SIZE:]
        signs = np.signbit(recent)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / len(recent)

        return SpectrumFrame(
            magnitude=magnitude,
            flux=flux,
            waveform=pcm[-self._fft_size:].copy(),
            # onset_strength computed downstream after HPSS split
            zcr=zcr,
        )

    def push_hop(self, hop: np.ndarray) -> SpectrumFrame:
        """Slide buffer and compute. Same interface as SpectrumEngine."""
        n = len(hop)
        self._buffer[:self._fft_size - n] = self._buffer[n:]
        self._buffer[self._fft_size - n:] = hop
        return self.compute(self._buffer)

    def reset(self) -> None:
        self._prev_spectrum = None
        self._buffer[:] = 0.0
