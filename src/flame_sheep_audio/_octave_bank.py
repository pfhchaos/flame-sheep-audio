"""Octave-bank spectrum engine — per-octave FFTs with matched resolution.

Runs a separate FFT per octave, each with a window length matched to
the frequency range. Low octaves get long windows (high frequency
resolution for kick/bass separation). High octaves get short windows
(high time resolution for transient detection).

This is conceptually equivalent to a streaming CQT but implemented as
a bank of standard FFTs — no specialized libraries required.

Produces the same SpectrumFrame interface as SpectrumEngine.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows

from ._spectrum import SpectrumFrame
from ._constants import SAMPLE_RATE, HOP_SIZE
from ._bands import a_weight_curve


def _build_octave_bank(
    sr: int = SAMPLE_RATE,
    hop: int = HOP_SIZE,
    n_octaves: int = 9,
    bins_per_octave: int = 12,
    fmin: float = 32.7,
) -> list[dict]:
    """Build the per-octave FFT configuration.

    Each octave gets the smallest FFT that provides at least
    bins_per_octave frequency bins within its range. Lower octaves
    get progressively larger FFTs.
    """
    octaves = []
    for oct_idx in range(n_octaves):
        f_lo = fmin * (2 ** oct_idx)
        f_hi = fmin * (2 ** (oct_idx + 1))
        octave_width = f_hi - f_lo

        # FFT size needed for bins_per_octave bins in this octave:
        # bin_width = sr / fft_size, need octave_width / bin_width >= bins_per_octave
        # fft_size >= sr * bins_per_octave / octave_width
        needed = int(np.ceil(sr * bins_per_octave / octave_width))
        # Round up to power of 2
        fft_size = max(hop, 1 << int(np.ceil(np.log2(max(needed, hop)))))
        fft_size = min(fft_size, 16384)  # cap to avoid excessive latency

        freqs = np.fft.rfftfreq(fft_size, 1 / sr)
        mask = (freqs >= f_lo) & (freqs < f_hi)

        octaves.append({
            'fft_size': fft_size,
            'window': windows.hann(fft_size, sym=False).astype(np.float32),
            'mask': mask,
            'n_fft_bins': int(mask.sum()),
            'f_lo': f_lo,
            'f_hi': f_hi,
        })

    return octaves


class OctaveBankEngine:
    """Per-octave FFT bank with log-spaced output bins.

    Each octave uses a different FFT size matched to its frequency
    range. Output is rebinned to bins_per_octave bins per octave.

    Same push_hop() interface as SpectrumEngine.
    """

    def __init__(self, n_octaves: int = 9, bins_per_octave: int = 12,
                 fmin: float = 32.7) -> None:
        self._bpo = bins_per_octave
        self._octaves = _build_octave_bank(
            n_octaves=n_octaves, bins_per_octave=bins_per_octave, fmin=fmin)
        self.n_bins = n_octaves * bins_per_octave

        # Per-octave sliding buffers
        self._buffers = [np.zeros(o['fft_size'], dtype=np.float32)
                         for o in self._octaves]

        # Build bin center frequencies
        self.bin_centers = np.zeros(self.n_bins, dtype=np.float32)
        for oct_idx, o in enumerate(self._octaves):
            for b in range(bins_per_octave):
                # Log-spaced within octave
                frac = b / bins_per_octave
                self.bin_centers[oct_idx * bins_per_octave + b] = \
                    o['f_lo'] * (2 ** frac)

        # A-weighting for onset strength
        self._a_weights = a_weight_curve(self.bin_centers)

        # Previous magnitude for flux computation
        self._prev_spectrum: np.ndarray | None = None

    def push_hop(self, hop: np.ndarray) -> SpectrumFrame:
        """Slide buffers and compute per-octave FFTs."""
        n = len(hop)
        magnitude = np.zeros(self.n_bins, dtype=np.float32)

        for oct_idx, (octave, buf) in enumerate(zip(self._octaves, self._buffers)):
            fft_size = octave['fft_size']

            # Slide buffer
            buf[:fft_size - n] = buf[n:]
            buf[fft_size - n:] = hop

            # FFT
            windowed = buf * octave['window']
            raw_mag = np.abs(np.fft.rfft(windowed)) / fft_size

            # Extract bins for this octave and rebin
            masked = raw_mag[octave['mask']]
            out_start = oct_idx * self._bpo

            if len(masked) >= self._bpo:
                # More FFT bins than output bins: RMS-average groups
                chunk_size = len(masked) // self._bpo
                for b in range(self._bpo):
                    s = b * chunk_size
                    e = s + chunk_size if b < self._bpo - 1 else len(masked)
                    magnitude[out_start + b] = float(np.sqrt(np.mean(masked[s:e] ** 2)))
            elif len(masked) > 0:
                # Fewer FFT bins than output: spread directly
                magnitude[out_start:out_start + len(masked)] = masked

        # Flux
        if self._prev_spectrum is not None:
            flux = np.maximum(magnitude - self._prev_spectrum, 0.0).astype(np.float32)
        else:
            flux = np.zeros(self.n_bins, dtype=np.float32)

        self._prev_spectrum = magnitude.copy()

        # ZCR on latest hop
        signs = np.signbit(hop)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / len(hop)

        return SpectrumFrame(
            magnitude=magnitude,
            flux=flux,
            waveform=hop.copy(),
            # onset_strength computed downstream after HPSS split
            zcr=zcr,
        )

    def compute(self, pcm: np.ndarray) -> SpectrumFrame:
        """Compute from a full PCM buffer.

        Feeds the buffer in HOP_SIZE chunks so low-frequency octaves
        get enough history. Returns the final frame.
        """
        # Feed in HOP_SIZE steps to fill the per-octave buffers
        pos = 0
        frame = None
        while pos < len(pcm):
            chunk = pcm[pos:pos + HOP_SIZE]
            if len(chunk) < HOP_SIZE:
                chunk = np.pad(chunk, (0, HOP_SIZE - len(chunk)))
            frame = self.push_hop(chunk)
            pos += HOP_SIZE
        if frame is None:
            frame = self.push_hop(np.zeros(HOP_SIZE, dtype=np.float32))
        return frame

    def reset(self) -> None:
        self._prev_spectrum = None
        for buf in self._buffers:
            buf[:] = 0.0
