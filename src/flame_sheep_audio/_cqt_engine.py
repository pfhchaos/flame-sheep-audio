"""CQT spectrum engine — wraps rt-cqt's SlidingCqt for real-time use.

Drop-in replacement for OctaveBankEngine. Same push_hop() interface,
same 108-bin output (9 octaves * 12 bins/octave).

Requires the prtcqt native module (rt-cqt Python bindings).
"""

from __future__ import annotations

import numpy as np

from ._spectrum import SpectrumFrame, SpectrumEngineBase
from ._constants import SAMPLE_RATE, HOP_SIZE
from ._bands import a_weight_curve

try:
    import prtcqt
    _HAS_CQT = True
except ImportError:
    _HAS_CQT = False


class CqtEngine(SpectrumEngineBase):
    """Real-time Constant Q Transform via rt-cqt's SlidingCqt.

    Same push_hop() interface as OctaveBankEngine.
    """

    def __init__(self, n_octaves: int = 9, bins_per_octave: int = 12,
                 fmin: float = 32.7) -> None:
        if not _HAS_CQT:
            raise ImportError("prtcqt not available — build rt-cqt Python bindings")

        self._bpo = bins_per_octave
        self._n_octaves = n_octaves
        self.n_bins = n_octaves * bins_per_octave

        # Init the sliding CQT
        # Only 12 and 24 bins_per_octave are supported by rt-cqt
        if bins_per_octave == 12:
            self._cqt = prtcqt.SlidingCqt12()
        elif bins_per_octave == 24:
            self._cqt = prtcqt.SlidingCqt24()
        else:
            raise ValueError(f"rt-cqt only supports 12 or 24 bins_per_octave, got {bins_per_octave}")

        self._cqt.init(float(SAMPLE_RATE), HOP_SIZE)

        # Build bin center frequencies from the CQT's own frequency map
        # CQT octave 0 = highest frequencies, octave N-1 = lowest
        # Within each octave, bins go ascending (low→high)
        self.bin_centers = np.zeros(self.n_bins, dtype=np.float32)
        for cqt_oct in range(n_octaves):
            our_oct = n_octaves - 1 - cqt_oct
            freqs = self._cqt.getOctaveBinFreqs(cqt_oct)
            self.bin_centers[our_oct * bins_per_octave:(our_oct + 1) * bins_per_octave] = freqs

        # A-weighting for onset strength
        self._a_weights = a_weight_curve(self.bin_centers)

        # Previous magnitude for flux computation
        self._prev_spectrum: np.ndarray | None = None

    def push_hop(self, hop: np.ndarray) -> SpectrumFrame:
        """Feed one hop of audio and return a SpectrumFrame."""
        # rt-cqt expects list of doubles (tolist is faster than numpy passthrough)
        self._cqt.inputBlock(hop.astype(np.float64).tolist(), len(hop))

        # Extract magnitudes and phase from all octaves (vectorized per octave)
        magnitude = np.zeros(self.n_bins, dtype=np.float32)
        phase = np.zeros(self.n_bins, dtype=np.float32)
        for cqt_oct in range(self._n_octaves):
            our_oct = self._n_octaves - 1 - cqt_oct
            vals = np.array(self._cqt.getOctaveValues(cqt_oct))
            start = our_oct * self._bpo
            magnitude[start:start + self._bpo] = np.abs(vals)
            phase[start:start + self._bpo] = np.angle(vals)

        # Flux
        if self._prev_spectrum is not None:
            flux = np.maximum(magnitude - self._prev_spectrum, 0.0).astype(np.float32)
        else:
            flux = np.zeros(self.n_bins, dtype=np.float32)
        self._prev_spectrum = magnitude.copy()

        # ZCR
        signs = np.signbit(hop)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / len(hop)

        return SpectrumFrame(
            magnitude=magnitude,
            flux=flux,
            waveform=hop.copy(),
            phase=phase,
            zcr=zcr,
        )

    def compute(self, pcm: np.ndarray) -> SpectrumFrame:
        """Compute from a full PCM buffer (feeds in HOP_SIZE chunks)."""
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
        # rt-cqt doesn't expose a reset — reinitialize
        if self._bpo == 12:
            self._cqt = prtcqt.SlidingCqt12()
        else:
            self._cqt = prtcqt.SlidingCqt24()
        self._cqt.init(float(SAMPLE_RATE), HOP_SIZE)
