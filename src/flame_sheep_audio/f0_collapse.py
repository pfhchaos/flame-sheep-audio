"""F0 harmonic collapsing — SpectrumFrame → SpectrumFrame transform.

Subtracts harmonic energy octave by octave, bottom up. Each octave's
magnitude is reduced by the energy already explained by fundamentals
in lower octaves. What remains is energy from new fundamentals at
that frequency.

Example: a 440Hz tone with harmonics at 880, 1320, 1760:
  - Octave containing 440Hz: full magnitude (fundamental)
  - Octave containing 880Hz: magnitude minus 440Hz's contribution → ~0
  - Octave containing 1320Hz: minus 440Hz's 3rd harmonic → ~0

A low-frequency hit at 60Hz with a harmonic at 120Hz:
  - Octave 0 (33-65Hz): energy survives
  - Octave 1 (65-131Hz): 120Hz minus 60Hz's 2nd harmonic → ~0

A mid-frequency hit at 200Hz with no fundamental below it:
  - Octave 2 (131-262Hz): nothing to subtract → energy survives

Works with any log-spaced bin layout (CQT, octave bank).
Assumes bins_per_octave bins per octave, evenly log-spaced.

Usage:
    collapse = F0Collapse(n_octaves=9, bins_per_octave=12)
    cleaned = collapse(percussive_frame)
"""

from __future__ import annotations

import numpy as np

from ._spectrum import SpectrumFrame
from .hpss import SpectrumTransform


class F0Collapse(SpectrumTransform):
    """Subtract harmonic energy from higher octaves, bottom up.

    Stateless — no history. Each frame is processed independently.
    Assumes log-spaced bins with fixed bins_per_octave.
    """

    def __init__(self, n_octaves: int = 9, bins_per_octave: int = 12,
                 n_harmonics: int = 5, decay: float = 0.5) -> None:
        """
        Args:
            n_octaves: number of octaves in the spectrum
            bins_per_octave: bins per octave (must match spectrum engine)
            n_harmonics: max harmonic number to subtract (2=octave only,
                5=up to 5th harmonic)
            decay: expected amplitude decay per harmonic step.
                0.5 means 2nd harmonic is 50% of fundamental,
                3rd is 25%, etc. Controls how much we subtract.
        """
        self._n_oct = n_octaves
        self._bpo = bins_per_octave
        self._n_bins = n_octaves * bins_per_octave
        self._n_harmonics = n_harmonics
        self._decay = decay

        # Precompute: for harmonic number n, the offset in bins and
        # expected amplitude ratio relative to fundamental.
        # 2nd harmonic = +12 bins (1 octave), 3rd = +~19 bins, etc.
        # In log2 space: harmonic n is at log2(n) octaves above fundamental
        # Bin offset = round(log2(n) * bins_per_octave)
        self._harmonic_offsets = []
        for n in range(2, n_harmonics + 2):
            offset = round(np.log2(n) * bins_per_octave)
            ratio = decay ** (n - 1)  # expected amplitude ratio
            if offset > 0 and offset not in [o for o, _, _ in self._harmonic_offsets]:
                self._harmonic_offsets.append((offset, n, ratio))

    def __call__(self, frame: SpectrumFrame) -> SpectrumFrame:
        """Subtract predicted harmonic energy from higher bins.

        For each bin, predict what its harmonics should look like
        (fundamental × decay^n), and subtract that prediction from
        the harmonic bins. What remains is energy not explained by
        any fundamental below — either a real fundamental at that
        frequency, or noise.

        Uses original (pre-subtraction) magnitudes for predictions
        so earlier subtractions don't cascade errors.
        """
        orig_mag = frame.magnitude
        orig_flux = frame.flux
        mag = orig_mag.copy()
        flux = orig_flux.copy()

        # Walk low to high. For each bin with energy, subtract its
        # predicted harmonic contributions from higher bins.
        # Use ORIGINAL magnitude for predictions to avoid cascading.
        for i in range(self._n_bins):
            if orig_mag[i] <= 0:
                continue
            for offset, _n, ratio in self._harmonic_offsets:
                j = i + offset
                if j >= self._n_bins:
                    break
                # Predicted harmonic amplitude = fundamental × decay ratio
                predicted = orig_mag[i] * ratio
                mag[j] = max(0.0, mag[j] - predicted)
                predicted_flux = orig_flux[i] * ratio
                flux[j] = max(0.0, flux[j] - predicted_flux)

        return SpectrumFrame(
            magnitude=mag.astype(np.float32),
            flux=flux.astype(np.float32),
            waveform=frame.waveform,
            zcr=frame.zcr,
        )

    def reset(self) -> None:
        pass  # stateless

    @property
    def harmonic_offsets(self) -> list[tuple[int, int, float]]:
        """List of (bin_offset, harmonic_number, decay_ratio) triples."""
        return list(self._harmonic_offsets)
