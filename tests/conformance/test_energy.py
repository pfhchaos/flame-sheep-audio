"""Energy analyzer conformance tests."""

import numpy as np
import pytest

from flame_sheep_audio.stability import MagnitudeStability
from flame_sheep_audio.energy import EnergyAnalyzer

BIN_COUNTS = [108, 1025, 64]


class TestEnergyConformance:
    """EnergyAnalyzer must work with any bin count."""

    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_update_with_stability(self, n_bins):
        """Energy update with stability must not crash on any bin count."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        energy = EnergyAnalyzer(freqs=freqs)
        stab = MagnitudeStability()
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        flux = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        energy.update(mag, flux, stability=stab)

    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_update_silent_start_with_stability(self, n_bins):
        """Energy + stability must work when audio starts with silence."""
        freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        energy = EnergyAnalyzer(freqs=freqs)
        stab = MagnitudeStability()
        silence = np.zeros(n_bins, dtype=np.float32)
        stab.update(silence)
        energy.update(silence, silence, stability=stab)
