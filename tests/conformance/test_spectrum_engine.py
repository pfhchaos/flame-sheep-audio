"""Spectrum engine conformance tests."""

import numpy as np
import pytest

from flame_sheep_audio._constants import HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumFrame, SpectrumEngineBase
from flame_sheep_audio._octave_bank import OctaveBankEngine
from flame_sheep_audio.stability import MagnitudeStability
from flame_sheep_audio.energy import EnergyAnalyzer

# Try importing optional engines
try:
    from flame_sheep_audio._cqt_engine import CqtEngine
    _HAS_CQT = True
except ImportError:
    _HAS_CQT = False

# All concrete SpectrumEngineBase subclasses.
# Adding a new engine? Add it here — the tests are automatic.
ENGINES = [("octave_bank", OctaveBankEngine)]
if _HAS_CQT:
    ENGINES.append(("cqt", CqtEngine))

def test_all_spectrum_engines_registered():
    """Ensure every SpectrumEngineBase subclass has a test entry."""
    from flame_sheep_audio._spectrum import SpectrumEngine
    # Exclude the FFT SpectrumEngine (it doesn't implement push_hop the same way)
    # and optional engines that aren't installed
    tested = {cls.__name__ for _, cls in ENGINES}
    concrete = set()
    for cls in SpectrumEngineBase.__subclasses__():
        if cls is SpectrumEngine:
            continue  # legacy FFT engine, separate interface
        try:
            # Only check if the class can be imported
            concrete.add(cls.__name__)
        except Exception:
            pass
    missing = concrete - tested
    assert not missing, f"SpectrumEngineBase subclasses without test entries: {missing}"


class TestSpectrumEngineConformance:
    """Every spectrum engine must produce consistent SpectrumFrame shapes."""

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_output_shape_consistency(self, name, cls):
        """All frame arrays must have the same length as n_bins."""
        engine = cls()
        hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
        frame = engine.push_hop(hop)
        assert frame.magnitude.shape == (engine.n_bins,), \
            f"{name}: magnitude {frame.magnitude.shape} != ({engine.n_bins},)"
        assert frame.flux.shape == (engine.n_bins,), \
            f"{name}: flux {frame.flux.shape} != ({engine.n_bins},)"
        assert frame.waveform.shape == (HOP_SIZE,)

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_silent_input(self, name, cls):
        """Engine must handle silence without crashing."""
        engine = cls()
        silence = np.zeros(HOP_SIZE, dtype=np.float32)
        frame = engine.push_hop(silence)
        assert frame.magnitude.shape == (engine.n_bins,)
        assert np.all(np.isfinite(frame.magnitude))

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_bin_centers_match_n_bins(self, name, cls):
        """bin_centers must have exactly n_bins elements."""
        engine = cls()
        assert len(engine.bin_centers) == engine.n_bins

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_compute_matches_push_hop(self, name, cls):
        """compute() must produce same shape as push_hop()."""
        engine = cls()
        pcm = np.random.randn(HOP_SIZE * 4).astype(np.float32) * 0.3
        frame = engine.compute(pcm)
        assert frame.magnitude.shape == (engine.n_bins,)

    @pytest.mark.parametrize("name,cls", ENGINES)
    def test_reset_allows_reuse(self, name, cls):
        """After reset, engine must work normally."""
        engine = cls()
        hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
        engine.push_hop(hop)
        engine.reset()
        frame = engine.push_hop(hop)
        assert frame.magnitude.shape == (engine.n_bins,)


# ===================================================================
# Integration: engine + stability + energy pipeline
# ===================================================================

class TestPipelineConformance:
    """Full pipeline must work for every engine x stability combination."""

    @pytest.mark.parametrize("eng_name,eng_cls", ENGINES)
    @pytest.mark.parametrize("stab_method", ["ema", "median", "shape"])
    def test_full_pipeline(self, eng_name, eng_cls, stab_method):
        """10 frames through engine -> stability -> energy must not crash."""
        engine = eng_cls()
        stab = MagnitudeStability(method=stab_method)
        energy = EnergyAnalyzer(freqs=engine.bin_centers)

        for i in range(10):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine.push_hop(hop)
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)

    @pytest.mark.parametrize("eng_name,eng_cls", ENGINES)
    @pytest.mark.parametrize("stab_method", ["ema", "median", "shape"])
    def test_pipeline_engine_hotswap(self, eng_name, eng_cls, stab_method):
        """Switching spectrum engine mid-stream must not crash.

        Stability and energy must adapt to the new bin count after
        the engine is swapped.
        """
        # Start with octave bank
        engine1 = OctaveBankEngine()
        stab = MagnitudeStability(method=stab_method)
        energy = EnergyAnalyzer(freqs=engine1.bin_centers)

        for _ in range(10):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine1.push_hop(hop)
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)

        # Swap to the test engine — stability and energy must be recreated
        # (they hold masks/freqs tied to bin count)
        engine2 = eng_cls()
        stab.reset()
        energy2 = EnergyAnalyzer(freqs=engine2.bin_centers)

        for _ in range(10):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine2.push_hop(hop)
            stab.update(frame.magnitude)
            energy2.update(frame.magnitude, frame.flux, stability=stab)

    @pytest.mark.parametrize("eng_name,eng_cls", ENGINES)
    @pytest.mark.parametrize("stab_method", ["ema", "median", "shape"])
    def test_pipeline_silent_start(self, eng_name, eng_cls, stab_method):
        """Pipeline must survive 5 silent frames then 5 signal frames."""
        engine = eng_cls()
        stab = MagnitudeStability(method=stab_method)
        energy = EnergyAnalyzer(freqs=engine.bin_centers)

        # Silent frames
        for _ in range(5):
            frame = engine.push_hop(np.zeros(HOP_SIZE, dtype=np.float32))
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)

        # Signal frames
        for _ in range(5):
            hop = np.random.randn(HOP_SIZE).astype(np.float32) * 0.3
            frame = engine.push_hop(hop)
            stab.update(frame.magnitude)
            energy.update(frame.magnitude, frame.flux, stability=stab)
