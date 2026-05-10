"""Stability interface conformance tests."""

import numpy as np
import pytest

from flame_sheep_audio.stability import (
    StabilityMethod,
    MagnitudeStability, _StabilityEMA, _StabilityMedian, _StabilityShape,
)

BIN_COUNTS = [108, 1025, 64]

# All concrete StabilityMethod subclasses with their factory functions.
# Adding a new stability method? Add it here — the tests are automatic.
STABILITY_CLASSES = [
    ("ema", lambda: _StabilityEMA(alpha=0.95)),
    ("median", lambda: _StabilityMedian(kernel_time=15, kernel_freq=7)),
    ("shape", lambda: _StabilityShape(alpha=0.95, kernel=7)),
]

def test_all_stability_methods_registered():
    """Ensure every StabilityMethod subclass has a test entry."""
    concrete = {cls.__name__ for cls in StabilityMethod.__subclasses__()}
    tested = {factory().__class__.__name__ for _, factory in STABILITY_CLASSES}
    missing = concrete - tested
    assert not missing, f"StabilityMethod subclasses without test entries: {missing}"


class TestStabilityConformance:
    """Every stability implementation must handle arbitrary bin counts."""

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_output_matches_input_size(self, name, factory, n_bins):
        """stability_per_bin() must return same size as input magnitude."""
        stab = factory()
        mag = np.random.randn(n_bins).astype(np.float32) * 0.01
        mag = np.abs(mag)
        stab.update(mag)
        result = stab.stability_per_bin()
        assert result.shape == (n_bins,), \
            f"{name}: expected ({n_bins},), got {result.shape}"

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_silent_first_frame(self, name, factory, n_bins):
        """Must handle silence on the first frame without size mismatch."""
        stab = factory()
        silence = np.zeros(n_bins, dtype=np.float32)
        stab.update(silence)
        result = stab.stability_per_bin()
        assert result.shape == (n_bins,), \
            f"{name}: silent frame returned {result.shape}, expected ({n_bins},)"

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_stability_per_bin_before_update(self, name, factory, n_bins):
        """stability_per_bin() before any update should not crash.

        May return a default size, but must not raise.
        """
        stab = factory()
        result = stab.stability_per_bin()
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_multiple_silent_then_signal(self, name, factory, n_bins):
        """Multiple silent frames followed by signal must work."""
        stab = factory()
        silence = np.zeros(n_bins, dtype=np.float32)
        for _ in range(5):
            stab.update(silence)
            result = stab.stability_per_bin()
            assert result.shape == (n_bins,)

        # Now feed signal
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        result = stab.stability_per_bin()
        assert result.shape == (n_bins,)

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_harmonic_rms_matches_input(self, name, factory, n_bins):
        """harmonic_rms() must work with masks matching input size."""
        stab = factory()
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        mask = np.zeros(n_bins, dtype=bool)
        mask[:n_bins // 2] = True
        result = stab.harmonic_rms(mag, mask)
        assert isinstance(result, float)

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_band_stability_matches_input(self, name, factory, n_bins):
        """band_stability() must work with masks matching input size."""
        stab = factory()
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        mask = np.ones(n_bins, dtype=bool)
        result = stab.band_stability(mask)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("name,factory", STABILITY_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_reset_then_different_size(self, name, factory, n_bins):
        """After reset, must accept a different bin count."""
        stab = factory()
        # Feed one size
        mag1 = np.abs(np.random.randn(64).astype(np.float32))
        stab.update(mag1)
        assert stab.stability_per_bin().shape == (64,)
        # Reset and feed different size
        stab.reset()
        mag2 = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag2)
        assert stab.stability_per_bin().shape == (n_bins,)


class TestMagnitudeStabilityConformance:
    """MagnitudeStability (composite) must respect bin counts for all methods."""

    @pytest.mark.parametrize("method", ["ema", "median", "shape"])
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_composite_output_size(self, method, n_bins):
        stab = MagnitudeStability(method=method)
        mag = np.abs(np.random.randn(n_bins).astype(np.float32))
        stab.update(mag)
        assert stab.stability_per_bin().shape == (n_bins,)

    @pytest.mark.parametrize("method", ["ema", "median", "shape"])
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_composite_silent_start(self, method, n_bins):
        stab = MagnitudeStability(method=method)
        stab.update(np.zeros(n_bins, dtype=np.float32))
        assert stab.stability_per_bin().shape == (n_bins,)
