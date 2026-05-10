"""Spectrum transform conformance tests."""

import numpy as np
import pytest

from flame_sheep_audio._constants import HOP_SIZE
from flame_sheep_audio._spectrum import SpectrumFrame
from flame_sheep_audio.stability import _StabilityEMA, _StabilityShape
from flame_sheep_audio.hpss import SpectrumTransform, LogMagnitudeTransform, HarmonicTransform, PercussiveTransform

BIN_COUNTS = [108, 1025, 64]

TRANSFORM_CLASSES = [
    ("log_magnitude", lambda: LogMagnitudeTransform()),
    ("harmonic", lambda: HarmonicTransform(_StabilityEMA(alpha=0.95))),
    ("percussive", lambda: PercussiveTransform(_StabilityEMA(alpha=0.95))),
    ("harmonic_shape", lambda: HarmonicTransform(_StabilityShape(alpha=0.95, kernel=3))),
    ("percussive_shape", lambda: PercussiveTransform(_StabilityShape(alpha=0.95, kernel=3))),
]


def test_all_transforms_registered():
    """Ensure every SpectrumTransform subclass has a test entry."""
    concrete = {cls.__name__ for cls in SpectrumTransform.__subclasses__()}
    tested = {factory().__class__.__name__ for _, factory in TRANSFORM_CLASSES}
    missing = concrete - tested
    assert not missing, f"SpectrumTransform subclasses without test entries: {missing}"


class TestSpectrumTransformConformance:
    """Every transform must preserve shapes and handle edge cases."""

    @pytest.mark.parametrize("name,factory", TRANSFORM_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_preserves_shape(self, name, factory, n_bins):
        """Output arrays must match input shape."""
        t = factory()
        frame = SpectrumFrame(
            magnitude=np.abs(np.random.randn(n_bins).astype(np.float32)),
            flux=np.abs(np.random.randn(n_bins).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        out = t(frame)
        assert out.magnitude.shape == (n_bins,)
        assert out.flux.shape == (n_bins,)
        assert out.waveform.shape == (HOP_SIZE,)

    @pytest.mark.parametrize("name,factory", TRANSFORM_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_silent_input(self, name, factory, n_bins):
        """Silence must not crash or produce NaN."""
        t = factory()
        frame = SpectrumFrame(
            magnitude=np.zeros(n_bins, dtype=np.float32),
            flux=np.zeros(n_bins, dtype=np.float32),
            waveform=np.zeros(HOP_SIZE, dtype=np.float32),
        )
        out = t(frame)
        assert np.all(np.isfinite(out.magnitude))
        assert np.all(np.isfinite(out.flux))

    @pytest.mark.parametrize("name,factory", TRANSFORM_CLASSES)
    def test_composable(self, name, factory):
        """transform(transform(frame)) must not crash."""
        t = factory()
        frame = SpectrumFrame(
            magnitude=np.abs(np.random.randn(108).astype(np.float32)),
            flux=np.abs(np.random.randn(108).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        out1 = t(frame)
        out2 = t(out1)  # feed output back in
        assert out2.magnitude.shape == (108,)

    MASKING_TRANSFORMS = [t for t in TRANSFORM_CLASSES if t[0] != "log_magnitude"]

    @pytest.mark.parametrize("name,factory", MASKING_TRANSFORMS)
    def test_output_bounded(self, name, factory):
        """Masking transforms: output magnitude must not exceed input."""
        t = factory()
        frame = SpectrumFrame(
            magnitude=np.abs(np.random.randn(108).astype(np.float32)),
            flux=np.abs(np.random.randn(108).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        # Feed a few frames so the estimator stabilizes
        for _ in range(10):
            out = t(frame)
        assert np.all(out.magnitude <= frame.magnitude + 1e-6)

    @pytest.mark.parametrize("name,factory", TRANSFORM_CLASSES)
    @pytest.mark.parametrize("n_bins", BIN_COUNTS)
    def test_reset_then_new_size(self, name, factory, n_bins):
        """After reset, must accept different bin count."""
        t = factory()
        frame1 = SpectrumFrame(
            magnitude=np.abs(np.random.randn(64).astype(np.float32)),
            flux=np.abs(np.random.randn(64).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        t(frame1)
        t.reset()
        frame2 = SpectrumFrame(
            magnitude=np.abs(np.random.randn(n_bins).astype(np.float32)),
            flux=np.abs(np.random.randn(n_bins).astype(np.float32)),
            waveform=np.random.randn(HOP_SIZE).astype(np.float32),
        )
        out = t(frame2)
        assert out.magnitude.shape == (n_bins,)
