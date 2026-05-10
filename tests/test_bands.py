"""Tests for spectral band definitions and utilities.

Covers frequency mask generation, A-weighting curve, BandConfig,
and SpringBand physics.
"""

import numpy as np
import pytest

from flame_sheep_audio._constants import SAMPLE_RATE, N_BINS, FREQS
from flame_sheep_audio._bands import (
    make_mask, make_weights, a_weight_curve, A_WEIGHTS,
    SpringBand,
)
from flame_sheep_audio._band_config import (
    BandConfig, EnergyBandDef, DetectionBandDef, default_band_config,
)


class TestMakeMask:
    """Test boolean mask generation from frequency ranges."""

    def test_shape(self):
        mask = make_mask(100, 1000)
        assert mask.shape == (N_BINS,)
        assert mask.dtype == bool

    def test_includes_range(self):
        mask = make_mask(100, 1000)
        in_range = (FREQS >= 100) & (FREQS < 1000)
        np.testing.assert_array_equal(mask, in_range)

    def test_excludes_outside(self):
        mask = make_mask(100, 200)
        assert not mask[0]
        assert not mask[-1]

    def test_full_range(self):
        mask = make_mask(0, SAMPLE_RATE / 2 + 1)
        assert mask.all()

    def test_empty_range(self):
        mask = make_mask(99999, 99999.1)
        assert not mask.any()


class TestMakeWeights:
    """Test normalized weight generation."""

    def test_sums_to_one(self):
        w = make_weights(100, 1000)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_zero_outside_range(self):
        w = make_weights(5000, 10000)
        low_bins = FREQS < 5000
        assert (w[low_bins] == 0).all()

    def test_uniform_within_range(self):
        w = make_weights(100, 1000)
        active = w[w > 0]
        assert np.allclose(active, active[0])

    def test_empty_range_all_zero(self):
        w = make_weights(99999, 99999.1)
        assert (w == 0).all()


class TestAWeighting:
    """Test A-weighting curve properties."""

    def test_1khz_is_unity(self):
        idx = np.argmin(np.abs(FREQS - 1000))
        assert abs(A_WEIGHTS[idx] - 1.0) < 0.05

    def test_low_freq_attenuated(self):
        idx_50hz = np.argmin(np.abs(FREQS - 50))
        idx_1khz = np.argmin(np.abs(FREQS - 1000))
        assert A_WEIGHTS[idx_50hz] < A_WEIGHTS[idx_1khz]

    def test_high_freq_attenuated(self):
        idx_15khz = np.argmin(np.abs(FREQS - 15000))
        idx_1khz = np.argmin(np.abs(FREQS - 1000))
        assert A_WEIGHTS[idx_15khz] < A_WEIGHTS[idx_1khz]

    def test_peak_near_2_4khz(self):
        mask_peak = (FREQS >= 2000) & (FREQS <= 5000)
        mask_low = (FREQS >= 50) & (FREQS <= 200)
        assert A_WEIGHTS[mask_peak].max() > A_WEIGHTS[mask_low].max()

    def test_shape_and_dtype(self):
        assert A_WEIGHTS.shape == (N_BINS,)
        assert A_WEIGHTS.dtype == np.float32

    def test_no_negative_values(self):
        assert (A_WEIGHTS >= 0).all()


class TestBandConfig:
    """Verify BandConfig structure and default config consistency."""

    def test_default_has_energy_and_detection_bands(self):
        cfg = default_band_config()
        assert len(cfg.energy_bands) >= 1
        assert len(cfg.detection_bands) >= 1

    def test_all_band_names_unique(self):
        cfg = default_band_config()
        names = cfg.all_band_names
        assert len(names) == len(set(names))

    def test_detection_band_names_subset(self):
        cfg = default_band_config()
        assert set(cfg.detection_band_names) <= set(cfg.all_band_names)

    def test_all_band_ranges_complete(self):
        cfg = default_band_config()
        for name in cfg.all_band_names:
            assert name in cfg.all_band_ranges
            lo, hi = cfg.all_band_ranges[name]
            assert lo < hi

    def test_detection_defaults_within_allowed(self):
        cfg = default_band_config()
        for band in cfg.detection_bands:
            d_lo, d_hi = band.freq_range
            a_lo, a_hi = band.allowed_range
            assert d_lo >= a_lo, \
                f'{band.name} default lo {d_lo} < allowed {a_lo}'

    def test_no_detection_band_gaps(self):
        """Adjacent detection bands should be contiguous or overlapping."""
        cfg = default_band_config()
        ranges = sorted(
            [b.freq_range for b in cfg.detection_bands],
            key=lambda r: r[0])
        for i in range(len(ranges) - 1):
            _, hi = ranges[i]
            lo_next, _ = ranges[i + 1]
            assert lo_next <= hi, \
                f'Gap between {ranges[i]} and {ranges[i+1]}'

    def test_custom_config(self):
        """A custom BandConfig should work with arbitrary bands."""
        cfg = BandConfig(
            energy_bands=(
                EnergyBandDef('low', (20, 500)),
                EnergyBandDef('high', (500, 20000)),
            ),
            detection_bands=(
                DetectionBandDef('pulse', (30, 200), (20, 300)),
            ),
        )
        assert cfg.all_band_names == ('low', 'high', 'pulse')
        assert cfg.detection_band_names == ('pulse',)

    def test_frozen(self):
        cfg = default_band_config()
        with pytest.raises(AttributeError):
            cfg.energy_bands = ()


class TestSpringBand:
    """Test spring-model adaptive band."""

    def test_initial_center(self):
        sb = SpringBand('low', default_range=(30, 200), allowed_range=(25, 150))
        assert sb.center == pytest.approx((30 + 200) / 2.0)

    def test_initial_mask_matches_default(self):
        sb = SpringBand('low', default_range=(30, 200), allowed_range=(25, 150))
        expected = make_mask(30, 200)
        np.testing.assert_array_equal(sb.mask, expected)

    def test_anchor_pulls_toward_default(self):
        sb = SpringBand('mid', default_range=(200, 1000),
                        allowed_range=(150, 2000))
        sb.center = sb.default_center + 200
        original = sb.center
        sb.apply_forces(anchor_k=1.0, flux_k=0.0,
                        neighbors=[], repulsion_k=0.0)
        assert abs(sb.center - sb.default_center) < abs(original - sb.default_center)

    def test_center_stays_in_allowed_range(self):
        sb = SpringBand('mid', default_range=(200, 1000),
                        allowed_range=(150, 2000))
        sb.center = 999999
        sb.apply_forces(anchor_k=0.01, flux_k=0.0,
                        neighbors=[], repulsion_k=0.0)
        half = sb.width
        assert sb.center >= sb.lo_allowed + half
        assert sb.center <= sb.hi_allowed - half

    def test_repulsion_pushes_bands_apart(self):
        low = SpringBand('low', default_range=(30, 200),
                         allowed_range=(25, 150))
        mid = SpringBand('mid', default_range=(200, 1000),
                         allowed_range=(150, 2000))
        mid.center = low.center + 10
        original_dist = abs(low.center - mid.center)
        low.apply_forces(anchor_k=0.0, flux_k=0.0,
                         neighbors=[mid], repulsion_k=100.0)
        new_dist = abs(low.center - mid.center)
        assert new_dist > original_dist

    def test_reset_restores_defaults(self):
        sb = SpringBand('low', default_range=(30, 200),
                        allowed_range=(25, 150))
        sb.center = 999
        sb.width = 1
        # Trigger lazy init of _flux_ema, then dirty it
        flux = np.ones(N_BINS, dtype=np.float32)
        stab = np.zeros(N_BINS, dtype=np.float32)
        sb.update_flux_ema(flux, stab)
        sb.reset()
        assert sb.center == sb.default_center
        assert sb.width == sb.default_width
        assert sb.flux_centroid() == sb.default_center

    def test_flux_centroid_default_when_empty(self):
        sb = SpringBand('low', default_range=(30, 200),
                        allowed_range=(25, 150))
        assert sb.flux_centroid() == sb.default_center

    def test_flux_ema_accumulates(self):
        sb = SpringBand('low', default_range=(30, 200),
                        allowed_range=(25, 150))
        flux = np.zeros(N_BINS, dtype=np.float32)
        target_idx = np.argmin(np.abs(FREQS - 100))
        flux[target_idx] = 1.0
        stability = np.zeros(N_BINS, dtype=np.float32)
        sb.update_flux_ema(flux, stability, alpha=0.0)
        centroid = sb.flux_centroid()
        assert abs(centroid - 100) < 30
