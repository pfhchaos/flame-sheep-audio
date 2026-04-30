"""Energy analyzer — sub-bass RMS, spectral centroid, percussiveness."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stability import MagnitudeStability

from ._constants import N_BINS, FFT_SIZE, FREQS, SAMPLE_RATE
from ._bands import make_mask, A_WEIGHTS
from ._band_config import BandConfig, default_band_config
from .config import cfg


class EnergyAnalyzer:
    """Track audio energy features for visual response.

    Provides:
      - rms: sub-bass energy (20-200Hz) — visceral loudness
      - centroid: spectral center of mass — bright vs dark
      - centroid_rms: energy around the centroid — how present the dominant voice is
      - percussiveness: flux/magnitude ratio — drums vs sustain
    """

    def __init__(self, alpha: float | None = None,
                 band_config: BandConfig | None = None) -> None:
        self._alpha = alpha if alpha is not None else cfg.energy.rms_alpha
        if band_config is None:
            band_config = default_band_config()
        self._band_config = band_config

        # Per-band RMS and harmonic RMS (masks built from config)
        self._masks = {name: make_mask(*rng)
                       for name, rng in band_config.all_band_ranges.items()}
        self._band_rms = {name: 0.0 for name in self._masks}
        self._band_harmonic_rms = {name: 0.0 for name in self._masks}

        # Slow envelope per band (asymmetric attack/release)
        self._slow_rms = {name: 0.0 for name in self._masks}
        self._slow_harmonic_rms = {name: 0.0 for name in self._masks}

        # Centroid tracking
        self._centroid = 1000.0
        self._prev_centroid = 1000.0
        self._centroid_rms = 0.0
        self._centroid_alpha = cfg.energy.centroid_alpha

        # Section change detection: dual-EMA on normalized centroid + energy
        self._centroid_fast_norm = 0.5
        self._centroid_slow_norm = 0.5
        self._energy_fast_norm = 0.0
        self._energy_slow_norm = 0.0
        self._SECTION_FAST_ALPHA = cfg.section.fast_alpha
        self._SECTION_SLOW_ALPHA = cfg.section.slow_alpha
        self._SECTION_W_CENTROID = cfg.section.weight_centroid
        self._SECTION_W_ENERGY = cfg.section.weight_energy
        # Normalization constants
        self._CENTROID_LOG_MIN = np.log2(20.0)
        self._CENTROID_LOG_RANGE = np.log2(20000.0) - np.log2(20.0)  # ~10 octaves
        self._ENERGY_REF = FFT_SIZE / 2.0

        # Percussiveness tracking
        self._percussiveness = 0.5
        self._perc_alpha = cfg.energy.percussiveness_alpha
        # Spectral shape distance — alternative percussiveness measure
        self._shape_ema: np.ndarray | None = None  # EMA of normalized spectrum
        self._shape_alpha = self._perc_alpha        # same smoothing as flux perc
        self._shape_percussiveness = 0.0

        # Harmonic energy (stability-weighted)
        self._harmonic_rms = 0.0
        self._harmonic_centroid_rms = 0.0

    def update(self, spectrum: np.ndarray, flux: np.ndarray | None = None,
               stability: MagnitudeStability | None = None) -> float:
        """Update all features from spectrum magnitude and flux.

        Args:
            spectrum: FFT magnitude (N_BINS,)
            flux: spectral flux (N_BINS,), optional — needed for percussiveness
            stability: MagnitudeStability reference, optional — for harmonic RMS

        Returns:
            Smoothed sub-bass RMS.
        """
        alpha = self._alpha

        # Per-band RMS and harmonic RMS (unified loop)
        slow_attack = cfg.energy.slow_attack_alpha
        slow_release = cfg.energy.slow_release_alpha
        for name, mask in self._masks.items():
            raw = float(np.sqrt(np.mean(spectrum[mask] ** 2)))
            self._band_rms[name] = alpha * self._band_rms[name] + (1 - alpha) * raw
            # Slow envelope: asymmetric attack/release
            rms = self._band_rms[name]
            sa = slow_attack if rms > self._slow_rms[name] else slow_release
            self._slow_rms[name] = sa * self._slow_rms[name] + (1 - sa) * rms
            if stability is not None:
                raw_h = stability.harmonic_rms(spectrum, mask)
                self._band_harmonic_rms[name] = (
                    alpha * self._band_harmonic_rms[name] + (1 - alpha) * raw_h)
                hrms = self._band_harmonic_rms[name]
                sha = slow_attack if hrms > self._slow_harmonic_rms[name] else slow_release
                self._slow_harmonic_rms[name] = (
                    sha * self._slow_harmonic_rms[name] + (1 - sha) * hrms)

        # Spectral centroid (A-weighted for perceptual accuracy)
        weighted_spec = spectrum * A_WEIGHTS
        mag_sum = weighted_spec.sum()
        if mag_sum > 1e-10:
            raw_centroid = float(np.sum(FREQS * weighted_spec) / mag_sum)
            self._prev_centroid = self._centroid
            self._centroid = (self._centroid_alpha * self._centroid
                              + (1 - self._centroid_alpha) * raw_centroid)

            # Section change dual-EMAs on normalized centroid + energy
            fa = self._SECTION_FAST_ALPHA
            sa = self._SECTION_SLOW_ALPHA
            # Normalize centroid to ~0-1 (log scale, 20 Hz = 0, 20 kHz = 1)
            c_norm = (np.log2(max(self._centroid, 20.0)) - self._CENTROID_LOG_MIN) / self._CENTROID_LOG_RANGE
            self._centroid_fast_norm = fa * self._centroid_fast_norm + (1 - fa) * c_norm
            self._centroid_slow_norm = sa * self._centroid_slow_norm + (1 - sa) * c_norm
            # Normalize energy by theoretical max
            oss = float(np.dot(spectrum, A_WEIGHTS))
            e_norm = oss / self._ENERGY_REF
            self._energy_fast_norm = fa * self._energy_fast_norm + (1 - fa) * e_norm
            self._energy_slow_norm = sa * self._energy_slow_norm + (1 - sa) * e_norm

            # RMS around centroid (±1 octave)
            lo_c = self._centroid / 2
            hi_c = self._centroid * 2
            centroid_mask = (FREQS >= lo_c) & (FREQS <= hi_c)
            if centroid_mask.any():
                raw_c_rms = float(np.sqrt(np.mean(spectrum[centroid_mask] ** 2)))
                self._centroid_rms = (self._centroid_alpha * self._centroid_rms
                                      + (1 - self._centroid_alpha) * raw_c_rms)
                if stability is not None:
                    raw_hc = stability.harmonic_rms(spectrum, centroid_mask)
                    self._harmonic_centroid_rms = (
                        self._centroid_alpha * self._harmonic_centroid_rms
                        + (1 - self._centroid_alpha) * raw_hc)

        # Percussiveness: flux / magnitude ratio (original method)
        if flux is not None and mag_sum > 1e-10:
            raw_perc = float(flux.sum() / mag_sum)
            self._percussiveness = (self._perc_alpha * self._percussiveness
                                     + (1 - self._perc_alpha) * raw_perc)

        # Spectral shape distance percussiveness:
        # 1. Maintain EMA of normalized spectrum ("what it usually looks like")
        # 2. Cosine distance between current frame and EMA shape
        # 3. EMA of those distances → percussiveness
        # Drums deviate from the running shape. Vibrato barely moves it.
        spec_norm = np.linalg.norm(spectrum)
        if spec_norm > 1e-10:
            normalized = spectrum / spec_norm
            if self._shape_ema is None:
                self._shape_ema = normalized.copy()
            else:
                # Distance from current frame to the running average shape
                shape_dist = 1.0 - float(np.dot(self._shape_ema, normalized))
                shape_dist = max(0.0, min(1.0, shape_dist))
                self._shape_percussiveness = (
                    self._shape_alpha * self._shape_percussiveness
                    + (1 - self._shape_alpha) * shape_dist)
                # Update shape EMA (unnormalized — the norm drift appears
                # to help speech/music discrimination)
                self._shape_ema = (self._shape_alpha * self._shape_ema
                                   + (1 - self._shape_alpha) * normalized)

        return self._band_rms.get('subbass', 0.0)

    @property
    def rms(self) -> float:
        """Sub-bass RMS (20-200Hz) — backwards compat alias."""
        return self._band_rms.get('subbass', 0.0)

    @property
    def band_rms_all(self) -> dict[str, float]:
        """Per-band RMS for all analysis bands."""
        return dict(self._band_rms)

    @property
    def band_harmonic_rms_all(self) -> dict[str, float]:
        """Per-band harmonic RMS for all analysis bands."""
        return dict(self._band_harmonic_rms)

    @property
    def band_slow_rms_all(self) -> dict[str, float]:
        """Per-band slow-envelope RMS (~2s attack, ~0.5s release)."""
        return dict(self._slow_rms)

    @property
    def band_slow_harmonic_rms_all(self) -> dict[str, float]:
        """Per-band slow-envelope harmonic RMS (~2s attack, ~0.5s release)."""
        return dict(self._slow_harmonic_rms)

    @property
    def centroid(self) -> float:
        """Spectral centroid in Hz — where the energy center of mass is."""
        return self._centroid

    @property
    def centroid_delta(self) -> float:
        """Absolute change in centroid since last frame (Hz)."""
        return abs(self._centroid - self._prev_centroid)

    @property
    def section_change(self) -> float:
        """Section change signal: weighted euclidean distance in normalized space.

        Uses dual-EMA (fast ~2s, slow ~15s) on normalized centroid (log Hz,
        0-1) and energy (fraction of theoretical max). Symmetric — loud→quiet
        and quiet→loud produce the same distance.

        Axes scaled by configurable weights (section.weight_centroid/energy).
        """
        dc = self._SECTION_W_CENTROID * (self._centroid_fast_norm - self._centroid_slow_norm)
        de = self._SECTION_W_ENERGY * (self._energy_fast_norm - self._energy_slow_norm)
        return float(np.sqrt(dc ** 2 + de ** 2))

    @property
    def section_fast(self) -> tuple[float, float]:
        """Fast EMA position in normalized (centroid, energy) space."""
        return (self._centroid_fast_norm, self._energy_fast_norm)

    @property
    def section_slow(self) -> tuple[float, float]:
        """Slow EMA position in normalized (centroid, energy) space."""
        return (self._centroid_slow_norm, self._energy_slow_norm)

    @property
    def centroid_rms(self) -> float:
        """RMS energy around the centroid (±1 octave)."""
        return self._centroid_rms

    @property
    def percussiveness(self) -> float:
        """Percussiveness measure — high = drums/transients, low = sustained tonal.

        Uses spectral shape distance (cosine distance between consecutive
        normalized spectra) by default. Falls back to flux/magnitude ratio
        if configured via energy.percussiveness_method = 'flux'.
        """
        method = getattr(cfg.energy, 'percussiveness_method', 'shape')
        if method == 'flux':
            return self._percussiveness
        return self._shape_percussiveness

    @property
    def flux_percussiveness(self) -> float:
        """Original flux/magnitude ratio percussiveness (for comparison)."""
        return self._percussiveness

    @property
    def shape_percussiveness(self) -> float:
        """Spectral shape distance percussiveness (for comparison)."""
        return self._shape_percussiveness

    @property
    def harmonic_rms(self) -> float:
        """Sub-bass harmonic RMS (backwards compat)."""
        return self._band_harmonic_rms.get('subbass', 0.0)

    @property
    def harmonic_centroid_rms(self) -> float:
        """RMS around centroid from stable (harmonic) bins only."""
        return self._harmonic_centroid_rms
