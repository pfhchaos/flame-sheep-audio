"""Energy analyzer — sub-bass RMS, spectral centroid, percussiveness."""

import numpy as np

from ._constants import N_BINS, FREQS, SAMPLE_RATE
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

    def __init__(self, alpha: float = None, band_config: BandConfig | None = None):
        self._alpha = alpha if alpha is not None else cfg.energy.rms_alpha
        if band_config is None:
            band_config = default_band_config()
        self._band_config = band_config

        # Per-band RMS and harmonic RMS (masks built from config)
        self._masks = {name: make_mask(*rng)
                       for name, rng in band_config.all_band_ranges.items()}
        self._band_rms = {name: 0.0 for name in self._masks}
        self._band_harmonic_rms = {name: 0.0 for name in self._masks}

        # Centroid tracking
        self._centroid = 1000.0
        self._prev_centroid = 1000.0
        self._centroid_rms = 0.0
        self._centroid_alpha = cfg.energy.centroid_alpha

        # Percussiveness tracking
        self._percussiveness = 0.5
        self._perc_alpha = cfg.energy.percussiveness_alpha

        # Harmonic energy (stability-weighted)
        self._harmonic_rms = 0.0
        self._harmonic_centroid_rms = 0.0

    def update(self, spectrum: np.ndarray, flux: np.ndarray | None = None,
               stability=None) -> float:
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
        for name, mask in self._masks.items():
            raw = float(np.sqrt(np.mean(spectrum[mask] ** 2)))
            self._band_rms[name] = alpha * self._band_rms[name] + (1 - alpha) * raw
            if stability is not None:
                raw_h = stability.harmonic_rms(spectrum, mask)
                self._band_harmonic_rms[name] = (
                    alpha * self._band_harmonic_rms[name] + (1 - alpha) * raw_h)

        # Spectral centroid (A-weighted for perceptual accuracy)
        weighted_spec = spectrum * A_WEIGHTS
        mag_sum = weighted_spec.sum()
        if mag_sum > 1e-10:
            raw_centroid = float(np.sum(FREQS * weighted_spec) / mag_sum)
            self._prev_centroid = self._centroid
            self._centroid = (self._centroid_alpha * self._centroid
                              + (1 - self._centroid_alpha) * raw_centroid)

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

        # Percussiveness: flux / magnitude ratio
        if flux is not None and mag_sum > 1e-10:
            raw_perc = float(flux.sum() / mag_sum)
            self._percussiveness = (self._perc_alpha * self._percussiveness
                                     + (1 - self._perc_alpha) * raw_perc)

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
    def centroid(self) -> float:
        """Spectral centroid in Hz — where the energy center of mass is."""
        return self._centroid

    @property
    def centroid_delta(self) -> float:
        """Absolute change in centroid since last frame (Hz)."""
        return abs(self._centroid - self._prev_centroid)

    @property
    def centroid_rms(self) -> float:
        """RMS energy around the centroid (±1 octave)."""
        return self._centroid_rms

    @property
    def percussiveness(self) -> float:
        """Flux/magnitude ratio — high = drums/transients, low = sustained tonal."""
        return self._percussiveness

    @property
    def harmonic_rms(self) -> float:
        """Sub-bass harmonic RMS (backwards compat)."""
        return self._band_harmonic_rms.get('subbass', 0.0)

    @property
    def harmonic_centroid_rms(self) -> float:
        """RMS around centroid from stable (harmonic) bins only."""
        return self._harmonic_centroid_rms
