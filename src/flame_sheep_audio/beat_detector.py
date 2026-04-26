"""Spectral flux beat detector — onset detection from FFT magnitude changes."""

import numpy as np
from collections import deque

from ._constants import N_BINS, HISTORY_LEN, FFT_SIZE, HOP_SIZE
from .config import cfg
from .tempo_scaler import TempoScaler
from ._band_config import BandConfig, default_band_config
from ._types import BeatEvent
from ._spectrum import SpectrumFrame
from ._bands import (
    AdaptiveBand, SpringBand, make_mask,
    ADAPT_ALPHA, ADAPT_FAST_ALPHA, ADAPT_INTERVAL, ADAPT_ANCHOR,
    SECTION_THRESHOLD, FAST_ADAPT_FRAMES,
)

_scaler = TempoScaler()


class FluxBeatDetector:
    """Detect beat onsets from spectral flux.

    Computes per-band mean flux (weighted or boolean-masked), compares
    to a rolling average, and fires a BeatEvent when flux exceeds the
    threshold. Supports adaptive band tracking.

    Usage:
        detector = FluxBeatDetector(adaptive=True)
        events = detector.detect(spectrum_frame)
    """

    # Non-configurable constants
    SHARPNESS_LOOKBACK = FFT_SIZE // HOP_SIZE  # span the full overlap attack ramp

    # Configurable constants read from cfg at access time
    @property
    def THRESHOLD(self): return cfg.detection.base_threshold
    @property
    def KICK_THRESHOLD(self): return cfg.detection.kick_threshold
    @property
    def COOLDOWN(self): return cfg.detection.cooldown_frames
    @property
    def KICK_COOLDOWN(self): return cfg.detection.kick_cooldown_frames
    @property
    def STABILITY_SCALING(self): return cfg.detection.stability_scaling
    @property
    def MIN_FLUX(self): return cfg.detection.min_flux
    @property
    def SHARPNESS(self): return cfg.detection.sharpness

    def __init__(self, adaptive: bool = False, sharpness: bool = True,
                 stability=None, band_config: BandConfig | None = None):
        self._adaptive = adaptive
        self._spring_bands_enabled = cfg.adaptive.enabled
        self._sharpness = sharpness
        self._stability = stability  # MagnitudeStability reference (optional)
        self._bpm = 0.0

        if band_config is None:
            band_config = default_band_config()
        self._band_config = band_config
        self._detection_names = list(band_config.detection_band_names)

        # Static band masks (built from config)
        self._bands = {b.name: make_mask(*b.freq_range)
                       for b in band_config.detection_bands}

        # Spring-model adaptive bands
        if self._spring_bands_enabled:
            self._spring_bands = {
                b.name: SpringBand(b.name, default_range=b.freq_range,
                                   allowed_range=b.allowed_range)
                for b in band_config.detection_bands
            }
            self._spring_frame = 0

        # Adaptive bands
        if adaptive:
            self._adaptive_bands = {
                b.name: AdaptiveBand(b.name, default_range=b.freq_range,
                                     allowed_range=b.allowed_range)
                for b in band_config.detection_bands
            }
            self._adapt_frame = 0
            self._adapt_alpha = ADAPT_ALPHA
            self._fast_adapt_remaining = 0

        # Per-band flux history
        self._flux_history = {
            name: deque(maxlen=HISTORY_LEN)
            for name in self._detection_names
        }

        # Per-band cooldown
        self._cooldown_frames = {name: 0 for name in self._detection_names}
        self._frame_count     = {name: 0 for name in self._detection_names}

        # Per-band recent flux (for attack sharpness lookback)
        self._recent_flux = {
            name: deque(maxlen=self.SHARPNESS_LOOKBACK + 1)
            for name in self._detection_names
        }

    @property
    def adaptive_bands(self):
        """Access adaptive band state (for tests/inspection)."""
        return self._adaptive_bands if self._adaptive else None

    def reset_bands(self):
        """Reset adaptive bands to defaults. Call on song change."""
        if self._adaptive:
            for ab in self._adaptive_bands.values():
                ab.reset()
            self._adapt_alpha = ADAPT_ALPHA
            self._fast_adapt_remaining = 0
        if self._spring_bands_enabled:
            for sb in self._spring_bands.values():
                sb.reset()

    def detect(self, frame: SpectrumFrame) -> list[BeatEvent]:
        """Detect beat onsets from a spectrum frame."""
        flux = frame.flux

        if self._adaptive:
            self._update_adaptive_bands(flux)

        if self._spring_bands_enabled:
            self._update_spring_bands(flux)

        events = []

        for band in self._detection_names:
            band_flux = self._band_flux(flux, band)
            hist = self._flux_history[band]
            recent = self._recent_flux[band]
            recent.append(band_flux)

            self._frame_count[band] += 1
            if band == 'kick' and self._bpm > 0:
                cd = _scaler.beats_to_frames(
                    self._bpm, cfg.detection.kick_cooldown_beat_fraction)
            else:
                cd = self.KICK_COOLDOWN if band == 'kick' else self.COOLDOWN
            in_cooldown = (self._frame_count[band]
                           - self._cooldown_frames[band]) < cd

            if len(hist) >= 10 and band_flux > self.MIN_FLUX and not in_cooldown:
                local_avg = float(np.mean(hist))

                # Attack sharpness gate
                if (self._sharpness and band != 'kick'
                        and len(recent) > self.SHARPNESS_LOOKBACK):
                    pre_attack = float(np.median(list(recent)[:-1]))
                    if pre_attack > self.MIN_FLUX:
                        sharpness_headroom = 1.0 / (1.0 + pre_attack * 10.0)
                        effective_sharpness = 1.0 + (self.SHARPNESS - 1.0) * sharpness_headroom
                        if band_flux / pre_attack < effective_sharpness:
                            hist.append(band_flux)
                            continue

                # Per-band threshold: base scaled by band width
                band_mask = self._bands[band]
                n_bins = int(band_mask.sum()) if hasattr(band_mask, 'sum') else np.count_nonzero(band_mask)
                thresh = self.THRESHOLD * max(1.0, np.sqrt(30.0 / max(n_bins, 1)))

                if self._stability is not None:
                    stab = self._stability.band_stability(self._bands[band])
                    headroom = 1.0 / (1.0 + local_avg * 10.0)
                    thresh *= (1.0 + stab * self.STABILITY_SCALING * headroom)
                if local_avg < self.MIN_FLUX:
                    # Near-silence: only fire if flux is substantially above
                    # noise floor, not just above MIN_FLUX
                    if band_flux > self.MIN_FLUX * 1000:
                        events.append(BeatEvent(kind=band, energy=1.0))
                        self._cooldown_frames[band] = self._frame_count[band]
                elif band_flux > local_avg * thresh:
                    normalized = min(1.0,
                        (band_flux / local_avg - thresh) / thresh)
                    events.append(BeatEvent(kind=band, energy=normalized))
                    self._cooldown_frames[band] = self._frame_count[band]

            hist.append(band_flux)

        return events

    def _band_flux(self, flux: np.ndarray, band: str) -> float:
        """Compute weighted mean flux for a band."""
        if self._adaptive:
            w = self._adaptive_bands[band].weights
            s = w.sum()
            return float(np.dot(flux, w) / s) if s > 0 else 0.0
        elif self._spring_bands_enabled and band in self._spring_bands:
            mask = self._spring_bands[band].mask
            return float(flux[mask].mean()) if mask.any() else 0.0
        else:
            mask = self._bands[band]
            return float(flux[mask].mean()) if mask.any() else 0.0

    def _update_spring_bands(self, flux: np.ndarray):
        """Update spring band positions from stability-weighted flux."""
        self._spring_frame += 1
        if self._spring_frame < cfg.adaptive.update_interval:
            return
        self._spring_frame = 0

        if self._stability is not None:
            stab = self._stability._fast.stability_per_bin()
        else:
            stab = np.zeros(N_BINS, dtype=np.float32)

        for sb in self._spring_bands.values():
            sb.update_flux_ema(flux, stab)

        # Bands ordered by default center frequency
        band_order = sorted(self._spring_bands.keys(),
                            key=lambda n: self._spring_bands[n].default_center)
        for i, name in enumerate(band_order):
            sb = self._spring_bands[name]
            neighbors = []
            if i > 0:
                neighbors.append(self._spring_bands[band_order[i-1]])
            if i < len(band_order) - 1:
                neighbors.append(self._spring_bands[band_order[i+1]])

            sb.apply_forces(
                anchor_k=cfg.adaptive.anchor_strength,
                flux_k=cfg.adaptive.flux_pull_strength,
                neighbors=neighbors,
                repulsion_k=cfg.adaptive.repulsion_strength,
            )

        for name in band_order:
            self._bands[name] = self._spring_bands[name].mask

    def _update_adaptive_bands(self, flux: np.ndarray):
        """EMA update of per-bin flux accumulators + periodic weight recompute."""
        alpha = self._adapt_alpha

        for ab in self._adaptive_bands.values():
            band_flux = flux * ab.allowed_mask
            ab.flux_accum = alpha * ab.flux_accum + (1 - alpha) * band_flux

        if self._fast_adapt_remaining > 0:
            self._fast_adapt_remaining -= 1
            if self._fast_adapt_remaining == 0:
                self._adapt_alpha = ADAPT_ALPHA

        self._adapt_frame += 1
        if self._adapt_frame < ADAPT_INTERVAL:
            return
        self._adapt_frame = 0

        for name in self._detection_names:
            ab = self._adaptive_bands[name]
            raw = ab.flux_accum * ab.allowed_mask
            peak = raw.max()
            old_weights = ab.weights.copy()

            if peak > 1e-7:
                normalized = raw / peak
                ab.weights = ((1 - ADAPT_ANCHOR) * normalized
                              + ADAPT_ANCHOR * ab.default_weights)
                s = ab.weights.sum()
                if s > 0:
                    ab.weights /= s
            else:
                ab.weights = ab.default_weights.copy()

            shift = np.sum(np.abs(ab.weights - old_weights))
            if shift > SECTION_THRESHOLD:
                self._adapt_alpha = ADAPT_FAST_ALPHA
                self._fast_adapt_remaining = FAST_ADAPT_FRAMES
