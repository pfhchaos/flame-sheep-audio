"""Shared types for audio analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._band_config import BandConfig


@dataclass
class BeatEvent:
    """A discrete audio event detected by the audio engine.

    Axes should ignore event kinds they don't recognize — new kinds
    can be added without updating every consumer.
    """
    kind: str       # detection band name (e.g. 'kick' | 'snare' | 'hihat') or 'song_start'
    energy: float   # normalized 0..1, how strong the onset was


@dataclass
class BandState:
    """Per-band audio metrics — same structure for every band."""
    rms: float = 0.0
    harmonic_rms: float = 0.0
    slow_rms: float = 0.0           # slow-attack envelope of RMS (~2s)
    slow_harmonic_rms: float = 0.0  # slow-attack envelope of harmonic RMS (~2s)
    onset_density: float = 0.0
    density_delta: float = 0.0      # first derivative of onset density


def _default_bands(band_config: BandConfig | None = None) -> dict[str, BandState]:
    if band_config is not None:
        return {name: BandState() for name in band_config.all_band_names}
    from ._band_config import default_band_config
    return {name: BandState() for name in default_band_config().all_band_names}


@dataclass
class AudioState:
    """Per-frame audio state passed to visual axes.

    Bundles discrete events with continuous features so axes get
    everything through one interface.
    """
    events: list[BeatEvent] = field(default_factory=list)

    # Per-band metrics (names determined by BandConfig)
    bands: dict[str, BandState] = field(default_factory=_default_bands)

    # Centroid (dynamic band — follows dominant frequency)
    centroid: float = 1000.0
    centroid_delta: float = 0.0
    centroid_rms: float = 0.0
    centroid_harmonic_rms: float = 0.0

    # Global
    percussiveness: float = 0.5
    section_change: float = 0.0      # centroid divergence (fast-slow EMA, normalized)
    bpm: float = 0.0
    effective_bpm: float = 120.0     # blended with default based on confidence
    tempo_confidence: float = 0.0    # ACF tempo tracker confidence (0..1)
    tempo_saturated: bool = False    # True when onset rate exceeds tracking range
    break_intensity: float = 0.0     # 0=normal, 1=deep break


@dataclass
class AudioSnapshot:
    """Atomic audio state snapshot from the audio thread.

    Returned by AudioProcessor.drain() — contains all accumulated beat events
    since the last drain, plus the latest spectrum/RMS/waveform/continuous features.
    """
    events: list[BeatEvent] = field(default_factory=list)
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    stability: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))  # per-bin 0..1, 1=harmonic

    # Per-band metrics
    bands: dict[str, BandState] = field(default_factory=_default_bands)

    # Centroid
    centroid: float = 1000.0
    centroid_delta: float = 0.0
    centroid_rms: float = 0.0
    centroid_harmonic_rms: float = 0.0

    # Global
    percussiveness: float = 0.5
    section_change: float = 0.0
    bpm: float = 0.0
    effective_bpm: float = 120.0
    tempo_confidence: float = 0.0
    tempo_saturated: bool = False
    mode: str = 'idle'               # 'idle' | 'energy' | 'beat'
    break_intensity: float = 0.0     # 0=normal, 1=deep break
