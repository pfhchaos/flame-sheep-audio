"""Band configuration — defines what bands the audio engine tracks.

Two band types:
  - DetectionBandDef: full onset detection, density tracking, spring adaptation
  - EnergyBandDef: RMS + harmonic RMS tracking only

The visualization declares what bands it needs at AudioProcessor construction
time. The engine doesn't care what they're for.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ._constants import SAMPLE_RATE


@dataclass(frozen=True)
class EnergyBandDef:
    """Band that only tracks RMS + harmonic RMS."""
    name: str
    freq_range: tuple[float, float]


@dataclass(frozen=True)
class DetectionBandDef:
    """Band with full onset detection + density tracking + spring adaptation."""
    name: str
    freq_range: tuple[float, float]      # default/static range
    allowed_range: tuple[float, float]   # hard clamp for spring drift


BandDef = EnergyBandDef | DetectionBandDef


@dataclass(frozen=True)
class BandConfig:
    """Complete band configuration for the audio engine."""
    energy_bands: tuple[EnergyBandDef, ...]
    detection_bands: tuple[DetectionBandDef, ...]

    @cached_property
    def all_band_names(self) -> tuple[str, ...]:
        return (tuple(b.name for b in self.energy_bands)
                + tuple(b.name for b in self.detection_bands))

    @cached_property
    def all_band_ranges(self) -> dict[str, tuple[float, float]]:
        return {b.name: b.freq_range
                for b in (*self.energy_bands, *self.detection_bands)}

    @cached_property
    def detection_band_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.detection_bands)


def default_band_config() -> BandConfig:
    """The default band layout for flame-sheep's visualization.

    3 detection bands matching the 3 visual axes:
      low  → GenomeAxis (morph/swap)
      mid  → PaletteAxis (palette walk)
      high → ZoomAxis (zoom pulse)

    high covers the old clap+hihat range (1000Hz+) since both
    fed the same axis anyway. The adaptive spring system will float
    each band to wherever the percussion actually is for a given song.
    """
    return BandConfig(
        energy_bands=(
            EnergyBandDef('subbass', (20, 200)),
        ),
        detection_bands=(
            DetectionBandDef('low',  (30, 200),   (25, 150)),
            DetectionBandDef('mid', (200, 1000),  (150, 2000)),
            DetectionBandDef('high', (1000, SAMPLE_RATE / 2),
                             (800, SAMPLE_RATE / 2)),
        ),
    )
