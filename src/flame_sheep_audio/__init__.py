"""
Audio analysis package for flame-sheep.

Re-exports public symbols so existing imports like
`from flame_sheep.audio import AudioProcessor, BeatEvent` continue to work.
"""

from __future__ import annotations

# Constants
from ._constants import (
    SAMPLE_RATE, DEFAULT_DEVICE, FFT_SIZE, HOP_SIZE, N_BINS, HISTORY_LEN, FREQS,
)

# Types
from ._types import BeatEvent, BandState, AudioState, AudioSnapshot
from ._spectrum import SpectrumEngine, SpectrumFrame

# Band configuration
from ._band_config import BandConfig, EnergyBandDef, DetectionBandDef, default_band_config
from ._bands import make_mask, make_weights, A_WEIGHTS

# Beat detection
from .beat_detector import FluxBeatDetector

# Energy analysis
from .energy import EnergyAnalyzer

# Signal sources
from .source import PipeWireSource, FeedSource

# Mode detection
from .mode import Mode, ModeDetector

# Processor classes
from .processor import AudioProcessor, SyntheticAudioProcessor, list_monitor_devices
