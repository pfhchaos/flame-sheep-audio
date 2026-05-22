"""Spectrum engine base interface + SpectrumFrame dataclass.

Concrete engine: see _cqt_engine.CqtEngine. The FFT-based reference
implementation was removed when CQT was committed to as the
production engine — its tests and tools were updated to use CQT, or
deleted if their purpose was finding the right engine in the first
place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass


@dataclass
class SpectrumFrame:
    """Output of a spectrum engine — computed once, consumed by all analyzers."""
    magnitude: np.ndarray    # (n_bins,) float32, magnitude spectrum
    flux: np.ndarray         # (n_bins,) float32, half-wave rectified spectral flux
    waveform: np.ndarray     # (HOP_SIZE,) float32, raw PCM window
    phase: np.ndarray | None = None  # (n_bins,) float32, spectral phase (optional, CQT only)
    onset_strength: float = 0.0  # A-weighted flux sum — scalar for tempo tracking
    zcr: float = 0.0             # zero-crossing rate (crossings per sample, speech/music discriminator)


class SpectrumEngineBase(ABC):
    """Interface for spectrum analysis engines.

    Implementations must:
      - Set n_bins (int) and bin_centers (float32 array) at construction
      - Return SpectrumFrames with arrays of size n_bins
      - Handle silence without crashing or producing NaN
      - Support reset() for state clearing
    """
    n_bins: int
    bin_centers: np.ndarray

    @abstractmethod
    def push_hop(self, hop: np.ndarray) -> SpectrumFrame:
        """Process one hop (HOP_SIZE samples) and return a SpectrumFrame."""
        ...

    @abstractmethod
    def compute(self, pcm: np.ndarray) -> SpectrumFrame:
        """Process a full PCM buffer, return the final SpectrumFrame."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state."""
        ...
