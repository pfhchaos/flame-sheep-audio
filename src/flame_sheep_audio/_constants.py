"""Audio processing constants."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE: int    = 48000
DEFAULT_DEVICE: str | int | None = None  # auto-detect PipeWire monitor source
BLOCK_SIZE: int     = 1024   # frames per sounddevice callback
FFT_SIZE: int       = 2048   # FFT window size
HOP_SIZE: int       = 512    # analysis hop size (10.7ms at 48kHz, 75% overlap)
N_BINS: int         = FFT_SIZE // 2 + 1
HISTORY_LEN: int    = 43     # ~0.46s of flux history at HOP_SIZE cadence

# Frequency bin array — shared across all instances
FREQS: NDArray[np.floating] = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
