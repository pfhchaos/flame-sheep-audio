"""Audio processing constants."""

import numpy as np

SAMPLE_RATE    = 48000
DEFAULT_DEVICE = None  # auto-detect PipeWire monitor source
BLOCK_SIZE     = 1024   # frames per sounddevice callback
FFT_SIZE       = 2048   # FFT window size
HOP_SIZE       = 512    # analysis hop size (10.7ms at 48kHz, 75% overlap)
N_BINS         = FFT_SIZE // 2 + 1
HISTORY_LEN    = 43     # ~0.46s of flux history at HOP_SIZE cadence

# Frequency bin array — shared across all instances
FREQS = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
