"""Pure-numpy forward pass for the BeatNet BDA model.

Designed to replace torch + madmom in the live wallpaper's runtime.
The .npz weights file ships:
  - conv1.{weight,bias}: Conv1d(1 → 2, kernel_size=10)
  - linear0.{weight,bias}: Linear(262 → 150)
  - lstm.weight_{ih,hh}_l{0,1}, lstm.bias_{ih,hh}_l{0,1}: LSTM(150, 150, 2)
  - linear.{weight,bias}: Linear(150 → 3)
  - filterbank: precomputed (705, 136) madmom log-filterbank
  - hanning_window: (1411,) hanning window for the STFT
  - _meta_*: architecture sizes for sanity-checking

Per-frame work (~1ms on CPU):
  features (272,) → conv1d → ReLU → max_pool(2) → flatten → linear0
   → 2-layer LSTM with carried (h, c) state → linear → softmax → (3,)

Verified bit-exact against the torch BDA model in
tests/eval/test_beatnet_lite.py — any divergence there means weights
or layer math diverged from torch's nn convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .rnn_forward import lstm_forward_step, sigmoid


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - x.max()
    e = np.exp(shifted)
    return e / e.sum()


class BeatNetLite:
    """Stateful BeatNet inference, pure numpy.

    Use `reset_state()` at the start of each track. `step(features)`
    consumes one (272,) feature vector and returns a (3,) probability
    vector with channels (beat, downbeat, non-beat).
    """

    def __init__(self, weights_path: Path):
        data = np.load(str(weights_path))
        # Cast everything to float32 explicitly — numpy ops will then
        # stay in float32, matching torch's default and keeping the
        # bit-exact test honest.
        def _f32(k: str) -> np.ndarray:
            return np.ascontiguousarray(data[k], dtype=np.float32)

        # Conv1d: torch layout (out_channels, in_channels=1, kernel_size).
        self._conv1_W = _f32('conv1.weight')        # (2, 1, 10)
        self._conv1_b = _f32('conv1.bias')           # (2,)
        # Linear: torch layout (out, in). We pre-transpose so the forward
        # is `x @ W.T + b` — match `linear_forward` from rnn_forward.
        self._linear0_W = _f32('linear0.weight')     # (150, 262)
        self._linear0_b = _f32('linear0.bias')       # (150,)

        # LSTM weights as-is from torch. lstm_forward_step expects
        # (4H, in) for W_ih, (4H, H) for W_hh, (4H,) for biases.
        self._lstm_W_ih = [
            _f32('lstm.weight_ih_l0'),  # (600, 150)
            _f32('lstm.weight_ih_l1'),  # (600, 150)
        ]
        self._lstm_W_hh = [
            _f32('lstm.weight_hh_l0'),
            _f32('lstm.weight_hh_l1'),
        ]
        self._lstm_b_ih = [
            _f32('lstm.bias_ih_l0'),
            _f32('lstm.bias_ih_l1'),
        ]
        self._lstm_b_hh = [
            _f32('lstm.bias_hh_l0'),
            _f32('lstm.bias_hh_l1'),
        ]

        self._linear_W = _f32('linear.weight')       # (3, 150)
        self._linear_b = _f32('linear.bias')         # (3,)

        # Streaming feature pipeline pieces.
        self.filterbank = _f32('filterbank')         # (705, 136)
        self.hanning_window = _f32('hanning_window')  # (1411,)

        # Architecture metadata sanity-check.
        meta = {k: int(data[k]) for k in data.files if k.startswith('_meta_')}
        self.dim_in = meta.get('_meta_dim_in', 272)
        self.dim_hd = meta.get('_meta_dim_hd', 150)
        self.num_layers = meta.get('_meta_num_layers', 2)
        self.n_out = meta.get('_meta_n_out', 3)
        self.sample_rate = meta.get('_meta_sample_rate', 22050)
        self.hop = meta.get('_meta_hop', 441)
        self.win = meta.get('_meta_win', 1411)

        # Sanity: shape contract.
        assert self._linear0_W.shape == (self.dim_hd, 262), (
            f'linear0 shape {self._linear0_W.shape} != expected '
            f'({self.dim_hd}, 262)')
        for i in range(self.num_layers):
            assert self._lstm_W_ih[i].shape == (4 * self.dim_hd, self.dim_hd), (
                f'lstm layer {i} W_ih shape mismatch')

        # State buffers (initialized by reset_state()).
        self.h: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.reset_state()

    def reset_state(self) -> None:
        """Reset LSTM hidden + cell state to zero. Call at track start."""
        self.h = [np.zeros(self.dim_hd, dtype=np.float32)
                  for _ in range(self.num_layers)]
        self.c = [np.zeros(self.dim_hd, dtype=np.float32)
                  for _ in range(self.num_layers)]

    def step(self, features: np.ndarray) -> np.ndarray:
        """Process one (dim_in,) feature vector. Returns (n_out,) probs.

        Channels (BeatNet convention): 0=beat, 1=downbeat, 2=non-beat.
        """
        assert features.shape == (self.dim_in,), (
            f'features shape {features.shape} != ({self.dim_in},)')
        x = features.astype(np.float32, copy=False)

        # Conv1d: input (1, dim_in), output (2, dim_in - 10 + 1 = 263).
        # No padding; stride=1.
        x_conv = np.empty((2, self.dim_in - 10 + 1), dtype=np.float32)
        for c_out in range(2):
            kernel = self._conv1_W[c_out, 0]   # (10,)
            # Manual cross-correlation: out[t] = sum_k(x[t+k] * w[k]).
            # np.correlate(x, w, mode='valid') gives exactly this with
            # output length len(x) - len(w) + 1.
            x_conv[c_out] = np.correlate(x, kernel, mode='valid') + self._conv1_b[c_out]

        # ReLU.
        np.maximum(x_conv, 0.0, out=x_conv)

        # MaxPool1d kernel_size=2 (default stride=2). Output (2, 131).
        # Pool over pairs of adjacent elements.
        L = (x_conv.shape[1] // 2) * 2   # truncate odd tail (262 here)
        x_pool = x_conv[:, :L].reshape(2, L // 2, 2).max(axis=2)   # (2, 131)

        # Flatten then linear0: (262,) → (150,).
        x_flat = x_pool.reshape(-1)
        x_lin = self._linear0_W @ x_flat + self._linear0_b

        # 2-layer LSTM with carried state.
        layer_input = x_lin
        for layer in range(self.num_layers):
            h_new, c_new = lstm_forward_step(
                layer_input, self.h[layer], self.c[layer],
                self._lstm_W_ih[layer], self._lstm_W_hh[layer],
                self._lstm_b_ih[layer], self._lstm_b_hh[layer])
            self.h[layer] = h_new
            self.c[layer] = c_new
            layer_input = h_new

        # Output linear: (150,) → (3,).
        logits = self._linear_W @ layer_input + self._linear_b
        # Softmax (numerically stable).
        return _softmax_stable(logits)
