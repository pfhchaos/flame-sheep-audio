"""CPU forward pass for the beat-RNN — pure numpy.

Used by both the offline eval CLI (tools/eval_beat_rnn_cpu.py) and the
BeatRNNDetector in the eval harness. Centralizes the layer math so they
don't drift apart and lets the detector run without a GPU context (the
training-time Vulkan path is wallpaper_ml's job).

The weight layout MUST match `build_beat_crnn` in wallpaper_ml.models:
1. VkLinear(input_size → proj_size) + ReLU
2. VkGRU(proj_size → hidden_size)
3. VkLinear(hidden_size → n_classes)

Continuous-activation reformulation uses n_classes=1 with sigmoid + BCE;
3-class softmax uses n_classes=3. Same forward path; only the head
differs in interpretation.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Preserve input dtype (float32 stays float32). The Python `1.0`
    # would otherwise upcast to float64 and break bit-exact tests
    # against torch float32 forwards.
    one = np.asarray(1.0, dtype=x.dtype)
    return one / (one + np.exp(-x))


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                   relu: bool = False) -> np.ndarray:
    y = x @ W + b
    if relu:
        np.maximum(y, 0.0, out=y)
    return y


def lstm_forward_step(x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                       W_ih: np.ndarray, W_hh: np.ndarray,
                       b_ih: np.ndarray, b_hh: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """One LSTM step in PyTorch's layout (split input/hidden biases).

    Gate order in W_ih / W_hh / biases: (input, forget, cell, output).
    W_ih: (4H, in)   W_hh: (4H, H)   b_ih, b_hh: (4H,)

    Returns (h_new, c_new).
    """
    H = h_prev.shape[0]
    gates = W_ih @ x + b_ih + W_hh @ h_prev + b_hh  # (4H,)
    i = sigmoid(gates[0:H])
    f = sigmoid(gates[H:2 * H])
    g = np.tanh(gates[2 * H:3 * H])
    o = sigmoid(gates[3 * H:4 * H])
    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def gru_forward_step(x: np.ndarray, h_prev: np.ndarray,
                     W: np.ndarray, U: np.ndarray, bias: np.ndarray,
                     hidden_size: int, input_size: int) -> np.ndarray:
    """One GRU step. PyTorch convention with split input/hidden biases.

    W: (3, input_size, hidden_size) — input projections for z, r, n
    U: (3, hidden_size, hidden_size) — hidden projections for z, r, n
    bias: (6, hidden_size) — [b_W_z, b_W_r, b_W_n, b_U_z, b_U_r, b_U_n]
    """
    wx_z, wx_r, wx_h = x @ W[0], x @ W[1], x @ W[2]
    uh_z, uh_r, uh_n = h_prev @ U[0], h_prev @ U[1], h_prev @ U[2]
    z = sigmoid(wx_z + uh_z + bias[0] + bias[3])
    r = sigmoid(wx_r + uh_r + bias[1] + bias[4])
    h_hat = np.tanh(wx_h + bias[2] + r * (uh_n + bias[5]))
    return (1.0 - z) * h_hat + z * h_prev


def unpack_weights(flat: np.ndarray, input_size: int, proj_size: int,
                   hidden_size: int, n_classes: int) -> tuple:
    """Slice the flat weight vector into the named arrays. Raises if the
    layout doesn't match (catches checkpoint/hyperparameter mismatch)."""
    offset = 0
    n_w = input_size * proj_size
    W_in = flat[offset:offset + n_w].reshape(input_size, proj_size); offset += n_w
    b_in = flat[offset:offset + proj_size]; offset += proj_size

    n_W = 3 * proj_size * hidden_size
    n_U = 3 * hidden_size * hidden_size
    n_bias = 6 * hidden_size
    W_gru = flat[offset:offset + n_W].reshape(3, proj_size, hidden_size); offset += n_W
    U_gru = flat[offset:offset + n_U].reshape(3, hidden_size, hidden_size); offset += n_U
    bias_gru = flat[offset:offset + n_bias].reshape(6, hidden_size); offset += n_bias

    n_w = hidden_size * n_classes
    W_out = flat[offset:offset + n_w].reshape(hidden_size, n_classes); offset += n_w
    b_out = flat[offset:offset + n_classes]; offset += n_classes

    if offset != len(flat):
        raise ValueError(
            f'Weight tail mismatch: parsed {offset} floats, file has '
            f'{len(flat)}. Check input_size/proj_size/hidden/n_classes.')
    return W_in, b_in, W_gru, U_gru, bias_gru, W_out, b_out


def forward_sequence(features: np.ndarray, weights: tuple,
                     hidden_size: int, proj_size: int) -> np.ndarray:
    """Run the full RNN over a (T, input_size) feature sequence.

    Returns the raw logits, shape (T, n_classes). Caller applies sigmoid
    (for n_classes=1 continuous) or softmax (for n_classes=3 classes).
    Hidden state initialized to zero — matches the training-time
    `reset_hidden()` at chunk boundaries.
    """
    W_in, b_in, W_gru, U_gru, bias_gru, W_out, b_out = weights
    T = features.shape[0]
    n_classes = b_out.shape[0]
    h = np.zeros(hidden_size, dtype=np.float32)
    logits = np.empty((T, n_classes), dtype=np.float32)
    for t in range(T):
        proj = linear_forward(features[t], W_in, b_in, relu=True)
        h = gru_forward_step(proj, h, W_gru, U_gru, bias_gru,
                              hidden_size, proj_size)
        logits[t] = linear_forward(h, W_out, b_out, relu=False)
    return logits


# ---------------------------------------------------------------------------
# Multi-depth variant
# ---------------------------------------------------------------------------
#
# Stacked-GRU layout (matches build_beat_crnn_multidepth in
# wallpaper_ml.models). save_weights() concat order:
#   [input_linear, gru_0, gru_1, ..., gru_{k-1}, head_0, ..., head_{k-1}]
# Each head is a (hidden_size, 1) linear with its own bias.
#
# Convention: head index 0 = shallowest GRU, k-1 = deepest. Each head
# reads from its own layer's output (one head per depth). The runtime
# detector maps head_idx → (downbeat / beat / onset) per the training
# convention in train_beat_rnn_continuous.MULTIDEPTH_HEAD_TO_LABEL_COL.


def unpack_weights_multidepth(flat: np.ndarray, input_size: int,
                              proj_size: int, hidden_size: int,
                              n_gru_layers: int,
                              n_heads: int | None = None) -> tuple:
    """Slice flat weights into (W_in, b_in, [gru_k …], [head_k …]) where
    each gru_k is (W, U, bias) and each head_k is (W, b).

    Raises ValueError if the parsed offset doesn't equal len(flat) — i.e.
    the caller passed wrong dims for this file. Same loud-failure
    discipline as the single-arch unpack_weights.
    """
    if n_heads is None:
        n_heads = n_gru_layers
    offset = 0

    # Input linear: (input_size → proj_size) + bias
    n_w = input_size * proj_size
    W_in = flat[offset:offset + n_w].reshape(input_size, proj_size)
    offset += n_w
    b_in = flat[offset:offset + proj_size]
    offset += proj_size

    # Stacked GRUs. Layer 0 consumes proj_size; later layers consume
    # hidden_size from the previous layer's output.
    grus = []
    for k in range(n_gru_layers):
        I_k = proj_size if k == 0 else hidden_size
        n_W = 3 * I_k * hidden_size
        n_U = 3 * hidden_size * hidden_size
        n_bias = 6 * hidden_size
        W_k = flat[offset:offset + n_W].reshape(3, I_k, hidden_size)
        offset += n_W
        U_k = flat[offset:offset + n_U].reshape(3, hidden_size, hidden_size)
        offset += n_U
        bias_k = flat[offset:offset + n_bias].reshape(6, hidden_size)
        offset += n_bias
        grus.append((W_k, U_k, bias_k))

    # Heads: each (hidden_size → 1) linear. Save order: head_0, head_1, ...
    heads = []
    for _ in range(n_heads):
        n_w = hidden_size * 1
        W_h = flat[offset:offset + n_w].reshape(hidden_size, 1)
        offset += n_w
        b_h = flat[offset:offset + 1]
        offset += 1
        heads.append((W_h, b_h))

    if offset != len(flat):
        raise ValueError(
            f'Multi-depth weight length mismatch: parsed {offset} floats, '
            f'file has {len(flat)}. Expected layout for input={input_size}, '
            f'proj={proj_size}, hidden={hidden_size}, '
            f'n_gru_layers={n_gru_layers}, n_heads={n_heads}.')
    return W_in, b_in, grus, heads


def step_multidepth(x: np.ndarray, hidden_states: list[np.ndarray],
                    weights: tuple) -> tuple[np.ndarray, list[np.ndarray]]:
    """One streaming-inference timestep through the multi-depth stack.

    x: (input_size,) — single frame's feature vector
    hidden_states: list of n_gru_layers (hidden_size,) arrays — updated
        in-place semantically but the function returns a new list so
        callers can choose whether to overwrite.
    weights: (W_in, b_in, grus, heads) from unpack_weights_multidepth.

    Returns (head_logits, new_hidden_states) where head_logits is
    shape (n_heads,) — caller applies sigmoid for the per-head
    probability.
    """
    W_in, b_in, grus, heads = weights
    proj = linear_forward(x, W_in, b_in, relu=True)

    new_hiddens: list[np.ndarray] = []
    layer_input = proj
    layer_outputs: list[np.ndarray] = []
    for (W_g, U_g, bias_g), h_prev in zip(grus, hidden_states):
        h_new = gru_forward_step(layer_input, h_prev, W_g, U_g, bias_g,
                                  hidden_size=h_prev.shape[0],
                                  input_size=layer_input.shape[0])
        new_hiddens.append(h_new)
        layer_outputs.append(h_new)
        layer_input = h_new

    # One head per layer (multi-depth design). head_k reads layer_outputs[k].
    head_logits = np.empty(len(heads), dtype=np.float32)
    for k, ((W_h, b_h), h_k) in enumerate(zip(heads, layer_outputs)):
        head_logits[k] = float((h_k @ W_h + b_h)[0])
    return head_logits, new_hiddens


def forward_sequence_multidepth(features: np.ndarray, weights: tuple,
                                hidden_size: int,
                                n_gru_layers: int) -> np.ndarray:
    """Run the multi-depth stack over (T, input_size) features.

    Returns logits shape (T, n_heads). Hidden states initialized to zero
    — matches the training-time reset_hidden() at chunk boundaries.
    """
    _, _, _, heads = weights
    n_heads = len(heads)
    T = features.shape[0]
    hiddens = [np.zeros(hidden_size, dtype=np.float32)
               for _ in range(n_gru_layers)]
    logits = np.empty((T, n_heads), dtype=np.float32)
    for t in range(T):
        logits_t, hiddens = step_multidepth(features[t], hiddens, weights)
        logits[t] = logits_t
    return logits


def peak_pick(scores: np.ndarray, threshold: float = 0.3,
              min_distance: int = 5) -> np.ndarray:
    """Find local maxima above threshold in a 1D score array.

    A frame t is a peak if scores[t] >= scores[t-1] and >= scores[t+1]
    and scores[t] >= threshold. min_distance enforces a refractory
    period — within min_distance frames of a previous peak, additional
    peaks are suppressed.

    Mirrors the peak_pick in tools/train_beat_rnn_continuous.py so the
    detector matches what validation reported during training.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) < 3:
        return np.array([], dtype=np.int64)
    interior = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    above_thresh = scores[1:-1] >= threshold
    candidates = np.where(interior & above_thresh)[0] + 1
    if min_distance > 0 and len(candidates) > 0:
        kept = [candidates[0]]
        for c in candidates[1:]:
            if c - kept[-1] >= min_distance:
                kept.append(c)
        candidates = np.array(kept, dtype=np.int64)
    return candidates


def load_weights_npz(path) -> np.ndarray:
    """Read a saved checkpoint and return the flat weights array.

    Accepts: .npz with 'weights' key (new format), .npz with raw arrays
    (training intermediate), or legacy .npy. Raises on anything else.
    """
    data = np.load(path)
    if hasattr(data, 'files'):
        if 'weights' in data.files:
            return np.asarray(data['weights'], dtype=np.float32)
        raise ValueError(f"{path}: .npz has no 'weights' key (keys: {data.files})")
    return np.asarray(data, dtype=np.float32)
