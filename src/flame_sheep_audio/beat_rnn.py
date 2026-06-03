"""Beat-RNN streaming detector — wraps a trained continuous-activation
beat-RNN for live use inside the daemon's BeatDetectorBase interface.

Architecture (matches `build_beat_crnn` in `wallpaper_ml`):
    Linear(216 → 32) + ReLU
    GRU(32 → 48)        — hidden state persists across frames
    Linear(48 → 1) + sigmoid
    → causal peak picker → BeatEvent

Input features: per-frame concatenation of [log_magnitude(108),
half-wave-rectified-diff-of-log-magnitude(108)], matching the training
pipeline in `tools/generate_beat_labels.py`. The training pipeline used
librosa CQT batched over each song; here the daemon's CQT engine feeds
us per-frame magnitudes and we transform on the fly.

Hidden state: training reset every 256 frames at chunk boundaries
(~2.73s). Live inference has no such reset by default. The
`auto_reset_frames` parameter lets the deployment match the training
regime; v1 default IS the training regime (reset every 256 frames),
matching the input distribution the model actually trained on. Future
calibration may relax this once the long-context behavior is
empirically validated (see `docs/beat_rnn_deploy_plan.md` Stage 1
acceptance).

Causal peak picker: maintains a `lookahead_frames`-sized rolling buffer
of activations. Emits a beat when the activation at position
`buffer[-lookahead_frames-1]` is a local maximum within ±lookahead and
above threshold. Latency = `lookahead_frames * HOP/SR` ≈ 100ms at
default settings.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from .beat_detector import BeatDetectorBase
from ._band_config import BandConfig, default_band_config
from ._spectrum import SpectrumFrame
from ._types import BeatEvent
from ._constants import FREQS


# Hyperparameters of the trained model (must match build_beat_crnn args
# used during training; see tools/train_beat_rnn_continuous.py).
#
# v2 architecture (multi-head):
#   - hidden_size 48 → 96: gives the GRU's shared representation room for
#     per-head feature subspaces (downbeat / beat / onset).
#   - proj_size 32 → 64: widens the input bottleneck so each head sees
#     more raw spectrum signal.
#   - n_classes 1 → 3: independent sigmoid heads with per-channel BCE
#     loss. Hierarchical labels — col 0 fires on downbeats, col 1 on
#     ALL beats (incl downbeats), col 2 on ALL onsets (incl beats). The
#     runtime classifier picks the most specific class that fired,
#     replacing the previous spectral-flux-based kind heuristic.
_INPUT_SIZE = 216  # fixed by the CQT engine (108 mag + 108 diff)

# Default architecture for checkpoints that don't carry their own dims
# (legacy pre-2026-05-29 files). New trainer output writes explicit
# input_size/proj_size/hidden_size/n_classes into the .npz; the runtime
# reads those when present and only falls back to these defaults for
# untagged legacy files.
_DEFAULT_PROJ_SIZE = 64
_DEFAULT_HIDDEN_SIZE = 48
_DEFAULT_N_CLASSES = 3

# Lookup for legacy files (no explicit dim metadata in the .npz).
# Maps flat-weight-array length → (proj_size, hidden_size, n_classes).
# Add entries here when adopting older checkpoints that need
# discovery support without a re-save.
_LEGACY_ARCH_BY_FLAT_LEN: dict[int, tuple[int, int, int]] = {
    18801: (32, 48, 1),  # v1 single-channel (beat_rnn_continuous.npz)
    30451: (64, 48, 3),  # v3 hierarchical 3-head (beat_rnn_3head.npz)
}

# Architecture markers stored in the .npz `architecture` field.
# Single = legacy + v3 (1 GRU + N-head linear); multidepth = stacked
# GRU, one head per layer (build_beat_crnn_multidepth in wallpaper_ml).
_ARCH_SINGLE = 'single'
_ARCH_MULTIDEPTH = 'multidepth'

# Output-channel semantics — match generate_beat_labels.py's labels_hier.
_COL_DOWNBEAT = 0
_COL_BEAT = 1
_COL_ONSET = 2

# CQT magnitude scale correction — band-aid for prtcqt/librosa.cqt
# normalization mismatch. The beat-RNN was trained on librosa.cqt
# magnitudes (range ~[0, 14], processed as log1p(mag * 10)). The
# daemon's CqtEngine wraps prtcqt and produces magnitudes ~77×
# smaller (range ~[0, 0.18]) for the same PCM. Without correction
# the model sees log-magnitude values ~5× compressed and operates
# well below its trained activation range.
#
# 770 = (daemon→librosa scale factor: 1/0.013 = 77) × (training-
# pipeline mag*10 multiplier). See tools/diagnose_cqt_skew.py for
# the empirical measurement that produced these constants.
#
# This is a band-aid until the next retrain uses daemon CQT directly
# (planned per docs/beat_rnn_iteration_plan.md — eliminates the
# train/serve representation skew entirely).
_CQT_SCALE_TO_TRAINING: float = 770.0


def _discover_arch(data, flat_len: int) -> tuple[int, int, int, int]:
    """Return (input_size, proj_size, hidden_size, n_classes) for the
    weights in `data` (an np.lib.npyio.NpzFile).

    Resolution order:
      1. Explicit fields in the .npz (preferred — trainer writes these
         as of 2026-05-29; everything new should carry them)
      2. Legacy lookup by `flat_len` for known pre-metadata checkpoints
      3. Raise — file is unknown and can't be loaded blind

    Why discover rather than hardcode constants: the runtime
    _DEFAULT_* numbers won't match every checkpoint, and there's no
    safe default for the dim that varies most (hidden_size — different
    runs train at 48 vs 96). Reading from the file makes the runtime
    schema-agnostic and shippable across architecture changes.
    """
    I = int(data['input_size']) if 'input_size' in data.files else _INPUT_SIZE
    explicit = {}
    for name, default in (('proj_size', _DEFAULT_PROJ_SIZE),
                          ('hidden_size', _DEFAULT_HIDDEN_SIZE),
                          ('n_classes', _DEFAULT_N_CLASSES)):
        if name in data.files:
            explicit[name] = int(data[name])
    if 'proj_size' in explicit and 'hidden_size' in explicit and 'n_classes' in explicit:
        return I, explicit['proj_size'], explicit['hidden_size'], explicit['n_classes']

    # Fall back to known-sizes lookup for legacy files
    if flat_len in _LEGACY_ARCH_BY_FLAT_LEN:
        P, H, C = _LEGACY_ARCH_BY_FLAT_LEN[flat_len]
        return I, P, H, C

    raise ValueError(
        f'Cannot determine beat-RNN architecture: weights file has '
        f'{flat_len}-element flat array, no explicit dim fields '
        f'(input_size/proj_size/hidden_size/n_classes), and length '
        f'does not match a known legacy configuration '
        f'{sorted(_LEGACY_ARCH_BY_FLAT_LEN.keys())}. Retrain (which '
        f'writes dims automatically) or add an entry to '
        f'_LEGACY_ARCH_BY_FLAT_LEN.'
    )


def _unpack_weights(flat: np.ndarray,
                    I: int, P: int, H: int, C: int) -> tuple:
    """Slice the flat weight vector into named arrays. Layout must match
    the order `build_beat_crnn` adds layers in. Dims (I, P, H, C) come
    from _discover_arch — see that function for resolution rules."""
    offset = 0
    W_in = flat[offset:offset + I * P].reshape(I, P); offset += I * P
    b_in = flat[offset:offset + P]; offset += P
    W_gru = flat[offset:offset + 3 * P * H].reshape(3, P, H); offset += 3 * P * H
    U_gru = flat[offset:offset + 3 * H * H].reshape(3, H, H); offset += 3 * H * H
    bias_gru = flat[offset:offset + 6 * H].reshape(6, H); offset += 6 * H
    W_out = flat[offset:offset + H * C].reshape(H, C); offset += H * C
    b_out = flat[offset:offset + C]; offset += C
    if offset != len(flat):
        raise ValueError(
            f'Weight length mismatch: parsed {offset}, file has {len(flat)}. '
            f'Expected layout for input={I}, proj={P}, hidden={H}, classes={C}.')
    return W_in, b_in, W_gru, U_gru, bias_gru, W_out, b_out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BeatRNNDetector(BeatDetectorBase):
    """Streaming beat detector backed by a trained continuous-activation
    RNN. CPU-only inference; the model is small enough (~19k params)
    that per-frame forward is sub-millisecond.

    Emits BeatEvents with kind='rnn' so consumers can distinguish from
    the flux/percentile detectors during side-by-side telemetry.
    """

    def __init__(self,
                 weights_path: str | Path,
                 threshold: float = 0.3,
                 downbeat_threshold: float | None = None,
                 beat_threshold: float | None = None,
                 onset_threshold: float | None = None,
                 min_peak_distance_frames: int = 9,  # ~100 ms at 93.75 fps
                 lookahead_frames: int = 9,           # ~100 ms latency
                 auto_reset_frames: int = 256,
                 band_config: BandConfig | None = None,
                 freqs: np.ndarray | None = None,
                 ) -> None:
        # Load + validate weights. Architecture dims discovered from
        # the file (explicit metadata preferred, legacy lookup fallback).
        weights_path = Path(weights_path)
        data = np.load(weights_path)
        if not hasattr(data, 'files') or 'weights' not in data.files:
            raise ValueError(
                f'{weights_path}: expected .npz with a "weights" array')
        flat = np.asarray(data['weights'], dtype=np.float32)
        I, P, H, C = _discover_arch(data, len(flat))
        self._input_size = I
        self._proj_size = P
        self._hidden_size = H
        self._n_classes = C
        (self._W_in, self._b_in,
         self._W_gru, self._U_gru, self._bias_gru,
         self._W_out, self._b_out) = _unpack_weights(flat, I, P, H, C)

        self._threshold = float(threshold)
        # Per-head thresholds default to the shared `threshold` for
        # back-compat. Override individually if one head is over/under
        # firing relative to ground truth — e.g. the downbeat head may
        # need a higher bar since its positive class is much rarer than
        # general beats.
        self._th_downbeat = float(downbeat_threshold if downbeat_threshold is not None else threshold)
        self._th_beat = float(beat_threshold if beat_threshold is not None else threshold)
        self._th_onset = float(onset_threshold if onset_threshold is not None else threshold)
        self._min_distance = int(min_peak_distance_frames)
        self._lookahead = int(lookahead_frames)
        self._auto_reset = int(auto_reset_frames)

        # Band masks for kind classification. The RNN gives us *when* a
        # beat happened; the per-band energy heuristic gives us *which*
        # of the three visual-axis tiers it should drive (low → genome
        # band_config / freqs accepted for API compatibility with the
        # other detectors, but the RNN doesn't use per-band features
        # for classification — the model heads do that — and emits all
        # events at fixed energy=1.0 (see emit site). Kept in the
        # signature in case future variants want per-band fallback.
        _ = (band_config, freqs)

        # Per-frame state
        self._h: np.ndarray = np.zeros(self._hidden_size, dtype=np.float32)
        self._prev_log_mag: np.ndarray | None = None
        self._frames_since_reset = 0
        # Diagnostic: track recent activations so we can periodically
        # log what the RNN is actually producing, even when no beats
        # are firing (zero events otherwise gives zero info on whether
        # the threshold is too high, the model is dead, etc.).
        self._diag_recent_activations: deque[float] = deque(maxlen=200)
        self._diag_log_interval = 200  # frames (~2 s at 93.75 fps)
        self._frames_since_diag_log = 0
        self._emissions_since_diag_log = 0

        # Peak-picker rolling buffer of activations (col 1, any-beat)
        self._buffer: deque[float] = deque(maxlen=2 * self._lookahead + 1)
        # Parallel buffers for the other two heads (downbeat / onset) so
        # the center-frame classifier can read them at emit time with the
        # same lookahead alignment as the peak-picked activation.
        self._downbeat_buffer: deque[float] = deque(
            maxlen=2 * self._lookahead + 1)
        self._onset_buffer: deque[float] = deque(
            maxlen=2 * self._lookahead + 1)
        # Position offset for emitted beats — running frame counter
        self._frame_idx = 0
        # Frames since last emitted beat (refractory enforcement)
        self._frames_since_beat = self._min_distance  # allow first beat

    # ---- BeatDetectorBase ----

    def reset_bands(self) -> None:
        """Reset all internal state. Called on song change."""
        self._h.fill(0.0)
        self._prev_log_mag = None
        self._frames_since_reset = 0
        self._buffer.clear()
        self._downbeat_buffer.clear()
        self._onset_buffer.clear()
        self._frames_since_beat = self._min_distance

    def _forward_step(self, x: np.ndarray) -> tuple[float, float, float]:
        """Single-frame forward → (downbeat_act, beat_act, onset_act).

        Single-arch implementation: input_linear → one GRU → one
        multi-head linear; columns laid out as [downbeat, beat, onset]
        per _COL_DOWNBEAT/_COL_BEAT/_COL_ONSET.

        Subclasses (e.g. MultiDepthBeatRNNDetector) override this with
        their own forward graph. The contract — returning the three
        head sigmoids in (downbeat, beat, onset) order — is what
        detect()'s peak-picker logic depends on.
        """
        # Linear → ReLU
        proj = x @ self._W_in + self._b_in
        np.maximum(proj, 0.0, out=proj)

        # GRU step (PyTorch convention: split input/hidden biases)
        wx_z = proj @ self._W_gru[0]
        wx_r = proj @ self._W_gru[1]
        wx_h = proj @ self._W_gru[2]
        uh_z = self._h @ self._U_gru[0]
        uh_r = self._h @ self._U_gru[1]
        uh_n = self._h @ self._U_gru[2]
        z = _sigmoid(wx_z + uh_z + self._bias_gru[0] + self._bias_gru[3])
        r = _sigmoid(wx_r + uh_r + self._bias_gru[1] + self._bias_gru[4])
        h_hat = np.tanh(wx_h + self._bias_gru[2] + r * (uh_n + self._bias_gru[5]))
        self._h = (1.0 - z) * h_hat + z * self._h

        # Linear out + sigmoid. Each output is an independent
        # classifier (no softmax). Layout:
        #   n_classes=3 (v3+): col 0 = downbeat, col 1 = any-beat,
        #                       col 2 = any-onset.
        #   n_classes=1 (v1):  col 0 = beat (legacy single-channel)
        all_logits = self._h @ self._W_out + self._b_out
        all_acts = _sigmoid(all_logits)
        if self._n_classes >= 3:
            return (float(all_acts[_COL_DOWNBEAT]),
                    float(all_acts[_COL_BEAT]),
                    float(all_acts[_COL_ONSET]))
        # v1 single-channel — treat the one output as "beat". Downbeat
        # and onset get 0 so the inference cascade falls back to
        # treating every fired event as a generic beat. Lower fidelity
        # than v3 but doesn't crash on a legacy weights file.
        beat = float(all_acts[0])
        return (0.0, beat, 0.0)

    def detect(self, frame: SpectrumFrame) -> list[BeatEvent]:
        """Process one spectrum frame; return any beat events that
        became confirmable now (with the buffered lookahead)."""
        mag = frame.magnitude
        if mag.shape != (108,):
            raise ValueError(
                f'BeatRNNDetector expects 108-bin magnitude (CQT), '
                f'got shape {mag.shape}. Daemon must use CqtEngine.')

        # Match training-pipeline transformation, with scale correction
        # for the librosa-vs-prtcqt magnitude convention mismatch:
        #   log_mag = log1p(mag * _CQT_SCALE_TO_TRAINING)
        #   diff = max(0, log_mag[t] - log_mag[t-1])
        log_mag = np.log1p(mag * _CQT_SCALE_TO_TRAINING).astype(np.float32)
        if self._prev_log_mag is None:
            diff = np.zeros_like(log_mag)
        else:
            diff = np.maximum(0.0, log_mag - self._prev_log_mag).astype(np.float32)
        self._prev_log_mag = log_mag

        x = np.concatenate([log_mag, diff])  # (216,)

        downbeat_act, activation, onset_act = self._forward_step(x)

        # Update counters
        self._frame_idx += 1
        self._frames_since_reset += 1
        self._frames_since_beat += 1
        self._frames_since_diag_log += 1
        self._diag_recent_activations.append(activation)

        # Periodic diagnostic: report activation distribution + recent
        # emission count. Lets you tell quickly whether 0 events means
        # "RNN dead" vs "RNN firing low" vs "RNN firing but peak-picker
        # rejecting" without instrumenting per-frame.
        if self._frames_since_diag_log >= self._diag_log_interval:
            if self._diag_recent_activations:
                acts = np.fromiter(self._diag_recent_activations,
                                    dtype=np.float32)
                log.debug(
                    '[beat_rnn diag] last %d frames: '
                    'activation max=%.3f mean=%.3f p90=%.3f  '
                    'threshold=%.2f  emissions=%d',
                    len(acts), float(acts.max()), float(acts.mean()),
                    float(np.percentile(acts, 90)),
                    self._threshold, self._emissions_since_diag_log)
            self._frames_since_diag_log = 0
            self._emissions_since_diag_log = 0

        # Periodic hidden-state reset (matches training regime by default)
        if self._auto_reset > 0 and self._frames_since_reset >= self._auto_reset:
            self._h.fill(0.0)
            self._frames_since_reset = 0

        # Causal peak picker: push all three head activations, check the
        # buffer's center frame (which has now seen lookahead frames on
        # both sides).
        self._buffer.append(activation)
        self._downbeat_buffer.append(downbeat_act)
        self._onset_buffer.append(onset_act)
        if len(self._buffer) < self._buffer.maxlen:
            return []  # buffer warming up; no decisions yet

        # The center frame is at index `lookahead` in the buffer
        # (with maxlen = 2*lookahead+1).
        L = self._lookahead
        center = self._buffer[L]
        center_downbeat = self._downbeat_buffer[L]
        center_onset = self._onset_buffer[L]

        # Pick the kind from the threshold hierarchy. Most specific wins:
        # downbeat > beat > onset > nothing. We peak-pick on whichever
        # head fires so onset-only events (palette triggers) aren't lost
        # just because the beat head was quiet.
        if center_downbeat >= self._th_downbeat:
            kind_role = 'downbeat'
            kind_band = 'low'
            peak_signal = center_downbeat
            head_buffer = self._downbeat_buffer
        elif center >= self._th_beat:
            kind_role = 'beat'
            kind_band = 'mid'
            peak_signal = center
            head_buffer = self._buffer
        elif center_onset >= self._th_onset:
            kind_role = 'onset'
            kind_band = 'high'
            peak_signal = center_onset
            head_buffer = self._onset_buffer
        else:
            return []

        # Peak-pick on the head that triggered (not always the beat head).
        is_peak = all(peak_signal >= head_buffer[i] for i in range(2 * L + 1)
                      if i != L)
        if not is_peak:
            return []
        if self._frames_since_beat < self._min_distance:
            return []
        self._frames_since_beat = 0
        self._emissions_since_diag_log += 1

        # Binary event energy (2026-06-02). Per-band flux z-score was
        # the original strength signal; in practice it 41%-saturated at
        # 1.0 anyway and was originally added to mask overfiring. With
        # tuned per-head thresholds the detector now only fires on
        # confident events, and continuous signals (RMS, iteration
        # scaling, brightness) carry the "loud vs quiet right now"
        # information frame-by-frame. Discrete events stay simple:
        # they happened, full strength.
        energy = 1.0
        log.debug(
            '[beat_rnn] emit kind=%s (band=%s) acts=(d=%.3f b=%.3f o=%.3f) '
            'frame=%d',
            kind_role, kind_band, center_downbeat, center, center_onset,
            self._frame_idx - L)
        # Emit with the band name (low/mid/high) so role_mapper's existing
        # band→role mapping handles the rest. role_mapper currently maps
        # low→downbeat, mid→backbeat, high→subdivision — which lines up
        # with our model semantics if you read "backbeat" loosely as
        # "any beat that isn't a downbeat" (downstream axes don't care
        # about the metrical-position distinction; they care which
        # visual axis to drive).
        return [BeatEvent(kind=kind_band, energy=energy)]


# ---------------------------------------------------------------------------
# Multi-depth variant — stacked GRUs, one head per layer
# ---------------------------------------------------------------------------

# Multi-depth → (downbeat, beat, onset) head-index mapping. Must mirror
# train_beat_rnn_continuous.MULTIDEPTH_HEAD_TO_LABEL_COL inverted:
#   head 0 (shallow) = onset    (col 2 in labels_hier)
#   head 1 (mid)     = beat     (col 1)
#   head 2 (deep)    = downbeat (col 0)
# Stored here so a future retrain that changes the mapping has one
# obvious place to update.
_MULTIDEPTH_HEAD_FOR_DOWNBEAT = 2
_MULTIDEPTH_HEAD_FOR_BEAT = 1
_MULTIDEPTH_HEAD_FOR_ONSET = 0


def _multidepth_param_count(I: int, P: int, H: int, L: int,
                            n_heads: int | None = None) -> int:
    """Total parameters in a multi-depth checkpoint of the given shape.

    Used by the file-length-based legacy detector when an architecture
    field is missing but the dim fields are present (we can still
    confirm the layout matches before committing to multidepth unpack).
    """
    if n_heads is None:
        n_heads = L
    # Input linear (I → P) + bias
    n = I * P + P
    # Stacked GRUs — layer 0 takes P, rest take H. Per layer:
    #   W (3 * in * H) + U (3 * H * H) + bias (6 * H)
    for k in range(L):
        in_k = P if k == 0 else H
        n += 3 * in_k * H + 3 * H * H + 6 * H
    # Heads: each (H → 1) + bias
    n += n_heads * (H + 1)
    return n


def _discover_arch_multidepth(data, flat_len: int
                              ) -> tuple[int, int, int, int, int]:
    """Return (input_size, proj_size, hidden_size, n_gru_layers, n_heads)
    for a multi-depth checkpoint.

    Requires explicit dim fields in the .npz — there's no legacy table
    for multi-depth since the architecture is new (2026-05-29). If you
    add a stacked-GRU checkpoint without the metadata, retrain or
    manually re-save with the dims.
    """
    required = ('input_size', 'proj_size', 'hidden_size', 'n_gru_layers')
    missing = [f for f in required if f not in data.files]
    if missing:
        raise ValueError(
            f'Multi-depth checkpoint missing required dim field(s): '
            f'{missing}. Multi-depth has no legacy-lookup fallback — '
            f'either retrain (writes dims automatically) or re-save '
            f'with the fields populated.')
    I = int(data['input_size'])
    P = int(data['proj_size'])
    H = int(data['hidden_size'])
    L = int(data['n_gru_layers'])
    # n_heads defaults to L (one head per layer, the canonical design).
    # Stored explicitly only if a future variant deviates.
    n_heads = int(data['n_heads']) if 'n_heads' in data.files else L
    expected = _multidepth_param_count(I, P, H, L, n_heads)
    if expected != flat_len:
        raise ValueError(
            f'Multi-depth weight length mismatch: file has {flat_len} '
            f'floats but layout (input={I}, proj={P}, hidden={H}, '
            f'layers={L}, heads={n_heads}) expects {expected}. '
            f'File metadata and weights are inconsistent.')
    return I, P, H, L, n_heads


class MultiDepthBeatRNNDetector(BeatRNNDetector):
    """Streaming beat detector backed by the multi-depth stacked-GRU
    model from build_beat_crnn_multidepth.

    Same BeatDetectorBase API as BeatRNNDetector — only the forward
    computation differs (stacked GRUs, one head per depth, head-index
    → (downbeat/beat/onset) routing per the training convention).

    Loads exclusively from .npz files with architecture='multidepth'
    and the n_gru_layers field set. The base BeatRNNDetector handles
    single-arch loads; the factory load_beat_rnn() picks between the
    two.
    """

    def __init__(self,
                 weights_path: str | Path,
                 threshold: float = 0.3,
                 downbeat_threshold: float | None = None,
                 beat_threshold: float | None = None,
                 onset_threshold: float | None = None,
                 min_peak_distance_frames: int = 9,
                 lookahead_frames: int = 9,
                 auto_reset_frames: int = 256,
                 band_config: BandConfig | None = None,
                 freqs: np.ndarray | None = None,
                 ) -> None:
        # Lazy import: keeps the base detector module free of an
        # eval-tree dependency; only multidepth requires the shared
        # numpy forward primitives.
        from flame_sheep.eval.rnn_forward import unpack_weights_multidepth

        weights_path = Path(weights_path)
        data = np.load(weights_path)
        if not hasattr(data, 'files') or 'weights' not in data.files:
            raise ValueError(
                f'{weights_path}: expected .npz with a "weights" array')
        flat = np.asarray(data['weights'], dtype=np.float32)
        I, P, H, L, n_heads = _discover_arch_multidepth(data, len(flat))
        # Store dims first so BeatRNNDetector's downstream init code
        # (peak picker buffers, band-flux history, etc.) sees a
        # consistent hidden_size.
        self._input_size = I
        self._proj_size = P
        self._hidden_size = H
        self._n_gru_layers = L
        self._n_heads = n_heads
        # n_classes preserved for diagnostic prints + tests that read it.
        self._n_classes = n_heads

        # Multi-depth weight unpack — list of GRU tuples, list of head tuples.
        (self._W_in, self._b_in,
         self._grus, self._heads) = unpack_weights_multidepth(flat, I, P, H, L,
                                                               n_heads=n_heads)

        # Per-layer hidden states — one per stack layer. Replaces the
        # base class's single self._h. Initialized below; reset_bands
        # zeroes them like the single-arch path.
        self._hiddens: list[np.ndarray] = [
            np.zeros(H, dtype=np.float32) for _ in range(L)]

        # Bypass the base class's __init__ weight-loading path and run
        # the rest of its setup (band masks, peak picker buffers, diag
        # counters). Factoring this out would have meant a larger
        # refactor of BeatRNNDetector; instead we replicate the
        # init body inline. Anything BeatRNNDetector.__init__ does
        # after weight-loading needs to also happen here.
        self._threshold = float(threshold)
        self._th_downbeat = float(downbeat_threshold if downbeat_threshold is not None else threshold)
        self._th_beat = float(beat_threshold if beat_threshold is not None else threshold)
        self._th_onset = float(onset_threshold if onset_threshold is not None else threshold)
        self._min_distance = int(min_peak_distance_frames)
        self._lookahead = int(lookahead_frames)
        self._auto_reset = int(auto_reset_frames)

        # band_config / freqs accepted for API parity (see base class
        # comment). All event energy is fixed at 1.0; per-band features
        # are unused.
        _ = (band_config, freqs)

        # Per-frame state (mirrors BeatRNNDetector's init body)
        self._prev_log_mag: np.ndarray | None = None
        self._frames_since_reset = 0
        self._diag_recent_activations: deque[float] = deque(maxlen=200)
        self._diag_log_interval = 200
        self._frames_since_diag_log = 0
        self._emissions_since_diag_log = 0
        self._buffer: deque[float] = deque(maxlen=2 * self._lookahead + 1)
        self._downbeat_buffer: deque[float] = deque(
            maxlen=2 * self._lookahead + 1)
        self._onset_buffer: deque[float] = deque(
            maxlen=2 * self._lookahead + 1)
        self._frame_idx = 0
        self._frames_since_beat = self._min_distance

    def reset_bands(self) -> None:
        """Reset all state. Zeroes every layer's hidden state."""
        for h in self._hiddens:
            h.fill(0.0)
        self._prev_log_mag = None
        self._frames_since_reset = 0
        self._buffer.clear()
        self._downbeat_buffer.clear()
        self._onset_buffer.clear()
        self._frames_since_beat = self._min_distance

    def _forward_step(self, x: np.ndarray) -> tuple[float, float, float]:
        """Multi-depth forward: input projection → 3 stacked GRUs → 3 heads.

        Head-index → (downbeat/beat/onset) per the training convention
        (_MULTIDEPTH_HEAD_FOR_* constants). Returns the same triple as
        BeatRNNDetector._forward_step so detect() doesn't need to
        branch on architecture.
        """
        # Lazy import keeps the module load cheap when nobody uses
        # multidepth; the function itself is hot (called per frame).
        from flame_sheep.eval.rnn_forward import step_multidepth

        weights = (self._W_in, self._b_in, self._grus, self._heads)
        head_logits, self._hiddens = step_multidepth(x, self._hiddens, weights)
        head_acts = _sigmoid(head_logits)
        return (float(head_acts[_MULTIDEPTH_HEAD_FOR_DOWNBEAT]),
                float(head_acts[_MULTIDEPTH_HEAD_FOR_BEAT]),
                float(head_acts[_MULTIDEPTH_HEAD_FOR_ONSET]))

    # The auto-reset hook in BeatRNNDetector.detect() does self._h.fill(0.0).
    # Override the underlying field with a property that fills every layer's
    # hidden when assigned/cleared — keeps detect()'s body architecture-free.
    @property
    def _h(self) -> np.ndarray:
        """Compatibility shim: detect()'s `self._h.fill(0.0)` path needs
        a target. Returning the deepest hidden is arbitrary but stable
        (.fill is the only operation the base class invokes on it
        outside _forward_step, which we override anyway)."""
        return self._hiddens[-1]

    @_h.setter
    def _h(self, value: np.ndarray) -> None:
        # detect() never reassigns self._h directly (only the base
        # class _forward_step does), so this setter is a safety net.
        # Treat any direct assignment as "reset all hiddens".
        for h in self._hiddens:
            h[:] = 0.0


# ---------------------------------------------------------------------------
# Factory — picks the right detector class based on the checkpoint
# ---------------------------------------------------------------------------

def load_beat_rnn(weights_path: str | Path, **kwargs) -> BeatRNNDetector:
    """Construct the right BeatRNN detector for the given checkpoint.

    Reads the .npz's `architecture` field if present; defaults to
    'single' for legacy files (the v1 + v3 layout). The 'multidepth'
    branch dispatches to MultiDepthBeatRNNDetector.

    All kwargs forward to the chosen detector's __init__. Same
    signature as BeatRNNDetector(weights_path, **kwargs) for the
    common case — callers in processor.py can swap the constructor
    call for this factory without other changes.
    """
    weights_path = Path(weights_path)
    data = np.load(weights_path)
    arch = _ARCH_SINGLE
    if hasattr(data, 'files') and 'architecture' in data.files:
        # np.savez stores Python strings as 0-d unicode arrays; str()
        # round-trips them. Legacy files without the field stay 'single'.
        arch = str(data['architecture'])
    # Close the .npz handle — the detector class loads it again itself.
    # np.load's NpzFile keeps the file descriptor open; explicit close
    # avoids "too many open files" if a long-running daemon repeatedly
    # reloads weights (e.g. hot-reload path).
    if hasattr(data, 'close'):
        data.close()

    if arch == _ARCH_MULTIDEPTH:
        return MultiDepthBeatRNNDetector(weights_path, **kwargs)
    if arch == _ARCH_SINGLE:
        return BeatRNNDetector(weights_path, **kwargs)
    raise ValueError(
        f'{weights_path}: unknown architecture {arch!r}. Supported: '
        f'{_ARCH_SINGLE!r}, {_ARCH_MULTIDEPTH!r}.')
