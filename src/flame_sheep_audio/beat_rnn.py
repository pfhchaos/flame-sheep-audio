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
        # axis, mid → palette, high → zoom). Mirrors the band-mask
        # convention used by PercentileBeatDetector.
        if band_config is None:
            band_config = default_band_config()
        self._band_names = list(band_config.detection_band_names)
        if freqs is None:
            # Caller didn't pass bin frequencies — derive from CqtEngine
            # since this detector only accepts 108-bin CQT magnitudes.
            from ._cqt_engine import CqtEngine
            freqs = CqtEngine().bin_centers
        self._band_masks = {
            b.name: (freqs >= b.freq_range[0]) & (freqs < b.freq_range[1])
            for b in band_config.detection_bands
        }

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
        # Parallel buffer of per-band flux — kept for energy magnitude
        # at emit time (model activation gives kind; band flux z-score
        # gives "how strong was this hit relative to recent context",
        # which is what downstream axes were tuned against).
        self._band_flux_buffer: deque[dict] = deque(
            maxlen=2 * self._lookahead + 1)
        # Per-band rolling flux history for relative-spike classification
        # (mirrors PercentileBeatDetector's per-band history). Without
        # this, classification falls back to absolute energy which is
        # spectrum-density-dominated and degenerate (mid always wins
        # because music has densest mid-band content).
        self._band_flux_history: dict[str, deque[float]] = {
            name: deque(maxlen=43)  # ~460 ms at 93.75 fps
            for name in self._band_names
        }
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
        self._band_flux_buffer.clear()
        for hist in self._band_flux_history.values():
            hist.clear()
        self._frames_since_beat = self._min_distance

    def _classify_band(self, band_flux: dict[str, float]
                        ) -> tuple[str, float]:
        """Pick (band_name, energy) by per-band *flux* relative to that
        band's recent history — matches PercentileBeatDetector's
        classification AND its energy convention so the RNN-driven
        events sit in the same regime as the existing detectors' events.

        For each band, compute how unusually high the current flux is
        compared to its recent history (z-score = (current - median) /
        std). Pick the band with the largest positive spike. The
        winning band's z-score, mapped to [0, 1], becomes the event
        energy — strong spikes (3σ+) clamp at 1.0, matching how
        PercentileBeatDetector reported energy as
        (flux - threshold) / threshold clamped at 1.

        Why z-score rather than sigmoid output: the model activation
        answers "is this a beat" (binary, gated by threshold). The
        z-score answers "how strong is this beat compared to recent
        local context" — which is what downstream axes were tuned
        against from PercentileBeatDetector. Using activation directly
        capped energy at ~0.6-0.7 since the model rarely outputs near
        1.0, causing visibly weaker downstream responses.
        """
        best_band = self._band_names[0]
        best_score = -float('inf')
        # Energy default for the warmup case where no band has enough
        # history to compute a z-score. Beats early in a song fall back
        # to a middling value so the system doesn't pin to 0 or 1.
        energy = 0.5
        for name in self._band_names:
            hist = self._band_flux_history[name]
            current = band_flux[name]
            if len(hist) < 5:
                # Warm-up: order bands by absolute flux so we don't
                # default to the first band.
                score = current
                this_energy: float | None = None
            else:
                arr = np.fromiter(hist, dtype=np.float32)
                ref = float(np.median(arr))
                spread = float(np.std(arr)) + 1e-9
                score = (current - ref) / spread
                # 3σ caps at 1.0 — empirically that's the strong-beat
                # regime; calibrate the divisor if downstream responses
                # are still off (lower = more aggressive = stronger
                # response per beat).
                this_energy = float(min(1.0, max(0.0, score / 3.0)))
            if score > best_score:
                best_score = score
                best_band = name
                if this_energy is not None:
                    energy = this_energy
        return best_band, energy

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

        # Linear out + sigmoid for ALL three heads. Each head is an
        # independent classifier (no softmax) — col 0 = downbeat, col 1
        # = any-beat, col 2 = any-onset. The peak-picker keys on col 1
        # (any-beat) since that's the broadest "something happened on a
        # beat" signal; col 0 and col 2 are used at classification time
        # to decide the most-specific kind that fired.
        all_logits = self._h @ self._W_out + self._b_out  # (3,)
        all_acts = _sigmoid(all_logits)  # (3,)
        activation = float(all_acts[_COL_BEAT])
        downbeat_act = float(all_acts[_COL_DOWNBEAT])
        onset_act = float(all_acts[_COL_ONSET])

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

        # Per-band flux for kind classification. Use the daemon's frame
        # flux (the same signal PercentileBeatDetector keys on), masked
        # by CQT bin band.
        band_flux_now = {}
        flux = frame.flux
        for name in self._band_names:
            mask = self._band_masks[name]
            band_flux_now[name] = (float(flux[mask].mean())
                                    if mask.any() else 0.0)
            # Update history with the just-now value (used by future
            # frames' classification — past values relative to which
            # the current spike is measured).
            self._band_flux_history[name].append(band_flux_now[name])

        # Causal peak picker: push all three head activations + band flux,
        # check the buffer's center frame (which has now seen lookahead
        # frames on both sides).
        self._buffer.append(activation)
        self._downbeat_buffer.append(downbeat_act)
        self._onset_buffer.append(onset_act)
        self._band_flux_buffer.append(band_flux_now)
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

        # Energy still comes from band-flux z-score (downstream axes were
        # tuned against PercentileBeatDetector's energy convention, not
        # against sigmoid activation magnitude). Use the band that
        # corresponds to the model's kind decision.
        _, energy = self._classify_band(self._band_flux_buffer[L])
        log.debug(
            '[beat_rnn] emit kind=%s (band=%s) acts=(d=%.3f b=%.3f o=%.3f) '
            'energy=%.3f frame=%d',
            kind_role, kind_band, center_downbeat, center, center_onset,
            energy, self._frame_idx - L)
        # Emit with the band name (low/mid/high) so role_mapper's existing
        # band→role mapping handles the rest. role_mapper currently maps
        # low→downbeat, mid→backbeat, high→subdivision — which lines up
        # with our model semantics if you read "backbeat" loosely as
        # "any beat that isn't a downbeat" (downstream axes don't care
        # about the metrical-position distinction; they care which
        # visual axis to drive).
        return [BeatEvent(kind=kind_band, energy=energy)]
