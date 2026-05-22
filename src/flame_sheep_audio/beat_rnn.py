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

from collections import deque
from pathlib import Path

import numpy as np

from .beat_detector import BeatDetectorBase
from ._band_config import BandConfig, default_band_config
from ._spectrum import SpectrumFrame
from ._types import BeatEvent
from ._constants import FREQS


# Hyperparameters of the trained model (must match build_beat_crnn args
# used during training; see tools/train_beat_rnn_continuous.py).
_INPUT_SIZE = 216
_PROJ_SIZE = 32
_HIDDEN_SIZE = 48
_N_CLASSES = 1


def _unpack_weights(flat: np.ndarray) -> tuple:
    """Slice the flat weight vector into named arrays. Layout must match
    the order `build_beat_crnn` adds layers in."""
    I, P, H, C = _INPUT_SIZE, _PROJ_SIZE, _HIDDEN_SIZE, _N_CLASSES
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
                 min_peak_distance_frames: int = 9,  # ~100 ms at 93.75 fps
                 lookahead_frames: int = 9,           # ~100 ms latency
                 auto_reset_frames: int = 256,
                 band_config: BandConfig | None = None,
                 freqs: np.ndarray | None = None,
                 ) -> None:
        # Load + validate weights
        weights_path = Path(weights_path)
        data = np.load(weights_path)
        if not hasattr(data, 'files') or 'weights' not in data.files:
            raise ValueError(
                f'{weights_path}: expected .npz with a "weights" array')
        flat = np.asarray(data['weights'], dtype=np.float32)
        (self._W_in, self._b_in,
         self._W_gru, self._U_gru, self._bias_gru,
         self._W_out, self._b_out) = _unpack_weights(flat)

        self._threshold = float(threshold)
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
        self._h: np.ndarray = np.zeros(_HIDDEN_SIZE, dtype=np.float32)
        self._prev_log_mag: np.ndarray | None = None
        self._frames_since_reset = 0

        # Peak-picker rolling buffer of activations
        self._buffer: deque[float] = deque(maxlen=2 * self._lookahead + 1)
        # Parallel buffer of per-band flux for kind classification at the
        # moment of emission. Same length as activation buffer.
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

    def _classify_band(self, band_flux: dict[str, float]) -> str:
        """Pick the kind by per-band *flux* relative to that band's
        recent history — matches PercentileBeatDetector's classification
        logic so the RNN-driven kind decisions sit in the same regime as
        the existing detectors' kind decisions.

        For each band, compute how unusually high the current flux is
        compared to its recent history (median + spread). Pick the band
        with the largest positive spike. This sidesteps the spectrum-
        density bias of classifying by absolute magnitude (mid always
        wins because mid has the most absolute content in most music).
        """
        best_band = self._band_names[0]
        best_spike = -float('inf')
        for name in self._band_names:
            hist = self._band_flux_history[name]
            current = band_flux[name]
            if len(hist) < 5:
                # Warm-up: fall back to absolute energy so we don't all
                # default to the first band.
                spike = current
            else:
                arr = np.fromiter(hist, dtype=np.float32)
                ref = float(np.median(arr))
                spread = float(np.std(arr)) + 1e-9
                # Z-score-ish: how many "typical noise widths" above the
                # median is the current flux?
                spike = (current - ref) / spread
            if spike > best_spike:
                best_spike = spike
                best_band = name
        return best_band

    def detect(self, frame: SpectrumFrame) -> list[BeatEvent]:
        """Process one spectrum frame; return any beat events that
        became confirmable now (with the buffered lookahead)."""
        mag = frame.magnitude
        if mag.shape != (108,):
            raise ValueError(
                f'BeatRNNDetector expects 108-bin magnitude (CQT), '
                f'got shape {mag.shape}. Daemon must use CqtEngine.')

        # Match training-pipeline transformation:
        #   log_mag = log1p(mag * 10.0)
        #   diff = max(0, log_mag[t] - log_mag[t-1])
        log_mag = np.log1p(mag * 10.0).astype(np.float32)
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

        # Linear out + sigmoid → activation in [0, 1]
        activation = float(_sigmoid(self._h @ self._W_out[:, 0] + self._b_out[0]))

        # Update counters
        self._frame_idx += 1
        self._frames_since_reset += 1
        self._frames_since_beat += 1

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

        # Causal peak picker: push activation, check the buffer's center
        # frame (which has now seen lookahead frames on both sides).
        self._buffer.append(activation)
        self._band_flux_buffer.append(band_flux_now)
        if len(self._buffer) < self._buffer.maxlen:
            return []  # buffer warming up; no decisions yet

        # The center frame is at index `lookahead` in the buffer
        # (with maxlen = 2*lookahead+1).
        L = self._lookahead
        center = self._buffer[L]
        if center < self._threshold:
            return []
        is_peak = all(center >= self._buffer[i] for i in range(2 * L + 1)
                      if i != L)
        if not is_peak:
            return []
        if self._frames_since_beat < self._min_distance:
            return []
        self._frames_since_beat = 0
        kind = self._classify_band(self._band_flux_buffer[L])
        return [BeatEvent(kind=kind, energy=center)]
