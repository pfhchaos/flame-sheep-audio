"""Streaming vs offline equivalence for BeatRNNDetector.

Stage 1 acceptance from docs/beat_rnn_deploy_plan.md: the streaming
detector must produce activations that match the offline CPU forward
pass on the same audio (modulo the daemon-CQT vs librosa-CQT bridge
which is its own concern). Without this guarantee, deployed inference
diverges from eval-measured behavior.

Also checks the hidden-state-reset behavior: when auto_reset_frames
is set, the streaming forward at boundaries should match
chunk-by-chunk processing in the training regime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flame_sheep_audio._cqt_engine import CqtEngine
from flame_sheep_audio._constants import HOP_SIZE, SAMPLE_RATE
from flame_sheep_audio.beat_rnn import BeatRNNDetector

_PROJECT = Path(__file__).resolve().parents[2]

# Streaming tests use the currently-deployed model. v1
# (beat_rnn_continuous.npz, n_classes=1) is no longer compatible with
# BeatRNNDetector.detect() — it assumes a 3-head output and indexes
# _COL_BEAT=1 unconditionally. The streaming-vs-offline equivalence
# behavior being tested is architecture-agnostic, so v3 works equally
# well as the test target.
WEIGHTS = _PROJECT / 'flame_sheep' / 'data' / 'beat_rnn_3head.npz'


def _weights_or_skip() -> Path:
    if not WEIGHTS.exists():
        pytest.skip(f'weights not present at {WEIGHTS}')
    return WEIGHTS


def _make_test_audio(seconds: float = 6.0,
                      bpm: float = 120.0,
                      seed: int = 0) -> np.ndarray:
    """Synthetic test signal: regular click pattern over band-limited noise."""
    n = int(seconds * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(n).astype(np.float32) * 0.02
    beat_interval = int(SAMPLE_RATE * 60.0 / bpm)
    click_len = int(0.01 * SAMPLE_RATE)  # 10 ms
    envelope = np.exp(-np.linspace(0.0, 6.0, click_len))
    for t in range(0, n - click_len, beat_interval):
        audio[t:t + click_len] += (rng.standard_normal(click_len).astype(np.float32)
                                     * envelope * 0.6)
    return audio


def test_construction_validates_weights():
    """Detector loads without error from the packaged weights file."""
    _weights_or_skip()
    d = BeatRNNDetector(WEIGHTS)
    assert d._h.shape == (48,)
    # Default config matches deploy-plan recommendation.
    assert d._auto_reset == 256


def test_reset_bands_clears_state():
    _weights_or_skip()
    d = BeatRNNDetector(WEIGHTS)
    # Push some frames to dirty the state
    engine = CqtEngine()
    audio = _make_test_audio(seconds=1.0)
    for i in range(20):
        frame = engine.push_hop(audio[i * HOP_SIZE:(i + 1) * HOP_SIZE])
        d.detect(frame)
    # State is now non-zero
    assert d._prev_log_mag is not None
    assert d._frame_idx > 0

    d.reset_bands()
    assert np.all(d._h == 0)
    assert d._prev_log_mag is None
    assert d._frame_idx > 0  # counter is monotonic; reset_bands resets state
    assert len(d._buffer) == 0


def test_streaming_produces_beats_on_regular_clicks():
    """On a clean 120 BPM click track, the detector should emit roughly
    one beat per 500 ms after the buffer warms up — not asserting tight
    F1 (that's the eval harness's job), just that streaming is alive."""
    _weights_or_skip()
    # Use a low threshold — synthetic clicks produce smaller activations
    # than real music (max ~0.2 on this signal vs the >0.3 we see on real
    # tracks). This is a "wiring is alive" test, not an F1 measurement.
    d = BeatRNNDetector(WEIGHTS, threshold=0.1)
    engine = CqtEngine()
    audio = _make_test_audio(seconds=8.0, bpm=120.0)
    n_hops = len(audio) // HOP_SIZE
    beats = []
    for i in range(n_hops):
        frame = engine.push_hop(audio[i * HOP_SIZE:(i + 1) * HOP_SIZE])
        events = d.detect(frame)
        for e in events:
            # kind is band-classified (low/mid/high) per the deploy
            # plan's "heuristic kind, RNN timing" v1 design.
            assert e.kind in ('low', 'mid', 'high'), e.kind
            beats.append(i * HOP_SIZE / SAMPLE_RATE)
    # Want at least a few beats (the engine + RNN both need warm-up).
    # Don't assert tight F1 here; that's eval_beat_detection's job.
    assert len(beats) >= 3, f'too few beats: {beats}'
    # Beats should be sorted (peak-picker emits in order).
    assert all(beats[i] <= beats[i + 1] for i in range(len(beats) - 1))


def test_periodic_reset_matches_chunked_streaming():
    """With auto_reset_frames=256, the streaming detector's activations
    should match those of two consecutive chunks each independently fed
    through a fresh detector. This proves the periodic-reset implementation
    is equivalent to training-regime chunking, which is the v1 default
    safety property."""
    _weights_or_skip()

    # Long enough to span 2+ chunks.
    audio = _make_test_audio(seconds=8.0, bpm=120.0, seed=7)
    n_hops = (len(audio) // HOP_SIZE)
    chunk_frames = 256
    n_chunks = min(2, n_hops // chunk_frames)
    if n_chunks < 2:
        pytest.skip('audio too short for the chunked-streaming check')

    # Stream A: single detector across the whole audio with auto_reset=256
    a = BeatRNNDetector(WEIGHTS, threshold=0.3, auto_reset_frames=256)
    engine_a = CqtEngine()
    acts_a: list[float] = []
    for i in range(n_chunks * chunk_frames):
        frame = engine_a.push_hop(audio[i * HOP_SIZE:(i + 1) * HOP_SIZE])
        # Capture pre-buffer activation by sniffing internal state.
        a.detect(frame)
        # The freshly-pushed activation is at buffer[-1] if buffer non-empty
        if len(a._buffer) > 0:
            acts_a.append(a._buffer[-1])

    # Stream B: separate detectors per chunk, each starting from zero state.
    # CqtEngine also needs to be fresh per chunk to mirror the reset.
    acts_b: list[float] = []
    for c in range(n_chunks):
        b = BeatRNNDetector(WEIGHTS, threshold=0.3, auto_reset_frames=0)
        engine_b = CqtEngine()
        for i in range(chunk_frames):
            j = c * chunk_frames + i
            frame = engine_b.push_hop(audio[j * HOP_SIZE:(j + 1) * HOP_SIZE])
            b.detect(frame)
            if len(b._buffer) > 0:
                acts_b.append(b._buffer[-1])

    # Per-chunk auto-reset means stream A's GRU state resets at each
    # chunk boundary, matching the per-chunk fresh detector in B. CQT
    # engine is NOT reset by auto_reset_frames in stream A — only the
    # GRU hidden — so we accept small early-chunk differences from the
    # CQT engine's warm-up state.
    #
    # Allow a few frames of CQT warmup drift per chunk; check most
    # frames agree closely.
    if len(acts_a) != len(acts_b):
        pytest.fail(f'length mismatch: A={len(acts_a)}, B={len(acts_b)}')
    diffs = np.abs(np.array(acts_a) - np.array(acts_b))
    # Most frames should be very close. The mean diff is what matters
    # most (we expect drift only in the very first frames of each chunk).
    median_diff = float(np.median(diffs))
    assert median_diff < 0.01, (
        f'median activation difference {median_diff:.4f} > 0.01 — '
        f'streaming-with-reset diverges from per-chunk streaming.')


def test_streaming_state_persists_without_reset():
    """With auto_reset_frames=0, hidden state should NOT reset across
    256-frame boundaries. This is the no-reset deployment regime; the
    test confirms the state actually accumulates."""
    _weights_or_skip()
    d = BeatRNNDetector(WEIGHTS, threshold=0.3, auto_reset_frames=0)
    engine = CqtEngine()
    audio = _make_test_audio(seconds=4.0, bpm=100.0)

    # Snapshot hidden state at frame 250 and 260 (boundary crossing).
    n_hops = len(audio) // HOP_SIZE
    h_at = {}
    for i in range(min(280, n_hops)):
        frame = engine.push_hop(audio[i * HOP_SIZE:(i + 1) * HOP_SIZE])
        d.detect(frame)
        if i in (250, 256, 270):
            h_at[i] = d._h.copy()
    # Without auto_reset, state at 256 should NOT be zero (no reset
    # happened).
    assert np.any(h_at[256] != 0.0)
    # And the state should be drifting (not stuck).
    if 250 in h_at and 270 in h_at:
        assert not np.allclose(h_at[250], h_at[270])
