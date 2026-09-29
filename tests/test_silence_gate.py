"""Regression tests for the silent-hop skip optimization in
AudioProcessor._audio_loop (committed HEAD 3fa9d30).

The threaded audio loop has a silence gate right after the AGC step:
`if not hop.any():`. Because the AGC zeros every sub-noise-floor sample,
a hop with no real audio content arrives here as all zeros. On such a
hop the loop SKIPS the entire analysis pipeline — CQT (spectrum_engine
.push_hop), CSD onset, log-mag / stability / energy, the beat detector
(feed_audio_hop / detect), and every tempo feed — and instead publishes
a zeroed, idle snapshot. On a non-silent hop it runs the full pipeline.

These tests pin BOTH directions of that gate:

  - silent hop -> expensive components NOT invoked (skip path taken)
  - loud hop   -> expensive components ARE invoked (full path taken)

The second direction guards against a gate that is too aggressive (one
that skips analysis always, not only on silence).

Driving the loop: the silence gate lives ONLY in the threaded
_audio_loop, not in the synchronous process() path the other pipeline
tests use, so these tests drive _audio_loop directly. FeedSource
.read_hop is non-blocking and returns None once the buffer is drained,
which breaks the loop — so after feeding exactly N hops, a direct
_audio_loop() call runs N iterations and then returns. No thread, fully
deterministic. Production code is not modified.

_sd_notify is patched to None so the watchdog ping in the silent path
has no systemd side effect during tests.
"""

from contextlib import ExitStack
from unittest import mock

import numpy as np

from flame_sheep_audio import HOP_SIZE, N_BINS

from audio_helpers import make_processor, make_silence, make_sine, feed_audio


def _spy_expensive(stack: ExitStack, proc):
    """Patch the pipeline's expensive entry points with call-recording
    wrappers that still delegate to the real implementation (wraps=),
    so the full path behaves normally while we observe whether it ran.

    Returns (push_hop, feed_audio_hop, detect) mocks.
    """
    stack.enter_context(
        mock.patch('flame_sheep_audio.processor._sd_notify', None))
    push_hop = stack.enter_context(mock.patch.object(
        proc._spectrum_engine, 'push_hop',
        wraps=proc._spectrum_engine.push_hop))
    feed_hop = stack.enter_context(mock.patch.object(
        proc._detector, 'feed_audio_hop',
        wraps=proc._detector.feed_audio_hop))
    detect = stack.enter_context(mock.patch.object(
        proc._detector, 'detect',
        wraps=proc._detector.detect))
    return push_hop, feed_hop, detect


def _run_loop_over_buffer(proc):
    """Drive _audio_loop synchronously over whatever has been fed.

    FeedSource.read_hop returns None when drained, which breaks the
    loop, so this processes exactly the hops currently buffered.
    """
    proc._running = True
    proc._audio_loop()


def test_silent_hop_skips_expensive_pipeline():
    """An all-zeros hop must NOT touch the expensive analysis stages."""
    proc = make_processor()

    # Seed published state as if a loud frame just ran, so we can also
    # observe the silent path zero it back to idle rather than freeze.
    proc._spectrum = np.ones(N_BINS, dtype=np.float32)

    feed_audio(proc, make_silence(HOP_SIZE))

    with ExitStack() as stack:
        push_hop, feed_hop, detect = _spy_expensive(stack, proc)
        _run_loop_over_buffer(proc)

    push_hop.assert_not_called()
    feed_hop.assert_not_called()
    detect.assert_not_called()

    # Idle publish: the stale loud spectrum is zeroed, mode stays idle,
    # no break intensity.
    assert not proc._spectrum.any(), \
        "silent hop should zero the published spectrum, not freeze it"
    assert proc._mode == 'idle'
    assert proc._break_intensity == 0.0


def test_loud_hop_runs_expensive_pipeline():
    """A real above-noise-floor hop must run the full pipeline.

    This proves the gate skips only on silence, not always — a guard
    against an over-aggressive gate that would starve the visualizer.
    """
    proc = make_processor()

    # A loud tone survives the AGC noise-floor gate, so hop.any() is
    # True and the silence gate falls through to full analysis. A few
    # hops give the streaming CQT normal input.
    feed_audio(proc, make_sine(220.0, HOP_SIZE * 4, amplitude=0.5))

    with ExitStack() as stack:
        push_hop, feed_hop, detect = _spy_expensive(stack, proc)
        _run_loop_over_buffer(proc)

    push_hop.assert_called()
    feed_hop.assert_called()
    detect.assert_called()
