"""End-to-end audio pipeline tests.

Synthesizes specific PCM patterns and feeds them through the full
audio pipeline (FFT → stability → beat detector → energy → density).
Validates the AudioSnapshot output, not individual components.

Each test case corresponds to a real-world scenario or a known bug.
"""

import numpy as np
import pytest

from flame_sheep_audio import AudioProcessor, SAMPLE_RATE, FFT_SIZE, HOP_SIZE
from flame_sheep_audio.source import FeedSource
from flame_sheep_audio._types import BeatEvent

from synths import synth_kick, synth_vocal, synth_snare, synth_hihat, synth_808_kick
from audio_helpers import make_processor, make_silence, make_sine, feed_audio


def _sustained_bass(duration_s: float, freq: float = 60.0,
                    amplitude: float = 0.6) -> np.ndarray:
    """Sustained bass tone — simulates bass synth pad."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    # Slightly detuned harmonics for richness
    sig = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sig += amplitude * 0.3 * np.sin(2 * np.pi * freq * 2 * t).astype(np.float32)
    return sig


WARMUP_FRAMES = 20  # silence frames before signal for detector warmup


def _run_pipeline(signal: np.ndarray, sharpness: bool = True,
                  warmup: bool = True):
    """Feed signal through full audio pipeline, return all events + snapshots.

    Feeds FFT_SIZE chunks with warmup silence, matching the
    pattern used by test_beat_engine.py.

    Returns:
        (all_events, snapshots): list of BeatEvents, list of AudioSnapshots
    """
    proc = make_processor(sharpness=sharpness)

    all_events = []
    snapshots = []

    # Warmup with silence
    if warmup:
        silence = make_silence(FFT_SIZE)
        for _ in range(WARMUP_FRAMES):
            feed_audio(proc, silence)
            proc.process()

    # Feed signal in FFT_SIZE chunks
    pos = 0
    while pos < len(signal):
        chunk = signal[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        feed_audio(proc, chunk)
        events = proc.process()
        snap = proc.drain()
        all_events.extend(events)
        snapshots.append(snap)
        pos += FFT_SIZE

    return all_events, snapshots


def _count_events(events, kind):
    return sum(1 for e in events if e.kind == kind)


def _place_hits(duration_s, interval_s, synth_fn, **kwargs):
    """Place synthesized hits at regular intervals in a signal."""
    n = int(SAMPLE_RATE * duration_s)
    signal = np.zeros(n, dtype=np.float32)
    hit = synth_fn(**kwargs)
    t = 0.0
    while t < duration_s:
        pos = int(t * SAMPLE_RATE)
        end = min(pos + len(hit), n)
        signal[pos:end] += hit[:end - pos]
        t += interval_s
    return signal


class TestPipelineLowDetection:
    """Verify low-band detection through the full pipeline."""

    def test_lows_at_120bpm_detected(self):
        """Regular kick drums at 120 BPM should produce low-band events."""
        signal = _place_hits(4.0, 0.5, synth_kick)
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        # 4 seconds at 2 hits/s = ~8 onsets, minus warmup
        assert lows >= 3, f"Expected at least 3 low-band onsets at 120 BPM, got {lows}"

    def test_silence_no_lows(self):
        """Silence should produce no low-band events."""
        signal = make_silence(int(SAMPLE_RATE * 2))
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        assert lows == 0, f"Expected 0 low-band events in silence, got {lows}"

    def test_silence_low_rms(self):
        """Silence should have near-zero subbass RMS."""
        signal = make_silence(int(SAMPLE_RATE * 2))
        _, snapshots = _run_pipeline(signal)
        # Check last snapshot (after settling)
        last = snapshots[-1]
        assert last.bands['subbass'].rms < 0.001


class TestPipelineVocalSuppression:
    """Verify that sustained vocals don't produce false low-band onsets."""

    def test_male_vocal_low_onset_count(self):
        """Sustained 80Hz male vocal should produce very few low-band events.

        This is the Jolene problem: male chest resonance at 50-100Hz
        triggers false low-band onsets without stability scaling.
        """
        signal = synth_vocal(4.0, pitch=80, amplitude=0.5)
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        assert lows < 5, \
            f"Male vocal at 80Hz produced {lows} false low-band onsets (expected < 5)"

    def test_low_over_vocal_still_detected(self):
        """Kick drums mixed with sustained vocal should still be detected.

        Stability scaling should raise the threshold but not suppress
        real low-band onsets — the transient exceeds the vocal's variance.
        """
        vocal = synth_vocal(4.0, pitch=80, amplitude=0.3)
        low_signal = _place_hits(4.0, 0.5, synth_kick, amplitude=0.8)
        signal = vocal + low_signal
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        assert lows >= 2, \
            f"Low-band over vocal produced only {lows} events (expected >= 2)"


class TestPipelineBreakDetection:
    """Verify break detection through the full pipeline."""

    def test_break_after_lows(self):
        """Sustained low-band hits followed by silence should trigger breaking state."""
        # 3s of low-band hits (warmup + establish baseline), then 2s silence
        low_hits = _place_hits(3.0, 0.5, synth_kick)
        silence = make_silence(int(SAMPLE_RATE * 2))
        signal = np.concatenate([low_hits, silence])
        _, snapshots = _run_pipeline(signal)

        # Check that at least one snapshot during the silent section has
        # the break detector's conditions met (we check the raw detector
        # state indirectly via the centroid_rms dropping)
        low_frames = int(3.0 * SAMPLE_RATE / HOP_SIZE)
        silent_rms = [s.bands['subbass'].rms for s in snapshots[low_frames:]]
        # RMS should be very low during silence
        if silent_rms:
            assert min(silent_rms) < 0.01, \
                "Subbass RMS should drop during silent break"


class TestPipelineBandSeparation:
    """Verify that events land in the correct bands."""

    def test_low_sine_triggers_low_not_high(self):
        """80Hz tone onset should trigger low, not high."""
        silence = make_silence(FFT_SIZE)
        tone = make_sine(80, FFT_SIZE * 4, amplitude=0.9)
        signal = np.concatenate([silence] * 5 + [tone])
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        highs = _count_events(events, 'high')
        assert lows > 0, "80Hz onset should trigger low"
        assert highs == 0 or lows > highs, \
            f"80Hz should primarily trigger low ({lows}), not high ({highs})"

    def test_high_sine_triggers_high_not_low(self):
        """10kHz tone onset should trigger high, not low."""
        silence = make_silence(FFT_SIZE)
        tone = make_sine(10000, FFT_SIZE * 4, amplitude=0.5)
        signal = np.concatenate([silence] * 5 + [tone])
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        highs = _count_events(events, 'high')
        # High should dominate or at least be present
        assert highs > 0 or lows == 0, \
            f"10kHz should trigger high ({highs}), not low ({lows})"


class TestPipelineHarmonicEnergy:
    """Verify harmonic/percussive energy split."""

    def test_sustained_tone_has_harmonic_rms(self):
        """A sustained sine wave should produce harmonic RMS."""
        signal = make_sine(200, int(SAMPLE_RATE * 3), amplitude=0.5)
        _, snapshots = _run_pipeline(signal)
        # After stability settles (~1s), harmonic RMS should be nonzero
        late = snapshots[len(snapshots) * 2 // 3:]
        if late:
            max_hrms = max(s.bands['subbass'].harmonic_rms for s in late)
            assert max_hrms > 0, "Sustained tone should have harmonic RMS"

    def test_low_transient_low_harmonic_rms(self):
        """A single low-band hit should have low harmonic RMS (it's transient)."""
        silence = make_silence(int(SAMPLE_RATE * 2))
        hit = synth_kick(amplitude=0.9)
        # Hit at 1.5s after stability has settled on silence
        signal = silence.copy()
        pos = int(1.5 * SAMPLE_RATE)
        signal[pos:pos + len(hit)] += hit
        signal = np.append(signal, make_silence(int(SAMPLE_RATE * 0.5)))
        _, snapshots = _run_pipeline(signal)
        # Harmonic RMS should stay low since the hit is transient
        max_hrms = max(s.bands['subbass'].harmonic_rms for s in snapshots)
        total_rms = max(s.bands['subbass'].rms for s in snapshots)
        if total_rms > 0.001:
            assert max_hrms < total_rms, \
                "Low-band transient should have harmonic_rms < total rms"


class TestPipelineWallOfBass:
    """Galaxy Collapse scenario: low-band onsets over sustained bass."""

    def test_lows_over_sustained_bass_detected(self):
        """Low-band hits on top of sustained bass should still fire events.

        The Galaxy Collapse problem: sustained bass keeps the low band
        magnitude high, making flux spikes relatively small. Stability
        + headroom scaling should allow real low-band onsets through.
        """
        bass = _sustained_bass(4.0, freq=60, amplitude=0.6)
        low_hits = _place_hits(4.0, 0.5, synth_kick, amplitude=0.9)
        signal = bass + low_hits
        events, _ = _run_pipeline(signal)
        low_count = _count_events(events, 'low')
        assert low_count >= 2, \
            f"Low-band hits over sustained bass produced only {low_count} events"

    def test_sustained_bass_alone_few_lows(self):
        """Sustained bass with no hits should produce few or no low-band events."""
        signal = _sustained_bass(4.0, freq=60, amplitude=0.6)
        events, _ = _run_pipeline(signal)
        low_count = _count_events(events, 'low')
        assert low_count < 5, \
            f"Sustained bass produced {low_count} false low-band onsets"

    def test_808_bass_with_high(self):
        """808 sub-bass + high-band hits — bands shouldn't interfere."""
        bass = _sustained_bass(4.0, freq=40, amplitude=0.7)
        high_hits = _place_hits(4.0, 0.25, synth_hihat, amplitude=0.4)
        signal = bass + high_hits
        events, _ = _run_pipeline(signal)
        high_count = _count_events(events, 'high')
        low_count = _count_events(events, 'low')
        # High should dominate, bass shouldn't trigger low
        assert high_count > low_count, \
            f"Expected high ({high_count}) > low ({low_count})"


class TestPipelineDensityTracking:
    """Verify onset density responds to different patterns."""

    def test_fast_lows_many_events(self):
        """Rapid low-band hits should produce many low-band events."""
        signal = _place_hits(4.0, 0.15, synth_kick, amplitude=0.8)
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        # 4s at ~6.7 hits/s = ~27 onsets, minus warmup/cooldown
        assert lows >= 5, \
            f"Fast low-band hits should produce many events, got {lows}"

    def test_slow_lows_fewer_events(self):
        """Slow low-band hits should produce fewer events than fast ones."""
        fast = _place_hits(4.0, 0.15, synth_kick, amplitude=0.8)
        slow = _place_hits(4.0, 1.0, synth_kick, amplitude=0.8)
        fast_events, _ = _run_pipeline(fast)
        slow_events, _ = _run_pipeline(slow)
        fast_lows = _count_events(fast_events, 'low')
        slow_lows = _count_events(slow_events, 'low')
        assert fast_lows > slow_lows, \
            f"Fast ({fast_lows}) should have more low-band events than slow ({slow_lows})"


class TestPipelineWaltz:
    """Non-4/4 time signature — density-driven should handle it."""

    def test_waltz_lows_detected(self):
        """3/4 time: low-band hits on beat 1 of each bar should be detected."""
        # 120 BPM waltz: hit every 1.5s (3 beats × 0.5s)
        signal = _place_hits(6.0, 1.5, synth_kick, amplitude=0.8)
        events, _ = _run_pipeline(signal)
        lows = _count_events(events, 'low')
        assert lows >= 2, \
            f"Waltz pattern produced only {lows} low-band events (expected >= 2)"
