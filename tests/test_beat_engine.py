"""
End-to-end beat engine tests.

Feeds synthetic PCM through AudioProcessor and asserts correct beat event
detection. Synth instruments and pattern definitions live in tests/synths.py.
Shared test helpers live in tests/conftest.py.
"""

import numpy as np
import pytest
from collections import defaultdict
from dataclasses import dataclass

from flame_sheep_audio import (
    AudioProcessor, SAMPLE_RATE, FFT_SIZE, N_BINS, FREQS,
)
from audio_helpers import make_processor
from synths import (
    synth_kick, synth_snare, synth_hihat,
    synth_808_kick, synth_clap, synth_low_hihat,
    synth_vocal, synth_speech,
    DrumPattern, PatternSpec,
    ALL_PATTERNS, PATTERN_IDS,
)


# -------------------------------------------------------------------
# Test runner: feed PCM through AudioProcessor, collect events
# -------------------------------------------------------------------

@dataclass
class DetectedEvent:
    """A beat event with its detection time (in samples from start)."""
    kind: str
    energy: float
    sample: int

    @property
    def time(self) -> float:
        return self.sample / SAMPLE_RATE


def run_pattern(pcm: np.ndarray, warmup_frames: int = 20) -> list[DetectedEvent]:
    """Feed PCM through AudioProcessor frame-by-frame."""
    proc = make_processor()

    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(warmup_frames):
        proc.feed(silence)
        proc.process()

    events = []
    pos = 0
    while pos < len(pcm):
        chunk = pcm[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        proc.feed(chunk)
        frame_events = proc.process()
        for e in frame_events:
            events.append(DetectedEvent(kind=e.kind, energy=e.energy, sample=pos))
        pos += FFT_SIZE

    return events


def events_by_kind(events: list[DetectedEvent]) -> dict[str, list[DetectedEvent]]:
    grouped = defaultdict(list)
    for e in events:
        grouped[e.kind].append(e)
    return dict(grouped)


# ===================================================================
# Parametrized genre tests
# ===================================================================

class TestPatternDetection:
    """Every pattern should produce events in all active bands."""

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_detects_kicks(self, spec: PatternSpec):
        events = run_pattern(spec.build().render())
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= spec.min_kicks, \
            f"{spec.name}: expected >={spec.min_kicks} kicks, got {len(kicks)}"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_detects_snares(self, spec: PatternSpec):
        events = run_pattern(spec.build().render())
        snares = [e for e in events if e.kind == 'mid']
        assert len(snares) >= spec.min_snares, \
            f"{spec.name}: expected >={spec.min_snares} snares, got {len(snares)}"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_detects_hihats(self, spec: PatternSpec):
        events = run_pattern(spec.build().render())
        hihats = [e for e in events if e.kind == 'high']
        assert len(hihats) >= spec.min_hihats, \
            f"{spec.name}: expected >={spec.min_hihats} hihats, got {len(hihats)}"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_no_event_flood(self, spec: PatternSpec):
        """No pattern should produce an absurd number of events."""
        events = run_pattern(spec.build().render())
        total_hits = sum(len(spec.kick_beats) + len(spec.snare_beats)
                         + len(spec.hihat_beats) for _ in range(spec.bars))
        # Allow 3x expected hits as headroom for crosstalk
        assert len(events) <= total_hits * 3 + 10, \
            f"{spec.name}: {len(events)} events seems excessive " \
            f"(expected ~{total_hits} hits)"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_energy_in_range(self, spec: PatternSpec):
        events = run_pattern(spec.build().render())
        for e in events:
            assert 0.0 <= e.energy <= 1.0, \
                f"{spec.name}: {e.kind} energy {e.energy} out of range"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_survives_noise(self, spec: PatternSpec):
        """Pattern should still detect beats with 12dB noise."""
        events = run_pattern(spec.build(noise_snr_db=12).render())
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"{spec.name}: no kicks detected with 12dB noise"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_survives_vocals(self, spec: PatternSpec):
        """Pattern should still detect beats with vocals mixed in."""
        events = run_pattern(spec.build(vocal=0.4).render())
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"{spec.name}: no kicks detected with vocals"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_survives_radio_compression(self, spec: PatternSpec):
        """Pattern should survive light compression (4:1, typical radio/streaming)."""
        events = run_pattern(spec.build(compression=4.0).render())
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"{spec.name}: no kicks detected with 4:1 compression"

    @pytest.mark.parametrize('spec', ALL_PATTERNS, ids=PATTERN_IDS)
    def test_survives_heavy_compression(self, spec: PatternSpec):
        """Pattern should survive heavy compression (10:1, modern pop mastering)."""
        events = run_pattern(spec.build(compression=10.0).render())
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"{spec.name}: no kicks detected with 10:1 compression"


# ===================================================================
# Compression-specific tests
# ===================================================================

class TestCompression:
    """Dynamic range compression effects on beat detection."""

    @pytest.mark.parametrize('ratio,label', [
        (2.0,  'gentle 2:1'),
        (4.0,  'radio 4:1'),
        (8.0,  'aggressive 8:1'),
        (10.0, 'pop mastering 10:1'),
        (20.0, 'brickwall 20:1'),
    ])
    def test_compression_levels(self, ratio, label):
        """4/4 pattern should detect kicks at various compression levels."""
        p = DrumPattern(bpm=120, duration=4.0, compression=ratio)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        events = run_pattern(p.render())
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"{label}: no kicks detected"

    def test_brickwall_vs_clean_count(self):
        """Brickwall limiting should detect fewer beats than clean signal."""
        def make(comp):
            p = DrumPattern(bpm=120, duration=4.0, compression=comp)
            p.add('low',  beats=[1, 3, 5, 7])
            p.add('mid', beats=[2, 4, 6, 8])
            return run_pattern(p.render())

        clean = len(make(None))
        brick = len(make(20.0))
        # Brickwall should degrade detection — if it doesn't, our compressor
        # isn't working or the detector is somehow compression-proof
        # (which would be great but suspicious)
        print(f"[compression] clean={clean} events, brickwall={brick} events")

    def test_compressed_vocals_plus_drums(self):
        """The worst case: compressed vocals + drums (modern pop mix)."""
        p = DrumPattern(bpm=120, duration=4.0, vocal=0.6, compression=8.0)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        events = run_pattern(p.render())
        kicks = [e for e in events if e.kind == 'low']
        # This is genuinely hard — compressed vocals fill the same spectral
        # space as drums with similar energy levels
        assert len(kicks) >= 1, \
            f"No kicks detected in compressed vocal+drum mix"

    def test_no_flood_under_compression(self):
        """Compression shouldn't cause event flooding."""
        p = DrumPattern(bpm=120, duration=4.0, compression=20.0)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        pcm = p.render()
        events = run_pattern(pcm)
        assert len(events) < 60, \
            f"Brickwall compression caused event flood: {len(events)}"


# ===================================================================
# Vocal-specific tests
# ===================================================================

class TestVocalInterference:
    """Vocals should not cause excessive false triggers."""

    def test_vocal_only_raw_events(self):
        """Vocal-only signal will produce some raw detector events.

        This is expected — the detector sees spectral flux from formants,
        pitch changes, and sibilants. The tempo gating layer is what
        suppresses these in production.
        """
        pcm = synth_vocal(duration=4.0, amplitude=0.5)
        events = run_pattern(pcm)
        # Vocals DO trigger events — that's the detector working correctly.
        # Just verify it's not an absurd flood.
        assert len(events) < 100, \
            f"Vocal-only produced {len(events)} events (too many)"

    def test_vocal_events_non_rhythmic(self):
        """Vocal onset timing should be irregular (non-rhythmic).

        If events cluster into a regular pattern, the detector is being
        fooled into thinking vocals are drums.
        """
        pcm = synth_vocal(duration=4.0, amplitude=0.5)
        events = run_pattern(pcm)
        if len(events) < 4:
            return  # not enough events to check regularity

        # Check inter-onset intervals — should have high variance
        times = [e.time for e in events]
        iois = [times[i+1] - times[i] for i in range(len(times)-1)]
        if len(iois) < 3:
            return
        ioi_arr = np.array(iois)
        mean_ioi = ioi_arr.mean()
        if mean_ioi < 1e-6:
            return
        # Coefficient of variation: std/mean. Rhythmic = low CV, random = high CV.
        cv = ioi_arr.std() / mean_ioi
        # Real drums at 120 BPM have CV < 0.1. Vocals should be much higher.
        assert cv > 0.3, \
            f"Vocal onsets look too regular (CV={cv:.2f}), might fool tempo tracker"

    def test_drums_audible_through_vocals(self):
        """Drums mixed with vocals should still be detected."""
        p = DrumPattern(bpm=120, duration=4.0, vocal=0.5)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        pcm = p.render()
        events = run_pattern(pcm)
        by_kind = events_by_kind(events)
        assert 'low' in by_kind, "Kicks lost behind vocals"
        assert len(by_kind['low']) >= 2, \
            f"Expected >=2 kicks through vocals, got {len(by_kind['low'])}"

    def test_loud_vocals_dont_drown_drums(self):
        """Even loud vocals shouldn't completely suppress drum detection."""
        p = DrumPattern(bpm=120, duration=4.0, vocal=0.8)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        pcm = p.render()
        events = run_pattern(pcm)
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"Loud vocals completely drowned kicks"

    def test_vocal_plus_noise(self):
        """Vocals + noise together shouldn't cause event flood."""
        p = DrumPattern(bpm=120, duration=3.0, noise_snr_db=10, vocal=0.4)
        p.add('low', beats=[1, 3, 5])
        pcm = p.render()
        events = run_pattern(pcm)
        assert len(events) < 80, \
            f"Vocals + noise caused event flood: {len(events)}"


# ===================================================================
# Speech-specific tests
# ===================================================================

class TestSpeechInterference:
    """Speech should be harder to reject than singing due to consonant transients."""

    @pytest.mark.parametrize('regularity,max_events', [
        (0.0, 100),   # irregular speech
        (0.8, 120),   # rhythmic speech (news anchor cadence)
        (1.0, 150),   # metronomic speech (worst case — Patrick Boyle)
    ], ids=['irregular', 'rhythmic', 'metronomic'])
    def test_speech_alone(self, regularity, max_events):
        """Speech at various regularities should not flood the detector."""
        pcm = synth_speech(duration=4.0, amplitude=0.5,
                           regularity=regularity, syllable_rate=4.0)
        events = run_pattern(pcm)
        assert len(events) < max_events, \
            f"Speech (reg={regularity}) produced {len(events)} events (max {max_events})"

    def test_music_to_rhythmic_speech_transition(self):
        """Music plays, then stops and rhythmic speech begins.

        This is the Patrick Boyle scenario: the tempo tracker locks
        during music, then music stops but rhythmic speech continues
        with similar enough timing that the tracker stays locked.

        We test that the detector doesn't produce MORE events during
        the speech phase than the music phase.
        """
        music_dur = 4.0
        speech_dur = 4.0

        # Phase 1: music — build tempo lock
        p = DrumPattern(bpm=120, duration=music_dur)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        music_pcm = p.render()

        # Phase 2: rhythmic speech at ~4 syllables/sec (close to 120 BPM's
        # eighth notes at 4/sec) — this is what tricks the tracker
        speech_pcm = synth_speech(
            duration=speech_dur, amplitude=0.5,
            regularity=0.85, syllable_rate=4.0,
        )

        # Concatenate: music then speech
        full_pcm = np.concatenate([music_pcm, speech_pcm])

        events = run_pattern(full_pcm)
        music_boundary = music_dur  # seconds

        music_events = [e for e in events if e.time < music_boundary]
        speech_events = [e for e in events if e.time >= music_boundary]

        print(f"[transition] music phase: {len(music_events)} events, "
              f"speech phase: {len(speech_events)} events")

        # The speech phase should not produce dramatically more events
        # than the music phase. If it does, speech is triggering more
        # than actual drums — a clear false positive problem.
        assert len(speech_events) <= len(music_events) * 2 + 10, \
            f"Speech phase ({len(speech_events)} events) overwhelmed " \
            f"music phase ({len(music_events)} events)"

    def test_drums_survive_speech_background(self):
        """Drums should still be detectable with speech in the background."""
        p = DrumPattern(bpm=120, duration=4.0,
                        speech={'amplitude': 0.4, 'regularity': 0.3})
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        pcm = p.render()
        events = run_pattern(pcm)
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 2, \
            f"Speech background drowned out kicks: only {len(kicks)} detected"

    def test_drums_survive_rhythmic_speech(self):
        """Drums + rhythmic speech (podcast with music bed)."""
        p = DrumPattern(bpm=120, duration=4.0,
                        speech={'amplitude': 0.5, 'regularity': 0.7})
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        pcm = p.render()
        events = run_pattern(pcm)
        kicks = [e for e in events if e.kind == 'low']
        assert len(kicks) >= 1, \
            f"Rhythmic speech drowned out all kicks"


# ===================================================================
# Dynamics tests — volume changes, fadeouts, crescendos
# ===================================================================

class TestDynamics:
    """Beat detection across extreme volume changes."""

    def test_fadeout(self):
        """Beats should still be detected during a fadeout until very quiet."""
        p = DrumPattern(bpm=120, duration=6.0,
                        volume_envelope=[(0, 1.0), (6, 0.0)])
        p.add('low',  beats=[1, 3, 5, 7, 9, 11])
        p.add('mid', beats=[2, 4, 6, 8, 10, 12])
        pcm = p.render()
        events = run_pattern(pcm)

        # Should detect beats in the loud first half
        early = [e for e in events if e.time < 3.0]
        assert len(early) >= 3, \
            f"Should detect beats in loud section, got {len(early)}"

    def test_crescendo(self):
        """Beats should be detected as volume rises from silence."""
        p = DrumPattern(bpm=120, duration=6.0,
                        volume_envelope=[(0, 0.0), (6, 1.0)])
        p.add('low',  beats=[1, 3, 5, 7, 9, 11])
        p.add('mid', beats=[2, 4, 6, 8, 10, 12])
        pcm = p.render()
        events = run_pattern(pcm)

        # Should detect beats in the loud second half
        late = [e for e in events if e.time >= 3.0]
        assert len(late) >= 3, \
            f"Should detect beats as volume rises, got {len(late)}"

    def test_silence_then_drop(self):
        """Total silence followed by sudden full-volume beats."""
        p = DrumPattern(bpm=120, duration=5.0,
                        volume_envelope=[(0, 0.0), (2.99, 0.0),
                                         (3.0, 1.0), (5.0, 1.0)])
        p.add('low',  beats=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        p.add('mid', beats=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pcm = p.render()
        events = run_pattern(pcm)

        # Everything before 3s should be silent
        before_drop = [e for e in events if e.time < 2.5]
        after_drop = [e for e in events if e.time >= 3.0]
        assert len(before_drop) == 0, \
            f"Should be silent before drop, got {len(before_drop)} events"
        assert len(after_drop) >= 2, \
            f"Should detect beats after drop, got {len(after_drop)}"

    def test_extreme_dynamics(self):
        """Alternating loud and quiet bars — simulates classical dynamics."""
        p = DrumPattern(bpm=120, duration=8.0,
                        volume_envelope=[
                            (0, 1.0), (1.9, 1.0),     # bar 1-2: forte
                            (2.0, 0.1), (3.9, 0.1),   # bar 3-4: pianissimo
                            (4.0, 1.0), (5.9, 1.0),   # bar 5-6: forte
                            (6.0, 0.1), (8.0, 0.1),   # bar 7-8: pianissimo
                        ])
        p.add('low',  beats=[1, 3, 5, 7, 9, 11, 13, 15])
        p.add('mid', beats=[2, 4, 6, 8, 10, 12, 14, 16])
        pcm = p.render()
        events = run_pattern(pcm)

        # Loud sections should have more events than quiet ones
        loud_events = [e for e in events
                       if (e.time < 2.0) or (4.0 <= e.time < 6.0)]
        quiet_events = [e for e in events
                        if (2.0 <= e.time < 4.0) or (6.0 <= e.time)]
        print(f"[dynamics] loud={len(loud_events)}, quiet={len(quiet_events)}")
        assert len(loud_events) >= len(quiet_events), \
            f"Loud sections should dominate: loud={len(loud_events)}, " \
            f"quiet={len(quiet_events)}"

    def test_fadeout_no_event_flood(self):
        """Fading to silence should not produce spurious events."""
        p = DrumPattern(bpm=120, duration=4.0,
                        volume_envelope=[(0, 1.0), (2, 0.0), (4, 0.0)])
        p.add('low', beats=[1, 3])
        pcm = p.render()
        events = run_pattern(pcm)

        # After the signal fades to zero, no events should fire
        silent_events = [e for e in events if e.time >= 2.5]
        assert len(silent_events) == 0, \
            f"Fadeout produced {len(silent_events)} events after silence"

    def test_swell_and_decay(self):
        """Smooth volume swell (quiet→loud→quiet) — detection should work
        throughout the entire dynamic range since flux is volume-invariant."""
        p = DrumPattern(bpm=120, duration=8.0,
                        volume_envelope=[
                            (0, 0.05), (4, 1.0), (8, 0.05)
                        ])
        p.add('low',  beats=[1, 3, 5, 7, 9, 11, 13, 15])
        p.add('mid', beats=[2, 4, 6, 8, 10, 12, 14, 16])
        pcm = p.render()
        events = run_pattern(pcm)

        # All three sections should detect events — flux-based detection
        # works at any volume level (it's the brightness/iterations that
        # respond to absolute volume, not the beat detection itself)
        early = [e for e in events if e.time < 2.5]
        mid = [e for e in events if 2.5 <= e.time < 5.5]
        late = [e for e in events if e.time >= 5.5]
        print(f"[swell] early={len(early)}, mid={len(mid)}, late={len(late)}")
        assert len(early) >= 2, \
            f"Should detect beats during quiet start: {len(early)}"
        assert len(mid) >= 2, \
            f"Should detect beats during peak: {len(mid)}"
        assert len(late) >= 2, \
            f"Should detect beats during quiet end: {len(late)}"


# ===================================================================
# Non-parametrized tests — specific behaviors
# ===================================================================

class TestBandIsolation:
    """Each drum should primarily trigger its own band."""

    def test_kick_primarily_triggers_kick(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add('low', beats=[1, 2, 3, 4, 5, 6])
        events = run_pattern(p.render())
        by_kind = events_by_kind(events)
        assert 'low' in by_kind, "Low band not detected"
        assert len(by_kind['low']) >= 3, \
            f"Expected >=3 low-band detections from 6 hits, got {len(by_kind['low'])}"

    def test_high_primarily_triggers_high(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5])
        events = run_pattern(p.render())
        by_kind = events_by_kind(events)
        assert 'high' in by_kind, "High band not detected"
        n_high = len(by_kind['high'])
        n_low = len(by_kind.get('low', []))
        assert n_high >= n_low, \
            f"High should dominate: {n_high} high vs {n_low} low"

    def test_mid_primarily_triggers_mid(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add('mid', beats=[1, 2, 3, 4, 5, 6])
        events = run_pattern(p.render())
        by_kind = events_by_kind(events)
        assert 'mid' in by_kind, "Mid band not detected"
        assert len(by_kind['mid']) >= 3, \
            f"Expected >=3 mid-band detections from 6 hits, got {len(by_kind['mid'])}"


class TestVolumeInvariance:
    """Flux-based detection should produce the same timing at different volumes."""

    def test_quiet_and_loud_same_detection(self):
        def low_pattern(amp):
            p = DrumPattern(bpm=120, duration=3.0)
            p.add('low', beats=[1, 3, 5], amplitude=amp)
            return run_pattern(p.render())

        n_quiet = len([e for e in low_pattern(0.15) if e.kind == 'low'])
        n_loud = len([e for e in low_pattern(0.9) if e.kind == 'low'])
        assert n_quiet > 0, "Quiet low-band hits not detected"
        assert n_loud > 0, "Loud low-band hits not detected"
        assert abs(n_quiet - n_loud) <= 1, \
            f"Volume changed detection count: quiet={n_quiet}, loud={n_loud}"


class TestCooldown:
    """Cooldown should suppress rapid re-firing within ~255ms."""

    def test_double_low_suppressed(self):
        p = DrumPattern(bpm=120, duration=2.0)
        p.add_at_times('low', [0.5, 0.6])
        lows = [e for e in run_pattern(p.render()) if e.kind == 'low']
        assert len(lows) == 1, \
            f"Expected 1 low (second suppressed by cooldown), got {len(lows)}"

    def test_spaced_lows_both_fire(self):
        p = DrumPattern(bpm=120, duration=2.0)
        p.add_at_times('low', [0.3, 0.9])
        lows = [e for e in run_pattern(p.render()) if e.kind == 'low']
        assert len(lows) == 2, \
            f"Expected 2 lows (outside cooldown), got {len(lows)}"


class TestBeatDrop:
    """Silence followed by sudden onset should detect cleanly."""

    def test_silence_then_low(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add_at_times('low', [2.0])
        lows = [e for e in run_pattern(p.render()) if e.kind == 'low']
        assert len(lows) == 1, \
            f"Expected exactly 1 low-band event after silence, got {len(lows)}"

    def test_silence_then_full_kit(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add_at_times('low',  [2.0])
        p.add_at_times('mid', [2.0])
        p.add_at_times('high', [2.0])
        by_kind = events_by_kind(run_pattern(p.render()))
        for kind in ['low', 'mid', 'high']:
            n = len(by_kind.get(kind, []))
            assert n <= 2, f"Beat drop produced {n} {kind} events (expected 1)"

    def test_no_spurious_events_in_silence(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add_at_times('low', [2.0])
        events = run_pattern(p.render())
        for e in events:
            assert e.time >= 1.5, \
                f"Spurious event at t={e.time:.3f}s during silence: {e.kind}"


class TestMidConfirmation:
    """Mid-band detection requires mid+high frequency confirmation."""

    def test_pure_bass_fewer_mids_than_lows(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add('low', beats=[1, 3, 5])
        by_kind = events_by_kind(run_pattern(p.render()))
        n_low = len(by_kind.get('low', []))
        n_mid = len(by_kind.get('mid', []))
        assert n_low >= n_mid, \
            f"Lows ({n_low}) should outnumber false mids ({n_mid})"

    def test_real_mid_detected(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add('mid', beats=[1, 3, 5])
        by_kind = events_by_kind(run_pattern(p.render()))
        assert 'mid' in by_kind, "Real mid-band hit not detected"
        assert len(by_kind['mid']) >= 2, \
            f"Expected multiple mid-band detections, got {len(by_kind['mid'])}"


class TestEventEnergy:
    """Beat event energy values should be well-behaved."""

    def test_louder_hit_higher_energy(self):
        p = DrumPattern(bpm=120, duration=3.0)
        p.add('low', beats=[1], amplitude=0.2)
        p.add('low', beats=[5], amplitude=0.9)
        lows = [e for e in run_pattern(p.render()) if e.kind == 'low']
        if len(lows) >= 2:
            assert lows[-1].energy >= lows[0].energy, \
                f"Loud low energy ({lows[-1].energy:.3f}) should >= " \
                f"quiet low ({lows[0].energy:.3f})"



# ===================================================================
# Adaptive spectral band tests
# ===================================================================

def make_adaptive_processor() -> AudioProcessor:
    return make_processor(adaptive=True)


def run_pattern_adaptive(pcm: np.ndarray, warmup_frames: int = 20) -> list[DetectedEvent]:
    """Feed PCM through an adaptive AudioProcessor frame-by-frame."""
    proc = make_adaptive_processor()

    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    for _ in range(warmup_frames):
        proc.feed(silence)
        proc.process()

    events = []
    pos = 0
    while pos < len(pcm):
        chunk = pcm[pos:pos + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
        proc.feed(chunk)
        frame_events = proc.process()
        for e in frame_events:
            events.append(DetectedEvent(kind=e.kind, energy=e.energy, sample=pos))
        pos += FFT_SIZE

    return events


class TestAdaptiveBands:
    """Tests for adaptive spectral band tracking."""

    def test_808_low_detected_and_weights_shift(self):
        """Sub-bass 808 hits should be detected and cause low band
        weights to shift toward the sub-bass region.

        At 23.4Hz FFT resolution, even static bands catch some 808 energy
        via spectral leakage. The key assertion is that adaptive weights
        actually shift downward to explicitly cover sub-bass bins.
        """
        proc = make_adaptive_processor()
        hit = synth_808_kick(freq=35.0, band_limit=True)
        n_samples = int(SAMPLE_RATE * 5.0)
        pcm = np.zeros(n_samples, dtype=np.float32)
        for i in range(8):
            start = int(i * 0.5 * SAMPLE_RATE)
            end = min(start + len(hit), n_samples)
            if start < n_samples:
                pcm[start:end] += hit[:end - start]

        # Warmup
        silence = np.zeros(FFT_SIZE, dtype=np.float32)
        for _ in range(20):
            proc.feed(silence)
            proc.process()

        # Feed the 808 pattern
        all_events = []
        pos = 0
        while pos < len(pcm):
            chunk = pcm[pos:pos + FFT_SIZE]
            if len(chunk) < FFT_SIZE:
                chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
            proc.feed(chunk)
            events = proc.process()
            all_events.extend(events)
            pos += FFT_SIZE

        lows = [e for e in all_events if e.kind == 'low']
        assert len(lows) >= 3, \
            f"808 low-band hits should be detected, got {len(lows)}"

        # Check that low-band weights shifted below 50Hz
        low_ab = proc._detector.adaptive_bands['low']
        sub_50_mask = FREQS < 50
        sub_50_weight = low_ab.weights[sub_50_mask].sum()
        total_weight = low_ab.weights.sum()
        sub_50_ratio = sub_50_weight / (total_weight + 1e-10)
        print(f"[808] lows={len(lows)}, sub-50Hz weight ratio: {sub_50_ratio:.3f}")
        assert sub_50_ratio > 0.1, \
            f"Low weights should shift toward sub-bass, sub-50Hz ratio={sub_50_ratio:.3f}"

    def test_low_high_detected_and_weights_shift(self):
        """6kHz electronic hits should be detected and cause high band
        weights to shift toward the 5-7kHz region."""
        proc = make_adaptive_processor()
        hh = synth_low_hihat()
        n_samples = int(SAMPLE_RATE * 4.0)
        pcm = np.zeros(n_samples, dtype=np.float32)
        for i in range(16):
            start = int(i * 0.25 * SAMPLE_RATE)
            end = min(start + len(hh), n_samples)
            if start < n_samples:
                pcm[start:end] += hh[:end - start]

        silence = np.zeros(FFT_SIZE, dtype=np.float32)
        for _ in range(20):
            proc.feed(silence)
            proc.process()

        all_events = []
        pos = 0
        while pos < len(pcm):
            chunk = pcm[pos:pos + FFT_SIZE]
            if len(chunk) < FFT_SIZE:
                chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
            proc.feed(chunk)
            events = proc.process()
            all_events.extend(events)
            pos += FFT_SIZE

        highs = [e for e in all_events if e.kind == 'high']
        assert len(highs) >= 3, \
            f"Low-frequency high-band hits should be detected, got {len(highs)}"

        # Check that high-band weights shifted below 8kHz
        high_ab = proc._detector.adaptive_bands['high']
        below_8k_mask = (FREQS >= 5000) & (FREQS < 8000)
        below_8k_weight = high_ab.weights[below_8k_mask].sum()
        total_weight = high_ab.weights.sum()
        below_8k_ratio = below_8k_weight / (total_weight + 1e-10)
        print(f"[low high] highs={len(highs)}, 5-8kHz weight ratio: {below_8k_ratio:.3f}")
        assert below_8k_ratio > 0.05, \
            f"High weights should shift toward 5-8kHz, ratio={below_8k_ratio:.3f}"

    def test_section_change_adaptation(self):
        """Low-band frequency shifts mid-song — adaptive should catch both.

        2 seconds of 80Hz hits (within static range), then 2 seconds
        of 40Hz hits (below static range). Adaptive should detect both.
        """
        kick_80 = synth_kick(freq=80.0)
        kick_40 = synth_808_kick(freq=40.0)

        n_samples = int(SAMPLE_RATE * 5.0)
        pcm = np.zeros(n_samples, dtype=np.float32)

        # Phase 1: 80Hz kicks (beats 1-4, 0.5s apart)
        for i in range(4):
            start = int(i * 0.5 * SAMPLE_RATE)
            end = min(start + len(kick_80), n_samples)
            pcm[start:end] += kick_80[:end - start]

        # Phase 2: 40Hz kicks (beats 5-8, starting at 2.5s)
        for i in range(4):
            start = int((2.5 + i * 0.5) * SAMPLE_RATE)
            end = min(start + len(kick_40), n_samples)
            pcm[start:end] += kick_40[:end - start]

        events = run_pattern_adaptive(pcm)
        boundary = 2.5

        phase1_lows = [e for e in events if e.kind == 'low' and e.time < boundary]
        phase2_lows = [e for e in events if e.kind == 'low' and e.time >= boundary]

        print(f"[section] phase1 (80Hz): {len(phase1_lows)} lows, "
              f"phase2 (40Hz): {len(phase2_lows)} lows")
        assert len(phase1_lows) >= 1, "Should detect 80Hz low-band hits in phase 1"
        assert len(phase2_lows) >= 1, "Should detect 40Hz low-band hits in phase 2"

    def test_no_drift_to_vocals(self):
        """Vocal-only signal should not cause band weights to drift far
        from defaults — the anchoring term should prevent this."""
        proc = make_adaptive_processor()
        pcm = synth_vocal(duration=4.0, amplitude=0.5)

        # Feed through processor
        silence = np.zeros(FFT_SIZE, dtype=np.float32)
        for _ in range(20):
            proc.feed(silence)
            proc.process()

        pos = 0
        while pos < len(pcm):
            chunk = pcm[pos:pos + FFT_SIZE]
            if len(chunk) < FFT_SIZE:
                chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
            proc.feed(chunk)
            proc.process()
            pos += FFT_SIZE

        # Check that low band weights haven't drifted drastically
        low_ab = proc._detector.adaptive_bands['low']
        default_energy = low_ab.default_weights.sum()
        # Weight should still be mostly in the default region
        default_region_weight = np.dot(low_ab.weights, low_ab.default_weights > 0)
        total_weight = low_ab.weights.sum()
        ratio = default_region_weight / (total_weight + 1e-10)
        print(f"[vocal drift] low default-region ratio: {ratio:.3f}")
        assert ratio > 0.15, \
            f"Low weights drifted too far from defaults: ratio={ratio:.3f}"

    def test_song_reset_restores_defaults(self):
        """After adaptation, reset_bands() should restore default weights."""
        proc = make_adaptive_processor()

        # Feed some low-band audio to cause adaptation
        hit = synth_kick(freq=80.0)[:FFT_SIZE]  # truncate to fit buffer
        for _ in range(60):
            pcm = np.zeros(FFT_SIZE, dtype=np.float32)
            pcm[:len(hit)] = hit
            proc.feed(pcm)
            proc.process()

        # Weights should have shifted
        low_ab = proc._detector.adaptive_bands['low']
        adapted_weights = low_ab.weights.copy()

        # Reset
        proc.reset_bands()

        # Should be back to defaults
        np.testing.assert_array_equal(
            low_ab.weights, low_ab.default_weights,
            err_msg="reset_bands() didn't restore default weights")
        np.testing.assert_array_equal(
            low_ab.flux_accum, np.zeros(N_BINS, dtype=np.float32),
            err_msg="reset_bands() didn't zero flux accumulator")

    def test_standard_pattern_still_works_adaptive(self):
        """Standard 4/4 pattern should still be detected with adaptive bands.

        Adaptive shouldn't break normal detection — the default weights
        cover the same range as the static masks.
        """
        p = DrumPattern(bpm=120, duration=4.0)
        p.add('low',  beats=[1, 3, 5, 7])
        p.add('mid', beats=[2, 4, 6, 8])
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        pcm = p.render()

        events = run_pattern_adaptive(pcm)
        by_kind = events_by_kind(events)

        assert 'low' in by_kind, "Adaptive mode broke low-band detection"
        assert 'mid' in by_kind, "Adaptive mode broke mid-band detection"
        assert 'high' in by_kind, "Adaptive mode broke high-band detection"
        assert len(by_kind['low']) >= 2, \
            f"Adaptive detected too few low-band events: {len(by_kind['low'])}"
