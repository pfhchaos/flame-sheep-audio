"""
Synthetic audio generators for beat detection testing.

Instruments: acoustic drums, electronic drums (808/clap), vocals, speech.
Effects: dynamic range compression, white noise, vocal/speech overlay.
Pattern builder: DrumPattern for composing multi-track PCM from beat positions.
Pattern library: PatternSpec definitions for parametrized tests.
"""

import numpy as np
from dataclasses import dataclass

from flame_sheep_audio import SAMPLE_RATE, FFT_SIZE


# -------------------------------------------------------------------
# Envelopes
# -------------------------------------------------------------------

def _envelope(n_samples: int, attack: int = 5) -> np.ndarray:
    """Exponential attack-decay envelope."""
    env = np.ones(n_samples, dtype=np.float32)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    decay = np.exp(-np.linspace(0, 6, n_samples - attack, dtype=np.float32))
    env[attack:] = decay
    return env


# -------------------------------------------------------------------
# Acoustic drum kit
# -------------------------------------------------------------------

def synth_kick(amplitude: float = 0.8, freq: float = 70.0,
               duration_ms: float = 80.0) -> np.ndarray:
    """Bass drum: band-limited low-frequency sine burst."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    sig = amplitude * np.sin(2 * np.pi * freq * t) * _envelope(n)
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spec[(freqs < 40) | (freqs > 120)] = 0
    return np.fft.irfft(spec, n=n).astype(np.float32) * amplitude


def synth_snare(amplitude: float = 0.6, duration_ms: float = 60.0) -> np.ndarray:
    """Snare: mid-freq tone + noise burst across 300-3000Hz."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    env = _envelope(n)
    tone = 0.4 * np.sin(2 * np.pi * 400 * t)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spec[(freqs < 300) | (freqs > 3000)] = 0
    noise = np.fft.irfft(spec, n=n).astype(np.float32)
    noise /= np.abs(noise).max() + 1e-8
    return (amplitude * (tone + 0.6 * noise) * env).astype(np.float32)


def synth_hihat(amplitude: float = 0.3, duration_ms: float = 20.0) -> np.ndarray:
    """Closed hihat: high-frequency noise burst (8kHz+)."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spec[freqs < 8000] = 0
    noise = np.fft.irfft(spec, n=n).astype(np.float32)
    noise /= np.abs(noise).max() + 1e-8
    return (amplitude * noise * _envelope(n)).astype(np.float32)


# -------------------------------------------------------------------
# Electronic drum kit
# -------------------------------------------------------------------

def synth_808_kick(amplitude: float = 0.9, freq: float = 35.0,
                   duration_ms: float = 150.0,
                   band_limit: bool = False) -> np.ndarray:
    """808-style sub-bass kick: sine with pitch drop, long decay.

    Args:
        band_limit: If True, strictly filter to 25-48Hz (for adaptive weight tests).
                    If False, allow natural harmonics (for realistic pattern tests).
    """
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    pitch = freq * (1.0 - 0.3 * t / (duration_ms / 1000))
    phase = np.cumsum(pitch) / SAMPLE_RATE * 2 * np.pi
    sig = (amplitude * np.sin(phase) * _envelope(n)).astype(np.float32)
    if band_limit:
        spec = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
        spec[(freqs < 25) | (freqs >= 48)] = 0
        sig = np.fft.irfft(spec, n=n).astype(np.float32) * amplitude
    return sig


def synth_clap(amplitude: float = 0.6, duration_ms: float = 40.0) -> np.ndarray:
    """Electronic clap: layered noise bursts in the mid band (200-2000Hz)."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    rng = np.random.default_rng(55)
    sig = np.zeros(n, dtype=np.float32)
    n_bursts = 4
    burst_len = int(SAMPLE_RATE * 0.003)
    spacing = int(SAMPLE_RATE * 0.005)
    for i in range(n_bursts):
        start = i * spacing
        end = min(start + burst_len, n)
        if start >= n:
            break
        burst = rng.standard_normal(end - start).astype(np.float32)
        b_spec = np.fft.rfft(burst)
        b_freqs = np.fft.rfftfreq(len(burst), 1.0 / SAMPLE_RATE)
        b_spec[(b_freqs < 200) | (b_freqs > 2000)] = 0
        burst = np.fft.irfft(b_spec, n=len(burst)).astype(np.float32)
        burst /= np.abs(burst).max() + 1e-8
        sig[start:end] += burst * (0.8 ** i)
    sig *= _envelope(n)
    sig /= np.abs(sig).max() + 1e-8
    return (amplitude * sig).astype(np.float32)


def synth_low_hihat(amplitude: float = 0.3, duration_ms: float = 25.0) -> np.ndarray:
    """Electronic hihat at ~6kHz (below the default 8kHz threshold)."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    rng = np.random.default_rng(88)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    spec[(freqs < 5000) | (freqs > 7000)] = 0
    noise = np.fft.irfft(spec, n=n).astype(np.float32)
    noise /= np.abs(noise).max() + 1e-8
    return (amplitude * noise * _envelope(n)).astype(np.float32)


# -------------------------------------------------------------------
# Vocal / speech synthesis
# -------------------------------------------------------------------

def synth_vocal(duration: float, amplitude: float = 0.4,
                pitch: float = 200.0, seed: int = 123) -> np.ndarray:
    """Synthetic vocal signal — formant synthesis with vibrato and consonants."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    vibrato = 1.0 + 0.03 * np.sin(2 * np.pi * 5.2 * t)
    phase = np.cumsum(pitch * vibrato) / SAMPLE_RATE * 2 * np.pi

    sig = np.zeros(n, dtype=np.float32)
    for harmonic in range(1, 8):
        h_freq = pitch * harmonic
        formant_gain = 0.0
        for f_center, f_width in [(800, 200), (1200, 300), (2800, 400)]:
            formant_gain += np.exp(-0.5 * ((h_freq - f_center) / f_width) ** 2)
        gain = (0.3 + 0.7 * formant_gain) / harmonic
        sig += gain * np.sin(harmonic * phase).astype(np.float32)

    phrase_len = int(SAMPLE_RATE * 0.8)
    gap_len = int(SAMPLE_RATE * 0.2)
    env = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        end = min(pos + phrase_len, n)
        chunk_len = end - pos
        phrase_env = np.ones(chunk_len, dtype=np.float32)
        ramp = min(int(SAMPLE_RATE * 0.05), chunk_len // 2)
        if ramp > 0:
            phrase_env[:ramp] = np.linspace(0, 1, ramp)
            phrase_env[-ramp:] = np.linspace(1, 0, ramp)
        env[pos:end] = phrase_env
        pos = end + gap_len

    sig *= env

    n_sibilants = int(duration * 4)
    for _ in range(n_sibilants):
        onset = int(rng.uniform(0, n - SAMPLE_RATE * 0.03))
        sib_len = int(SAMPLE_RATE * rng.uniform(0.015, 0.04))
        sib_end = min(onset + sib_len, n)
        sib = rng.standard_normal(sib_end - onset).astype(np.float32)
        sib_spec = np.fft.rfft(sib)
        sib_freqs = np.fft.rfftfreq(len(sib), 1.0 / SAMPLE_RATE)
        sib_spec[sib_freqs < 4000] = 0
        sib = np.fft.irfft(sib_spec, n=len(sib)).astype(np.float32)
        sib /= np.abs(sib).max() + 1e-8
        sig[onset:sib_end] += 0.3 * sib * _envelope(sib_end - onset)

    sig /= np.abs(sig).max() + 1e-8
    return (amplitude * sig).astype(np.float32)


def synth_speech(duration: float, amplitude: float = 0.4,
                 syllable_rate: float = 4.0, regularity: float = 0.0,
                 pitch: float = 150.0, seed: int = 456) -> np.ndarray:
    """Synthetic speech signal — rapid formant transitions with consonant bursts.

    Args:
        duration: Length in seconds.
        amplitude: Output amplitude.
        syllable_rate: Average syllables per second (~4 for normal speech).
        regularity: 0.0 = natural irregular timing, 1.0 = perfectly metronomic.
        pitch: Base fundamental frequency in Hz.
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    sig = np.zeros(n, dtype=np.float32)

    syllable_period = 1.0 / syllable_rate
    syllable_times = []
    pos_t = 0.1
    while pos_t < duration - 0.1:
        syllable_times.append(pos_t)
        if regularity >= 0.99:
            pos_t += syllable_period
        else:
            jitter = rng.exponential(syllable_period * 0.3) * (1 - regularity)
            pos_t += syllable_period * regularity + jitter + syllable_period * 0.3

    phrase_len = int(rng.integers(3, 8))
    pitch_contour = np.ones(n, dtype=np.float32) * pitch
    for i, st in enumerate(syllable_times):
        phrase_pos = (i % phrase_len) / max(phrase_len - 1, 1)
        pitch_shift = pitch * 0.15 * np.sin(np.pi * phrase_pos)
        if i % phrase_len == 0:
            phrase_len = int(rng.integers(3, 8))
        s_start = int(st * SAMPLE_RATE)
        s_end = min(s_start + int(syllable_period * SAMPLE_RATE), n)
        pitch_contour[s_start:s_end] += pitch_shift

    phase = np.cumsum(pitch_contour) / SAMPLE_RATE * 2 * np.pi

    vowel_formants = [
        [(800, 150), (1200, 200), (2600, 300)],
        [(300, 100), (2300, 200), (3000, 300)],
        [(500, 120), (1000, 200), (2500, 300)],
        [(400, 100), (2000, 250), (2800, 300)],
        [(350, 100), (700, 150), (2700, 300)],
    ]

    for i, st in enumerate(syllable_times):
        s_start = int(st * SAMPLE_RATE)
        vowel_dur = rng.uniform(0.08, 0.2)
        s_end = min(s_start + int(vowel_dur * SAMPLE_RATE), n)
        if s_start >= n:
            break

        formants = vowel_formants[i % len(vowel_formants)]
        chunk_len = s_end - s_start
        chunk = np.zeros(chunk_len, dtype=np.float32)
        for harmonic in range(1, 10):
            h_freq = pitch * harmonic
            formant_gain = 0.0
            for f_center, f_width in formants:
                formant_gain += np.exp(-0.5 * ((h_freq - f_center) / f_width) ** 2)
            gain = (0.2 + 0.8 * formant_gain) / harmonic
            chunk += gain * np.sin(harmonic * phase[s_start:s_end]).astype(np.float32)

        env = np.ones(chunk_len, dtype=np.float32)
        ramp = min(int(SAMPLE_RATE * 0.01), chunk_len // 4)
        if ramp > 0:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        sig[s_start:s_end] += chunk * env * 0.6

        if rng.random() < 0.7:
            cons_type = rng.choice(['plosive', 'fricative', 'sibilant'])
            cons_dur = int(SAMPLE_RATE * rng.uniform(0.01, 0.03))
            cons_start = max(0, s_start - cons_dur)
            cons_end = s_start
            if cons_end > cons_start:
                cons_len = cons_end - cons_start
                cons = rng.standard_normal(cons_len).astype(np.float32)
                cons_spec = np.fft.rfft(cons)
                cons_freqs = np.fft.rfftfreq(cons_len, 1.0 / SAMPLE_RATE)

                if cons_type == 'plosive':
                    cons_spec[cons_freqs < 100] *= 0.5
                elif cons_type == 'fricative':
                    cons_spec[cons_freqs < 500] = 0
                    cons_spec[cons_freqs > 6000] *= 0.3
                else:
                    cons_spec[cons_freqs < 4000] = 0

                cons = np.fft.irfft(cons_spec, n=cons_len).astype(np.float32)
                cons /= np.abs(cons).max() + 1e-8
                sig[cons_start:cons_end] += cons * _envelope(cons_len) * 0.5

    sig /= np.abs(sig).max() + 1e-8
    return (amplitude * sig).astype(np.float32)


# -------------------------------------------------------------------
# Dynamic range compression
# -------------------------------------------------------------------

def compress(pcm: np.ndarray, ratio: float = 4.0,
             threshold_db: float = -12.0,
             attack_ms: float = 5.0, release_ms: float = 50.0,
             makeup: bool = True) -> np.ndarray:
    """Apply dynamic range compression to PCM signal.

    Uses block-based envelope following for performance (~50x faster
    than per-sample Python loop). Processes in 64-sample blocks.
    """
    peak = np.abs(pcm).max()
    if peak < 1e-10:
        return pcm.copy()

    threshold = peak * (10 ** (threshold_db / 20))
    # Block-based: compute envelope per block, apply gain per block
    block_size = 64
    n_blocks = (len(pcm) + block_size - 1) // block_size

    # Compute per-block peak levels
    padded = np.pad(pcm, (0, n_blocks * block_size - len(pcm)))
    blocks = padded.reshape(n_blocks, block_size)
    block_levels = np.abs(blocks).max(axis=1)

    # Smooth envelope across blocks
    attack_coeff = np.exp(-block_size / (SAMPLE_RATE * attack_ms / 1000))
    release_coeff = np.exp(-block_size / (SAMPLE_RATE * release_ms / 1000))
    envelope = np.empty(n_blocks, dtype=np.float64)
    env = 0.0
    for i in range(n_blocks):
        level = block_levels[i]
        if level > env:
            env = attack_coeff * env + (1 - attack_coeff) * level
        else:
            env = release_coeff * env + (1 - release_coeff) * level
        envelope[i] = env

    # Compute per-block gain
    gain = np.ones(n_blocks, dtype=np.float32)
    above = envelope > threshold
    if above.any():
        gain_db = np.zeros(n_blocks)
        gain_db[above] = (1 - 1/ratio) * 20 * np.log10(
            envelope[above] / threshold + 1e-10)
        gain = (10 ** (-gain_db / 20)).astype(np.float32)

    # Apply gain per block (broadcast across samples in each block)
    out = (blocks * gain[:, np.newaxis]).reshape(-1)[:len(pcm)]

    if makeup:
        out_peak = np.abs(out).max()
        if out_peak > 1e-10:
            out *= peak / out_peak

    return out.astype(np.float32)


# -------------------------------------------------------------------
# Drum pattern builder
# -------------------------------------------------------------------

class DrumPattern:
    """
    Build a multi-track drum pattern and render to PCM.

    Usage:
        p = DrumPattern(bpm=120, duration=4.0)
        p.add('low', beats=[1, 3])
        p.add('mid', beats=[2, 4])
        p.add('high', beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        pcm = p.render()
    """

    KIT_ACOUSTIC = {
        'low':  synth_kick,
        'mid': synth_snare,
        'high': synth_hihat,
    }

    KIT_ELECTRONIC = {
        'low':  synth_808_kick,
        'mid': synth_clap,
        'high': synth_low_hihat,
    }

    SYNTHS = KIT_ACOUSTIC

    def __init__(self, bpm: float = 120.0, duration: float = 4.0,
                 noise_snr_db: float | None = None,
                 vocal: float | None = None,
                 speech: dict | None = None,
                 compression: float | None = None,
                 kit: dict | None = None,
                 volume_envelope: list[tuple[float, float]] | None = None):
        """
        Args:
            volume_envelope: List of (time_sec, gain) keyframes, linearly
                interpolated. E.g. [(0, 1.0), (4, 0.0)] for a 4s fadeout.
                Applied after all mixing, before compression.
        """
        self.bpm = bpm
        self.duration = duration
        self.noise_snr_db = noise_snr_db
        self.vocal = vocal
        self.speech = speech
        self.compression = compression
        self.volume_envelope = volume_envelope
        self._kit = kit or self.KIT_ACOUSTIC
        self._hits: list[tuple[str, float, float]] = []
        self.beat_duration = 60.0 / bpm

    def add(self, kind: str, beats: list[float], amplitude: float | None = None):
        """Add hits at given beat positions (1-indexed, fractional ok)."""
        amp = amplitude or {'low': 0.8, 'mid': 0.6, 'high': 0.3}[kind]
        for b in beats:
            t = (b - 1) * self.beat_duration
            self._hits.append((kind, t, amp))

    def add_at_times(self, kind: str, times: list[float], amplitude: float | None = None):
        """Add hits at absolute times in seconds."""
        amp = amplitude or {'low': 0.8, 'mid': 0.6, 'high': 0.3}[kind]
        for t in times:
            self._hits.append((kind, t, amp))

    def render(self) -> np.ndarray:
        """Render pattern to mono float32 PCM."""
        n_samples = int(SAMPLE_RATE * self.duration)
        pcm = np.zeros(n_samples, dtype=np.float32)

        for kind, t, amp in self._hits:
            synth = self._kit[kind]
            hit = synth(amplitude=amp)
            start = int(t * SAMPLE_RATE)
            end = min(start + len(hit), n_samples)
            if start < n_samples:
                pcm[start:end] += hit[:end - start]

        if self.vocal is not None:
            pcm += synth_vocal(self.duration, amplitude=self.vocal)

        if self.speech is not None:
            pcm += synth_speech(self.duration, **self.speech)

        if self.noise_snr_db is not None:
            signal_rms = np.sqrt(np.mean(pcm ** 2)) + 1e-10
            noise_rms = signal_rms / (10 ** (self.noise_snr_db / 20))
            rng = np.random.default_rng(77)
            noise = (rng.standard_normal(n_samples) * noise_rms).astype(np.float32)
            pcm += noise

        # Volume envelope — applied after mixing, before compression
        if self.volume_envelope is not None:
            times = np.array([t for t, _ in self.volume_envelope])
            gains = np.array([g for _, g in self.volume_envelope])
            t_axis = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE
            env = np.interp(t_axis, times, gains).astype(np.float32)
            pcm *= env

        if self.compression is not None:
            pcm = compress(pcm, ratio=self.compression)

        return np.clip(pcm, -1.0, 1.0)


# -------------------------------------------------------------------
# Pattern spec + library
# -------------------------------------------------------------------

@dataclass
class PatternSpec:
    """Describes a drum pattern for parametrized testing."""
    name: str
    bpm: float
    bars: int
    bar_length: int
    kick_beats: list[float]
    snare_beats: list[float]
    hihat_beats: list[float]
    kick_amp: float = 0.8
    snare_amp: float = 0.6
    hihat_amp: float = 0.3
    min_kicks: int = 2
    min_snares: int = 1
    min_hihats: int = 1
    kit: dict | None = None

    def build(self, noise_snr_db: float | None = None,
              vocal: float | None = None,
              speech: dict | None = None,
              compression: float | None = None,
              volume_envelope: list[tuple[float, float]] | None = None) -> DrumPattern:
        total_beats = self.bars * self.bar_length
        duration = total_beats * (60.0 / self.bpm) + 0.5
        p = DrumPattern(bpm=self.bpm, duration=duration,
                        noise_snr_db=noise_snr_db, vocal=vocal,
                        speech=speech, compression=compression,
                        kit=self.kit, volume_envelope=volume_envelope)

        for bar in range(self.bars):
            offset = bar * self.bar_length
            if self.kick_beats:
                p.add('low',  [offset + b for b in self.kick_beats],
                      amplitude=self.kick_amp)
            if self.snare_beats:
                p.add('mid', [offset + b for b in self.snare_beats],
                      amplitude=self.snare_amp)
            if self.hihat_beats:
                p.add('high', [offset + b for b in self.hihat_beats],
                      amplitude=self.hihat_amp)
        return p


# --- Acoustic patterns ---

FOUR_FOUR = PatternSpec(
    name='4/4 rock', bpm=120, bars=4, bar_length=4,
    kick_beats=[1, 3], snare_beats=[2, 4],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    min_kicks=3, min_snares=2, min_hihats=3,
)

FOUR_FOUR_FAST = PatternSpec(
    name='4/4 fast punk', bpm=180, bars=4, bar_length=4,
    kick_beats=[1, 2, 3, 4], snare_beats=[2, 4],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    min_kicks=3, min_snares=2, min_hihats=3,
)

WALTZ = PatternSpec(
    name='3/4 waltz', bpm=140, bars=6, bar_length=3,
    kick_beats=[1], snare_beats=[2, 3], hihat_beats=[1, 2, 3],
    min_kicks=3, min_snares=3, min_hihats=3,
)

WALTZ_SLOW = PatternSpec(
    name='3/4 slow waltz', bpm=84, bars=4, bar_length=3,
    kick_beats=[1], snare_beats=[2, 3], hihat_beats=[1, 2, 3],
    min_kicks=2, min_snares=2, min_hihats=2,
)

VIENNESE_WALTZ = PatternSpec(
    name='3/4 viennese waltz', bpm=180, bars=6, bar_length=3,
    kick_beats=[1], snare_beats=[2, 3], hihat_beats=[1, 2, 3],
    kick_amp=0.9, snare_amp=0.4, hihat_amp=0.25,
    min_kicks=3, min_snares=2, min_hihats=2,
)

SIX_EIGHT = PatternSpec(
    name='6/8 compound', bpm=120, bars=4, bar_length=6,
    kick_beats=[1, 4], snare_beats=[4],
    hihat_beats=[1, 2, 3, 4, 5, 6],
    min_kicks=3, min_snares=2, min_hihats=3,
)

BOSSA_NOVA = PatternSpec(
    name='bossa nova', bpm=130, bars=4, bar_length=4,
    kick_beats=[1, 3.5], snare_beats=[2, 4],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    min_kicks=2, min_snares=2, min_hihats=3,
)

HALF_TIME = PatternSpec(
    name='half-time', bpm=140, bars=4, bar_length=4,
    kick_beats=[1], snare_beats=[3], hihat_beats=[1, 2, 3, 4],
    min_kicks=2, min_snares=2, min_hihats=2,
)

# --- Electronic patterns ---

_EKIT = DrumPattern.KIT_ELECTRONIC

TRAP = PatternSpec(
    name='trap 808', bpm=140, bars=4, bar_length=4,
    kick_beats=[1, 2.5], snare_beats=[3],
    hihat_beats=[1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
                 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75],
    kick_amp=0.9, snare_amp=0.7, hihat_amp=0.25,
    min_kicks=2, min_snares=2, min_hihats=3, kit=_EKIT,
)

FOUR_ON_FLOOR_808 = PatternSpec(
    name='4otf 808', bpm=128, bars=4, bar_length=4,
    kick_beats=[1, 2, 3, 4], snare_beats=[2, 4],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    kick_amp=0.9, snare_amp=0.6, hihat_amp=0.3,
    min_kicks=3, min_snares=2, min_hihats=3, kit=_EKIT,
)

DNB_ELECTRONIC = PatternSpec(
    name='dnb electronic', bpm=174, bars=4, bar_length=4,
    kick_beats=[1, 2.75], snare_beats=[2, 4],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    kick_amp=0.9, snare_amp=0.7, hihat_amp=0.25,
    min_kicks=2, min_snares=2, min_hihats=2, kit=_EKIT,
)

REGGAETON = PatternSpec(
    name='reggaeton dembow', bpm=95, bars=4, bar_length=4,
    kick_beats=[1, 2.75], snare_beats=[1.75, 3.75],
    hihat_beats=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
    kick_amp=0.9, snare_amp=0.6, hihat_amp=0.3,
    min_kicks=2, min_snares=2, min_hihats=3, kit=_EKIT,
)

ALL_PATTERNS = [
    FOUR_FOUR, FOUR_FOUR_FAST,
    WALTZ, WALTZ_SLOW, VIENNESE_WALTZ,
    SIX_EIGHT, BOSSA_NOVA, HALF_TIME,
    TRAP, FOUR_ON_FLOOR_808, DNB_ELECTRONIC, REGGAETON,
]

PATTERN_IDS = [p.name for p in ALL_PATTERNS]
