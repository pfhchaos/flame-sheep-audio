"""BeatNet-Lite adapter for the daemon's BeatDetectorBase interface.

Wraps `flame_sheep_audio.beatnet_lite.BeatNetLite` (pure numpy, ~2 MB
install) as a daemon-compatible detector. The daemon feeds raw audio
hops via `feed_audio_hop()`; this adapter buffers + resamples 48kHz →
22050Hz, runs BeatNet's sliding-STFT feature pipeline + LSTM forward
when the buffer crosses a BeatNet hop boundary (~every 2 daemon hops),
and emits BeatEvents on peak-picked activations.

Band mapping (BeatNet → daemon band names):
  downbeat (channel 1) → 'low'   (rarest hit, biggest visual)
  beat-not-downbeat    → 'mid'   (palette walk)
  high-frequency onset → not produced by BeatNet; 'high' is silent
                         unless a hybrid path is added later.

Latency budget (vs the live audio stream):
  + ~32 ms — BeatNet's intrinsic window-center delay (centered framing
              on a 1411-sample window at 22050 Hz)
  + ~20 ms — one BeatNet hop of peak-pick lookahead (a peak frame is
              emitted only after the next frame confirms it's a maximum)
  ≈ 52 ms total — perceptible but acceptable for a wallpaper viz.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ._constants import HOP_SIZE, SAMPLE_RATE
from ._types import BeatEvent
from ._spectrum import SpectrumFrame
from .beat_detector import BeatDetectorBase


# Resample fraction: daemon → BeatNet. 22050/48000 = 147/320 reduced.
_BEATNET_SR = 22050
_DAEMON_SR = SAMPLE_RATE   # 48000
# Using gcd: 48000 = 320*150, 22050 = 147*150 → down 320 / up 147.
_RESAMPLE_DOWN = 320
_RESAMPLE_UP = 147

_BEATNET_HOP = 441            # samples @ 22050 Hz = 20 ms
_BEATNET_WIN = 1411           # samples @ 22050 Hz = 64 ms

# Audio buffer size: enough for one full STFT window + one extra hop
# of margin (handles boundary alignment when the buffer crosses a
# BeatNet hop). Two hops past the window gives generous headroom.
_AUDIO_BUFFER_SAMPLES = _BEATNET_WIN + 2 * _BEATNET_HOP


class BeatNetLiveDetector(BeatDetectorBase):
    """Live-streaming BeatNet for the daemon.

    Wires BeatNetLite into the daemon's per-hop audio loop. The
    `detect(csd_frame)` method ignores its input frame entirely;
    all work happens in `feed_audio_hop`. BeatEvents accumulate
    between detect() calls and are returned on the next detect().

    Construction reads the lite weights from this package's data dir
    (`flame_sheep_audio/data/beatnet_m{N}_lite.npz`) by default
    (configurable via the daemon's `cfg.detector.beatnet_weights_path`).
    """

    def __init__(self,
                 model_index: int = 1,
                 weights_path: Path | None = None,
                 peak_threshold: float = 0.3,
                 min_distance_frames: int = 3,
                 downbeat_threshold: float | None = None,
                 emit_high_band: bool = False,
                 use_particle_filter: bool = False,
                 *, band_config=None, freqs=None):
        # band_config + freqs accepted for factory-protocol compatibility
        # but BeatNet doesn't use them — the spectrum / band structure
        # is internal to the model.
        if weights_path is None:
            # Package default: the lite weights ship in this package's own
            # data dir (0.4 relocation — each package carries what it loads).
            # An explicit weights_path (from cfg.detector.beatnet_weights_path)
            # still overrides this.
            weights_path = (Path(__file__).resolve().parent
                            / 'data' / f'beatnet_m{model_index}_lite.npz')
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f'BeatNet lite weights not found: {weights_path}. '
                f'Run tools/export_beatnet_lite_weights.py to generate.')

        # Lazy import — keeps numpy-only BeatNet code out of the daemon's
        # import cost when this detector isn't selected.
        from .beatnet_lite import BeatNetLite
        self._lite = BeatNetLite(weights_path)
        self._lite.reset_state()

        self._peak_threshold = float(peak_threshold)
        self._min_distance = int(min_distance_frames)
        # Downbeat channel needs its own threshold — typically the
        # downbeat probability is lower than beat probability.
        self._downbeat_threshold = (
            float(downbeat_threshold) if downbeat_threshold is not None
            else self._peak_threshold)
        self._emit_high_band = bool(emit_high_band)

        # Optional particle-filter decoder — BeatNet's intended causal
        # post-processing. Adds tempo + meter priors that suppress
        # naive-peak-pick spurious fires. Heavier per-hop (~5-10ms typical)
        # vs the ~0.3ms naive path.
        self._use_pf = bool(use_particle_filter)
        self._pf = None
        if self._use_pf:
            self._pf = self._make_particle_filter()
            self._pf_path_len = 0
            self._pf_step_times: list[float] = []   # last N step times for live latency

        # Per-hop timing accumulators (live diagnostics; surfaced via
        # last_timing). Keep last N for percentile readouts.
        self._lite_step_times: list[float] = []

        # Latest activation values — used to set BeatEvent.energy
        # from `_drain_pf_events` when the PF path emits a new entry.
        self._last_beat_act = 0.0
        self._last_db_act = 0.0

        # State for resample + buffer.
        self._audio_buf = np.zeros(_AUDIO_BUFFER_SAMPLES, dtype=np.float32)
        self._buf_fill = 0                # samples currently in buffer
        self._next_emit_ref = _BEATNET_WIN // 2  # next absolute time we'll emit a frame
                                                  # centered (in resampled-sample space)
        self._total_resampled = 0          # cumulative resampled-sample count
        self._prev_log_spec = None         # for diff in the LOG_SPECT pipeline

        # Peak-pick state (with 1-frame lookahead).
        # We hold the last 2 activations; peak is the middle one.
        self._beat_history = deque(maxlen=3)
        self._downbeat_history = deque(maxlen=3)
        self._frames_since_beat = 1000
        self._frames_since_downbeat = 1000

        # Events buffered between detect() calls.
        self._pending: list[BeatEvent] = []

        # Pre-cache filterbank + hanning views from the lite weights.
        self._filterbank = self._lite.filterbank
        self._hanning = self._lite.hanning_window

        # Placeholder for daemon-side bpm injection (matches the
        # FluxBeatDetector / PercentileBeatDetector convention).
        self._bpm = 120.0

    @staticmethod
    def _patch_numpy_for_pf() -> None:
        """BeatNet's particle filter uses `np.in1d` (removed in numpy 2.0).
        Monkeypatch it to `np.isin` so the cascade runs. Idempotent."""
        import numpy as _np
        if not hasattr(_np, 'in1d'):
            _np.in1d = _np.isin

    def _make_particle_filter(self):
        """Construct BeatNet's particle_filter_cascade in causal mode.

        50 fps to match our BeatNet hop. mode=None silences upstream's
        `print('beat!')` stdout-spam (which fires inside `process()`
        only for mode in ('stream', 'realtime')).

        The fast vectorized `process()` is monkey-patched in immediately
        — it's mathematically equivalent (same per-state categorical
        sampling, just batched per unique source) and gives 5-10× on
        the tail. See `_pf_fast.install_fast_process`.
        """
        self._patch_numpy_for_pf()
        from BeatNet.particle_filtering_cascade import particle_filter_cascade
        from ._pf_fast import install_fast_process
        pf = particle_filter_cascade(
            beats_per_bar=[],
            particle_size=1500,
            down_particle_size=250,
            min_bpm=55.0, max_bpm=215.0,
            fps=50, plot=[],
            mode=None)
        install_fast_process(pf)
        return pf

    # ---- BeatDetectorBase ----

    def feed_audio_hop(self, hop: np.ndarray) -> None:
        """Receive one daemon hop (HOP_SIZE @ SAMPLE_RATE). Downsamples
        + appends to the BeatNet-rate buffer; flushes complete BeatNet
        hops through the model and accumulates events."""
        # Downsample 48kHz → 22050Hz. resample_poly is FIR + decimate.
        resampled = resample_poly(hop, up=_RESAMPLE_UP, down=_RESAMPLE_DOWN)
        # Append to rolling buffer.
        n_new = len(resampled)
        if self._buf_fill + n_new <= len(self._audio_buf):
            self._audio_buf[self._buf_fill:self._buf_fill + n_new] = resampled
            self._buf_fill += n_new
        else:
            # Shift old samples out — keep the most recent WIN-1 samples.
            shift = self._buf_fill + n_new - len(self._audio_buf)
            self._audio_buf[:-shift] = self._audio_buf[shift:].copy()
            self._buf_fill -= shift
            self._audio_buf[self._buf_fill:self._buf_fill + n_new] = resampled
            self._buf_fill += n_new
        self._total_resampled += n_new

        # Emit zero or more BeatNet frames. A frame at absolute resampled-
        # time `t_ref` requires the buffer to contain audio through
        # `t_ref + WIN/2`. Emit as many as we now have data for.
        while (self._total_resampled
               >= self._next_emit_ref + _BEATNET_WIN // 2):
            self._process_one_frame()
            self._next_emit_ref += _BEATNET_HOP

    def _process_one_frame(self) -> None:
        """Compute one BeatNet frame at the current emit ref position
        and update the peak-detection state machine."""
        # Frame center is at self._next_emit_ref in absolute resampled
        # time. The buffer represents the most recent `_buf_fill` samples
        # ending at absolute time `self._total_resampled`. So the frame
        # we want covers absolute samples
        # [next_emit - WIN/2, next_emit + WIN/2), which inside the buffer
        # starts at offset `self._buf_fill - (self._total_resampled - (next_emit - WIN/2))`.
        half = _BEATNET_WIN // 2
        abs_start = self._next_emit_ref - half
        abs_end = abs_start + _BEATNET_WIN
        buf_start = self._buf_fill - (self._total_resampled - abs_start)
        buf_end = buf_start + _BEATNET_WIN
        if buf_start < 0 or buf_end > self._buf_fill:
            # Buffer hasn't accumulated enough history yet — emit zero
            # (warmup). Frame skipped this step; state machine sees no peak.
            self._beat_history.append(0.0)
            self._downbeat_history.append(0.0)
            self._frames_since_beat += 1
            self._frames_since_downbeat += 1
            self._check_peak()
            return

        frame = self._audio_buf[buf_start:buf_end]
        windowed = frame * self._hanning
        fft_mag = np.abs(np.fft.rfft(windowed, n=_BEATNET_WIN))[:-1]
        log_spec = np.log10(1.0 + fft_mag @ self._filterbank).astype(np.float32)
        if self._prev_log_spec is None:
            diff = np.zeros_like(log_spec)
        else:
            diff = np.maximum(log_spec - self._prev_log_spec, 0.0)
        features = np.concatenate([log_spec, diff])    # (272,)
        import time
        t_lite = time.monotonic()
        probs = self._lite.step(features)               # (3,) [beat, downbeat, non-beat]
        self._lite_step_times.append(time.monotonic() - t_lite)
        if len(self._lite_step_times) > 500:
            del self._lite_step_times[:250]
        self._prev_log_spec = log_spec

        if self._use_pf:
            # Particle filter path — feed (beat, downbeat) activations
            # into the cascade; emit events from new path entries.
            beat_act = float(probs[0] + probs[1])    # 1 - non_beat (BeatNet's beat-or-downbeat channel)
            db_act = float(probs[1])                   # downbeat channel
            act_step = np.array([[beat_act, db_act]], dtype=np.float64)
            # Stash latest activation so _drain_pf_events can use it
            # as the BeatEvent.energy (the PF decides emission on the
            # current frame's activation; the upstream module doesn't
            # expose per-event confidence directly).
            self._last_beat_act = beat_act
            self._last_db_act = db_act
            t_pf = time.monotonic()
            self._pf.process(act_step)
            self._pf_step_times.append(time.monotonic() - t_pf)
            if len(self._pf_step_times) > 500:
                del self._pf_step_times[:250]
            self._drain_pf_events()
            return

        # Naive peak-pick path (no PF). Per-frame local max with refractory.
        beat_score = float(1.0 - probs[2])
        downbeat_score = float(probs[1])
        self._beat_history.append(beat_score)
        self._downbeat_history.append(downbeat_score)
        self._frames_since_beat += 1
        self._frames_since_downbeat += 1
        self._check_peak()

    def _drain_pf_events(self) -> None:
        """Emit BeatEvents for any new entries appended to the PF's path
        since our last drain.

        path[i] = [time_seconds, kind] where kind=1 is downbeat,
        kind=2 is beat. Energy in BeatEvent we set from the activation
        value at the firing frame — best-effort, since the PF doesn't
        report confidence per emission.
        """
        path = self._pf.path
        new_len = len(path)
        if new_len <= self._pf_path_len:
            return
        for i in range(self._pf_path_len, new_len):
            row = path[i]
            kind_code = int(row[1])
            if kind_code == 1:
                kind = 'low'
                # Downbeat: use the downbeat-channel activation as
                # confidence proxy (clamped to [0,1]).
                energy = min(1.0, max(0.0, float(self._last_db_act)))
            elif kind_code == 2:
                kind = 'mid'
                # Non-downbeat beat: use the any-beat activation
                # (1 - non_beat). Tends to be 0.5-1.0 at firing time.
                energy = min(1.0, max(0.0, float(self._last_beat_act)))
            else:
                continue
            self._pending.append(BeatEvent(kind=kind, energy=energy))
        self._pf_path_len = new_len

    def _check_peak(self) -> None:
        """1-frame-lookahead peak detection on the activation
        histories. When history holds >=3 frames, the middle one is a
        peak candidate if it's larger than its neighbors and above
        threshold.

        Downbeat classification uses BOTH:
          1. Absolute floor: d_curr >= downbeat_threshold (filters noise)
          2. Channel dominance: 2 * d_curr > b_curr, i.e. probs[1] > probs[0]
             — the downbeat channel beats the beat channel after the
             non-beat slice is removed.

        Without (2), every beat where d_curr happens to exceed
        downbeat_threshold gets misclassified as a downbeat (since the
        beat-channel softmax probability often distributes across both
        channels). Empirically gave all-low events in the early live
        test 2026-06-08.
        """
        if len(self._beat_history) < 3:
            return
        b_prev, b_curr, b_next = list(self._beat_history)
        if (b_curr >= self._peak_threshold
                and b_curr > b_prev and b_curr >= b_next
                and self._frames_since_beat >= self._min_distance):
            d_curr = self._downbeat_history[1]
            downbeat_dominates = 2.0 * d_curr > b_curr
            if (downbeat_dominates
                    and d_curr >= self._downbeat_threshold
                    and self._frames_since_downbeat >= self._min_distance):
                kind = 'low'   # downbeat
                self._frames_since_downbeat = 0
            else:
                kind = 'mid'
            self._pending.append(BeatEvent(kind=kind, energy=b_curr))
            self._frames_since_beat = 0

    def detect(self, frame: SpectrumFrame) -> list[BeatEvent]:
        """Drain and return accumulated events. The `frame` argument
        from the daemon pipeline is unused — BeatNet has its own
        STFT path on the raw audio."""
        out = self._pending
        self._pending = []
        return out

    def reset_bands(self) -> None:
        """Reset on song change. Clears LSTM state + history + buffer."""
        self._lite.reset_state()
        self._audio_buf[:] = 0
        self._buf_fill = 0
        self._next_emit_ref = _BEATNET_WIN // 2
        self._total_resampled = 0
        self._prev_log_spec = None
        self._beat_history.clear()
        self._downbeat_history.clear()
        self._frames_since_beat = 1000
        self._frames_since_downbeat = 1000
        self._pending.clear()
        if self._use_pf:
            # Cleanest reset: rebuild the cascade (it has internal
            # particle states + counter that aren't simply zero-able).
            self._pf = self._make_particle_filter()
            self._pf_path_len = 0

    def hint_tempo(self, bpm: float,
                    strength: float = 0.5,
                    relative_tolerance: float = 0.04) -> bool:
        """Bias the PF toward `bpm` by reseating a fraction of its
        particles onto states whose beat-period matches the hint.

        Parameters
        ----------
        bpm : float
            Hinted tempo in BPM. Must be inside the PF's [min_bpm, max_bpm]
            range (currently 55-215); silently a no-op otherwise.
        strength : float, in [0.0, 1.0]
            Fraction of particles to relocate. 0.5 (default) replaces half;
            1.0 hard-overrides the entire population (PF can still drift
            but loses all prior history). 0.1-0.2 is a gentle prior.
        relative_tolerance : float
            How close a state's interval must be to the hint to be a
            valid relocation target. Default 4% matches the "exact"
            tempo-accuracy band the eval uses.

        Returns
        -------
        bool
            True if at least one particle was relocated; False if PF
            inactive, hint out of range, or no candidate states matched.

        Notes
        -----
        Implementation: builds the set of states whose `state_intervals`
        match the hint period within `relative_tolerance`, then randomly
        samples `int(strength * N_particles)` of them into the existing
        particles array (positions chosen randomly so we don't bias a
        particular bar-phase). The PF's next observation step re-weights
        and resamples normally — a wrong hint will be corrected by the
        activations, not stuck forever.

        Connects to `current_tempo()` — calling `hint_tempo(bpm)` then
        reading `current_tempo()` will return ≈ bpm IF the relocated
        particles dominate. Confidence will start low and rise as the
        observation steps reinforce.
        """
        if not self._use_pf or self._pf is None:
            return False
        if bpm <= 0.0:
            return False
        fps = float(self._pf.fps)
        target_interval_frames = 60.0 * fps / bpm

        intervals = np.asarray(self._pf.st.state_intervals, dtype=np.float64)
        # Avoid division by zero on any 0-interval slots (shouldn't exist
        # in BarStateSpace but defensive).
        nonzero = intervals > 0
        rel_dist = np.full(intervals.shape, np.inf, dtype=np.float64)
        rel_dist[nonzero] = (
            np.abs(intervals[nonzero] - target_interval_frames)
            / target_interval_frames)
        candidate_states = np.where(rel_dist < relative_tolerance)[0]
        if candidate_states.size == 0:
            return False

        n_particles = len(self._pf.particles)
        n_replace = int(max(0, min(n_particles, strength * n_particles)))
        if n_replace == 0:
            return False

        # Pick random positions to overwrite (uniform; we don't
        # discriminate by current particle state since the goal is just
        # to inject the hint without committing fully).
        rng = getattr(self._pf, '_fast_rng', None) or np.random.default_rng()
        positions = rng.choice(n_particles, size=n_replace, replace=False)
        new_states = rng.choice(candidate_states, size=n_replace, replace=True)
        # Use ndarray-backed assignment. self._pf.particles is np.ndarray
        # already; in-place modify keeps any external view valid.
        self._pf.particles[positions] = new_states
        return True

    def hint_tempo_octaves(self, bpm: float,
                             strength: float = 0.5,
                             octaves: tuple[float, ...] = (0.5, 1.0, 2.0),
                             relative_tolerance: float = 0.04) -> int:
        """Hint multiple tempo octaves simultaneously, letting the PF
        pick the right one from activations.

        Useful when the upstream tracker (e.g. BTrack) is good at the
        tempo SHAPE but error-prone on the octave choice. Pull
        particles toward all candidate octaves; the next observation
        steps reweight; the wrong octaves lose particles, the right one
        wins.

        Parameters
        ----------
        bpm : float
            Base hinted tempo.
        strength : float
            Total fraction of particles to relocate across all octaves.
            Each octave gets `strength / len(octaves)`.
        octaves : tuple of float
            Multipliers to apply to `bpm`. Default (0.5, 1.0, 2.0) covers
            half, true, and double — the three most-common BTrack errors.
            Add 1/3 and 3 for triplet-feel coverage if you want them.
        relative_tolerance : float
            Same meaning as `hint_tempo`.

        Returns
        -------
        int
            Number of octaves successfully hinted (≤ len(octaves)).
            Octaves that fall outside the PF's BPM range are silently
            skipped.

        Notes
        -----
        `hint_tempo` is called once per octave with strength /
        len(octaves), so total reseated ≈ strength * n_particles
        (small statistical overlap between draws is possible — under
        4% expected when octaves don't reuse the same state space).
        """
        if not octaves:
            return 0
        per_octave = strength / len(octaves)
        n_ok = 0
        for r in octaves:
            if self.hint_tempo(bpm * r, strength=per_octave,
                                relative_tolerance=relative_tolerance):
                n_ok += 1
        return n_ok

    def current_tempo(self) -> dict[str, float] | None:
        """Read the particle filter's current tempo estimate.

        Returns None if the PF isn't active or hasn't received enough
        frames to converge. Otherwise returns:
          bpm: median-particle's beat period → BPM (50 fps × 60 / interval)
          confidence: particle clustering 0..1 (1 = all particles agree,
            0 = uniform spread across 300 tempi)
          n_particles: total particle count (for context)

        The PF tracks 300 tempi between min_bpm=55 and max_bpm=215, so
        BPM resolution is ~0.5. Confidence is computed from the
        weighted concentration around the mode: count_at_mode / total.
        """
        if not self._use_pf or self._pf is None:
            return None
        particles = self._pf.particles
        if particles is None or len(particles) == 0:
            return None
        state_intervals = self._pf.st.state_intervals
        median_particle = int(np.median(particles))
        interval = float(state_intervals[median_particle])
        if interval <= 0:
            return None
        # Most particles concentrate around the dominant interval —
        # treat confidence as fraction sharing the same interval as the
        # median particle.
        mode_interval = state_intervals[median_particle]
        same_tempo = int((state_intervals[particles] == mode_interval).sum())
        confidence = same_tempo / len(particles)
        return {
            'bpm': 60.0 * float(self._pf.fps) / interval,
            'confidence': float(confidence),
            'n_particles': float(len(particles)),
        }

    def current_meter(self) -> dict[str, float] | None:
        """Read the particle filter's current meter (beats-per-bar)
        estimate.

        The PF maintains a separate state space for meter alongside the
        tempo state space. With our default config (beats_per_bar=[],
        min=2, max=4), it considers duple (2/4), triple (3/4 / waltz),
        and quadruple (4/4) meters. Down-particles cluster onto the
        states that best fit the observed downbeat activations.

        Returns None if PF inactive. Otherwise:
          beats_per_bar: modal beats-per-bar value (2, 3, or 4)
          confidence:    fraction of down-particles agreeing (0..1).
                         Uniform startup gives ~0.33; a locked 4/4 track
                         typically reads 0.85-1.0.
          n_particles:   total down-particle count (250 in default config)

        Notes
        -----
        Locks within ~3 seconds on a clear-meter track. On an
        ambiguous track (heavy syncopation, 6/8 polymeter, blast
        beats) confidence stays lower. Use the confidence to decide
        whether to act on the reading.
        """
        if not self._use_pf or self._pf is None:
            return None
        down_particles = getattr(self._pf, 'down_particles', None)
        if down_particles is None or len(down_particles) == 0:
            return None
        intervals = np.asarray(self._pf.st2.state_intervals)
        # Each particle's beats-per-bar is the interval at its state.
        bpb_per_particle = intervals[down_particles]
        unique, counts = np.unique(bpb_per_particle, return_counts=True)
        modal_idx = int(np.argmax(counts))
        modal_bpb = int(unique[modal_idx])
        confidence = float(counts[modal_idx]) / len(down_particles)
        return {
            'beats_per_bar': float(modal_bpb),
            'confidence': confidence,
            'n_particles': float(len(down_particles)),
        }

    def latency_snapshot(self) -> dict[str, float]:
        """Per-component live latency snapshot for the daemon's
        observability. Returns ms for lite-model step and (if active)
        particle-filter step. Empty buckets while history is short.
        """
        import numpy as _np
        snap: dict[str, float] = {}
        if self._lite_step_times:
            arr = _np.asarray(self._lite_step_times, dtype=_np.float64) * 1000.0
            snap['lite_mean_ms'] = float(arr.mean())
            snap['lite_p95_ms'] = float(_np.percentile(arr, 95))
            snap['lite_p99_ms'] = float(_np.percentile(arr, 99))
            snap['lite_max_ms'] = float(arr.max())
        if self._use_pf and self._pf_step_times:
            arr = _np.asarray(self._pf_step_times, dtype=_np.float64) * 1000.0
            snap['pf_mean_ms'] = float(arr.mean())
            snap['pf_p95_ms'] = float(_np.percentile(arr, 95))
            snap['pf_p99_ms'] = float(_np.percentile(arr, 99))
            snap['pf_max_ms'] = float(arr.max())
        return snap
