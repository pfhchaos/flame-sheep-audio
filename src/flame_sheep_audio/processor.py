"""
AudioProcessor — orchestrates audio analysis pipeline.

Wires together spectrum engine, beat detection, energy analysis,
onset density tracking, and tempo estimation. Runs in a daemon thread
(PipeWireSource) or synchronously (FeedSource for tests).
"""

import logging

log = logging.getLogger(__name__)

import threading
import time

import numpy as np
import sounddevice as sd

from ._constants import SAMPLE_RATE, FFT_SIZE, N_BINS, HOP_SIZE
from ._types import BeatEvent, BandState, AudioSnapshot
from ._spectrum import SpectrumEngine
from .beat_detector import FluxBeatDetector
from .energy import EnergyAnalyzer
from .source import PipeWireSource, FeedSource
from .onset_density import OnsetDensityTracker
from .stability import MagnitudeStability
from .tempo_acf import AutocorrelationTempoTracker
from ._band_config import BandConfig, default_band_config
from ._bands import A_WEIGHTS


class AudioProcessor:
    """
    Orchestrates audio analysis: signal source -> spectrum -> detection + energy.

    Two modes determined by source type:
      - Threaded (PipeWireSource): audio analysis runs in a daemon thread at
        HOP_SIZE cadence. Render thread calls drain() to get accumulated state.
      - Synchronous (FeedSource): process() works as before for deterministic tests.
    """

    def __init__(self, device: str | int | None = None, adaptive: bool = False,
                 sharpness: bool = True, source=None,
                 band_config: BandConfig | None = None):
        if band_config is None:
            band_config = default_band_config()
        self._band_config = band_config

        self._source = source or PipeWireSource(device=device)
        self._spectrum_engine = SpectrumEngine()
        self._stability = MagnitudeStability()
        self._detector = FluxBeatDetector(adaptive=adaptive, sharpness=sharpness,
                                          stability=self._stability,
                                          band_config=band_config)
        self._energy = EnergyAnalyzer(band_config=band_config)
        self._tempo = AutocorrelationTempoTracker(
            hop_duration=HOP_SIZE / SAMPLE_RATE)
        self._density = OnsetDensityTracker(band_config=band_config)

        # Auto-detect: FeedSource is synchronous, everything else is threaded
        self._threaded = not isinstance(self._source, FeedSource)

        # Shared state (lock-protected, read by drain(), written by audio thread or process())
        self._lock     = threading.Lock()
        self._spectrum = np.zeros(N_BINS, dtype=np.float32)
        self._waveform = np.zeros(FFT_SIZE, dtype=np.float32)
        self._centroid = 1000.0
        self._centroid_delta = 0.0
        self._centroid_rms = 0.0
        self._centroid_harmonic_rms = 0.0
        self._percussiveness = 0.5
        self._bands = {name: BandState() for name in band_config.all_band_names}
        self._bpm = 0.0
        self._effective_bpm = 120.0
        self._tempo_saturated = False
        self._pending_events: list[BeatEvent] = []

        # Thread state
        self._thread: threading.Thread | None = None
        self._running = False

    def reset_bands(self):
        """Reset adaptive bands to defaults. Call on song change."""
        self._detector.reset_bands()

    def song_started(self):
        """Signal new song — reset tempo + density trackers."""
        self._tempo.song_started()
        self._density.reset()

    def hint_tempo(self, bpm: float):
        """Provide tempo hint from external source."""
        self._tempo.hint_tempo(bpm)

    def reset_tempo(self):
        """Reset tempo + density state (e.g., on seek)."""
        self._tempo.reset()
        self._density.reset()

    def start(self):
        self._source.start()
        if self._threaded:
            self._running = True
            self._thread = threading.Thread(
                target=self._audio_loop, daemon=True, name='audio-analysis')
            self._thread.start()
            log.info('Audio thread started (hop=%d, %.1fms)', HOP_SIZE,
                     HOP_SIZE / SAMPLE_RATE * 1000)

    def stop(self):
        self._running = False
        self._source.stop()  # unblocks read_hop via stop_event
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def feed(self, pcm: np.ndarray):
        """Feed PCM samples into the source buffer."""
        self._source.feed(pcm)

    # ------------------------------------------------------------------
    # Threaded mode: audio loop + drain
    # ------------------------------------------------------------------

    def _audio_loop(self):
        """Runs in daemon thread. Reads hops, analyses, publishes."""
        detection_names = set(self._band_config.detection_band_names)
        while self._running:
            hop = self._source.read_hop(HOP_SIZE)
            if hop is None:
                break  # source stopped

            now = time.perf_counter()
            frame = self._spectrum_engine.push_hop(hop)
            self._stability.update(frame.magnitude)
            self._energy.update(frame.magnitude, frame.flux,
                                stability=self._stability)
            events = self._detector.detect(frame)

            # Feed ACF tempo tracker with onset strength (A-weighted flux sum)
            onset_strength = float(np.dot(frame.flux, A_WEIGHTS))
            self._tempo.feed(onset_strength)

            # Feed density tracker
            for event in events:
                if event.kind in detection_names:
                    self._density.process_onset(event.kind, now)
            self._density.update(now)
            self._detector._bpm = self._tempo.effective_bpm

            with self._lock:
                self._pending_events.extend(events)
                self._spectrum[:] = frame.magnitude
                self._waveform[:] = frame.waveform
                self._centroid = self._energy.centroid
                self._centroid_delta = self._energy.centroid_delta
                self._centroid_rms = self._energy.centroid_rms
                self._centroid_harmonic_rms = self._energy.harmonic_centroid_rms
                self._percussiveness = self._energy.percussiveness
                self._bpm = self._tempo.bpm
                self._effective_bpm = self._tempo.effective_bpm
                self._tempo_saturated = self._tempo.saturated
                # Build per-band state
                band_rms = self._energy.band_rms_all
                band_hrms = self._energy.band_harmonic_rms_all
                densities = self._density.densities
                density_deltas = self._density.density_deltas
                for name in self._bands:
                    self._bands[name] = BandState(
                        rms=band_rms.get(name, 0.0),
                        harmonic_rms=band_hrms.get(name, 0.0),
                        onset_density=densities.get(name, 0.0),
                        density_delta=density_deltas.get(name, 0.0),
                    )

    def drain(self) -> AudioSnapshot:
        """Atomically read and clear accumulated audio state.

        In threaded mode: returns events accumulated since last drain.
        In sync mode: calls process() once, wraps result in AudioSnapshot.
        """
        if not self._threaded:
            events = self.process()
            return self._build_snapshot(events)

        with self._lock:
            snap = self._build_snapshot(self._pending_events)
            self._pending_events = []
            return snap

    def _build_snapshot(self, events: list[BeatEvent]) -> AudioSnapshot:
        """Build AudioSnapshot from current shared state (call under lock)."""
        bands = {name: BandState(rms=bs.rms, harmonic_rms=bs.harmonic_rms,
                                 onset_density=bs.onset_density,
                                 density_delta=bs.density_delta)
                 for name, bs in self._bands.items()}
        return AudioSnapshot(
            events=events,
            spectrum=self._spectrum.copy(),
            waveform=self._waveform.copy(),
            bands=bands,
            centroid=self._centroid,
            centroid_delta=self._centroid_delta,
            centroid_rms=self._centroid_rms,
            centroid_harmonic_rms=self._centroid_harmonic_rms,
            percussiveness=self._percussiveness,
            bpm=self._bpm,
            effective_bpm=self._effective_bpm,
            tempo_saturated=self._tempo_saturated,
        )

    # ------------------------------------------------------------------
    # Synchronous mode: process (for FeedSource / tests)
    # ------------------------------------------------------------------

    def process(self) -> list[BeatEvent]:
        """Run one analysis frame synchronously. Returns beat events.

        For FeedSource/test use. In threaded mode, use drain() instead.
        """
        pcm = self._source.read()
        if pcm is None:
            return []

        frame = self._spectrum_engine.compute(pcm)
        self._stability.update(frame.magnitude)
        self._energy.update(frame.magnitude, frame.flux,
                            stability=self._stability)

        events = self._detector.detect(frame)

        # Feed ACF tempo tracker
        onset_strength = float(np.dot(frame.flux, A_WEIGHTS))
        self._tempo.feed(onset_strength)

        # Feed density tracker
        now = time.perf_counter()
        detection_names = set(self._band_config.detection_band_names)
        for event in events:
            if event.kind in detection_names:
                self._density.process_onset(event.kind, now)
        self._density.update(now)

        with self._lock:
            self._spectrum[:] = frame.magnitude
            self._waveform[:] = frame.waveform
            self._centroid = self._energy.centroid
            self._centroid_delta = self._energy.centroid_delta
            self._centroid_rms = self._energy.centroid_rms
            self._centroid_harmonic_rms = self._energy.harmonic_centroid_rms
            self._percussiveness = self._energy.percussiveness
            band_rms = self._energy.band_rms_all
            band_hrms = self._energy.band_harmonic_rms_all
            densities = self._density.densities
            density_deltas = self._density.density_deltas
            for name in self._bands:
                self._bands[name] = BandState(
                    rms=band_rms.get(name, 0.0),
                    harmonic_rms=band_hrms.get(name, 0.0),
                    onset_density=densities.get(name, 0.0),
                    density_delta=density_deltas.get(name, 0.0),
                )

        return events

    @property
    def spectrum(self) -> np.ndarray:
        """Latest FFT magnitude spectrum, N_BINS long. For GPU texture upload."""
        with self._lock:
            return self._spectrum.copy()

    @property
    def waveform(self) -> np.ndarray:
        """Latest raw PCM window. For GPU texture upload if desired."""
        with self._lock:
            return self._waveform.copy()



class SyntheticAudioProcessor:
    """
    Drop-in replacement for AudioProcessor that generates a predictable
    metronome signal instead of capturing real audio.

    Useful for visual/integration testing: you know exactly when beats
    should fire and can judge whether the app responds correctly.

    Pattern (all timings in seconds, relative to start()):
      kick  — every `kick_interval`  seconds  (default 0.5s = 120 bpm)
      snare — every `snare_interval` seconds  (default 1.0s, on the 2 and 4)
      hihat — every `hihat_interval` seconds  (default 0.25s = 8th notes)

    A fake spectrum is synthesised so the audio visualiser on the GPU
    (tonemap.frag pulse effect) still animates.
    """

    def __init__(
        self,
        kick_interval:  float = 0.5,
        snare_interval: float = 1.0,
        hihat_interval: float = 0.25,
        bpm_label:      str   = '120 bpm',
        clock=None,
        band_config: BandConfig | None = None,
    ):
        if band_config is None:
            band_config = default_band_config()
        self._band_config = band_config

        self.kick_interval  = kick_interval
        self.snare_interval = snare_interval
        self.hihat_interval = hihat_interval
        self.bpm_label      = bpm_label
        self._clock         = clock  # callable returning seconds, or None for perf_counter

        self._start_time: float | None = None
        self._last: dict[str, float]   = {'kick': -1.0, 'snare': -1.0, 'hihat': -1.0}
        self._spectrum = np.zeros(N_BINS, dtype=np.float32)
        self._rms      = 0.5  # synthetic audio is "always playing"

    def start(self):
        if self._clock is not None:
            self._start_time = self._clock()
        else:
            import time
            self._start_time = time.perf_counter()
        log.info(f'[synthetic audio] {self.bpm_label}  '
              f'kick={self.kick_interval:.2f}s  '
              f'snare={self.snare_interval:.2f}s  '
              f'hihat={self.hihat_interval:.2f}s')

    def stop(self):
        pass  # nothing to close

    def reset_bands(self):
        pass

    def song_started(self):
        pass

    def hint_tempo(self, bpm: float):
        pass

    def reset_tempo(self):
        pass

    def process(self) -> list[BeatEvent]:
        if self._start_time is None:
            return []

        if self._clock is not None:
            now = self._clock() - self._start_time
        else:
            import time
            now = time.perf_counter() - self._start_time
        events  = []
        self._spectrum[:] = 0.0

        for band, interval in [
            ('kick',  self.kick_interval),
            ('snare', self.snare_interval),
            ('hihat', self.hihat_interval),
        ]:
            # Fire when we cross a beat boundary since last call
            beat_num_now  = int(now / interval)
            beat_num_last = int(self._last[band] / interval) if self._last[band] >= 0 else -1
            if beat_num_now > beat_num_last:
                events.append(BeatEvent(kind=band, energy=1.0))
                self._last[band] = now
                # Inject energy into the matching spectrum region so the
                # GPU pulse effect fires visually too
                freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
                if band == 'kick':
                    mask = (freqs >= 50) & (freqs < 100)
                elif band == 'snare':
                    mask = (freqs >= 150) & (freqs < 800)
                else:
                    mask = freqs >= 8000
                self._spectrum[mask] = 1.0

        return events

    def drain(self) -> AudioSnapshot:
        """Wrap process() into AudioSnapshot for uniform API with AudioProcessor."""
        events = self.process()
        # Synthetic: put fake RMS in subbass band so drift detection works
        bands = {name: BandState() for name in self._band_config.all_band_names}
        if 'subbass' in bands:
            bands['subbass'] = BandState(rms=self._rms, harmonic_rms=self._rms)
        return AudioSnapshot(
            events=events,
            spectrum=self._spectrum.copy(),
            waveform=np.zeros(FFT_SIZE, dtype=np.float32),
            bands=bands,
        )

    @property
    def spectrum(self) -> np.ndarray:
        return self._spectrum.copy()

    @property
    def rms(self) -> float:
        return self._rms


def list_monitor_devices() -> list[dict]:
    """Helper: list available input devices, highlighting monitor sinks."""
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            devices.append({'index': i, 'name': d['name'], 'device': d})
    return devices
