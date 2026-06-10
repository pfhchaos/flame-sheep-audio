"""BeatNet particle-filter tempo tracker.

Wraps a `BeatNetLiveDetector` instance and exposes its PF tempo state
through the daemon's `TempoTrackerBase` interface. Smooths the PF's
~23 BPM raw jitter via a two-stage EMA so the wallpaper sees a stable
BPM + a clean first derivative.

Architecture: the detector is the source of audio + activations + PF
state. This tracker just reads the PF's `current_tempo()` snapshot
each tick. `feed()` / `feed_audio()` from the daemon's audio loop
become "ticks" — no audio is reprocessed.

When `kind = "beatnet_lite"` in `cfg.detector`, the processor
substitutes this tracker for BTrack so the daemon's tempo output
becomes "PF tempo, smoothed." BTrack stays available as a hint
source — see `confidence_driven_hint` (callable from the audio loop).
"""
from __future__ import annotations

import logging

from .tempo import TempoTrackerBase
from ._constants import HOP_SIZE, SAMPLE_RATE

log = logging.getLogger(__name__)


class BeatNetTempoTracker(TempoTrackerBase):
    """PF-driven tempo with EMA smoothing + first derivative.

    Parameters
    ----------
    detector : BeatNetLiveDetector
        Active detector whose PF state we read. Must have
        `use_particle_filter=True`; otherwise `current_tempo()`
        returns None and this tracker stays at its default 120 BPM.
    smoothing_seconds : float
        EMA time constant for the raw PF tempo. Default 2.0s.
        Kills the ~23 BPM jitter (alternating between adjacent
        quantum bins) to a few BPM.
    delta_smoothing_seconds : float
        EMA time constant for the BPM derivative. Default 5.0s.
        Longer than smoothing_seconds because derivative noise scales
        as 1/dt; need a wider window for a clean rate.
    default_bpm : float
        Reported BPM before the PF has converged or when its
        confidence is below `min_confidence`. Default 120.
    min_confidence : float
        Floor on PF confidence to consume its output. Default 0.05.
        Below this, the smoothed BPM holds its previous value rather
        than tracking noise.
    """

    def __init__(self,
                 detector,
                 smoothing_seconds: float = 2.0,
                 delta_smoothing_seconds: float = 5.0,
                 default_bpm: float = 120.0,
                 min_confidence: float = 0.05,
                 hop_size: int = HOP_SIZE,
                 sample_rate: int = SAMPLE_RATE):
        self._det = detector
        self._hop_sec = hop_size / sample_rate
        self._tau_bpm = max(self._hop_sec, smoothing_seconds)
        self._tau_delta = max(self._hop_sec, delta_smoothing_seconds)
        self._default_bpm = float(default_bpm)
        self._min_conf = float(min_confidence)

        # State.
        self._smoothed_bpm = self._default_bpm
        self._smoothed_delta = 0.0
        self._last_confidence = 0.0
        self._frame_count = 0
        # Saturation: PF saturates if observed BPM consistently rails
        # against min/max. Track over a short window.
        self._saturated_count = 0

    # ---- daemon ticks (feed) ----

    def feed(self, onset_strength: float, onset_density: float = 0.0) -> None:
        """Tick. Audio path runs via the detector's feed_audio_hop;
        this just samples the PF state and updates EMAs."""
        self._tick()

    def feed_audio(self, hop) -> None:
        """Same as `feed` — audio is already being processed by the
        shared BeatNetLiveDetector. Implemented for interface parity
        in case the processor calls feed_audio instead."""
        self._tick()

    def _tick(self) -> None:
        snap = self._det.current_tempo()
        self._frame_count += 1
        if snap is None:
            return
        confidence = float(snap.get('confidence', 0.0))
        self._last_confidence = confidence
        if confidence < self._min_conf:
            # Hold previous smoothed value but still decay the delta
            # toward zero so the daemon doesn't report a stale rate
            # when tempo lock is lost.
            alpha_d = self._hop_sec / self._tau_delta
            self._smoothed_delta = (1.0 - alpha_d) * self._smoothed_delta
            return

        raw_bpm = float(snap.get('bpm', self._default_bpm))
        # Saturation detection: PF clipping against its 55/215 limits.
        if raw_bpm <= 55.5 or raw_bpm >= 214.5:
            self._saturated_count = min(self._saturated_count + 1, 100)
        else:
            self._saturated_count = max(0, self._saturated_count - 1)

        # EMA for BPM.
        alpha_b = self._hop_sec / self._tau_bpm
        prev_bpm = self._smoothed_bpm
        new_bpm = (1.0 - alpha_b) * prev_bpm + alpha_b * raw_bpm
        # Raw step derivative in BPM/sec.
        step_delta = (new_bpm - prev_bpm) / self._hop_sec
        # EMA for derivative.
        alpha_d = self._hop_sec / self._tau_delta
        self._smoothed_delta = (
            (1.0 - alpha_d) * self._smoothed_delta + alpha_d * step_delta)
        self._smoothed_bpm = new_bpm

    # ---- properties (TempoTrackerBase contract) ----

    @property
    def bpm(self) -> float:
        return self._smoothed_bpm

    @property
    def effective_bpm(self) -> float:
        # No prior blending here — PF tempo is already a learned-prior
        # estimate. Return the smoothed value directly.
        return self._smoothed_bpm

    @property
    def confidence(self) -> float:
        return self._last_confidence

    @property
    def phase(self) -> float:
        # PF doesn't expose phase directly. Could derive from the
        # particle median's position within its beat-period if needed.
        return 0.0

    @property
    def bpm_delta(self) -> float:
        """First derivative of tempo in BPM/sec (smoothed via EMA)."""
        return self._smoothed_delta

    @property
    def saturated(self) -> bool:
        """True if PF tempo has been railed against its [min_bpm, max_bpm]
        boundary for ~hop_sec * saturated_count seconds."""
        return self._saturated_count > 30

    # ---- lifecycle ----

    def reset(self) -> None:
        self._smoothed_bpm = self._default_bpm
        self._smoothed_delta = 0.0
        self._last_confidence = 0.0
        self._frame_count = 0
        self._saturated_count = 0

    def song_started(self) -> None:
        """Soft reset on song change. Forward to the detector so its
        PF re-initializes too; otherwise the previous song's tempo
        lock persists into the new song."""
        self.reset()
        if hasattr(self._det, 'reset_bands'):
            self._det.reset_bands()

    def hint_tempo(self, bpm: float) -> None:
        """Forward a tempo hint to the detector's PF (multi-octave by
        default so the PF picks the right octave from activations).
        Useful for: song-start MPRIS metadata, confidence-driven
        warm-starts from BTrack."""
        if hasattr(self._det, 'hint_tempo_octaves'):
            n_ok = self._det.hint_tempo_octaves(bpm)
            log.debug('[hint] PF multi-octave injection: bpm=%.1f, '
                      '%d/3 octaves accepted (others outside PF range)',
                      bpm, n_ok)
        else:
            log.debug('[hint] detector has no hint_tempo_octaves; '
                      'tempo=%.1f dropped', bpm)
