"""Audio-level AGC — applied at PCM source, before any analysis.

Purpose: normalize away the mix engineer's mastering decisions and the
listener's speaker volume, leave the musical dynamics intact. Every
downstream consumer (CQT, FFT, RMS, beat detectors, energy analyzer,
mode detector, RNN) sees the same level baseline regardless of how
loud the user has their speakers turned up.

Design choices (per the AGC architecture discussion):
- **Sample-level noise floor**: PCM samples below `noise_floor` are
  zeroed. Single point where "this is silence" is defined; downstream
  detectors don't need their own min_flux/min_amplitude thresholds.
- **Multi-minute EMA**: tracks song-master + listener-baseline
  loudness, not within-song dynamics. Choruses and quiet verses pass
  through untouched as musical intensity.
- **Skip update on silence**: silent blocks don't drag the EMA toward
  zero. Pauses between songs preserve the previous loudness baseline.
- **Persistence across daemon restarts**: saved EMA state means each
  launch starts at the user's typical level baseline, not from a cold
  default. Self-calibrates over weeks of use.
- **Inter-song dynamics preserved**: no per-song reset. The 2-minute
  EMA naturally smooths cross-song mastering differences without
  fully erasing them, which is roughly what the ear does.

Constants worth knowing:
- `target_rms = 0.1`     ≈ -20 dBFS  — typical "normalized" audio level
- `noise_floor = 0.001`  ≈ -60 dBFS — well below quiet audio
- `time_constant = 120s` — multi-minute, ignores section dynamics
- `save_interval = 30s`  — single-float disk write, basically free
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_TARGET_RMS: float = 0.1
_DEFAULT_NOISE_FLOOR: float = 0.001
_DEFAULT_TIME_CONSTANT_SEC: float = 120.0
_DEFAULT_SAVE_INTERVAL_SEC: float = 30.0


def _state_path_default() -> Path:
    return (Path.home() / '.local' / 'share' / 'flame-sheep'
            / 'agc_state.json')


class AudioLevelAgc:
    """Process PCM blocks; return normalized PCM and report current
    gain. Sample-rate-aware so the time constant means what it says
    in seconds, not "samples."
    """

    def __init__(self,
                 sample_rate: int,
                 target_rms: float = _DEFAULT_TARGET_RMS,
                 noise_floor: float = _DEFAULT_NOISE_FLOOR,
                 time_constant_sec: float = _DEFAULT_TIME_CONSTANT_SEC,
                 save_interval_sec: float = _DEFAULT_SAVE_INTERVAL_SEC,
                 persist_path: Path | None | str = 'default',
                 ) -> None:
        self.sample_rate = int(sample_rate)
        self.target_rms = float(target_rms)
        self.noise_floor = float(noise_floor)
        self.time_constant_sec = float(time_constant_sec)
        self.save_interval_sec = float(save_interval_sec)

        if persist_path == 'default':
            persist_path = _state_path_default()
        self.persist_path: Path | None = (Path(persist_path)
                                          if persist_path else None)

        # Load saved slow_rms if available, else bootstrap at target.
        self.slow_rms: float = self._load_or_bootstrap()
        # Last applied gain (published for diagnostics).
        self.current_gain: float = self._compute_gain(self.slow_rms)
        self._last_save_monotonic = time.monotonic()
        # Diagnostic: periodic gain/slow_rms log so the operator can
        # see whether AGC has settled, where it's settled, and how it's
        # tracking the audio's actual level. Without this, a quiet
        # detector gives no information about whether the gain is the
        # cause.
        self._last_diag_monotonic = self._last_save_monotonic
        self._diag_interval_sec = 5.0

    # ---- public API ----

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Normalize a block of PCM samples in-place-safe (returns new
        array). pcm: float32 1D, range ~[-1, 1]."""
        if pcm.size == 0:
            return pcm
        # 1. Sample-level noise floor — zero out anything quieter than
        # noise_floor in absolute terms.
        out = np.where(np.abs(pcm) < self.noise_floor, 0.0, pcm).astype(
            np.float32, copy=False)

        # 2. Compute block RMS over non-zero samples (the zeros above
        # represent silence and shouldn't count). If everything zeroed,
        # block is silent — skip the EMA update.
        nonzero_mask = out != 0.0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero > 0:
            block_rms = float(np.sqrt(
                (out * out).sum() / max(n_nonzero, 1)))
            # Only update EMA on blocks that have real audio content.
            if block_rms > self.noise_floor:
                # alpha calibrated so the EMA reaches 63% of a step
                # change after time_constant_sec.
                block_duration = pcm.size / self.sample_rate
                alpha = 1.0 - math.exp(-block_duration
                                        / self.time_constant_sec)
                self.slow_rms = ((1.0 - alpha) * self.slow_rms
                                  + alpha * block_rms)

        # 3. Apply gain. Clamp the denominator at noise_floor so very
        # quiet baselines don't produce extreme gains.
        self.current_gain = self._compute_gain(self.slow_rms)
        out = out * self.current_gain

        # 4. Periodic persistence so crashes only lose a few minutes
        # of adaptation, not a session.
        now = time.monotonic()
        if (self.persist_path is not None
                and now - self._last_save_monotonic
                >= self.save_interval_sec):
            self._save()
            self._last_save_monotonic = now

        # 5. Periodic diagnostic log so the operator can see what the
        # AGC is doing without instrumenting each consumer.
        if now - self._last_diag_monotonic >= self._diag_interval_sec:
            log.debug(
                '[agc] slow_rms=%.4f gain=%.2f target=%.3f noise_floor=%.4f',
                self.slow_rms, self.current_gain, self.target_rms,
                self.noise_floor)
            self._last_diag_monotonic = now

        return out

    def save(self) -> None:
        """Force a save now. Useful for graceful shutdown."""
        if self.persist_path is not None:
            self._save()

    # ---- internals ----

    def _compute_gain(self, slow_rms: float) -> float:
        return self.target_rms / max(slow_rms, self.noise_floor)

    def _load_or_bootstrap(self) -> float:
        if self.persist_path is None or not self.persist_path.exists():
            return self.target_rms  # bootstrap at target → gain = 1.0
        try:
            data = json.loads(self.persist_path.read_text())
            v = float(data['slow_rms'])
            # Sanity: reject obviously-broken saved values.
            if v <= 0 or v > 10.0:
                log.warning('[agc] saved slow_rms=%g out of range; '
                            'bootstrapping at target', v)
                return self.target_rms
            log.info('[agc] loaded slow_rms=%.5f from %s', v,
                     self.persist_path)
            return v
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            log.warning('[agc] failed to load %s: %s; bootstrapping',
                        self.persist_path, e)
            return self.target_rms

    def _save(self) -> None:
        assert self.persist_path is not None
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'slow_rms': float(self.slow_rms),
                'target_rms': float(self.target_rms),
                'noise_floor': float(self.noise_floor),
                'time_constant_sec': float(self.time_constant_sec),
                # Wall-clock so a stale state file is recognizable.
                'saved_unix': time.time(),
            }
            tmp = self.persist_path.with_suffix(self.persist_path.suffix + '.tmp')
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.persist_path)
        except OSError as e:
            log.warning('[agc] failed to save %s: %s', self.persist_path, e)
