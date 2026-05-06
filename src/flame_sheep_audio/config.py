"""Audio engine configuration — loads tuning constants from TOML.

All audio analysis tuning constants live here. Components read from
the global `cfg` object instead of defining their own class-level constants.

Config is loaded from (in order):
  1. Built-in defaults (DEFAULTS dict below)
  2. ~/.config/flame-sheep/audio.toml (user overrides)

Call `cfg.reload()` to re-read from disk.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / '.config' / 'flame-sheep' / 'audio.toml'

DEFAULTS = {
    'detection': {
        'base_threshold': 1.5,          # flux multiplier (future: Percentile)
        'low_threshold': 3.5,           # low-band flux multiplier (future: Percentile)
        'cooldown': 0.25,               # beats — unified cooldown (1/16th note)
        'stability_scaling': 1.0,       # stability → threshold scaling (raw float)
        'sharpness': 3.0,              # attack sharpness gate (future: Percentile)
        'min_flux': 1e-7,              # absolute noise floor
    },
    'stability': {
        'method': 'median',        # 'ema', 'median' (causal HPSS), or 'shape' (experimental)
        'hpss_ema_window': 0.5,    # beats — 95% decay window for fast stability EMA
        'slow_window': 2.0,        # seconds — section-level stability

        'hpss_time_window': 1.0,   # beats of history for HPSS time median
        'hpss_freq_kernel': 15,    # bins for HPSS frequency median (not tempo-scaled)
    },
    'energy': {
        'rms_smoothing': 0.1,              # seconds — band RMS output smoothing
        'centroid_smoothing': 0.1,         # seconds — spectral centroid smoothing
        'novelty_window': 0.5,             # beats — spectral novelty EMA window
        'percussiveness_method': 'shape',  # 'shape' (spectral distance) or 'flux' (original)
        'slow_attack': 2.0,               # seconds — slow envelope attack
        'slow_release': 0.5,              # seconds — slow envelope release
    },
    'section': {
        'fast_window': 2.0,         # seconds — fast EMA for section change
        'slow_window': 15.0,        # seconds — slow EMA for section change
        'weight_centroid': 1.0,     # tunable axis coefficient
        'weight_energy': 1.0,       # tunable axis coefficient
    },
    'density': {
        'window': 1.0,              # seconds — onset counting window (already in seconds)
        'delta_window': 2.0,        # seconds — density delta lookback (already in seconds)
        'fast_smoothing': 0.1,      # seconds — density output for visualization
        'slow_smoothing': 2.0,      # seconds — density for tempo tracker
    },
    'tempo': {
        'min_bpm': 60,
        'max_bpm': 400,
        'default_bpm': 120,
        'window_seconds': 8.0,
        'update_interval': 0.5,
        'prior_center': 110.0,
        'prior_width': 1.4,
        'smooth_alpha': 0.8,
        'confidence_threshold': 0.15,
        'lock_threshold': 0.5,
        'unlock_threshold': 0.2,
    },
    'breaks': {
        'enabled': True,
        'activation_window': 2.0,           # beats — quiet time before break activates
        'bass_activation_window': 1.5,      # beats — sub-bass quiet time
        'drop_energy_ratio': 0.15,          # ratio (future: Percentile)
        'subbass_drop_ratio': 0.10,         # ratio (future: Percentile)
        'min_lows_before_break': 8,         # count — prevent false triggers on intros
        'cooldown': 15.0,                   # seconds — time after break ends
    },
    'adaptive': {
        'enabled': False,
        'update_interval': 9,
        'anchor_strength': 0.3,
        'repulsion_strength': 0.1,
        'flux_pull_strength': 0.5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursively for nested dicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _to_namespace(d: dict) -> SimpleNamespace:
    """Convert nested dict to nested SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


class Config:
    """Global configuration with dot-access and hot reload."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = dict(DEFAULTS)
        self._ns: SimpleNamespace = _to_namespace(self._data)
        self._reload_callbacks: list[Callable[[], None]] = []
        self._load_user_config()

    def _load_user_config(self) -> None:
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'rb') as f:
                    user = tomllib.load(f)
                self._data = _deep_merge(DEFAULTS, user)
                self._ns = _to_namespace(self._data)
                log.info(f'Loaded audio config from {CONFIG_PATH}')
            except Exception:
                log.exception(f'Failed to load {CONFIG_PATH}, using defaults')

    def on_reload(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called after config is reloaded."""
        self._reload_callbacks.append(callback)

    def reload(self) -> None:
        """Re-read config from disk and notify listeners."""
        self._data = dict(DEFAULTS)
        self._ns = _to_namespace(self._data)
        self._load_user_config()
        log.info('Audio config reloaded')
        for cb in self._reload_callbacks:
            cb()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)


# Global config instance
cfg = Config()
