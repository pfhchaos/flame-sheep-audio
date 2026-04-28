"""Audio engine configuration — loads tuning constants from TOML.

All audio analysis tuning constants live here. Components read from
the global `cfg` object instead of defining their own class-level constants.

Config is loaded from (in order):
  1. Built-in defaults (DEFAULTS dict below)
  2. ~/.config/flame-sheep/audio.toml (user overrides)

Call `cfg.reload()` to re-read from disk.
"""

import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / '.config' / 'flame-sheep' / 'audio.toml'

DEFAULTS = {
    'detection': {
        'base_threshold': 1.5,
        'kick_threshold': 3.5,
        'cooldown_frames': 12,
        'kick_cooldown_frames': 8,
        'kick_cooldown_beat_fraction': 0.4,
        'stability_scaling': 1.0,
        'sharpness': 3.0,
        'min_flux': 1e-7,
    },
    'stability': {
        'fast_alpha': 0.95,
        'slow_alpha': 0.995,
    },
    'energy': {
        'rms_alpha': 0.9,
        'centroid_alpha': 0.85,
        'percussiveness_alpha': 0.92,
        'slow_attack_alpha': 0.995,    # ~2s half-life at 93fps (HOP cadence)
        'slow_release_alpha': 0.98,    # ~0.5s half-life
    },
    'density': {
        'window': 1.0,
        'delta_window': 2.0,
        'alpha': 0.9,
    },
    'tempo': {
        'min_bpm': 60,
        'max_bpm': 400,
        'default_bpm': 120,
    },
    'breaks': {
        'enabled': False,
        'quiet_threshold_frames': 60,
        'bass_quiet_threshold_frames': 45,
        'drop_energy_ratio': 0.15,
        'subbass_drop_ratio': 0.10,
        'min_kicks_before_break': 8,
        'cooldown_seconds': 15.0,
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

    def __init__(self):
        self._data = dict(DEFAULTS)
        self._ns = _to_namespace(self._data)
        self._reload_callbacks: list = []
        self._load_user_config()

    def _load_user_config(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'rb') as f:
                    user = tomllib.load(f)
                self._data = _deep_merge(DEFAULTS, user)
                self._ns = _to_namespace(self._data)
                log.info(f'Loaded audio config from {CONFIG_PATH}')
            except Exception:
                log.exception(f'Failed to load {CONFIG_PATH}, using defaults')

    def on_reload(self, callback):
        """Register a callback to be called after config is reloaded."""
        self._reload_callbacks.append(callback)

    def reload(self):
        """Re-read config from disk and notify listeners."""
        self._data = dict(DEFAULTS)
        self._ns = _to_namespace(self._data)
        self._load_user_config()
        log.info('Audio config reloaded')
        for cb in self._reload_callbacks:
            cb()

    def __getattr__(self, name):
        return getattr(self._ns, name)


# Global config instance
cfg = Config()
