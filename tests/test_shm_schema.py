"""Contract tests for shm_layout.generate_schema (the dbus GetSchema payload).

The schema must carry everything the client needs to interpret the shm
spectrum — including per-bin center frequencies — so the client never has to
reconstruct daemon-internal state (it used to build a CqtEngine just to read
bin_centers, which silently went wrong if the daemon's engine params differed).
"""
from __future__ import annotations

import json

import numpy as np

from flame_sheep_audio.shm_layout import compute_layout, generate_schema


def _layout():
    return compute_layout(108, ['low', 'mid', 'high'])


def test_schema_carries_bin_freqs():
    freqs = np.linspace(20, 20000, 108).astype(np.float32)
    schema = generate_schema(_layout(), freqs)
    assert schema['bin_freqs'] is not None
    assert len(schema['bin_freqs']) == 108
    assert schema['bin_freqs'][0] == float(freqs[0])


def test_schema_bin_freqs_are_json_safe():
    # Must be plain floats, not numpy scalars, or the dbus JSON dump breaks.
    freqs = np.linspace(20, 20000, 108).astype(np.float32)
    schema = generate_schema(_layout(), freqs)
    assert all(type(f) is float for f in schema['bin_freqs'])
    round_tripped = json.loads(json.dumps(schema))
    assert round_tripped['bin_freqs'][:3] == schema['bin_freqs'][:3]
