"""Shared memory layout for the audio daemon.

Defines the binary struct layout for zero-copy IPC between the audio
daemon and consumers (wallpaper, debug visualizer, plugins).

Writer side (daemon): ShmWriter.write_snapshot() + write_event()
Reader side (consumer): ShmReader.read_snapshot() + read_new_events()

Layout is computed at runtime based on spectrum bin count and band names.
Schema is JSON-serializable for dbus GetSchema() negotiation.

No synchronization — atomic sequence counter for staleness detection.
"""

from __future__ import annotations

import mmap
import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from ._types import AudioSnapshot, BeatEvent, BandState

SHM_NAME = 'flame-sheep-audio'
SHM_SIZE = 16384
MAX_BINS = 1025     # FFT max (2048/2+1), CQT is 108
MAX_BANDS = 8
EVENT_RING_SIZE = 32

# Band entry: name[16] + 6 floats (rms, harmonic_rms, slow_rms, slow_harmonic_rms, onset_density, density_delta)
BAND_ENTRY_SIZE = 16 + 6 * 4  # 40 bytes

# Event entry: timestamp_ns(u64) + energy(f32) + kind_id(u8) + pad(3) + event_seq(u64)
EVENT_ENTRY_SIZE = 8 + 4 + 1 + 3 + 8  # 24 bytes

# Mode encoding
MODE_MAP = {'idle': 0, 'energy': 1, 'beat': 2}
MODE_UNMAP = {v: k for k, v in MODE_MAP.items()}


@dataclass
class ShmLayout:
    """Computed field offsets for the shared memory region."""
    n_bins: int
    band_names: list[str]
    # All offsets in bytes
    seq: int = 0                    # uint64
    spectrum: int = 0               # float32[n_bins]
    stability: int = 0             # float32[n_bins]
    sustained: int = 0             # float32[n_bins]
    centroid: int = 0              # float32
    centroid_delta: int = 0        # float32
    centroid_rms: int = 0          # float32
    centroid_harmonic_rms: int = 0 # float32
    slow_centroid_harmonic_rms: int = 0  # float32
    percussiveness: int = 0        # float32
    spectral_novelty: int = 0      # float32
    section_change: int = 0        # float32
    bpm: int = 0                   # float32
    effective_bpm: int = 0         # float32
    tempo_confidence: int = 0      # float32
    tempo_saturated: int = 0       # uint8
    mode: int = 0                  # uint8
    break_intensity: int = 0       # float32
    n_bands: int = 0               # uint32
    bands: int = 0                 # BandEntry[MAX_BANDS]
    event_head: int = 0            # uint32
    event_ring: int = 0            # EventEntry[EVENT_RING_SIZE]
    actual_n_bins: int = 0         # uint32 (stored bin count for consumer)
    total_size: int = 0


def compute_layout(n_bins: int, band_names: list[str]) -> ShmLayout:
    """Compute byte offsets for all fields given bin count and band config."""
    layout = ShmLayout(n_bins=n_bins, band_names=list(band_names))
    off = 0

    layout.seq = off;               off += 8   # uint64
    layout.actual_n_bins = off;      off += 4   # uint32
    layout.spectrum = off;           off += n_bins * 4
    layout.stability = off;          off += n_bins * 4
    layout.sustained = off;          off += n_bins * 4
    layout.centroid = off;           off += 4
    layout.centroid_delta = off;     off += 4
    layout.centroid_rms = off;       off += 4
    layout.centroid_harmonic_rms = off; off += 4
    layout.slow_centroid_harmonic_rms = off; off += 4
    layout.percussiveness = off;     off += 4
    layout.spectral_novelty = off;   off += 4
    layout.section_change = off;     off += 4
    layout.bpm = off;                off += 4
    layout.effective_bpm = off;      off += 4
    layout.tempo_confidence = off;   off += 4
    layout.tempo_saturated = off;    off += 1
    layout.mode = off;               off += 1
    off += 2  # padding to align float32
    layout.break_intensity = off;    off += 4
    layout.n_bands = off;            off += 4
    layout.bands = off;              off += MAX_BANDS * BAND_ENTRY_SIZE
    layout.event_head = off;         off += 4
    layout.event_ring = off;         off += EVENT_RING_SIZE * EVENT_ENTRY_SIZE
    layout.total_size = off

    assert off <= SHM_SIZE, f'Layout exceeds SHM_SIZE: {off} > {SHM_SIZE}'
    return layout


def generate_schema(layout: ShmLayout) -> dict:
    """Generate JSON-serializable schema for dbus GetSchema()."""
    return {
        'version': 1,
        'shm_name': SHM_NAME,
        'size': SHM_SIZE,
        'n_bins': layout.n_bins,
        'band_names': layout.band_names,
        'fields': {
            'seq': {'offset': layout.seq, 'type': 'u64', 'count': 1},
            'n_bins': {'offset': layout.actual_n_bins, 'type': 'u32', 'count': 1},
            'spectrum': {'offset': layout.spectrum, 'type': 'f32', 'count': layout.n_bins},
            'stability': {'offset': layout.stability, 'type': 'f32', 'count': layout.n_bins},
            'sustained': {'offset': layout.sustained, 'type': 'f32', 'count': layout.n_bins},
            'centroid': {'offset': layout.centroid, 'type': 'f32', 'count': 1},
            'centroid_delta': {'offset': layout.centroid_delta, 'type': 'f32', 'count': 1},
            'centroid_rms': {'offset': layout.centroid_rms, 'type': 'f32', 'count': 1},
            'centroid_harmonic_rms': {'offset': layout.centroid_harmonic_rms, 'type': 'f32', 'count': 1},
            'slow_centroid_harmonic_rms': {'offset': layout.slow_centroid_harmonic_rms, 'type': 'f32', 'count': 1},
            'percussiveness': {'offset': layout.percussiveness, 'type': 'f32', 'count': 1},
            'spectral_novelty': {'offset': layout.spectral_novelty, 'type': 'f32', 'count': 1},
            'section_change': {'offset': layout.section_change, 'type': 'f32', 'count': 1},
            'bpm': {'offset': layout.bpm, 'type': 'f32', 'count': 1},
            'effective_bpm': {'offset': layout.effective_bpm, 'type': 'f32', 'count': 1},
            'tempo_confidence': {'offset': layout.tempo_confidence, 'type': 'f32', 'count': 1},
            'tempo_saturated': {'offset': layout.tempo_saturated, 'type': 'u8', 'count': 1},
            'mode': {'offset': layout.mode, 'type': 'u8', 'count': 1},
            'break_intensity': {'offset': layout.break_intensity, 'type': 'f32', 'count': 1},
        },
        'bands': {
            'offset': layout.bands,
            'n_bands_offset': layout.n_bands,
            'stride': BAND_ENTRY_SIZE,
            'max_bands': MAX_BANDS,
            'entry_fields': [
                {'name': 'name', 'rel_offset': 0, 'type': 'char16'},
                {'name': 'rms', 'rel_offset': 16, 'type': 'f32'},
                {'name': 'harmonic_rms', 'rel_offset': 20, 'type': 'f32'},
                {'name': 'slow_rms', 'rel_offset': 24, 'type': 'f32'},
                {'name': 'slow_harmonic_rms', 'rel_offset': 28, 'type': 'f32'},
                {'name': 'onset_density', 'rel_offset': 32, 'type': 'f32'},
                {'name': 'density_delta', 'rel_offset': 36, 'type': 'f32'},
            ],
        },
        'events': {
            'head_offset': layout.event_head,
            'ring_offset': layout.event_ring,
            'ring_size': EVENT_RING_SIZE,
            'entry_size': EVENT_ENTRY_SIZE,
            'entry_fields': [
                {'name': 'timestamp_ns', 'rel_offset': 0, 'type': 'u64'},
                {'name': 'energy', 'rel_offset': 8, 'type': 'f32'},
                {'name': 'kind_id', 'rel_offset': 12, 'type': 'u8'},
                {'name': 'event_seq', 'rel_offset': 16, 'type': 'u64'},
            ],
        },
    }


class ShmWriter:
    """Daemon-side shared memory writer."""

    def __init__(self, layout: ShmLayout, buf: mmap.mmap | memoryview):
        self._layout = layout
        self._buf = buf
        self._seq = 0
        self._event_head = 0
        self._event_seq = 0
        # Build band name → kind_id mapping
        self._band_ids = {name: i for i, name in enumerate(layout.band_names)}
        # Write static fields
        struct.pack_into('I', self._buf, layout.actual_n_bins, layout.n_bins)
        struct.pack_into('I', self._buf, layout.n_bands, len(layout.band_names))
        # Write band names
        for i, name in enumerate(layout.band_names):
            name_bytes = name.encode('utf-8')[:15].ljust(16, b'\x00')
            off = layout.bands + i * BAND_ENTRY_SIZE
            self._buf[off:off + 16] = name_bytes

    def write_snapshot(self, snap: AudioSnapshot) -> None:
        """Write continuous audio state to shared memory."""
        L = self._layout
        buf = self._buf

        # Spectrum arrays
        n = min(len(snap.spectrum), L.n_bins)
        if n > 0:
            buf[L.spectrum:L.spectrum + n * 4] = snap.spectrum[:n].tobytes()
        if len(snap.stability) >= n and n > 0:
            buf[L.stability:L.stability + n * 4] = snap.stability[:n].tobytes()
        if len(snap.sustained) >= n and n > 0:
            buf[L.sustained:L.sustained + n * 4] = snap.sustained[:n].tobytes()

        # Scalars
        struct.pack_into('f', buf, L.centroid, snap.centroid)
        struct.pack_into('f', buf, L.centroid_delta, snap.centroid_delta)
        struct.pack_into('f', buf, L.centroid_rms, snap.centroid_rms)
        struct.pack_into('f', buf, L.centroid_harmonic_rms, snap.centroid_harmonic_rms)
        struct.pack_into('f', buf, L.slow_centroid_harmonic_rms, snap.slow_centroid_harmonic_rms)
        struct.pack_into('f', buf, L.percussiveness, snap.percussiveness)
        struct.pack_into('f', buf, L.spectral_novelty, snap.spectral_novelty)
        struct.pack_into('f', buf, L.section_change, snap.section_change)
        struct.pack_into('f', buf, L.bpm, snap.bpm)
        struct.pack_into('f', buf, L.effective_bpm, snap.effective_bpm)
        struct.pack_into('f', buf, L.tempo_confidence, snap.tempo_confidence)
        struct.pack_into('B', buf, L.tempo_saturated, 1 if snap.tempo_saturated else 0)
        struct.pack_into('B', buf, L.mode, MODE_MAP.get(snap.mode, 0))
        struct.pack_into('f', buf, L.break_intensity, snap.break_intensity)

        # Per-band state
        for i, name in enumerate(L.band_names):
            bs = snap.bands.get(name)
            if bs is None:
                continue
            off = L.bands + i * BAND_ENTRY_SIZE + 16  # skip name
            struct.pack_into('ffffff', buf, off,
                             bs.rms, bs.harmonic_rms, bs.slow_rms,
                             bs.slow_harmonic_rms, bs.onset_density, bs.density_delta)

        # Bump sequence counter last (ensures consumer sees consistent data)
        self._seq += 1
        struct.pack_into('Q', buf, L.seq, self._seq)

    def write_event(self, event: BeatEvent) -> None:
        """Write a beat event to the ring buffer."""
        L = self._layout
        idx = self._event_head % EVENT_RING_SIZE
        off = L.event_ring + idx * EVENT_ENTRY_SIZE
        self._event_seq += 1

        kind_id = self._band_ids.get(event.kind, 255)
        ts = time.monotonic_ns()

        struct.pack_into('QfB3xQ', self._buf, off,
                         ts, event.energy, kind_id, self._event_seq)

        self._event_head += 1
        struct.pack_into('I', self._buf, L.event_head, self._event_head)


class ShmReader:
    """Consumer-side shared memory reader."""

    def __init__(self, layout: ShmLayout, buf: mmap.mmap | memoryview):
        self._layout = layout
        self._buf = buf
        self._last_seq = 0
        self._last_event_head = 0
        self._last_event_seq = 0
        # Build kind_id → name mapping
        self._kind_names = {i: name for i, name in enumerate(layout.band_names)}

    def has_new_data(self) -> bool:
        """Check if the daemon has written new data since last read."""
        seq = struct.unpack_from('Q', self._buf, self._layout.seq)[0]
        return seq != self._last_seq

    def read_snapshot(self) -> AudioSnapshot:
        """Read current continuous state from shared memory."""
        L = self._layout
        buf = self._buf
        n = L.n_bins

        # Read seq first
        self._last_seq = struct.unpack_from('Q', buf, L.seq)[0]

        # Arrays
        spectrum = np.frombuffer(buf, dtype=np.float32, count=n, offset=L.spectrum).copy()
        stability = np.frombuffer(buf, dtype=np.float32, count=n, offset=L.stability).copy()
        sustained = np.frombuffer(buf, dtype=np.float32, count=n, offset=L.sustained).copy()

        # Scalars
        centroid = struct.unpack_from('f', buf, L.centroid)[0]
        centroid_delta = struct.unpack_from('f', buf, L.centroid_delta)[0]
        centroid_rms = struct.unpack_from('f', buf, L.centroid_rms)[0]
        centroid_h_rms = struct.unpack_from('f', buf, L.centroid_harmonic_rms)[0]
        slow_c_h_rms = struct.unpack_from('f', buf, L.slow_centroid_harmonic_rms)[0]
        percussiveness = struct.unpack_from('f', buf, L.percussiveness)[0]
        spectral_novelty = struct.unpack_from('f', buf, L.spectral_novelty)[0]
        section_change = struct.unpack_from('f', buf, L.section_change)[0]
        bpm = struct.unpack_from('f', buf, L.bpm)[0]
        effective_bpm = struct.unpack_from('f', buf, L.effective_bpm)[0]
        tempo_confidence = struct.unpack_from('f', buf, L.tempo_confidence)[0]
        tempo_saturated = bool(struct.unpack_from('B', buf, L.tempo_saturated)[0])
        mode_id = struct.unpack_from('B', buf, L.mode)[0]
        mode = MODE_UNMAP.get(mode_id, 'idle')
        break_intensity = struct.unpack_from('f', buf, L.break_intensity)[0]

        # Bands
        bands = {}
        for i, name in enumerate(L.band_names):
            off = L.bands + i * BAND_ENTRY_SIZE + 16
            rms, h_rms, s_rms, s_h_rms, density, d_delta = struct.unpack_from(
                'ffffff', buf, off)
            bands[name] = BandState(
                rms=rms, harmonic_rms=h_rms, slow_rms=s_rms,
                slow_harmonic_rms=s_h_rms, onset_density=density,
                density_delta=d_delta)

        # Events
        events = self.read_new_events()

        return AudioSnapshot(
            events=events,
            spectrum=spectrum,
            waveform=np.zeros(0, dtype=np.float32),  # not in shmem
            stability=stability,
            sustained=sustained,
            bands=bands,
            centroid=centroid,
            centroid_delta=centroid_delta,
            centroid_rms=centroid_rms,
            centroid_harmonic_rms=centroid_h_rms,
            slow_centroid_harmonic_rms=slow_c_h_rms,
            percussiveness=percussiveness,
            spectral_novelty=spectral_novelty,
            section_change=section_change,
            bpm=bpm,
            effective_bpm=effective_bpm,
            tempo_confidence=tempo_confidence,
            tempo_saturated=tempo_saturated,
            mode=mode,
            break_intensity=break_intensity,
        )

    def read_new_events(self) -> list[BeatEvent]:
        """Read new events from the ring buffer since last call."""
        L = self._layout
        head = struct.unpack_from('I', self._buf, L.event_head)[0]

        if head == self._last_event_head:
            return []

        events = []
        # Read from last_event_head to head (may wrap)
        n_new = min(head - self._last_event_head, EVENT_RING_SIZE)
        start = head - n_new

        for i in range(n_new):
            idx = (start + i) % EVENT_RING_SIZE
            off = L.event_ring + idx * EVENT_ENTRY_SIZE
            ts, energy, kind_id, event_seq = struct.unpack_from(
                'QfB3xQ', self._buf, off)

            # Skip events we've already seen (by event_seq)
            if event_seq <= self._last_event_seq:
                continue

            kind = self._kind_names.get(kind_id, f'unknown_{kind_id}')
            events.append(BeatEvent(kind=kind, energy=energy))
            self._last_event_seq = event_seq

        self._last_event_head = head
        return events
