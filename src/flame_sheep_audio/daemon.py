"""Audio engine daemon — runs AudioProcessor and publishes via shmem + dbus.

Usage:
    python -m flame_sheep_audio.daemon [--device NAME]

Consumers connect via:
    1. dbus GetSchema() → JSON layout description
    2. mmap shared memory region
    3. Poll shmem for continuous data + event ring buffer
    4. Listen dbus signals for discrete events (backup)
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import logging
import os
import signal
import sys
import time
from pathlib import Path

from .shm_layout import (
    SHM_NAME, SHM_SIZE,
    compute_layout, generate_schema, ShmWriter,
)

log = logging.getLogger(__name__)


class AudioDaemon:
    """Audio analysis daemon with shmem + dbus output."""

    def __init__(self, device: str | int | None = None):
        from .processor import AudioProcessor
        from ._band_config import default_band_config
        from ._cqt_engine import CqtEngine

        engine = CqtEngine()
        log.info('spectrum engine: CQT')

        self._processor = AudioProcessor(device=device, spectrum_engine=engine)
        self._band_config = default_band_config()
        n_bins = engine.n_bins

        # Shared memory — use /dev/shm directly to avoid multiprocessing
        # resource tracker, which can unlink shm from other processes on crash.
        import mmap as _mmap
        self._shm_path = f'/dev/shm/{SHM_NAME}'
        # Clean up stale shm file
        if os.path.exists(self._shm_path):
            os.unlink(self._shm_path)
        fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, SHM_SIZE)
        self._mmap = _mmap.mmap(fd, SHM_SIZE)
        os.close(fd)

        self._layout = compute_layout(n_bins, list(self._band_config.all_band_names))
        self._writer = ShmWriter(self._layout, self._mmap)
        atexit.register(self._cleanup)

        # D-Bus service
        schema = generate_schema(self._layout)
        from .dbus_service import AudioDbusService
        self._dbus = AudioDbusService(schema, SHM_NAME)

        self._running = False
        self._last_bpm = 0.0
        self._last_mode = ''

    def run(self) -> None:
        """Run the daemon main loop. Blocks until stopped."""
        self._processor.start()
        self._dbus.start()
        self._running = True

        log.info('audio daemon started (pid=%d)', __import__('os').getpid())

        try:
            while self._running:
                snap = self._processor.drain_blocking(timeout=0.1)

                # Write continuous state to shmem
                self._writer.write_snapshot(snap)

                # Write events to ring buffer + emit dbus signals
                for event in snap.events:
                    self._writer.write_event(event)
                    self._dbus.emit_beat(event.kind, event.energy)

                # Emit mode/tempo changes
                self._dbus.emit_mode_change(snap.mode)
                if abs(snap.bpm - self._last_bpm) > 1.0:
                    self._last_bpm = snap.bpm
                    self._dbus.emit_tempo_update(snap.bpm, snap.tempo_confidence)

        except KeyboardInterrupt:
            log.info('audio daemon interrupted')
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._processor.stop()
        self._dbus.stop()
        log.info('audio daemon stopped')

    def _cleanup(self) -> None:
        """Cleanup shared memory on exit."""
        try:
            self._mmap.close()
        except Exception:
            pass
        try:
            os.unlink(self._shm_path)
        except Exception:
            pass


def _acquire_single_instance_lock() -> bool:
    """Pidfile + flock so a second daemon refuses to start instead of
    fighting the first for the audio device.

    Returns True if we got the lock (proceed); False if another
    instance already holds it (caller should exit 0 — not an error,
    just a no-op duplicate). The fd stays open at module scope; the
    kernel releases it on process exit.

    Mirror of flame_sheep.process_util.acquire_single_instance_lock —
    inlined here because flame_sheep_audio can't import flame_sheep
    (the dependency runs the other way). Keep the two in sync."""
    xdg = os.environ.get('XDG_RUNTIME_DIR')
    base = Path(xdg) if xdg else Path(f'/tmp/flame-sheep-{os.getuid()}')
    runtime_dir = base / 'flame-sheep'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / 'audio.pid'
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            holder = os.read(fd, 32).decode('ascii', errors='replace').strip()
        except OSError:
            holder = '?'
        os.close(fd)
        print(f'[audio daemon] another instance already running '
              f'(pid={holder}); exiting', file=sys.stderr)
        return False
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, f'{os.getpid()}\n'.encode('ascii'))
    # Stash on the module so GC can't close it.
    global _lock_fd
    _lock_fd = fd
    return True


_lock_fd: int | None = None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Flame Sheep Audio Daemon')
    parser.add_argument('--device', type=str, default=None,
                        help='audio device name or index')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    # Refuse to start a duplicate. Two daemons fighting for the same
    # PortAudio device would either cause one to fail with a cryptic
    # ALSA error or — worse — both bind successfully and produce
    # interleaved garbage to the same shmem.
    if not _acquire_single_instance_lock():
        sys.exit(0)

    # CLI --device wins; otherwise fall back to cfg.input.device
    # (audio.toml [input] device = "..."); else None → sounddevice default.
    from .config import cfg
    device = args.device
    if device is None:
        device = getattr(cfg.input, 'device', None)
    daemon = AudioDaemon(device=device)

    def _handle_signal(signum, frame):
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    daemon.run()


if __name__ == '__main__':
    main()
