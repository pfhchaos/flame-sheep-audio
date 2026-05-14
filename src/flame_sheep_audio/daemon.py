"""Audio engine daemon — runs AudioProcessor and publishes via shmem + dbus.

Usage:
    python -m flame_sheep_audio.daemon [--device NAME] [--engine cqt|octave_bank]

Consumers connect via:
    1. dbus GetSchema() → JSON layout description
    2. mmap shared memory region
    3. Poll shmem for continuous data + event ring buffer
    4. Listen dbus signals for discrete events (backup)
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
import time

from .shm_layout import (
    SHM_NAME, SHM_SIZE,
    compute_layout, generate_schema, ShmWriter,
)

log = logging.getLogger(__name__)


class AudioDaemon:
    """Audio analysis daemon with shmem + dbus output."""

    def __init__(self, device: str | int | None = None,
                 spectrum_engine: str = 'cqt'):
        from .processor import AudioProcessor
        from ._band_config import default_band_config

        # Build spectrum engine
        engine = None
        if spectrum_engine == 'cqt':
            try:
                from ._cqt_engine import CqtEngine
                engine = CqtEngine()
                log.info('spectrum engine: CQT')
            except ImportError:
                log.warning('CQT unavailable, falling back to octave bank')
        if engine is None:
            from ._octave_bank import OctaveBankEngine
            engine = OctaveBankEngine()
            log.info('spectrum engine: octave bank')

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
                snap = self._processor.drain()
                if snap is None:
                    time.sleep(0.005)
                    continue

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

                # Sleep to match analysis cadence (~10ms)
                time.sleep(0.005)

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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Flame Sheep Audio Daemon')
    parser.add_argument('--device', type=str, default=None,
                        help='audio device name or index')
    parser.add_argument('--engine', type=str, default='cqt',
                        choices=['cqt', 'octave_bank'],
                        help='spectrum engine (default: cqt)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    daemon = AudioDaemon(device=args.device, spectrum_engine=args.engine)

    def _handle_signal(signum, frame):
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    daemon.run()


if __name__ == '__main__':
    main()
