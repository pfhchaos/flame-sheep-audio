"""D-Bus service for the audio daemon.

Provides schema negotiation (methods) and discrete event signals.
Continuous data goes through shmem — dbus is for low-frequency events
and consumer discovery only.

Bus name: com.flameSheep.AudioEngine
Object path: /com/flameSheep/AudioEngine
"""

from __future__ import annotations

import json
import logging
import threading

log = logging.getLogger(__name__)

BUS_NAME = 'com.flameSheep.AudioEngine'
OBJ_PATH = '/com/flameSheep/AudioEngine'
IFACE = 'com.flameSheep.AudioEngine'


class AudioDbusService:
    """D-Bus service for audio daemon discovery and event signals."""

    def __init__(self, schema: dict, shm_name: str):
        self._schema_json = json.dumps(schema)
        self._shm_name = shm_name
        self._obj = None
        self._loop = None
        self._thread: threading.Thread | None = None
        self._last_mode: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name='audio-dbus')
        self._thread.start()

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.quit()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import dbus
            import dbus.service
            from dbus.mainloop.glib import DBusGMainLoop
            from gi.repository import GLib

            DBusGMainLoop(set_as_default=True)
            bus = dbus.SessionBus()
            dbus.service.BusName(BUS_NAME, bus)

            # Define the service object inline (needs dbus imported)
            schema_json = self._schema_json
            shm_name = self._shm_name

            class Obj(dbus.service.Object):
                @dbus.service.method(IFACE, out_signature='s')
                def GetSchema(self):
                    return schema_json

                @dbus.service.method(IFACE, out_signature='s')
                def GetShmName(self):
                    return shm_name

                @dbus.service.signal(IFACE, signature='sd')
                def BeatEvent(self, kind, energy):
                    pass

                @dbus.service.signal(IFACE, signature='s')
                def ModeChange(self, mode):
                    pass

                @dbus.service.signal(IFACE, signature='ss')
                def SongStart(self, title, artist):
                    pass

                @dbus.service.signal(IFACE, signature='dd')
                def TempoUpdate(self, bpm, confidence):
                    pass

            self._obj = Obj(bus, OBJ_PATH)
            self._loop = GLib.MainLoop()
            log.info('dbus service started: %s', BUS_NAME)
            self._loop.run()
        except Exception:
            log.exception('dbus service failed to start')

    def emit_beat(self, kind: str, energy: float) -> None:
        if self._obj is not None:
            try:
                self._obj.BeatEvent(kind, energy)
            except Exception:
                pass

    def emit_mode_change(self, mode: str) -> None:
        if mode == self._last_mode:
            return
        self._last_mode = mode
        if self._obj is not None:
            try:
                self._obj.ModeChange(mode)
            except Exception:
                pass

    def emit_song_start(self, title: str, artist: str) -> None:
        if self._obj is not None:
            try:
                self._obj.SongStart(title, artist)
            except Exception:
                pass

    def emit_tempo_update(self, bpm: float, confidence: float) -> None:
        if self._obj is not None:
            try:
                self._obj.TempoUpdate(bpm, confidence)
            except Exception:
                pass
