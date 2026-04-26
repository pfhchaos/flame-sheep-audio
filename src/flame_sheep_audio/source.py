"""Signal sources — abstract audio input from capture method.

PipeWireSource: real audio from sounddevice/PipeWire.
FeedSource: manual PCM injection for testing.
"""

import threading
import numpy as np
from collections import deque
from typing import Protocol

import sounddevice as sd

from ._constants import SAMPLE_RATE, BLOCK_SIZE, FFT_SIZE


class SignalSource(Protocol):
    """Protocol for audio signal sources."""

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> np.ndarray | None:
        """Return FFT_SIZE samples if available, else None."""
        ...

    def read_hop(self, n: int) -> np.ndarray | None:
        """Return exactly n new samples. May block (PipeWire) or return None (Feed)."""
        ...

    def feed(self, pcm: np.ndarray) -> None:
        """Push PCM samples (for test sources). May be a no-op."""
        ...


class PipeWireSource:
    """Real audio capture via sounddevice (PipeWire/PulseAudio/ALSA)."""

    def __init__(self, device: str | int | None = None):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._buffer = deque(maxlen=SAMPLE_RATE // 2)  # ~0.5s buffer
        self._new_samples = 0

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype='float32',
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        with self._cond:
            self._buffer.extend(indata[:, 0])
            self._new_samples += frames
            self._cond.notify_all()

    def start(self):
        self._stream.start()

    def stop(self):
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()  # unblock any waiting read_hop
        self._stream.stop()
        self._stream.close()

    def read(self) -> np.ndarray | None:
        """Return FFT_SIZE samples if enough new audio has arrived."""
        with self._lock:
            if len(self._buffer) < FFT_SIZE or self._new_samples < BLOCK_SIZE:
                return None
            self._new_samples = 0
            return np.array(self._buffer, dtype=np.float32)[-FFT_SIZE:]

    def read_hop(self, n: int) -> np.ndarray | None:
        """Block until n new samples are available, return them.
        Returns None if the source has been stopped."""
        with self._cond:
            while self._new_samples < n:
                if self._stop_event.is_set():
                    return None
                self._cond.wait(timeout=0.1)
            # Clamp _new_samples to buffer length: if the deque overflowed
            # (e.g., during a GPU stall), old samples were silently dropped
            # and _new_samples would exceed len(buffer), causing negative
            # start indices and corrupted reads.
            buf_len = len(self._buffer)
            if self._new_samples > buf_len:
                self._new_samples = buf_len
            # Take the n oldest unconsumed samples from the buffer
            start = buf_len - self._new_samples
            samples = np.array(list(self._buffer)[start:start + n], dtype=np.float32)
            self._new_samples -= n
            return samples

    def feed(self, pcm: np.ndarray):
        """Push PCM samples directly (bypass sounddevice)."""
        with self._cond:
            self._buffer.extend(pcm)
            self._new_samples += len(pcm)
            self._cond.notify_all()


class FeedSource:
    """Test signal source — accepts PCM via feed(), no audio hardware."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=SAMPLE_RATE // 2)
        self._new_samples = 0

    def start(self): pass
    def stop(self): pass

    def read(self) -> np.ndarray | None:
        with self._lock:
            if len(self._buffer) < FFT_SIZE or self._new_samples < BLOCK_SIZE:
                return None
            self._new_samples = 0
            return np.array(self._buffer, dtype=np.float32)[-FFT_SIZE:]

    def read_hop(self, n: int) -> np.ndarray | None:
        """Non-blocking: return n samples if available, else None."""
        with self._lock:
            if self._new_samples < n:
                return None
            buf_list = list(self._buffer)
            start = len(buf_list) - self._new_samples
            samples = np.array(buf_list[start:start + n], dtype=np.float32)
            self._new_samples -= n
            return samples

    def feed(self, pcm: np.ndarray):
        with self._lock:
            self._buffer.extend(pcm)
            self._new_samples += len(pcm)
