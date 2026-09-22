"""Audio capture sources.

Every source runs a background thread that pushes mono float32 blocks at
``SAMPLE_RATE`` into a bounded queue.  When the consumer falls behind the
oldest block is dropped rather than letting latency grow without bound --
stale subtitles are worse than missing ones.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from ..config import SAMPLE_RATE, AudioConfig
from . import dsp
from .devices import DeviceInfo, list_devices, pick_device

log = logging.getLogger(__name__)

#: ~8 seconds of 32 ms blocks. Deep enough to ride out a GC pause.
_QUEUE_DEPTH = 256


class CaptureError(RuntimeError):
    """Raised when a capture source cannot be opened."""


class AudioSource(ABC):
    """A background producer of 16 kHz mono float32 blocks."""

    def __init__(self, gain_db: float = 0.0) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gain_db = gain_db
        self.dropped_blocks = 0
        self.description = self.__class__.__name__

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, name="audio-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        """Pop the next block, or None if nothing arrived within ``timeout``."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # -- producer side --------------------------------------------------
    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception:
            log.exception("capture thread died: %s", self.description)

    @abstractmethod
    def _run(self) -> None:
        """Read from the device until ``self._stop`` is set."""

    def _emit(self, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        block = dsp.apply_gain_db(mono, self._gain_db)
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Drop the oldest block so we track live audio instead of drifting.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(block)
            except (queue.Empty, queue.Full):
                pass
            self.dropped_blocks += 1
            if self.dropped_blocks % 50 == 1:
                log.warning(
                    "audio queue full, dropped %d blocks (consumer too slow)",
                    self.dropped_blocks,
                )


class PyAudioWPatchSource(AudioSource):
    """WASAPI capture via PyAudioWPatch. Handles loopback and microphones."""

    def __init__(self, device: DeviceInfo, block_ms: int, gain_db: float) -> None:
        super().__init__(gain_db)
        self.device = device
        self.block_ms = block_ms
        self.description = f"pyaudiowpatch:{device.name}"

    def _run(self) -> None:
        import pyaudiowpatch as pyaudio  # type: ignore

        pa = pyaudio.PyAudio()
        stream = None
        try:
            rate = self.device.sample_rate
            channels = self.device.channels
            frames = max(64, int(rate * self.block_ms / 1000))
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=int(self.device.ident),
                frames_per_buffer=frames,
            )
            log.info(
                "capturing %s @ %d Hz x%d (%d frames/block)",
                self.device.name,
                rate,
                channels,
                frames,
            )
            while not self._stop.is_set():
                raw = stream.read(frames, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.float32)
                mono = dsp.to_mono(samples, channels)
                self._emit(dsp.resample(mono, rate, SAMPLE_RATE))
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()


class SoundcardSource(AudioSource):
    """WASAPI capture via the pure-Python ``soundcard`` package."""

    def __init__(self, device: DeviceInfo, block_ms: int, gain_db: float) -> None:
        super().__init__(gain_db)
        self.device = device
        self.block_ms = block_ms
        self.description = f"soundcard:{device.name}"

    def _run(self) -> None:
        import warnings

        import soundcard as sc  # type: ignore

        # soundcard warns "data discontinuity in recording" whenever WASAPI
        # reports a glitch, which happens routinely as a stream starts. It is
        # harmless for captioning, so silence it rather than printing a
        # console warning every time capture starts.
        warnings.filterwarnings(
            "ignore", message="data discontinuity", module="soundcard"
        )
        mic = sc.get_microphone(self.device.ident, include_loopback=True)
        rate = self.device.sample_rate
        frames = max(64, int(rate * self.block_ms / 1000))
        log.info(
            "capturing %s @ %d Hz (%d frames/block)", self.device.name, rate, frames
        )
        with mic.recorder(samplerate=rate, blocksize=frames) as rec:
            while not self._stop.is_set():
                data = rec.record(numframes=frames)  # (frames, channels) float32
                arr = np.asarray(data, dtype=np.float32)
                channels = arr.shape[1] if arr.ndim == 2 else 1
                mono = dsp.to_mono(arr.reshape(-1), channels)
                self._emit(dsp.resample(mono, rate, SAMPLE_RATE))


class WavFileSource(AudioSource):
    """Replays a WAV file in real time. Used for testing without a device."""

    def __init__(
        self, path: str, block_ms: int = 32, gain_db: float = 0.0, loop: bool = False
    ) -> None:
        super().__init__(gain_db)
        self.path = path
        self.block_ms = block_ms
        self.loop = loop
        self.description = f"wav:{path}"

    def _run(self) -> None:
        audio = dsp.read_wav_mono(self.path, SAMPLE_RATE)
        frames = max(64, int(SAMPLE_RATE * self.block_ms / 1000))
        log.info("replaying %s (%.1fs)", self.path, audio.size / SAMPLE_RATE)
        while not self._stop.is_set():
            next_deadline = time.perf_counter()
            for start in range(0, audio.size, frames):
                if self._stop.is_set():
                    return
                self._emit(audio[start : start + frames])
                # Pace the replay so downstream timing matches live capture.
                next_deadline += frames / SAMPLE_RATE
                delay = next_deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if not self.loop:
                return


class SilenceSource(AudioSource):
    """Emits digital silence. Lets the UI run with no audio stack at all."""

    def __init__(self, block_ms: int = 32) -> None:
        super().__init__(0.0)
        self.block_ms = block_ms
        self.description = "silence"

    def _run(self) -> None:
        frames = max(64, int(SAMPLE_RATE * self.block_ms / 1000))
        block = np.zeros(frames, dtype=np.float32)
        while not self._stop.is_set():
            self._emit(block.copy())
            time.sleep(frames / SAMPLE_RATE)


def create_source(cfg: AudioConfig) -> AudioSource:
    """Build the capture source described by ``cfg``, raising on failure."""
    devices = list_devices(cfg.backend)
    if not devices:
        raise CaptureError(
            "No capture devices found. Install a capture backend with:\n"
            "  pip install PyAudioWPatch   (preferred)\n"
            "  pip install soundcard       (pure-Python fallback)"
        )

    device = pick_device(devices, cfg.source, cfg.device_name)
    if device is None:
        kind = "loopback" if cfg.source == "loopback" else "input"
        raise CaptureError(
            f"No {kind} device matched {cfg.device_name!r}. "
            f"Available: {', '.join(d.label for d in devices)}"
        )

    if device.backend == "pyaudiowpatch":
        return PyAudioWPatchSource(device, cfg.block_ms, cfg.gain_db)
    return SoundcardSource(device, cfg.block_ms, cfg.gain_db)
