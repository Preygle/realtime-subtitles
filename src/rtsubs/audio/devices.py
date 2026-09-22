"""Device enumeration across the supported capture backends.

Both backends are optional imports.  ``list_devices`` returns whatever is
reachable and never raises, so the control panel can always be opened even if
no audio library is installed yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    #: Backend that owns this device: "pyaudiowpatch" or "soundcard".
    backend: str
    #: Backend-specific handle (index for PyAudio, id string for soundcard).
    ident: str
    name: str
    channels: int
    sample_rate: int
    #: True when this device captures system output rather than a microphone.
    is_loopback: bool
    is_default: bool = False

    @property
    def label(self) -> str:
        tag = "loopback" if self.is_loopback else "input"
        star = " *" if self.is_default else ""
        return f"[{tag}] {self.name}{star}"


def _pyaudiowpatch_devices() -> list[DeviceInfo]:
    try:
        import pyaudiowpatch as pyaudio  # type: ignore
    except ImportError:
        return []

    out: list[DeviceInfo] = []
    pa = pyaudio.PyAudio()
    try:
        try:
            default_speaker = pa.get_default_wasapi_loopback()
            default_loopback_idx = int(default_speaker["index"])
        except (OSError, LookupError, ValueError):
            default_loopback_idx = -1

        try:
            default_input_idx = int(pa.get_default_input_device_info()["index"])
        except (OSError, LookupError, ValueError):
            default_input_idx = -1

        for info in pa.get_device_info_generator():
            channels = int(info.get("maxInputChannels", 0))
            if channels <= 0:
                continue
            idx = int(info["index"])
            is_loopback = bool(info.get("isLoopbackDevice", False))
            out.append(
                DeviceInfo(
                    backend="pyaudiowpatch",
                    ident=str(idx),
                    name=str(info.get("name", f"device {idx}")),
                    channels=channels,
                    sample_rate=int(info.get("defaultSampleRate", 48000)),
                    is_loopback=is_loopback,
                    is_default=idx
                    == (default_loopback_idx if is_loopback else default_input_idx),
                )
            )
    except Exception:  # pragma: no cover - driver-dependent
        log.exception("failed to enumerate PyAudioWPatch devices")
    finally:
        pa.terminate()
    return out


def _soundcard_devices() -> list[DeviceInfo]:
    try:
        import soundcard as sc  # type: ignore
    except (ImportError, OSError):
        return []

    out: list[DeviceInfo] = []
    try:
        default_speaker_name = ""
        try:
            default_speaker_name = sc.default_speaker().name
        except Exception:
            pass
        default_mic_name = ""
        try:
            default_mic_name = sc.default_microphone().name
        except Exception:
            pass

        # include_loopback surfaces one pseudo-microphone per speaker, which is
        # how soundcard exposes WASAPI loopback.
        for mic in sc.all_microphones(include_loopback=True):
            is_loopback = bool(getattr(mic, "isloopback", False))
            ref = default_speaker_name if is_loopback else default_mic_name
            out.append(
                DeviceInfo(
                    backend="soundcard",
                    ident=str(mic.id),
                    name=str(mic.name),
                    channels=int(getattr(mic, "channels", 2) or 2),
                    sample_rate=48000,
                    is_loopback=is_loopback,
                    is_default=bool(ref) and ref in str(mic.name),
                )
            )
    except Exception:  # pragma: no cover - driver-dependent
        log.exception("failed to enumerate soundcard devices")
    return out


def list_devices(backend: str = "auto") -> list[DeviceInfo]:
    """Enumerate capture devices, preferring PyAudioWPatch when present."""
    devices: list[DeviceInfo] = []
    if backend in ("auto", "pyaudiowpatch"):
        devices += _pyaudiowpatch_devices()
    if backend in ("auto", "soundcard"):
        # In auto mode soundcard is only a fallback; duplicate device lists
        # from two backends would just confuse the picker.
        if backend == "soundcard" or not devices:
            devices += _soundcard_devices()
    return devices


def pick_device(
    devices: list[DeviceInfo], source: str, name_filter: str = ""
) -> DeviceInfo | None:
    """Choose a device matching ``source`` ("loopback"/"input") and a name hint."""
    want_loopback = source == "loopback"
    candidates = [d for d in devices if d.is_loopback == want_loopback]
    if not candidates:
        return None

    if name_filter:
        needle = name_filter.casefold()
        exact = [d for d in candidates if d.name.casefold() == needle]
        if exact:
            return exact[0]
        partial = [d for d in candidates if needle in d.name.casefold()]
        if partial:
            return partial[0]

    for d in candidates:
        if d.is_default:
            return d
    return candidates[0]


def available_backends() -> list[str]:
    found = []
    try:
        import pyaudiowpatch  # type: ignore  # noqa: F401

        found.append("pyaudiowpatch")
    except ImportError:
        pass
    try:
        import soundcard  # type: ignore  # noqa: F401

        found.append("soundcard")
    except (ImportError, OSError):
        pass
    return found
