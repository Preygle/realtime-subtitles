"""Small signal-processing helpers: downmix, resample, WAV encoding.

Resampling prefers ``soxr`` (fast, high quality) and falls back to a polyphase
implementation from ``scipy``, then to plain linear interpolation, so the
pipeline still runs on a bare numpy install.
"""

from __future__ import annotations

import io
import logging
import math
import wave

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - import-time capability probe
    import soxr  # type: ignore

    _RESAMPLER = "soxr"
except ImportError:  # pragma: no cover
    try:
        from scipy.signal import resample_poly  # type: ignore

        _RESAMPLER = "scipy"
    except ImportError:
        _RESAMPLER = "linear"

log.debug("resampler backend: %s", _RESAMPLER)


def resampler_name() -> str:
    return _RESAMPLER


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Downmix an interleaved frame buffer to a 1-D mono float32 array."""
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    if channels <= 1:
        return flat
    usable = (flat.size // channels) * channels
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return flat[:usable].reshape(-1, channels).mean(axis=1)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio between two sample rates."""
    x = np.asarray(samples, dtype=np.float32)
    if src_rate == dst_rate or x.size == 0:
        return x

    if _RESAMPLER == "soxr":
        return soxr.resample(x, src_rate, dst_rate).astype(np.float32, copy=False)

    if _RESAMPLER == "scipy":
        g = math.gcd(int(src_rate), int(dst_rate))
        return resample_poly(x, dst_rate // g, src_rate // g).astype(
            np.float32, copy=False
        )

    # Linear fallback. Audible aliasing on big ratios, but never wrong enough
    # to break the pipeline.
    n_out = int(round(x.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0.0, x.size - 1.0, n_out, dtype=np.float64)
    return np.interp(idx, np.arange(x.size, dtype=np.float64), x).astype(np.float32)


def pcm16_bytes_to_float(raw: bytes) -> np.ndarray:
    """Decode little-endian int16 PCM into float32 in [-1, 1]."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(raw, dtype="<i2")
    return (ints.astype(np.float32) / 32768.0).copy()


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Clip and quantise float32 audio to int16."""
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2")


def apply_gain_db(samples: np.ndarray, gain_db: float) -> np.ndarray:
    if not gain_db:
        return samples
    return np.asarray(samples, dtype=np.float32) * (10.0 ** (gain_db / 20.0))


def dbfs(samples: np.ndarray) -> float:
    """RMS level of a block in dBFS; -inf-ish for digital silence."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Wrap mono float32 audio in an in-memory 16-bit PCM WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_pcm16(samples).tobytes())
    return buf.getvalue()


def read_wav_mono(path: str, target_rate: int) -> np.ndarray:
    """Load a WAV file as mono float32 at ``target_rate`` (for file replay)."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    return resample(to_mono(data, channels), rate, target_rate)
