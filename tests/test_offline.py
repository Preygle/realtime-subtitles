"""Tests for subtitling an existing video or audio file."""

from __future__ import annotations

import subprocess
import threading

import numpy as np
import pytest

from rtsubs.audio import dsp
from rtsubs.config import SAMPLE_RATE, AppConfig
from rtsubs.offline import MediaError, decode_audio, find_ffmpeg, subtitle_file


def _speech(duration: float, freq: float = 180.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)
    harmonics = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    )
    return (0.35 * envelope * harmonics / 1.75).astype(np.float32)


def _silence(duration: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)


@pytest.fixture
def media(tmp_path):
    """A WAV with two spoken stretches separated by a pause."""
    audio = np.concatenate(
        [_silence(0.5), _speech(2.0), _silence(0.9), _speech(1.6), _silence(0.6)]
    )
    path = tmp_path / "meeting recording.wav"
    path.write_bytes(dsp.encode_wav(audio, SAMPLE_RATE))
    return path


def _config(formats=("srt", "txt")):
    cfg = AppConfig()
    cfg.asr.backend = "mock"
    cfg.asr.language = "Japanese"
    cfg.translate.backend = "passthrough"
    cfg.vad.engine = "energy"
    cfg.export.formats = list(formats)
    return cfg


# ---------------------------------------------------------------- decode
def test_decode_wav_without_ffmpeg(media):
    audio = decode_audio(media)
    assert audio.dtype == np.float32
    assert abs(audio.size / SAMPLE_RATE - 5.6) < 0.1


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(MediaError, match="not found"):
        decode_audio(tmp_path / "nope.mp4")


def test_unreadable_media_is_reported_clearly(tmp_path):
    if find_ffmpeg() is None:
        pytest.skip("ffmpeg not installed")
    fake = tmp_path / "broken.mp4"
    fake.write_text("this is not a video")
    with pytest.raises(MediaError):
        decode_audio(fake)


def test_decodes_a_real_video_container(tmp_path):
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg not installed")
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=16000", "-t", "2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video)],
        capture_output=True, check=True,
    )
    audio = decode_audio(video)
    assert abs(audio.size / SAMPLE_RATE - 2.0) < 0.2


# -------------------------------------------------------------- results
def test_writes_subtitles_named_after_the_media(media):
    result = subtitle_file(media, _config())

    assert result.lines == 2
    assert set(result.outputs) == {"srt", "txt"}
    # Named after the media and beside it, so players find them.
    assert result.outputs["srt"] == media.with_suffix(".srt")
    assert result.outputs["srt"].exists()
    assert result.media_seconds > 5
    assert result.speed > 0

    srt = result.outputs["srt"].read_text("utf-8-sig")
    assert srt.startswith("1\n00:00:")
    assert srt.count(" --> ") >= 2


def test_output_folder_can_be_chosen(media, tmp_path):
    out = tmp_path / "subs"
    result = subtitle_file(media, _config(["srt"]), output_dir=out)
    assert result.outputs["srt"].parent == out
    assert result.outputs["srt"].name == "meeting recording.srt"


def test_formats_argument_overrides_config(media):
    result = subtitle_file(media, _config(["srt", "txt"]), formats=["vtt"])
    assert set(result.outputs) == {"vtt"}
    assert result.outputs["vtt"].read_text("utf-8").startswith("WEBVTT")


def test_text_transcript_uses_media_positions(media):
    result = subtitle_file(media, _config(["txt"]))
    txt = result.outputs["txt"].read_text("utf-8")
    assert "times are positions in the media" in txt
    # First line starts within the first couple of seconds of the media.
    assert "[00:00:0" in txt


def test_detected_languages_are_counted(media):
    result = subtitle_file(media, _config(["srt"]))
    assert result.languages == {"Japanese": 2}


def test_cancelling_stops_the_job(media):
    cancel = threading.Event()
    cancel.set()
    result = subtitle_file(media, _config(["srt"]), cancel=cancel)
    assert result.cancelled and result.lines == 0


def test_progress_reaches_the_end(media):
    seen = []
    subtitle_file(media, _config(["srt"]), on_progress=seen.append)
    assert seen and seen[-1].fraction == 1.0
    assert all(0.0 <= p.fraction <= 1.0 for p in seen)
    assert any(p.line for p in seen)


def test_silent_media_writes_nothing(tmp_path):
    quiet = tmp_path / "quiet.wav"
    quiet.write_bytes(dsp.encode_wav(_silence(3.0), SAMPLE_RATE))
    result = subtitle_file(quiet, _config(["srt"]))
    assert result.lines == 0 and result.outputs == {}
    assert not quiet.with_suffix(".srt").exists()


def test_unavailable_asr_is_reported_before_work_starts(media):
    cfg = _config(["srt"])
    cfg.asr.backend = "llamacpp"
    cfg.asr.base_url = "http://127.0.0.1:1"  # nothing listening
    with pytest.raises(MediaError, match="not available"):
        subtitle_file(media, cfg)
