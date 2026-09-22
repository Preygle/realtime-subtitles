"""Tests for saving subtitles and transcripts to disk."""

from __future__ import annotations

import datetime as dt
import re

import pytest

from rtsubs.config import ExportConfig
from rtsubs.export import TranscriptRecorder, split_cues, wrap_cue
from rtsubs.pipeline import SubtitleEvent

START = dt.datetime(2026, 9, 22, 14, 3, 0)


def _event(text, start, duration, source="", language="Japanese"):
    return SubtitleEvent(
        utterance_id=1,
        source_text=source or text,
        text=text,
        language=language,
        is_final=True,
        start_time=start,
        duration=duration,
    )


def _recorder(tmp_path, formats=("srt", "vtt", "txt"), include_original=True):
    cfg = ExportConfig(
        enabled=True,
        formats=list(formats),
        directory=str(tmp_path),
        include_original=include_original,
    )
    return TranscriptRecorder(cfg, started_at=START, target_language="English")


# ------------------------------------------------------------------ SRT
def test_srt_numbering_and_timestamps(tmp_path):
    rec = _recorder(tmp_path, formats=["srt"])
    rec.write(_event("It might rain tomorrow.", 1.25, 2.0))
    rec.write(_event("You'd better bring an umbrella.", 3.5, 1.8))
    rec.close()

    srt = rec.paths["srt"].read_text(encoding="utf-8-sig")
    assert srt.startswith("1\n00:00:01,250 --> 00:00:03,250\nIt might rain tomorrow.\n\n")
    assert "2\n00:00:03,500 --> 00:00:05,300\nYou'd better bring an umbrella.\n" in srt
    # Written with a BOM so Windows players detect UTF-8.
    assert rec.paths["srt"].read_bytes().startswith(b"\xef\xbb\xbf")


def test_cues_never_overlap(tmp_path):
    rec = _recorder(tmp_path, formats=["srt"])
    rec.write(_event("First line.", 0.0, 3.0))
    rec.write(_event("Carried-over tail starts early.", 2.7, 2.0))  # overlaps the first
    rec.close()

    times = re.findall(r"(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})", rec.paths["srt"].read_text("utf-8-sig"))
    assert times[1][0] >= times[0][1]


def test_very_short_lines_get_a_minimum_duration(tmp_path):
    rec = _recorder(tmp_path, formats=["srt"])
    rec.write(_event("Yes.", 5.0, 0.2))
    rec.close()
    assert "00:00:05,000 --> 00:00:06,000" in rec.paths["srt"].read_text("utf-8-sig")


def test_long_line_is_split_into_timed_cues(tmp_path):
    long = " ".join(["word"] * 60)  # ~300 characters
    rec = _recorder(tmp_path, formats=["srt"])
    rec.write(_event(long, 10.0, 12.0))
    rec.close()

    srt = rec.paths["srt"].read_text("utf-8-sig")
    cues = srt.strip().split("\n\n")
    assert len(cues) >= 3
    for cue in cues:
        text_lines = cue.split("\n")[2:]
        assert len(text_lines) <= 2
        assert all(len(line) <= 42 for line in text_lines)
    # The cues together still span the utterance.
    assert "00:00:10,000 -->" in cues[0]
    assert cues[-1].split("\n")[1].endswith("00:00:22,000")


def test_split_and_wrap_handle_unspaced_scripts():
    japanese = "明日は雨が降るかもしれません" * 8
    chunks = split_cues(japanese)
    assert "".join(chunks) == japanese
    assert all(len(c) <= 84 for c in chunks)
    assert wrap_cue(chunks[0]).count("\n") == 1


def test_wrap_balances_two_lines():
    wrapped = wrap_cue("So Alibaba just dropped a new model, and it is extremely flexible.")
    first, second = wrapped.split("\n")
    assert abs(len(first) - len(second)) < 12


# ------------------------------------------------------------------ VTT
def test_vtt_header_and_dot_milliseconds(tmp_path):
    rec = _recorder(tmp_path, formats=["vtt"])
    rec.write(_event("Hello.", 0.5, 1.5))
    rec.close()
    vtt = rec.paths["vtt"].read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT\n\n00:00:00.500 --> 00:00:02.000\nHello.\n")


# ------------------------------------------------------------------ TXT
def test_text_transcript_uses_clock_times_and_original(tmp_path):
    rec = _recorder(tmp_path, formats=["txt"])
    rec.write(_event("It might rain tomorrow.", 65.0, 2.0, source="明日は雨が降るかもしれません。"))
    rec.close()

    txt = rec.paths["txt"].read_text(encoding="utf-8")
    assert txt.startswith("Transcript - 2026-09-22 14:03\n")
    assert "subtitles in English" in txt
    # 14:03:00 + 65 s of audio
    assert "[14:04:05] It might rain tomorrow.\n" in txt
    assert "(Japanese: 明日は雨が降るかもしれません。)" in txt


def test_text_transcript_can_omit_original(tmp_path):
    rec = _recorder(tmp_path, formats=["txt"], include_original=False)
    rec.write(_event("It might rain tomorrow.", 1.0, 2.0, source="明日は雨が降るかもしれません。"))
    rec.close()
    assert "明日" not in rec.paths["txt"].read_text(encoding="utf-8")


def test_untranslated_lines_are_not_duplicated(tmp_path):
    rec = _recorder(tmp_path, formats=["txt"])
    rec.write(_event("Already English.", 1.0, 1.0, source="Already English.", language="English"))
    rec.close()
    assert rec.paths["txt"].read_text(encoding="utf-8").count("Already English.") == 1


# ------------------------------------------------------------- lifecycle
def test_lines_are_on_disk_before_close(tmp_path):
    """A crash mid-meeting must not lose what was already said."""
    rec = _recorder(tmp_path, formats=["srt", "txt"])
    rec.write(_event("Saved immediately.", 1.0, 1.0))
    assert "Saved immediately." in rec.paths["srt"].read_text("utf-8-sig")
    assert "Saved immediately." in rec.paths["txt"].read_text("utf-8")
    rec.close()


def test_empty_session_leaves_no_files(tmp_path):
    rec = _recorder(tmp_path)
    paths = list(rec.paths.values())
    rec.close()
    assert not any(p.exists() for p in paths)


def test_blank_text_is_skipped(tmp_path):
    rec = _recorder(tmp_path, formats=["srt"])
    rec.write(_event("   ", 1.0, 1.0))
    rec.write(_event("Real line.", 2.0, 1.0))
    rec.close()
    assert rec.paths["srt"].read_text("utf-8-sig").startswith("1\n")
    assert rec.lines_written == 1


def test_unknown_formats_are_ignored(tmp_path):
    rec = _recorder(tmp_path, formats=["srt", "docx", "srt"])
    assert list(rec.paths) == ["srt"]
    rec.close()


# ------------------------------------------------------------ pipeline
def test_pipeline_saves_finished_lines(tmp_path):
    """End to end: captioning with export on writes every finished line."""
    import time

    import numpy as np

    from rtsubs.audio import dsp
    from rtsubs.audio.capture import WavFileSource
    from rtsubs.config import SAMPLE_RATE, AppConfig
    from rtsubs.pipeline import SubtitlePipeline

    t = np.arange(int(SAMPLE_RATE * 2.0), dtype=np.float32) / SAMPLE_RATE
    tone = (0.3 * (0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)) * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    quiet = np.zeros(int(SAMPLE_RATE * 0.9), dtype=np.float32)
    wav = tmp_path / "clip.wav"
    wav.write_bytes(dsp.encode_wav(np.concatenate([quiet, tone, quiet, tone, quiet]), SAMPLE_RATE))

    cfg = AppConfig()
    cfg.asr.backend = "mock"
    cfg.translate.backend = "passthrough"
    cfg.vad.engine = "energy"
    cfg.export.enabled = True
    cfg.export.formats = ["srt", "txt"]
    cfg.export.directory = str(tmp_path / "out")

    events = []
    pipeline = SubtitlePipeline(cfg, on_event=events.append, source=WavFileSource(str(wav)))
    pipeline.start()
    deadline = time.time() + 20
    while time.time() < deadline and pipeline._source.running:
        time.sleep(0.05)
    time.sleep(1.5)
    pipeline.stop()

    finals = [e for e in events if e.is_final]
    srt_files = list((tmp_path / "out").glob("*.srt"))
    assert len(finals) == 2 and len(srt_files) == 1
    srt = srt_files[0].read_text("utf-8-sig")
    assert srt.count(" --> ") >= 2
    for e in finals:
        assert e.text.split()[0] in srt
    # The first cue starts roughly where the first tone starts (0.9 s),
    # minus the VAD's 0.3 s pre-roll.
    first_start = re.search(r"00:00:(\d\d),(\d{3}) -->", srt)
    assert 0.3 <= int(first_start.group(1)) + int(first_start.group(2)) / 1000 <= 1.2
