"""Subtitle an existing video or audio file.

The live pipeline is paced by the clock: audio arrives in real time, so a
one-hour video would take an hour. Here the same stages -- Silero VAD ->
segmenter -> ASR -> translation -> subtitle files -- run as fast as the GPU
allows, which on the test machine is roughly 12x faster than real time.

ffmpeg does the decoding, so any container it understands works (mp4, mkv,
mov, webm, mp3, m4a, wav...).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .asr import create_asr
from .audio import dsp
from .audio.segmenter import Segmenter
from .audio.vad import create_vad
from .config import SAMPLE_RATE, AppConfig
from .export import TranscriptRecorder
from .languages import canonical_name, same_language
from .pipeline import SubtitleEvent
from .translate import TranslationRequest, create_translator

log = logging.getLogger(__name__)

MEDIA_SUFFIXES = (
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".flv", ".wmv",
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma",
)


class MediaError(RuntimeError):
    """Raised when the media cannot be read or decoded."""


@dataclass
class FileResult:
    media: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    lines: int = 0
    media_seconds: float = 0.0
    speech_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def speed(self) -> float:
        """How many times faster than real time the job ran."""
        return self.media_seconds / self.elapsed_seconds if self.elapsed_seconds else 0.0


@dataclass
class Progress:
    """One progress update, for a UI or the console."""

    fraction: float
    media_position: float
    media_seconds: float
    line: str = ""
    language: str = ""
    eta_seconds: float = 0.0


def find_ffmpeg(name: str = "ffmpeg") -> str | None:
    """Locate ffmpeg on PATH, or in the usual Windows install spots."""
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path(r"C:\ffmpeg\bin"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg" / "bin",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
    ]
    for folder in candidates:
        exe = folder / (f"{name}.exe" if sys.platform == "win32" else name)
        if exe.is_file():
            return str(exe)
    return None


def decode_audio(media: Path, ffmpeg: str | None = None) -> np.ndarray:
    """Decode any media file to 16 kHz mono float32."""
    media = Path(media)
    if not media.is_file():
        raise MediaError(f"File not found: {media}")

    # A plain WAV needs no external tool at all.
    if media.suffix.lower() == ".wav":
        try:
            return dsp.read_wav_mono(str(media), SAMPLE_RATE)
        except Exception as exc:
            raise MediaError(f"Could not read {media.name}: {exc}") from exc

    exe = ffmpeg or find_ffmpeg()
    if not exe:
        raise MediaError(
            "ffmpeg is needed to read this file but was not found.\n"
            "Install it with:  winget install Gyan.FFmpeg\n"
            "(or put ffmpeg.exe on your PATH)"
        )

    # Decode to a temporary WAV rather than a pipe: ffmpeg cannot write a
    # correct RIFF length to a stream, and some readers reject that.
    handle, temp_name = tempfile.mkstemp(suffix=".wav", prefix="rtsubs_")
    os.close(handle)
    temp = Path(temp_name)
    try:
        result = subprocess.run(
            [exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(media),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(temp)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not temp.stat().st_size:
            detail = (result.stderr or "").strip().splitlines()
            raise MediaError(
                f"ffmpeg could not decode {media.name}"
                + (f": {detail[-1]}" if detail else "")
            )
        return dsp.read_wav_mono(str(temp), SAMPLE_RATE)
    finally:
        try:
            temp.unlink()
        except OSError:
            pass


def subtitle_file(
    media: str | Path,
    cfg: AppConfig,
    output_dir: str | Path | None = None,
    formats: list[str] | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    cancel: threading.Event | None = None,
) -> FileResult:
    """Transcribe, translate and write subtitle files for ``media``.

    Subtitles land next to the media by default and are named after it, so
    players load them automatically.
    """
    media = Path(media).expanduser()
    cancel = cancel or threading.Event()
    report = on_progress or (lambda p: None)
    started = time.perf_counter()

    audio = decode_audio(media)
    duration = audio.size / SAMPLE_RATE
    result = FileResult(media=media, media_seconds=duration)
    if duration <= 0:
        raise MediaError(f"{media.name} contains no audio")

    segmenter = Segmenter(create_vad(cfg.vad), cfg.vad, cfg.segmenter)
    segments = []
    for start in range(0, audio.size, 512):
        if cancel.is_set():
            result.cancelled = True
            return result
        segments += [e for e in segmenter.feed(audio[start : start + 512]) if e.is_final]
    segments += segmenter.flush()
    result.speech_seconds = sum(s.duration for s in segments)

    asr = create_asr(cfg.asr)
    ok, detail = asr.health()
    if not ok:
        raise MediaError(f"Speech recognition is not available: {detail}")
    translator = create_translator(cfg.translate)

    export_cfg = cfg.export
    if formats:
        export_cfg = type(export_cfg)(
            enabled=True,
            formats=list(formats),
            directory=export_cfg.directory,
            include_original=export_cfg.include_original,
        )
    recorder = TranscriptRecorder(
        export_cfg,
        source_description=media.name,
        target_language=cfg.translate.target_language,
        stem=media.stem,
        directory=output_dir or media.parent,
        clock_times=False,
    )

    history: list[tuple[str, str]] = []
    try:
        for index, segment in enumerate(segments, 1):
            if cancel.is_set():
                result.cancelled = True
                break

            transcript = asr.transcribe(
                segment.audio, language=cfg.asr.language, context=cfg.asr.context_prompt
            )
            text = transcript.text.strip()
            if not text:
                continue
            language = canonical_name(transcript.language, default="")
            result.languages[language or "?"] = result.languages.get(language or "?", 0) + 1

            translated = text
            needs_translation = not transcript.already_translated and not (
                cfg.translate.skip_if_target
                and same_language(language, cfg.translate.target_language)
            )
            if needs_translation:
                try:
                    out = translator.translate(
                        TranslationRequest(
                            text=text,
                            source_language=language,
                            target_language=cfg.translate.target_language,
                            history=tuple(history[-cfg.translate.history_turns :]),
                            is_final=True,
                        )
                    )
                    if out.text:
                        translated = out.text
                        history.append((text, out.text))
                except Exception as exc:
                    # Keep the transcript rather than losing the line entirely.
                    log.warning("translation failed, keeping source text: %s", exc)

            event = SubtitleEvent(
                utterance_id=index,
                source_text=text,
                text=translated,
                language=language,
                is_final=True,
                start_time=segment.start_time,
                duration=segment.duration,
            )
            recorder.write(event)
            result.lines += 1

            elapsed = time.perf_counter() - started
            done = min(1.0, segment.start_time / duration if duration else 1.0)
            report(
                Progress(
                    fraction=done,
                    media_position=segment.start_time,
                    media_seconds=duration,
                    line=translated,
                    language=language,
                    eta_seconds=(elapsed / done - elapsed) if done > 0.02 else 0.0,
                )
            )
    finally:
        recorder.close()
        for backend in (asr, translator):
            try:
                backend.close()
            except Exception:
                pass

    result.outputs = dict(recorder.paths) if result.lines else {}
    result.elapsed_seconds = time.perf_counter() - started
    if not result.cancelled:
        report(Progress(fraction=1.0, media_position=duration, media_seconds=duration))
    return result
