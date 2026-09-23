"""Save finished subtitle lines to disk as they happen.

Three formats, any combination:

``srt``  SubRip, the subtitle format every player and editor understands.
``vtt``  WebVTT, for browsers and video platforms.
``txt``  A plain transcript with clock times, meant for reading, searching or
         pasting into a summarizer.

Every line is appended and flushed immediately, so a crash or a closed laptop
loses at most the sentence being spoken. Timings in subtitle files are
relative to when capture started, so they line up with a recording started at
the same moment; the text transcript uses wall-clock times instead, which is
what you want when reviewing a meeting.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from .config import PROJECT_ROOT, ExportConfig

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import SubtitleEvent

log = logging.getLogger(__name__)

FORMATS = ("srt", "vtt", "txt")

#: Standard subtitle limits: about 42 characters per line, two lines per cue.
_MAX_LINE = 42
#: Cues shorter than this flash past; stretch them if there is room.
_MIN_CUE_SEC = 1.0


def _timestamp(seconds: float, sep: str) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _is_spaced(text: str) -> bool:
    """False for scripts written without spaces (Chinese, Japanese, Thai)."""
    return " " in text.strip()


def _rows(text: str) -> list[str]:
    """Fill rows of at most _MAX_LINE characters, breaking between words.

    Scripts without spaces are broken between characters, and a single word
    longer than a row is hard-split rather than allowed to overflow.
    """
    text = " ".join(text.split())
    if not text:
        return []
    if not _is_spaced(text):
        return [text[i : i + _MAX_LINE] for i in range(0, len(text), _MAX_LINE)]

    rows: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > _MAX_LINE:
            if current:
                rows.append(current)
                current = ""
            rows.append(word[:_MAX_LINE])
            word = word[_MAX_LINE:]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= _MAX_LINE:
            current = candidate
        else:
            rows.append(current)
            current = word
    if current:
        rows.append(current)
    return rows


def split_cues(text: str) -> list[str]:
    """Split a long line into cue-sized chunks that each fit on two rows.

    Rows are laid out first and then paired, so every chunk is guaranteed to
    fit the two-row, 42-character standard. (Splitting by an 84-character
    budget first doesn't: the space where the row breaks counts too.)
    """
    rows = _rows(text)
    joiner = " " if _is_spaced(text) else ""
    return [joiner.join(rows[i : i + 2]) for i in range(0, len(rows), 2)]


def wrap_cue(text: str) -> str:
    """Lay a cue out on at most two rows, balanced when the words allow it."""
    if len(text) <= _MAX_LINE:
        return text
    if not _is_spaced(text):
        half = (len(text) + 1) // 2
        return f"{text[:half]}\n{text[half:]}"
    words = text.split(" ")
    best, best_gap = None, len(text)
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        gap = abs(len(first) - len(second))
        if len(first) <= _MAX_LINE and len(second) <= _MAX_LINE and gap < best_gap:
            best, best_gap = f"{first}\n{second}", gap
    # No balanced split fits: fall back to filling rows left to right.
    return best if best is not None else "\n".join(_rows(text))


class TranscriptRecorder:
    """Writes one session's finished lines to the configured formats."""

    def __init__(
        self,
        cfg: ExportConfig,
        started_at: dt.datetime | None = None,
        source_description: str = "",
        target_language: str = "",
        stem: str = "",
        directory: Path | str = "",
        clock_times: bool = True,
    ) -> None:
        self.cfg = cfg
        self.started_at = started_at or dt.datetime.now()
        self.source_description = source_description
        self.target_language = target_language
        #: Subtitling a file wants times counted from the start of the media,
        #: not the wall clock it happened to be processed at.
        self.clock_times = clock_times

        folder = Path(directory or cfg.directory)
        if not folder.is_absolute():
            folder = PROJECT_ROOT / folder
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder

        # Files are named after the media when subtitling a file, so players
        # pick the subtitles up automatically, and after the time otherwise.
        stem = stem or self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.paths: dict[str, Path] = {}
        self._files: dict[str, TextIO] = {}
        formats = [f for f in cfg.formats if f in FORMATS]
        for fmt in dict.fromkeys(formats):  # de-duplicate, keep order
            path = folder / f"{stem}.{fmt}"
            # BOM on .srt: some Windows players otherwise misread UTF-8 text.
            encoding = "utf-8-sig" if fmt == "srt" else "utf-8"
            self._files[fmt] = path.open("w", encoding=encoding, newline="\n")
            self.paths[fmt] = path

        self._cue_index = 0
        self._last_end = 0.0
        self.lines_written = 0
        self._write_headers()

    # ------------------------------------------------------------------
    def _write_headers(self) -> None:
        if "vtt" in self._files:
            self._files["vtt"].write("WEBVTT\n\n")
        if "txt" in self._files:
            when = self.started_at.strftime("%Y-%m-%d %H:%M")
            header = [f"Transcript - {when}"]
            if not self.clock_times:
                header.append("times are positions in the media")
            details = []
            if self.source_description:
                details.append(f"audio: {self.source_description}")
            if self.target_language:
                details.append(f"subtitles in {self.target_language}")
            if details:
                header.append(" | ".join(details))
            self._files["txt"].write("\n".join(header) + "\n\n")
        self._flush()

    def write(self, event: "SubtitleEvent") -> None:
        text = " ".join(event.text.split())
        if not text or not self._files:
            return

        # Lines can overlap slightly when a long utterance was split with a
        # carried-over audio tail; subtitle players dislike overlapping cues.
        start = max(event.start_time, self._last_end)
        end = max(start + _MIN_CUE_SEC, event.start_time + event.duration)

        if "srt" in self._files or "vtt" in self._files:
            self._write_cues(text, start, end)
        if "txt" in self._files:
            self._write_text(event, text)

        self._last_end = end
        self.lines_written += 1
        self._flush()

    def _write_cues(self, text: str, start: float, end: float) -> None:
        chunks = split_cues(text)
        total_chars = sum(len(c) for c in chunks) or 1
        cursor = start
        for chunk in chunks:
            # Share the line's time between its cues by length.
            share = (end - start) * len(chunk) / total_chars
            cue_start, cue_end = cursor, cursor + share
            cursor = cue_end
            body = wrap_cue(chunk)
            self._cue_index += 1
            if "srt" in self._files:
                self._files["srt"].write(
                    f"{self._cue_index}\n"
                    f"{_timestamp(cue_start, ',')} --> {_timestamp(cue_end, ',')}\n"
                    f"{body}\n\n"
                )
            if "vtt" in self._files:
                self._files["vtt"].write(
                    f"{_timestamp(cue_start, '.')} --> {_timestamp(cue_end, '.')}\n"
                    f"{body}\n\n"
                )

    def _write_text(self, event: "SubtitleEvent", text: str) -> None:
        if self.clock_times:
            stamp = (self.started_at + dt.timedelta(seconds=event.start_time)).strftime(
                "%H:%M:%S"
            )
        else:
            stamp = _timestamp(event.start_time, ".")[:8]  # position in the media
        line = f"[{stamp}] {text}\n"
        source = " ".join(event.source_text.split())
        if self.cfg.include_original and source and source != text:
            language = f"{event.language}: " if event.language else ""
            line += f"           ({language}{source})\n"
        self._files["txt"].write(line)

    def _flush(self) -> None:
        for handle in self._files.values():
            handle.flush()

    def close(self) -> None:
        for handle in self._files.values():
            try:
                handle.close()
            except OSError:
                pass
        self._files.clear()
        # An empty session leaves nothing useful behind; don't litter the
        # folder with header-only files.
        if self.lines_written == 0:
            for path in self.paths.values():
                try:
                    path.unlink()
                except OSError:
                    pass

    def describe(self) -> str:
        names = ", ".join(p.name for p in self.paths.values())
        return f"{names} in {self.folder}"
