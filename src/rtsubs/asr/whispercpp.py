"""Whisper through a whisper.cpp server -- single-pass speech translation.

Whisper's ``translate`` task transcribes any supported language and emits
**English**, in one model pass. That is normally a limitation; for this app it
is exactly the goal, and it removes the second model entirely:

    R2T2 path : audio -> R2T2 (:8090) -> source text -> LLM (:8081) -> English
    Whisper   : audio -> whisper.cpp (:8082) ----------------------> English

Lower latency and roughly half the VRAM, at two costs worth knowing about:

* Whisper re-decodes its whole window each call, so a growing partial can
  *revise* words it already emitted. R2T2 is append-only and never does. For a
  caption you read while watching something, that stability is the thing R2T2
  buys you.
* Whisper's built-in translation is weaker than a dedicated LLM translator,
  and it can only target English.

Set ``asr.whisper_translate`` to false to get plain transcription in the source
language and run the normal translation stage after it.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import requests

from ..audio import dsp
from ..config import SAMPLE_RATE, AsrConfig
from ..languages import canonical_name, lookup
from .base import AsrBackend, Transcript, parse_asr_output

log = logging.getLogger(__name__)


class WhisperCppAsr(AsrBackend):
    name = "whispercpp"

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.whisper_base_url.rstrip("/")
        self.translate = cfg.whisper_translate
        self._session = requests.Session()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def transcribe(
        self, audio: np.ndarray, language: str = "", context: str = ""
    ) -> Transcript:
        if audio.size == 0:
            return Transcript(text="")

        wav = dsp.encode_wav(audio, SAMPLE_RATE)
        files = {"file": ("segment.wav", wav, "audio/wav")}
        data: dict[str, str] = {
            "response_format": "json",
            "temperature": "0.0",
            "temperature_inc": "0.2",
            "no_timestamps": "true",
            # Whisper wants an ISO code, or "auto" to detect.
            "language": self._iso_code(language),
            "translate": "true" if self.translate else "false",
        }
        if context:
            data["prompt"] = context

        started = time.perf_counter()
        with self._lock:
            response = self._session.post(
                f"{self.base_url}/inference",
                files=files,
                data=data,
                timeout=self.cfg.timeout_sec,
            )
        latency = time.perf_counter() - started

        if response.status_code >= 400:
            raise RuntimeError(
                f"whisper.cpp returned {response.status_code}: {response.text[:300]}"
            )

        raw_text, detected = self._extract(response)
        text, tag_language = parse_asr_output(raw_text)
        text = _strip_whisper_noise(text)

        return Transcript(
            text=text,
            language=detected or tag_language or canonical_name(language, ""),
            latency=latency,
            raw=raw_text,
            already_translated=self.translate,
        )

    @staticmethod
    def _iso_code(language: str) -> str:
        found = lookup(language)
        return found.iso if found else "auto"

    @staticmethod
    def _extract(response: requests.Response) -> tuple[str, str]:
        """Return (text, detected_language_name) from a whisper.cpp reply."""
        try:
            payload = response.json()
        except ValueError:
            return response.text, ""

        if not isinstance(payload, dict):
            return response.text, ""

        text = ""
        for key in ("text", "transcription"):
            if isinstance(payload.get(key), str):
                text = payload[key]
                break
        else:
            # verbose_json-style replies carry a segment list instead.
            segments = payload.get("segments")
            if isinstance(segments, list):
                text = "".join(
                    seg.get("text", "")
                    for seg in segments
                    if isinstance(seg, dict)
                )

        # Present only on verbose_json; harmless when missing.
        detected = canonical_name(payload.get("language"), default="")
        return text, detected


    # ------------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        mode = "translate to English" if self.translate else "transcribe only"
        try:
            # whisper.cpp serves a web UI at "/" and has no /health endpoint,
            # so any HTTP response at all proves a server is listening.
            self._session.get(self.base_url, timeout=3.0)
            return True, f"whisper.cpp ok at {self.base_url} ({mode})"
        except requests.RequestException as exc:
            return False, f"cannot reach {self.base_url}: {exc.__class__.__name__}"

    def close(self) -> None:
        self._session.close()


#: Whisper emits these when fed silence or music. They are not speech.
_HALLUCINATION_MARKERS = (
    "[blank_audio]",
    "[silence]",
    "[music]",
    "(music)",
    "[applause]",
    "(applause)",
    "[inaudible]",
    "[ silence ]",
    "[sound]",
    "[noise]",
)


def _strip_whisper_noise(text: str) -> str:
    """Drop Whisper's bracketed non-speech annotations.

    Whisper reliably emits things like ``[BLANK_AUDIO]`` or ``[Music]`` on
    near-silent input. The segmenter's VAD filters most of that out already,
    but music beds and room tone still get through, and a caption reading
    "[Music]" every few seconds is worse than no caption.
    """
    cleaned = text.strip()
    lowered = cleaned.casefold()
    for marker in _HALLUCINATION_MARKERS:
        if lowered == marker:
            return ""
        # Also strip a marker that merely leads or trails real speech.
        if lowered.startswith(marker):
            cleaned = cleaned[len(marker) :].strip()
            lowered = cleaned.casefold()
        if lowered.endswith(marker):
            cleaned = cleaned[: len(cleaned) - len(marker)].strip()
            lowered = cleaned.casefold()
    return cleaned
