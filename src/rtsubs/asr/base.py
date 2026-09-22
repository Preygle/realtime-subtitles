"""ASR backend interface.

Keeping this narrow is what lets the llama.cpp/Vulkan backend be swapped for a
native R2T2 backend (vLLM, or the project's own WebSocket server) later without
touching the pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Transcript:
    text: str
    #: Language the model reported, as a canonical name. "" when unknown.
    language: str = ""
    #: Round-trip time of the request, in seconds.
    latency: float = 0.0
    #: Raw backend response, kept for the debug log.
    raw: str = ""
    #: True when ``text`` is already in the target language because the model
    #: did recognition and translation in one pass (Whisper's translate task).
    #: The pipeline then skips the translation stage entirely.
    already_translated: bool = False


class AsrBackend(ABC):
    """Transcribes a chunk of 16 kHz mono float32 audio."""

    name = "asr"

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "",
        context: str = "",
    ) -> Transcript:
        """Transcribe ``audio``.

        ``language`` is a language *name* ("Chinese"); empty means auto-detect.
        ``context`` is a hotword/continuation hint the model may use as a prompt.
        """

    def health(self) -> tuple[bool, str]:
        """Return (reachable, human-readable detail) for the control panel."""
        return True, "ok"

    def close(self) -> None:
        """Release sockets or model handles."""


# ----------------------------------------------------------------------
# Output cleanup
#
# Qwen3-ASR-family models in llama.cpp prefix the transcript with a detected
# language tag, e.g. "language English<asr_text>hello there". Clients that do
# not expect it end up displaying the tag (ggml-org/llama.cpp#26749), so we
# strip it here and recover the detected language from it.

_LANG_TAG = re.compile(
    r"^\s*(?:language\s*[:\s]\s*)?(?P<lang>[A-Za-z][A-Za-z \-]{1,24}?)\s*"
    r"<\s*asr_text\s*>\s*",
    re.IGNORECASE,
)
_STRAY_TAGS = re.compile(r"<\s*/?\s*(?:asr_text|\|?endoftext\|?|im_end)\s*>", re.IGNORECASE)
#: Models sometimes emit a bare "<|...|>" control token at the edges.
_CONTROL = re.compile(r"<\|[^|>]*\|>")


def parse_asr_output(raw: str) -> tuple[str, str]:
    """Split a raw backend string into (clean_text, detected_language_name).

    >>> parse_asr_output("language English<asr_text>hello there")
    ('hello there', 'English')
    >>> parse_asr_output("  plain text ")
    ('plain text', '')
    """
    from ..languages import canonical_name

    text = raw or ""
    detected = ""

    match = _LANG_TAG.match(text)
    if match:
        detected = canonical_name(match.group("lang"), default="")
        text = text[match.end() :]

    text = _STRAY_TAGS.sub("", text)
    text = _CONTROL.sub("", text)
    return text.strip(), detected
