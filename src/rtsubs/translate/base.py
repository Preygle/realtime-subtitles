"""Translation backend interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Translation:
    text: str
    latency: float = 0.0
    #: True when the stage deliberately passed the text through untranslated
    #: (source already in the target language, or passthrough backend).
    skipped: bool = False


@dataclass
class TranslationRequest:
    text: str
    #: Canonical source language name; "" when unknown.
    source_language: str
    target_language: str
    #: Recent (source, target) pairs, oldest first, for pronoun continuity.
    history: tuple[tuple[str, str], ...] = ()
    #: False for the still-changing live line, which may be a partial sentence.
    is_final: bool = True


class Translator(ABC):
    name = "translator"

    @abstractmethod
    def translate(self, request: TranslationRequest) -> Translation:
        """Translate ``request.text`` into ``request.target_language``."""

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def close(self) -> None:
        """Release sockets or model handles."""


# ----------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
#: Instruction-tuned models love to preface the answer.
_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is)\s+(?:the\s+)?"
    r"(?:english\s+)?translation\s*[:\-]?|translation\s*[:\-]|english\s*[:\-])\s*",
    re.IGNORECASE,
)


def clean_translation(raw: str) -> str:
    """Strip reasoning blocks, preambles and stray quoting from LLM output."""
    text = _THINK_BLOCK.sub("", raw or "")
    text = _UNCLOSED_THINK.sub("", text)
    text = text.strip()
    text = _PREAMBLE.sub("", text).strip()

    # Models often wrap the whole answer in quotes; drop them only when they
    # enclose the entire string, so real quoted speech survives.
    for opener, closer in (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»")):
        if len(text) >= 2 and text.startswith(opener) and text.endswith(closer):
            inner = text[1:-1]
            if opener not in inner and closer not in inner:
                text = inner.strip()
                break

    # Collapse the newlines an LLM may add; subtitles are laid out by the UI.
    return " ".join(text.split())
