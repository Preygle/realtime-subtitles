"""No-op translator: shows the transcript in its original language."""

from __future__ import annotations

from ..config import TranslateConfig
from .base import Translation, TranslationRequest, Translator


class PassthroughTranslator(Translator):
    name = "passthrough"

    def __init__(self, cfg: TranslateConfig) -> None:
        self.cfg = cfg

    def translate(self, request: TranslationRequest) -> Translation:
        return Translation(text=request.text.strip(), skipped=True)

    def health(self) -> tuple[bool, str]:
        return True, "passthrough (transcription only, no translation)"
