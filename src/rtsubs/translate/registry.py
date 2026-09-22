"""Factory mapping config strings to translation backends."""

from __future__ import annotations

from ..config import TranslateConfig
from .base import Translation, TranslationRequest, Translator, clean_translation

#: Backend ids offered in the control panel.
BACKEND_IDS = ("llamacpp", "nllb", "passthrough")


def create_translator(cfg: TranslateConfig) -> Translator:
    backend = (cfg.backend or "llamacpp").lower()
    if backend in ("llamacpp", "llm"):
        from .llm import LlmTranslator

        return LlmTranslator(cfg)
    if backend in ("nllb", "ct2"):
        from .nllb import NllbTranslator

        return NllbTranslator(cfg)
    if backend in ("passthrough", "none", "off"):
        return PassthroughTranslator(cfg)
    raise ValueError(
        f"unknown translation backend {cfg.backend!r}; expected one of {BACKEND_IDS}"
    )


from .passthrough import PassthroughTranslator  # noqa: E402  (circular-safe)

__all__ = [
    "Translator",
    "Translation",
    "TranslationRequest",
    "clean_translation",
    "BACKEND_IDS",
    "create_translator",
]
