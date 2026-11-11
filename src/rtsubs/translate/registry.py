"""Factory mapping config strings to translation backends."""

from __future__ import annotations

from ..config import TranslateConfig
from .base import Translation, TranslationRequest, Translator, clean_translation

#: Backend ids offered in the control panel.
BACKEND_IDS = ("llamacpp", "index", "nllb", "passthrough")

#: The GGUF each llama-server-backed translator expects on port 8081, so the
#: launcher loads the model that matches the chosen backend.
SERVER_MODEL_FILES = {
    "llamacpp": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "llm": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "index": "Index-Translate-2B.Q8_0.gguf",
}


def server_model_file(cfg: TranslateConfig) -> str:
    """Model file the translation server must load, or "" if none is needed."""
    return SERVER_MODEL_FILES.get((cfg.backend or "llamacpp").lower(), "")


def create_translator(cfg: TranslateConfig) -> Translator:
    backend = (cfg.backend or "llamacpp").lower()
    if backend in ("llamacpp", "llm"):
        from .llm import LlmTranslator

        return LlmTranslator(cfg)
    if backend == "index":
        from .index_translate import IndexTranslator

        return IndexTranslator(cfg)
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
