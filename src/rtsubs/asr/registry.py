"""Factory mapping config strings to ASR backends."""

from __future__ import annotations

from ..config import AsrConfig
from .base import AsrBackend, Transcript, parse_asr_output

#: Backend ids offered in the control panel.
BACKEND_IDS = ("llamacpp", "whispercpp", "mock")


def create_asr(cfg: AsrConfig) -> AsrBackend:
    backend = (cfg.backend or "llamacpp").lower()
    if backend == "llamacpp":
        from .llamacpp import LlamaCppAsr

        return LlamaCppAsr(cfg)
    if backend in ("whispercpp", "whisper"):
        from .whispercpp import WhisperCppAsr

        return WhisperCppAsr(cfg)
    if backend == "mock":
        from .mock import MockAsr

        return MockAsr(cfg)
    raise ValueError(
        f"unknown ASR backend {cfg.backend!r}; expected one of {BACKEND_IDS}"
    )


__all__ = ["AsrBackend", "Transcript", "BACKEND_IDS", "create_asr", "parse_asr_output"]
