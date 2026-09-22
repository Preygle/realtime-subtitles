"""Translation via an instruction-tuned LLM on a llama.cpp server.

This is the default because it handles every language R2T2 can transcribe with
one model, keeps cross-segment context so pronouns and terminology stay stable,
and can be told to produce subtitle-shaped output (short, no commentary).

Runs as a *second* ``llama-server`` instance on its own port so the ASR model
stays resident; a 4B Q4_K_M translator plus R2T2 Q8_0 fits comfortably in 10 GB
of VRAM.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from ..config import TranslateConfig
from .base import Translation, TranslationRequest, Translator, clean_translation

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a live subtitle translator. Translate the user's text into "
    "{target}. Rules:\n"
    "1. Output ONLY the translation. No notes, no romanisation, no quotes.\n"
    "2. Keep it short and natural, the way a subtitle reads.\n"
    "3. Preserve names, numbers and technical terms.\n"
    "4. If the text is already in {target}, repeat it unchanged.\n"
    "5. The text may be a mid-sentence fragment. Translate just the fragment; "
    "do not invent an ending."
)


class LlmTranslator(Translator):
    name = "llamacpp"

    def __init__(self, cfg: TranslateConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self._session = requests.Session()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def translate(self, request: TranslationRequest) -> Translation:
        text = request.text.strip()
        if not text:
            return Translation(text="", skipped=True)

        messages = self._build_messages(request)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
            # Qwen3 and friends: keep reasoning off, it wrecks latency.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        started = time.perf_counter()
        with self._lock:
            response = self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.cfg.timeout_sec,
            )
        latency = time.perf_counter() - started

        if response.status_code >= 400:
            raise RuntimeError(
                f"translator returned {response.status_code}: {response.text[:300]}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"unexpected translator response: {response.text[:300]}"
            ) from exc

        return Translation(text=clean_translation(content or ""), latency=latency)

    def _build_messages(self, request: TranslationRequest) -> list[dict[str, str]]:
        target = request.target_language
        system = _SYSTEM_PROMPT.format(target=target)
        if request.source_language:
            system += f"\nThe source language is {request.source_language}."

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        # Replay recent pairs as turns so the model picks up the register and
        # keeps terminology consistent across subtitle lines.
        for source_text, target_text in request.history[-self.cfg.history_turns :]:
            if source_text and target_text:
                messages.append({"role": "user", "content": source_text})
                messages.append({"role": "assistant", "content": target_text})

        messages.append({"role": "user", "content": request.text.strip()})
        return messages

    # ------------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=3.0)
            if r.status_code == 200:
                return True, f"translator ok at {self.base_url}"
            return False, f"/health returned {r.status_code}"
        except requests.RequestException as exc:
            return False, f"cannot reach {self.base_url}: {exc.__class__.__name__}"

    def close(self) -> None:
        self._session.close()
