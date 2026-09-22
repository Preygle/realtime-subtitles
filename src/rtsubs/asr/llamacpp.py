"""R2T2 through a llama.cpp server.

``llama-server`` exposes the Qwen3-ASR family (which Confucius4-R2T2 is built
on) at the OpenAI-compatible ``/v1/audio/transcriptions`` endpoint once the
model is loaded together with its ``mmproj`` audio encoder.

This is a per-segment request rather than R2T2's native append-only streaming
state: llama.cpp does not expose ``unfixed_token_num`` / LSP. The segmenter
compensates by feeding short, VAD-bounded chunks and re-sending the in-progress
utterance for the live line.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import requests

from ..audio import dsp
from ..config import SAMPLE_RATE, AsrConfig
from .base import AsrBackend, Transcript, parse_asr_output

log = logging.getLogger(__name__)


class LlamaCppAsr(AsrBackend):
    name = "llamacpp"

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self._session = requests.Session()
        # One in-flight request at a time; llama-server serialises anyway and
        # this keeps the connection pool from thrashing.
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
            "model": self.cfg.model,
            "response_format": "json",
            "temperature": "0",
        }
        if language:
            data["language"] = language
        if context:
            # Doubles as R2T2's hotword/context hint.
            data["prompt"] = context

        started = time.perf_counter()
        with self._lock:
            response = self._session.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=self.cfg.timeout_sec,
            )
        latency = time.perf_counter() - started

        if response.status_code >= 400:
            raise RuntimeError(
                f"ASR server returned {response.status_code}: {response.text[:300]}"
            )

        raw_text = self._extract_text(response)
        text, detected = parse_asr_output(raw_text)
        return Transcript(
            text=text,
            language=detected or language,
            latency=latency,
            raw=raw_text,
        )

    @staticmethod
    def _extract_text(response: requests.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                payload = response.json()
            except ValueError:
                return response.text
            if isinstance(payload, dict):
                for key in ("text", "transcription", "content"):
                    if isinstance(payload.get(key), str):
                        return payload[key]
                # Some builds answer in chat-completion shape.
                choices = payload.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        message = first.get("message") or {}
                        if isinstance(message.get("content"), str):
                            return message["content"]
                        if isinstance(first.get("text"), str):
                            return first["text"]
            return response.text
        return response.text

    # ------------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=3.0)
            if r.status_code == 200:
                return True, f"llama-server ok at {self.base_url}"
            return False, f"/health returned {r.status_code}"
        except requests.RequestException as exc:
            return False, f"cannot reach {self.base_url}: {exc.__class__.__name__}"

    def close(self) -> None:
        self._session.close()
