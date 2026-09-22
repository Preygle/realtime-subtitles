"""A fake ASR backend.

Lets the whole pipeline -- capture, VAD, segmentation, translation plumbing,
overlay rendering -- be exercised end to end before any model weights are
downloaded. It produces deterministic pseudo-text whose length tracks the
duration of the segment, and simulates a plausible amount of latency.
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np

from ..config import SAMPLE_RATE, AsrConfig
from .base import AsrBackend, Transcript

log = logging.getLogger(__name__)

#: Short phrases in a few scripts, so the overlay's font handling gets tested.
_PHRASE_POOL: dict[str, list[str]] = {
    "English": ["the quick brown fox", "jumps over", "the lazy dog by the river"],
    "Chinese": ["今天天气很好", "我们一起去公园", "这个模型运行得很快"],
    "Japanese": ["今日はいい天気です", "一緒に公園へ行きましょう"],
    "Spanish": ["hace buen tiempo hoy", "vamos juntos al parque"],
    "French": ["il fait beau aujourd'hui", "allons ensemble au parc"],
}


class MockAsr(AsrBackend):
    name = "mock"

    def __init__(self, cfg: AsrConfig, latency: float = 0.12) -> None:
        self.cfg = cfg
        self.simulated_latency = latency

    def transcribe(
        self, audio: np.ndarray, language: str = "", context: str = ""
    ) -> Transcript:
        started = time.perf_counter()
        time.sleep(self.simulated_latency)

        lang = language or "English"
        pool = _PHRASE_POOL.get(lang, _PHRASE_POOL["English"])

        duration = audio.size / SAMPLE_RATE
        # Roughly two words per second, so partials visibly grow.
        n_words = max(1, int(duration * 2))

        # Seed from the audio content so the same segment yields the same text,
        # and a growing partial keeps its existing prefix stable.
        digest = hashlib.sha256(
            np.asarray(audio[:4096], dtype=np.float32).tobytes()
        ).digest()

        words: list[str] = []
        for phrase in pool:
            words.extend(phrase.split())
        if not words:
            words = ["..."]

        offset = digest[0] % len(words)
        picked = [words[(offset + i) % len(words)] for i in range(n_words)]
        text = " ".join(picked)

        return Transcript(
            text=text,
            language=lang,
            latency=time.perf_counter() - started,
            raw=f"language {lang}<asr_text>{text}",
        )

    def health(self) -> tuple[bool, str]:
        return True, "mock backend (no model required)"
