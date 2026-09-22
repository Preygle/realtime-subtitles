"""Translation via NLLB-200 on CTranslate2.

Runs on the CPU, so the GPU stays entirely dedicated to R2T2. Much lower
latency per sentence than an LLM and covers 200 languages, at the cost of no
cross-segment context and weaker handling of idiom.

Needs a CTranslate2-converted model directory; see scripts/convert_nllb.py.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..config import PROJECT_ROOT, TranslateConfig
from ..languages import to_flores
from .base import Translation, TranslationRequest, Translator

log = logging.getLogger(__name__)


class NllbTranslator(Translator):
    name = "nllb"

    def __init__(self, cfg: TranslateConfig) -> None:
        self.cfg = cfg

        model_dir = Path(cfg.ct2_model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        if not (model_dir / "model.bin").is_file():
            raise FileNotFoundError(
                f"No CTranslate2 NLLB model at {model_dir}. "
                "Run: python scripts/convert_nllb.py"
            )

        import ctranslate2  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        self._translator = ctranslate2.Translator(
            str(model_dir),
            device=cfg.ct2_device,
            compute_type=cfg.ct2_compute_type,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.ct2_tokenizer)
        self._lock = threading.Lock()
        log.info(
            "NLLB loaded from %s (%s/%s)",
            model_dir,
            cfg.ct2_device,
            cfg.ct2_compute_type,
        )

    # ------------------------------------------------------------------
    def translate(self, request: TranslationRequest) -> Translation:
        text = request.text.strip()
        if not text:
            return Translation(text="", skipped=True)

        src_code = to_flores(request.source_language, default="eng_Latn")
        tgt_code = to_flores(request.target_language, default="eng_Latn")

        started = time.perf_counter()
        with self._lock:
            # NLLB expects the source language token to lead the sequence.
            self._tokenizer.src_lang = src_code
            tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(text)
            )
            results = self._translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_code]],
                beam_size=2,
                max_decoding_length=256,
            )
            hypothesis = results[0].hypotheses[0]
            # Drop the target-language token the prefix forced.
            if hypothesis and hypothesis[0] == tgt_code:
                hypothesis = hypothesis[1:]
            output = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(hypothesis),
                skip_special_tokens=True,
            )
        latency = time.perf_counter() - started

        return Translation(text=output.strip(), latency=latency)

    def health(self) -> tuple[bool, str]:
        return True, f"NLLB on {self.cfg.ct2_device}/{self.cfg.ct2_compute_type}"

    def close(self) -> None:
        self._translator = None
