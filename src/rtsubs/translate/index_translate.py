"""Translation with Index-Translate, a dedicated translation model.

Index-Translate (IndexTeam / bilibili) is a 2B or 9B model trained only to
translate, covering 150 languages. It runs on the same llama-server slot as
the general-purpose LLM translator (port 8081); only the prompt differs.

It is trained on one fixed instruction, so this backend sends exactly the
official default prompt -- a single user message with greedy decoding --
instead of the LLM translator's system prompt and conversation history.
See https://github.com/bilibili/Index-Translate/blob/main/docs/prompts.md
"""

from __future__ import annotations

from .base import TranslationRequest
from .llm import LlmTranslator

#: The official English form of the default translation prompt. It expects a
#: full language name ("English", not "en"), which is what the config stores.
_PROMPT = (
    "Translate the following text into {target}. "
    "Output the translation directly, without any explanation:\n\n{text}"
)


class IndexTranslator(LlmTranslator):
    name = "index"

    def _temperature(self) -> float:
        # The model card recommends greedy decoding.
        return 0.0

    def _build_messages(self, request: TranslationRequest) -> list[dict[str, str]]:
        prompt = _PROMPT.format(target=request.target_language, text=request.text.strip())
        return [{"role": "user", "content": prompt}]
