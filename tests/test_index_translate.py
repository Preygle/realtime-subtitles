"""Tests for the Index-Translate backend, against a stub llama-server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rtsubs.config import TranslateConfig
from rtsubs.translate import TranslationRequest, create_translator
from rtsubs.translate.index_translate import IndexTranslator


class _Stub(BaseHTTPRequestHandler):
    reply = "It might rain tomorrow."
    last: dict = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).last = {"path": self.path, "body": body}
        payload = {"choices": [{"message": {"role": "assistant", "content": type(self).reply}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Stub.reply = "It might rain tomorrow."
    _Stub.last = {}
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def _translate(url, text="明日は雨が降るかもしれません。", **kwargs):
    translator = create_translator(TranslateConfig(backend="index", base_url=url))
    request = TranslationRequest(
        text=text, source_language="Japanese", target_language="English", **kwargs
    )
    return translator, translator.translate(request)


def test_registry_builds_index_backend():
    assert isinstance(create_translator(TranslateConfig(backend="index")), IndexTranslator)


@pytest.mark.parametrize(
    "backend,model",
    [
        ("index", "Index-Translate-2B.Q8_0.gguf"),
        ("llamacpp", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        ("llm", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        ("nllb", ""),  # runs on the CPU: no translation server to start
        ("passthrough", ""),
    ],
)
def test_launcher_loads_the_model_the_translator_needs(backend, model):
    from rtsubs.translate.registry import server_model_file

    assert server_model_file(TranslateConfig(backend=backend)) == model


def test_sends_the_official_prompt_as_one_user_message(server):
    _translate(server, history=(("前の文。", "The previous line."),))
    body = _Stub.last["body"]

    assert _Stub.last["path"] == "/v1/chat/completions"
    # Exactly the model's trained instruction: no system prompt, no history.
    assert body["messages"] == [
        {
            "role": "user",
            "content": "Translate the following text into English. Output the translation "
            "directly, without any explanation:\n\n明日は雨が降るかもしれません。",
        }
    ]


def test_uses_greedy_decoding_with_thinking_off(server):
    _translate(server)
    body = _Stub.last["body"]
    assert body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_output_is_cleaned_like_other_llm_backends(server):
    _Stub.reply = '<think></think>\n"It might rain tomorrow."'
    _, result = _translate(server)
    assert result.text == "It might rain tomorrow."


def test_blank_text_is_not_sent(server):
    _, result = _translate(server, text="   ")
    assert result.skipped
    assert _Stub.last == {}
