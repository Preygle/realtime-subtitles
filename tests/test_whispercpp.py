"""Tests for the whisper.cpp backend, against a stub HTTP server.

These pin the wire contract (endpoint, form fields, response shapes) so a
change to the backend cannot silently stop matching whisper-server's API.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from rtsubs.asr.whispercpp import WhisperCppAsr, _strip_whisper_noise
from rtsubs.config import SAMPLE_RATE, AppConfig, AsrConfig
from rtsubs.pipeline import SubtitlePipeline


class _StubHandler(BaseHTTPRequestHandler):
    #: Set per-test: the JSON body the stub replies with.
    reply: dict | str = {"text": "hello world"}
    status: int = 200
    #: Filled in by the handler so tests can assert on what was sent.
    last_request: dict = {}

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html>whisper.cpp</html>")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Crude multipart parse -- enough to assert on the form fields.
        fields: dict[str, str] = {}
        text = body.decode("utf-8", errors="replace")
        for chunk in text.split("--"):
            if 'name="' not in chunk:
                continue
            name = chunk.split('name="', 1)[1].split('"', 1)[0]
            value = chunk.split("\r\n\r\n", 1)[-1].rsplit("\r\n", 1)[0]
            fields[name] = value

        type(self).last_request = {
            "path": self.path,
            "fields": fields,
            "has_file": b"segment.wav" in body,
            "wav_magic": b"RIFF" in body,
        }

        payload = type(self).reply
        raw = json.dumps(payload) if isinstance(payload, dict) else payload
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw.encode("utf-8"))


@pytest.fixture
def stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _StubHandler.reply = {"text": "hello world"}
    _StubHandler.status = 200
    _StubHandler.last_request = {}
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _audio(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _backend(url: str, **kwargs) -> WhisperCppAsr:
    cfg = AsrConfig(backend="whispercpp", whisper_base_url=url, **kwargs)
    return WhisperCppAsr(cfg)


# ------------------------------------------------------------------ wire
def test_posts_wav_to_inference_endpoint(stub_server):
    _, url = stub_server
    backend = _backend(url)
    result = backend.transcribe(_audio(), language="Japanese")

    request = _StubHandler.last_request
    assert request["path"] == "/inference"
    assert request["has_file"] and request["wav_magic"]
    assert result.text == "hello world"


def test_language_name_is_converted_to_iso_code(stub_server):
    _, url = stub_server
    _backend(url).transcribe(_audio(), language="Japanese")
    assert _StubHandler.last_request["fields"]["language"] == "ja"


def test_empty_language_becomes_auto(stub_server):
    _, url = stub_server
    _backend(url).transcribe(_audio(), language="")
    assert _StubHandler.last_request["fields"]["language"] == "auto"


def test_translate_flag_follows_config(stub_server):
    _, url = stub_server
    _backend(url, whisper_translate=True).transcribe(_audio())
    assert _StubHandler.last_request["fields"]["translate"] == "true"

    _backend(url, whisper_translate=False).transcribe(_audio())
    assert _StubHandler.last_request["fields"]["translate"] == "false"


def test_context_is_sent_as_prompt(stub_server):
    _, url = stub_server
    _backend(url).transcribe(_audio(), context="Kubernetes, Anthropic")
    assert _StubHandler.last_request["fields"]["prompt"] == "Kubernetes, Anthropic"


# -------------------------------------------------------------- results
def test_translate_mode_marks_transcript_as_already_translated(stub_server):
    _, url = stub_server
    assert _backend(url, whisper_translate=True).transcribe(_audio()).already_translated
    assert not _backend(url, whisper_translate=False).transcribe(_audio()).already_translated


def test_detected_language_from_verbose_json(stub_server):
    _, url = stub_server
    _StubHandler.reply = {"text": "konnichiwa", "language": "ja"}
    result = _backend(url).transcribe(_audio())
    assert result.language == "Japanese"


def test_segment_list_response_is_joined(stub_server):
    _, url = stub_server
    _StubHandler.reply = {
        "segments": [{"text": "hello "}, {"text": "there"}],
        "language": "en",
    }
    assert _backend(url).transcribe(_audio()).text == "hello there"


def test_http_error_raises(stub_server):
    _, url = stub_server
    _StubHandler.status = 500
    _StubHandler.reply = "model not loaded"
    with pytest.raises(RuntimeError, match="500"):
        _backend(url).transcribe(_audio())


def test_empty_audio_short_circuits(stub_server):
    _, url = stub_server
    assert _backend(url).transcribe(np.zeros(0, dtype=np.float32)).text == ""
    # Nothing should have been sent.
    assert _StubHandler.last_request == {}


def test_health_reports_reachability(stub_server):
    _, url = stub_server
    ok, detail = _backend(url).health()
    assert ok and "translate to English" in detail

    bad_ok, bad_detail = _backend("http://127.0.0.1:1").health()
    assert not bad_ok and "cannot reach" in bad_detail


# --------------------------------------------------- hallucination filter
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[BLANK_AUDIO]", ""),
        ("[Music]", ""),
        ("(applause)", ""),
        ("[ Silence ]", ""),
        ("[Music] and then he spoke", "and then he spoke"),
        ("he stopped talking [Music]", "he stopped talking"),
        ("a normal sentence", "a normal sentence"),
        # A bracketed word inside real speech must survive.
        ("the [sic] original text", "the [sic] original text"),
    ],
)
def test_strip_whisper_noise(raw, expected):
    assert _strip_whisper_noise(raw) == expected


def test_blank_audio_is_not_emitted_as_a_subtitle(stub_server):
    _, url = stub_server
    _StubHandler.reply = {"text": "[BLANK_AUDIO]"}
    assert _backend(url).transcribe(_audio()).text == ""


# ------------------------------------------------------------- pipeline
def test_pipeline_skips_translation_in_single_pass_mode(stub_server):
    """The translator must never see text Whisper already rendered to English."""
    _, url = stub_server
    _StubHandler.reply = {"text": "already in english", "language": "ja"}

    cfg = AppConfig()
    cfg.asr.backend = "whispercpp"
    cfg.asr.whisper_base_url = url
    cfg.asr.whisper_translate = True
    cfg.translate.backend = "passthrough"
    cfg.vad.engine = "energy"

    events: list = []
    # SilenceSource keeps the test off real audio hardware, which CI runners
    # do not have; the segment under test is injected directly below.
    from rtsubs.audio.capture import SilenceSource

    pipeline = SubtitlePipeline(cfg, on_event=events.append, source=SilenceSource())

    calls = []

    class SpyTranslator:
        name = "spy"

        def translate(self, request):
            calls.append(request)
            raise AssertionError("translation stage must be bypassed")

        def health(self):
            return True, "spy"

        def close(self):
            pass

    from rtsubs.audio.segmenter import Segment
    from rtsubs.pipeline import _Job

    pipeline._asr = pipeline._asr or None
    pipeline.start()
    pipeline._translator = SpyTranslator()
    try:
        pipeline._asr_queue.put_final(
            _Job(Segment(1, _audio(1.5), True, 0.0, 1.5), time.perf_counter())
        )
        deadline = time.time() + 5
        while time.time() < deadline and not events:
            time.sleep(0.05)
    finally:
        pipeline.stop()

    assert calls == []
    assert len(events) == 1
    assert events[0].text == "already in english"
    assert events[0].language == "Japanese"
    assert pipeline.stats.translate_errors == 0
