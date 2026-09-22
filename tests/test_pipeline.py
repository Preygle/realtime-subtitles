"""Tests for the parts of the pipeline that run without models or hardware."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rtsubs.asr.base import parse_asr_output
from rtsubs.audio import dsp
from rtsubs.audio.capture import SilenceSource, WavFileSource
from rtsubs.audio.segmenter import Segmenter
from rtsubs.audio.vad import FRAME_SAMPLES, EnergyVad, create_vad
from rtsubs.config import SAMPLE_RATE, AppConfig, SegmenterConfig, VadConfig
from rtsubs.languages import canonical_name, lookup, same_language, to_flores
from rtsubs.pipeline import SubtitlePipeline
from rtsubs.translate.base import clean_translation


# ---------------------------------------------------------------- helpers
def speech_like(duration: float, freq: float = 180.0, amp: float = 0.35) -> np.ndarray:
    """A harmonic, amplitude-modulated tone that the energy VAD treats as speech."""
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)
    harmonics = (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    )
    return (amp * envelope * harmonics / 1.75).astype(np.float32)


def silence(duration: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)


# ------------------------------------------------------------------- dsp
def test_to_mono_downmixes_interleaved_stereo():
    stereo = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert np.allclose(dsp.to_mono(stereo, 2), [0.5, 0.5])


def test_to_mono_handles_ragged_tail():
    # A short read can end mid-frame; the partial frame must be dropped.
    assert dsp.to_mono(np.ones(5, dtype=np.float32), 2).size == 2


@pytest.mark.parametrize("src,dst", [(48000, 16000), (44100, 16000), (16000, 16000)])
def test_resample_length_and_shape(src, dst):
    seconds = 0.5
    out = dsp.resample(np.zeros(int(src * seconds), dtype=np.float32), src, dst)
    assert abs(out.size - dst * seconds) <= 32
    assert out.dtype == np.float32


def test_wav_round_trip_preserves_signal():
    original = speech_like(0.5)
    wav = dsp.encode_wav(original, SAMPLE_RATE)
    assert wav[:4] == b"RIFF"

    import io
    import wave

    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnchannels() == 1
        decoded = dsp.pcm16_bytes_to_float(handle.readframes(handle.getnframes()))

    assert decoded.size == original.size
    # int16 quantisation error only.
    assert np.max(np.abs(decoded - original)) < 1e-3


def test_dbfs_reports_silence_and_full_scale():
    assert dsp.dbfs(np.zeros(1000, dtype=np.float32)) <= -120.0
    assert dsp.dbfs(np.ones(1000, dtype=np.float32)) == pytest.approx(0.0, abs=0.1)


# ------------------------------------------------------------------- vad
def test_energy_vad_separates_speech_from_silence():
    vad = EnergyVad()
    loud = speech_like(0.2)
    quiet = silence(0.2)

    # Prime the noise-floor tracker on silence first.
    for i in range(0, quiet.size - FRAME_SAMPLES, FRAME_SAMPLES):
        vad.probability(quiet[i : i + FRAME_SAMPLES])

    speech_hits = sum(
        vad.probability(loud[i : i + FRAME_SAMPLES]) >= 0.5
        for i in range(0, loud.size - FRAME_SAMPLES, FRAME_SAMPLES)
    )
    assert speech_hits > 0


def test_create_vad_falls_back_when_silero_missing():
    # models/silero_vad.onnx is not downloaded in CI; auto must not raise.
    assert create_vad(VadConfig(engine="auto")).name in ("silero", "energy")


def test_create_vad_raises_when_silero_explicitly_requested_and_absent(tmp_path):
    from rtsubs.audio import vad as vad_module

    if vad_module.SILERO_MODEL_PATH.is_file():
        pytest.skip("silero model is present")
    with pytest.raises(Exception):
        create_vad(VadConfig(engine="silero"))


# ------------------------------------------------------------- segmenter
def _run_segmenter(stream: np.ndarray, block: int = 512):
    vad_cfg = VadConfig(engine="energy")
    segmenter = Segmenter(EnergyVad(), vad_cfg, SegmenterConfig())
    events = []
    for i in range(0, stream.size, block):
        events.extend(segmenter.feed(stream[i : i + block]))
    events.extend(segmenter.flush())
    return events


def test_segmenter_splits_on_pauses():
    stream = np.concatenate(
        [silence(0.4), speech_like(2.0), silence(0.9), speech_like(1.5), silence(0.9)]
    )
    finals = [e for e in _run_segmenter(stream) if e.is_final]
    assert len(finals) == 2
    assert all(e.duration > 0.5 for e in finals)


def test_segmenter_emits_growing_partials_before_the_final():
    stream = np.concatenate([silence(0.3), speech_like(3.0), silence(0.9)])
    events = _run_segmenter(stream)
    partials = [e for e in events if not e.is_final]

    assert len(partials) >= 2
    # Each partial must be a strict superset of the previous one, which is what
    # keeps the on-screen live line from flickering.
    durations = [e.duration for e in partials]
    assert durations == sorted(durations)
    assert all(e.utterance_id == partials[0].utterance_id for e in partials)


def test_segmenter_drops_sub_threshold_blips():
    stream = np.concatenate([silence(0.5), speech_like(0.05), silence(0.9)])
    assert [e for e in _run_segmenter(stream) if e.is_final] == []


def test_segmenter_force_commits_long_speech():
    cfg = SegmenterConfig(max_segment_sec=2.0)
    segmenter = Segmenter(EnergyVad(), VadConfig(engine="energy"), cfg)
    stream = np.concatenate([silence(0.3), speech_like(7.0)])

    events = []
    for i in range(0, stream.size, 512):
        events.extend(segmenter.feed(stream[i : i + 512]))

    forced = [e for e in events if e.is_final and e.forced]
    assert len(forced) >= 2
    assert all(e.duration <= cfg.max_segment_sec + 0.1 for e in forced)
    # Audio is carried across a forced cut so split words survive.
    assert any(e.continued for e in events if e.is_final)


def test_segmenter_reset_clears_state():
    segmenter = Segmenter(EnergyVad(), VadConfig(engine="energy"), SegmenterConfig())
    segmenter.feed(speech_like(1.0))
    segmenter.reset()
    assert not segmenter.in_speech
    assert segmenter.flush() == []


# ------------------------------------------------------------- languages
@pytest.mark.parametrize(
    "value,expected",
    [
        ("Chinese", "Chinese"),
        ("zh", "Chinese"),
        ("zh-CN", "Chinese"),
        ("zho_Hans", "Chinese"),
        ("mandarin", "Chinese"),
        ("ENGLISH", "English"),
        ("en-US", "English"),
        ("farsi", "Persian"),
        ("nonsense", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_language_lookup_is_forgiving(value, expected):
    assert canonical_name(value) == expected


def test_flores_codes_for_nllb():
    assert to_flores("Japanese") == "jpn_Jpan"
    assert to_flores("unknown-language") == "eng_Latn"


def test_same_language_matches_across_spellings():
    assert same_language("en", "English")
    assert not same_language("Chinese", "English")
    # An unknown source must not be assumed to match the target.
    assert not same_language("", "English")


def test_every_language_is_self_consistent():
    from rtsubs.languages import LANGUAGES

    assert len({lang.iso for lang in LANGUAGES}) == len(LANGUAGES)
    for lang in LANGUAGES:
        assert lookup(lang.iso) is lang
        assert lookup(lang.flores) is lang


# ----------------------------------------------------------- asr parsing
@pytest.mark.parametrize(
    "raw,text,language",
    [
        ("language English<asr_text>hello there", "hello there", "English"),
        ("language Chinese<asr_text>你好", "你好", "Chinese"),
        ("Japanese<asr_text>こんにちは", "こんにちは", "Japanese"),
        ("language: Spanish <asr_text> hola ", "hola", "Spanish"),
        ("plain text", "plain text", ""),
        ("language English<asr_text>x<|endoftext|>", "x", "English"),
        ("", "", ""),
    ],
)
def test_parse_asr_output(raw, text, language):
    assert parse_asr_output(raw) == (text, language)


# --------------------------------------------------- translation cleanup
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Here is the translation: Hello world", "Hello world"),
        ("<think>reasoning</think>\nHello world", "Hello world"),
        ('"Hello world"', "Hello world"),
        ("Translation: Hello\nworld", "Hello world"),
        ("Sure! Here's the English translation - Good morning", "Good morning"),
        # Internal quotes must survive.
        ('He said "stop" and left', 'He said "stop" and left'),
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_clean_translation(raw, expected):
    assert clean_translation(raw) == expected


# ---------------------------------------------------------------- audio
def test_silence_source_produces_blocks():
    source = SilenceSource(block_ms=32)
    source.start()
    try:
        blocks = [source.read(1.0) for _ in range(3)]
    finally:
        source.stop()
    assert all(b is not None and b.size == FRAME_SAMPLES for b in blocks)
    assert not source.running


def test_capture_source_drops_oldest_when_consumer_stalls():
    """A stalled consumer must cost us old audio, not unbounded memory."""
    from rtsubs.audio.capture import _QUEUE_DEPTH

    source = SilenceSource()
    block = np.zeros(512, dtype=np.float32)
    # Drive _emit directly so the test does not depend on thread scheduling.
    for i in range(_QUEUE_DEPTH + 10):
        source._emit(block + i)

    assert source.dropped_blocks == 10
    assert source._queue.qsize() == _QUEUE_DEPTH
    # The surviving head is block 10, i.e. the oldest ten were shed.
    assert source.read(0.1)[0] == pytest.approx(10.0)


# -------------------------------------------------------------- pipeline
def test_pipeline_end_to_end_with_mock_backends(tmp_path):
    wav = tmp_path / "clip.wav"
    stream = np.concatenate(
        [silence(0.4), speech_like(2.0), silence(0.9), speech_like(1.6), silence(0.8)]
    )
    wav.write_bytes(dsp.encode_wav(stream, SAMPLE_RATE))

    cfg = AppConfig()
    cfg.asr.backend = "mock"
    cfg.asr.language = "Chinese"
    cfg.translate.backend = "passthrough"
    cfg.vad.engine = "energy"

    events = []
    pipeline = SubtitlePipeline(
        cfg,
        on_event=events.append,
        source=WavFileSource(str(wav)),
    )
    pipeline.start()
    deadline = time.time() + 20
    while time.time() < deadline and pipeline._source.running:
        time.sleep(0.05)
    time.sleep(1.5)
    pipeline.stop()

    finals = [e for e in events if e.is_final]
    assert len(finals) == 2
    assert all(e.text for e in finals)
    assert all(e.language == "Chinese" for e in finals)
    assert pipeline.stats.asr_errors == 0
    assert pipeline.stats.translate_errors == 0


def test_pipeline_survives_a_failing_asr_backend():
    """An exploding backend must be logged, not crash the worker thread."""

    class Boom:
        name = "boom"

        def transcribe(self, *a, **k):
            raise RuntimeError("model exploded")

        def health(self):
            return True, "ok"

        def close(self):
            pass

    cfg = AppConfig()
    cfg.asr.backend = "mock"
    cfg.translate.backend = "passthrough"
    cfg.vad.engine = "energy"

    statuses = []
    pipeline = SubtitlePipeline(
        cfg,
        on_event=lambda e: None,
        on_status=lambda level, msg: statuses.append((level, msg)),
        source=SilenceSource(),
    )
    pipeline.start()
    pipeline._asr = Boom()  # swap in after start so health() passes
    try:
        from rtsubs.audio.segmenter import Segment

        pipeline._asr_queue.put_final(
            _job(Segment(1, speech_like(1.0), True, 0.0, 1.0))
        )
        time.sleep(1.0)
    finally:
        pipeline.stop()

    assert pipeline.stats.asr_errors >= 1
    assert any(level == "error" for level, _ in statuses)


def _job(segment):
    from rtsubs.pipeline import _Job

    return _Job(segment, time.perf_counter())


# ----------------------------------------------------------------- config
def test_config_partial_merge_keeps_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"overlay": {"font_size": 44}, "log_level": "DEBUG"}')

    cfg = AppConfig.load(path)
    assert cfg.overlay.font_size == 44
    assert cfg.log_level == "DEBUG"
    # Untouched fields keep their defaults.
    assert cfg.overlay.max_lines == AppConfig().overlay.max_lines
    assert cfg.asr.backend == AppConfig().asr.backend


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    cfg = AppConfig()
    cfg.asr.language = "Japanese"
    cfg.overlay.vertical_anchor = 0.5
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert loaded.asr.language == "Japanese"
    assert loaded.overlay.vertical_anchor == 0.5


def test_config_load_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json")
    assert AppConfig.load(path).asr.backend == AppConfig().asr.backend


# ----------------------------------------------------- real-speech checks
SPEECH_FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "speech_en.wav"


def test_silero_detects_real_speech():
    """Regression: without Silero v5's 64-sample context, speech scored ~0.1.

    That made the whole app silent once the Silero model was downloaded. The
    synthetic tones used elsewhere cannot catch this -- Silero is trained to
    reject them -- so this uses a short clip of real (TTS) speech.
    """
    from rtsubs.audio import vad as vad_module

    if not vad_module.SILERO_MODEL_PATH.is_file():
        pytest.skip("silero model not downloaded")
    pytest.importorskip("onnxruntime")

    audio = dsp.read_wav_mono(str(SPEECH_FIXTURE), SAMPLE_RATE)
    silero = vad_module.SileroVad()
    probs = [
        silero.probability(audio[i : i + FRAME_SAMPLES])
        for i in range(0, audio.size - FRAME_SAMPLES, FRAME_SAMPLES)
    ]
    voiced = sum(p >= 0.5 for p in probs) / len(probs)
    assert voiced > 0.4, f"only {voiced:.0%} of speech frames detected"

    # And it must still reject silence.
    silero.reset()
    quiet = [silero.probability(np.zeros(FRAME_SAMPLES, np.float32)) for _ in range(20)]
    assert max(quiet) < 0.5


def test_segmenter_finds_real_speech_with_silero():
    from rtsubs.audio import vad as vad_module

    if not vad_module.SILERO_MODEL_PATH.is_file():
        pytest.skip("silero model not downloaded")
    pytest.importorskip("onnxruntime")

    audio = np.concatenate(
        [silence(0.5), dsp.read_wav_mono(str(SPEECH_FIXTURE), SAMPLE_RATE), silence(1.0)]
    )
    segmenter = Segmenter(vad_module.SileroVad(), VadConfig(), SegmenterConfig())
    events = []
    for i in range(0, audio.size, 512):
        events.extend(segmenter.feed(audio[i : i + 512]))
    events.extend(segmenter.flush())
    assert [e for e in events if e.is_final], "no utterance detected in real speech"
