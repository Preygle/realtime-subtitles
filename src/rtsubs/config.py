"""Configuration objects for the realtime subtitle pipeline.

Everything is a plain dataclass so the config can be round-tripped to JSON and
edited by hand.  ``AppConfig.load`` merges a user file over the defaults, so a
partial config file only needs to mention what it overrides.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

# The pipeline resamples everything to this rate before it reaches the VAD or
# the ASR backend.  R2T2/Qwen3-ASR expects 16 kHz mono.
SAMPLE_RATE = 16000


@dataclass
class AudioConfig:
    #: "loopback" captures what is playing on the system, "input" a microphone.
    source: str = "loopback"
    #: Substring matched against device names; empty means "system default".
    device_name: str = ""
    #: Capture backend: "auto", "pyaudiowpatch" or "soundcard".
    backend: str = "auto"
    #: Size of a capture read in milliseconds.
    block_ms: int = 32
    #: Extra gain applied after downmixing, in dB. Useful for quiet sources.
    gain_db: float = 0.0


@dataclass
class VadConfig:
    #: "silero" (ONNX, much better) or "energy" (no download required).
    engine: str = "auto"
    #: Probability above which a frame counts as speech (silero only).
    threshold: float = 0.5
    #: Energy VAD threshold in dBFS (energy engine only).
    energy_dbfs: float = -42.0
    #: Trailing silence that closes an utterance.
    min_silence_ms: int = 480
    #: Speech shorter than this is discarded as noise.
    min_speech_ms: int = 220
    #: Audio kept before speech onset so the first phoneme is not clipped.
    speech_pad_ms: int = 300


@dataclass
class SegmenterConfig:
    #: Force a commit if someone talks this long without pausing.
    max_segment_sec: float = 12.0
    #: How often an in-progress utterance is re-transcribed for the live line.
    partial_interval_ms: int = 700
    #: Don't ask for a partial until the utterance has at least this much audio.
    min_partial_ms: int = 600
    #: Audio carried into the next segment after a forced commit, for context.
    carry_over_ms: int = 320


@dataclass
class AsrConfig:
    #: Backend id registered in rtsubs.asr.registry.
    backend: str = "llamacpp"
    base_url: str = "http://127.0.0.1:8090"
    model: str = "Confucius4-R2T2"
    #: --- whispercpp backend ---
    #: Kept separate from base_url so switching backends in the UI does not
    #: require retyping a port.
    whisper_base_url: str = "http://127.0.0.1:8082"
    #: Use Whisper's translate task (any language -> English) in one pass.
    #: When true the pipeline skips the separate translation stage.
    whisper_translate: bool = True
    #: Source language name (e.g. "Chinese"); empty string means auto-detect.
    language: str = ""
    #: Hotwords / topic hint forwarded to the model as a prompt.
    context_prompt: str = ""
    timeout_sec: float = 30.0
    #: Number of previously committed transcripts fed back as context.
    #: Off by default: R2T2 treats the context as text it may continue, and in
    #: testing it re-emitted earlier sentences into new transcripts (seen on
    #: Chinese), which then got translated twice. Hotwords via context_prompt
    #: are unaffected.
    history_turns: int = 0


@dataclass
class TranslateConfig:
    #: "llamacpp", "nllb" or "passthrough".
    backend: str = "llamacpp"
    target_language: str = "English"
    #: --- llamacpp backend ---
    base_url: str = "http://127.0.0.1:8081"
    model: str = "qwen3-4b-instruct"
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_sec: float = 30.0
    #: Previous sentence pairs given to the LLM so pronouns stay consistent.
    history_turns: int = 2
    #: --- nllb (CTranslate2) backend ---
    ct2_model_dir: str = "models/nllb-200-distilled-600M-ct2"
    ct2_tokenizer: str = "facebook/nllb-200-distilled-600M"
    ct2_device: str = "cpu"
    ct2_compute_type: str = "int8"
    #: --- shared ---
    #: Minimum gap between live-line translations (overlay.live_line = "translated").
    partial_throttle_ms: int = 450
    #: Skip translation when the detected source language already matches.
    skip_if_target: bool = True


@dataclass
class OverlayConfig:
    font_family: str = "Segoe UI"
    font_size: int = 20
    #: Background opacity of the caption box, 0.0 = fully transparent.
    background_opacity: float = 0.55
    text_color: str = "#FFFFFF"
    partial_color: str = "#C8D4E8"
    outline_width: int = 3
    #: Fraction of screen height the caption box sits at (0 = top, 1 = bottom).
    vertical_anchor: float = 0.86
    #: Fraction of screen width the caption box occupies.
    width_fraction: float = 0.78
    max_lines: int = 2
    #: What to show while someone is still mid-sentence:
    #:   "off"        -- nothing; each sentence appears once, when it is finished
    #:   "original"   -- the untranslated transcript, growing as they speak
    #:   "translated" -- a translation that is redone as the sentence grows
    #: "translated" is the default: text appears while the person is still
    #: talking. Languages that put the verb last (Japanese, Korean, Turkish,
    #: Hindi...) make it reword itself as the sentence completes; "off" avoids
    #: that, but then nothing shows until the speaker pauses.
    live_line: str = "translated"
    #: Each finished line stays up for len(text) / reading_cps seconds, clamped
    #: to [line_min_sec, line_max_sec]. 15 characters/second is a comfortable
    #: subtitle reading speed.
    reading_cps: float = 15.0
    line_min_sec: float = 2.0
    line_max_sec: float = 6.0
    #: When locked the overlay ignores the mouse entirely (click-through).
    locked: bool = True
    #: Also render the original-language text above the translation.
    show_source_text: bool = False


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "AppConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        cfg = cls()
        if p.is_file():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return cfg
            cfg.apply(raw)
        return cfg

    def apply(self, raw: dict[str, Any]) -> None:
        """Merge a (possibly partial) dict into this config, in place."""
        for key, value in raw.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if dataclasses.is_dataclass(current) and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if hasattr(current, sub_key):
                        setattr(current, sub_key, sub_value)
            else:
                setattr(self, key, value)
