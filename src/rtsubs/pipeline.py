"""The realtime pipeline: capture -> VAD -> ASR -> translation -> callbacks.

Four stages run concurrently so a slow model never stalls audio capture:

    capture thread  ->  audio thread   ->  ASR thread   ->  translate thread
    (AudioSource)       (VAD/segment)      (R2T2)           (LLM or NLLB)

The queues between stages are *latest-wins for partials, FIFO for finals*. A
final transcript must never be dropped -- it is the committed subtitle line --
but a stale partial is worthless, so when the ASR stage falls behind, only the
newest in-progress snapshot survives. This is what keeps subtitles tracking
live audio instead of drifting further behind as the backlog grows.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .asr import AsrBackend, Transcript, create_asr
from .audio.capture import AudioSource, create_source
from .audio.segmenter import Segment, Segmenter
from .audio.vad import create_vad
from .config import SAMPLE_RATE, AppConfig
from .languages import canonical_name, same_language
from .translate import TranslationRequest, Translator, create_translator

log = logging.getLogger(__name__)


@dataclass
class SubtitleEvent:
    """One subtitle update handed to the UI."""

    utterance_id: int
    #: Transcript in the spoken language.
    source_text: str
    #: Text to display (translated, or the source when translation is skipped).
    text: str
    #: Detected or configured source language name.
    language: str
    is_final: bool
    asr_latency: float = 0.0
    translate_latency: float = 0.0
    #: Wall-clock seconds from end of the audio segment to this event.
    total_latency: float = 0.0
    #: When the line was spoken, in seconds of audio since capture started,
    #: and how long it lasted. Used to time subtitle files.
    start_time: float = 0.0
    duration: float = 0.0


@dataclass
class PipelineStats:
    segments_transcribed: int = 0
    finals_emitted: int = 0
    partials_emitted: int = 0
    asr_errors: int = 0
    translate_errors: int = 0
    dropped_partials: int = 0
    last_asr_latency: float = 0.0
    last_translate_latency: float = 0.0
    last_total_latency: float = 0.0
    audio_blocks: int = 0

    def snapshot(self) -> dict[str, float | int]:
        return dict(self.__dict__)


@dataclass
class _Job:
    segment: Segment
    #: Time the segment's audio ended, used for end-to-end latency.
    captured_at: float
    transcript: Transcript | None = None


class _StageQueue:
    """Bounded FIFO for finals plus a single latest-wins partial slot."""

    def __init__(self) -> None:
        self._finals: deque[_Job] = deque()
        self._partial: _Job | None = None
        self._cv = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put_final(self, job: _Job) -> None:
        with self._cv:
            self._finals.append(job)
            self._cv.notify()

    def put_partial(self, job: _Job) -> None:
        with self._cv:
            if self._partial is not None:
                self.dropped += 1
            self._partial = job
            self._cv.notify()

    def get(self, timeout: float = 0.25) -> _Job | None:
        with self._cv:
            if not self._finals and self._partial is None and not self._closed:
                self._cv.wait(timeout)
            # Finals first: committed lines must not queue behind live updates.
            if self._finals:
                return self._finals.popleft()
            if self._partial is not None:
                job, self._partial = self._partial, None
                return job
            return None

    def clear(self) -> None:
        with self._cv:
            self._finals.clear()
            self._partial = None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class SubtitlePipeline:
    """Owns every worker thread and turns audio into :class:`SubtitleEvent`s."""

    def __init__(
        self,
        cfg: AppConfig,
        on_event: Callable[[SubtitleEvent], None],
        on_status: Callable[[str, str], None] | None = None,
        source: AudioSource | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_event = on_event
        self._on_status = on_status or (lambda level, msg: None)
        self._explicit_source = source

        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self._source: AudioSource | None = None
        self._segmenter: Segmenter | None = None
        self._asr: AsrBackend | None = None
        self._translator: Translator | None = None

        self._asr_queue = _StageQueue()
        self._translate_queue = _StageQueue()

        # Recent committed (source, target) pairs given to the translator.
        self._history: deque[tuple[str, str]] = deque(maxlen=8)
        # Recent committed transcripts fed back to the ASR as a context prompt.
        self._asr_history: deque[str] = deque(maxlen=8)
        self._last_partial_translation = 0.0
        self._lock = threading.Lock()
        #: Saves finished lines to disk when export is enabled.
        self._recorder = None

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()

        self._asr = create_asr(self.cfg.asr)
        ok, detail = self._asr.health()
        self._on_status("info" if ok else "error", f"ASR: {detail}")
        if not ok and self.cfg.asr.backend != "mock":
            raise RuntimeError(f"ASR backend not ready -- {detail}")

        self._translator = create_translator(self.cfg.translate)
        ok, detail = self._translator.health()
        self._on_status("info" if ok else "error", f"Translator: {detail}")
        if not ok and self.cfg.translate.backend == "llamacpp":
            raise RuntimeError(f"Translation backend not ready -- {detail}")

        vad = create_vad(self.cfg.vad)
        self._on_status("info", f"VAD: {vad.name}")
        self._segmenter = Segmenter(vad, self.cfg.vad, self.cfg.segmenter)

        self._source = self._explicit_source or create_source(self.cfg.audio)
        self._source.start()
        self._on_status("info", f"Audio: {self._source.description}")

        if self.cfg.export.enabled:
            from .export import TranscriptRecorder

            try:
                self._recorder = TranscriptRecorder(
                    self.cfg.export,
                    source_description=self._source.description.split(":", 1)[-1],
                    target_language=self.cfg.translate.target_language,
                )
                self._on_status("info", f"Saving transcript: {self._recorder.describe()}")
            except OSError as exc:
                # Captioning still works; only saving is unavailable.
                self._on_status("error", f"Cannot save transcript: {exc}")

        for target, name in (
            (self._audio_loop, "rtsubs-audio"),
            (self._asr_loop, "rtsubs-asr"),
            (self._translate_loop, "rtsubs-translate"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        if not self._threads and self._source is None:
            return
        self._stop.set()
        self._asr_queue.close()
        self._translate_queue.close()

        if self._source is not None:
            self._source.stop()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()

        for backend in (self._asr, self._translator):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    log.exception("error closing %s", backend.name)
        self._asr = self._translator = None
        self._source = None
        self._asr_queue.clear()
        self._translate_queue.clear()
        # Closed only after the worker threads have flushed the last line.
        if self._recorder is not None:
            recorder, self._recorder = self._recorder, None
            recorder.close()
            if recorder.lines_written:
                self._on_status(
                    "info",
                    f"Saved {recorder.lines_written} lines: {recorder.describe()}",
                )
        self._on_status("info", "Pipeline stopped")

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._asr_history.clear()

    # -- stage 1: VAD + segmentation ------------------------------------
    def _audio_loop(self) -> None:
        assert self._source is not None and self._segmenter is not None
        while not self._stop.is_set():
            block = self._source.read(timeout=0.25)
            if block is None:
                continue
            self.stats.audio_blocks += 1
            try:
                segments = self._segmenter.feed(block)
            except Exception:
                log.exception("segmenter failed")
                continue
            for segment in segments:
                # With the live line off nobody will see in-progress lines, so
                # don't spend GPU time transcribing them.
                if not segment.is_final and self.cfg.overlay.live_line == "off":
                    continue
                self._enqueue(self._asr_queue, _Job(segment, time.perf_counter()))

        # Commit whatever was mid-utterance when the user hit stop.
        try:
            for segment in self._segmenter.flush():
                self._enqueue(self._asr_queue, _Job(segment, time.perf_counter()))
        except Exception:
            log.exception("segmenter flush failed")

    @staticmethod
    def _enqueue(queue: _StageQueue, job: _Job) -> None:
        if job.segment.is_final:
            queue.put_final(job)
        else:
            queue.put_partial(job)

    # -- stage 2: ASR ---------------------------------------------------
    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            job = self._asr_queue.get(timeout=0.25)
            if job is None:
                continue
            try:
                transcript = self._run_asr(job.segment)
            except Exception as exc:
                self.stats.asr_errors += 1
                log.exception("ASR failed")
                self._on_status("error", f"ASR failed: {exc}")
                continue

            if not transcript.text:
                continue

            self.stats.segments_transcribed += 1
            self.stats.last_asr_latency = transcript.latency
            job.transcript = transcript

            if job.segment.is_final:
                with self._lock:
                    self._asr_history.append(transcript.text)
            self._enqueue(self._translate_queue, job)

        self.stats.dropped_partials = self._asr_queue.dropped

    def _run_asr(self, segment: Segment) -> Transcript:
        assert self._asr is not None
        cfg = self.cfg.asr

        context_parts: list[str] = []
        if cfg.context_prompt:
            context_parts.append(cfg.context_prompt)
        if cfg.history_turns > 0:
            with self._lock:
                recent = list(self._asr_history)[-cfg.history_turns :]
            context_parts.extend(recent)

        return self._asr.transcribe(
            segment.audio,
            language=cfg.language,
            context=" ".join(context_parts).strip(),
        )

    # -- stage 3: translation -------------------------------------------
    def _translate_loop(self) -> None:
        while not self._stop.is_set():
            job = self._translate_queue.get(timeout=0.25)
            if job is None or job.transcript is None:
                continue

            segment, transcript = job.segment, job.transcript
            source_language = canonical_name(transcript.language, default="")

            if not segment.is_final:
                action = self._live_line_action(transcript)
                if action == "drop":
                    continue
                if action == "source":
                    self._emit(job, transcript.text, 0.0, source_language)
                    continue

            if not self._should_translate(source_language, transcript):
                self._emit(job, transcript.text, 0.0, source_language)
                continue

            try:
                result = self._run_translation(transcript, segment, source_language)
            except Exception as exc:
                self.stats.translate_errors += 1
                log.exception("translation failed")
                self._on_status("error", f"Translation failed: {exc}")
                # Better to show untranslated text than nothing at all.
                self._emit(job, transcript.text, 0.0, source_language)
                continue

            self.stats.last_translate_latency = result.latency
            if segment.is_final and result.text and not result.skipped:
                with self._lock:
                    self._history.append((transcript.text, result.text))

            self._emit(job, result.text or transcript.text, result.latency, source_language)

    def _live_line_action(self, transcript: Transcript) -> str:
        """Decide what to do with an in-progress line: drop, source or translate."""
        mode = self.cfg.overlay.live_line
        if mode == "off":
            # The setting was switched off while this line was in flight.
            return "drop"
        if mode == "original" or transcript.already_translated:
            return "source"
        # "translated": rate-limit so live updates can't starve finished
        # lines. A throttled update is dropped, never shown untranslated --
        # flashing the source language between translations is exactly the
        # flicker this mode should avoid.
        now = time.perf_counter()
        throttle = self.cfg.translate.partial_throttle_ms
        if (now - self._last_partial_translation) * 1000 < throttle:
            return "drop"
        self._last_partial_translation = now
        return "translate"

    def _should_translate(self, source_language: str, transcript: Transcript) -> bool:
        cfg = self.cfg.translate
        # Whisper's translate task already produced target-language text, so a
        # second pass would just translate English into English.
        if transcript.already_translated:
            return False
        if cfg.skip_if_target and same_language(source_language, cfg.target_language):
            return False
        return True

    def _run_translation(
        self, transcript: Transcript, segment: Segment, source_language: str
    ):
        assert self._translator is not None
        with self._lock:
            history = tuple(self._history)[-self.cfg.translate.history_turns :]
        return self._translator.translate(
            TranslationRequest(
                text=transcript.text,
                source_language=source_language,
                target_language=self.cfg.translate.target_language,
                history=history,
                is_final=segment.is_final,
            )
        )

    # -- output ----------------------------------------------------------
    def _emit(
        self, job: _Job, text: str, translate_latency: float, language: str
    ) -> None:
        assert job.transcript is not None
        segment = job.segment
        total = time.perf_counter() - job.captured_at

        if segment.is_final:
            self.stats.finals_emitted += 1
        else:
            self.stats.partials_emitted += 1
        self.stats.last_total_latency = total

        event = SubtitleEvent(
            utterance_id=segment.utterance_id,
            source_text=job.transcript.text,
            text=text,
            language=language,
            is_final=segment.is_final,
            asr_latency=job.transcript.latency,
            translate_latency=translate_latency,
            total_latency=total,
            start_time=segment.start_time,
            duration=segment.duration,
        )
        if event.is_final and self._recorder is not None:
            try:
                self._recorder.write(event)
            except Exception:
                # A full disk must not stop the subtitles themselves.
                log.exception("could not write transcript line")
        try:
            self._on_event(event)
        except Exception:
            log.exception("subtitle event handler raised")


__all__ = ["SubtitlePipeline", "SubtitleEvent", "PipelineStats", "SAMPLE_RATE"]
