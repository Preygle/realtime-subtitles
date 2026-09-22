"""Turns a stream of audio blocks into utterances.

The segmenter is the piece that recovers most of what we lose by not having
R2T2's native append-only streaming state. It runs a small state machine over
VAD frames and emits two kinds of event:

``partial``
    The utterance so far, re-emitted every ``partial_interval_ms``. Drives the
    live, still-changing subtitle line.

``final``
    A completed utterance, closed by a pause or by the max-length guard. This
    is what gets committed to the overlay and never revised.

Because each segment is transcribed independently, a forced (non-pause) commit
carries a short audio tail into the next segment so a word split across the
boundary is not lost.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import SAMPLE_RATE, SegmenterConfig, VadConfig
from .vad import FRAME_MS, FRAME_SAMPLES, Vad

log = logging.getLogger(__name__)

_ids = itertools.count(1)


@dataclass
class Segment:
    """One utterance (or a snapshot of one in progress)."""

    #: Stable id shared by every partial and the final of one utterance.
    utterance_id: int
    audio: np.ndarray
    is_final: bool
    #: Seconds since the pipeline started, at the first sample of the segment.
    start_time: float
    duration: float
    #: True when the segment was cut by the length guard, not by a pause.
    forced: bool = False
    #: True when this utterance continues audio carried over from a forced cut.
    continued: bool = False

    @property
    def kind(self) -> str:
        return "final" if self.is_final else "partial"


@dataclass
class _Utterance:
    utterance_id: int
    start_time: float
    chunks: list[np.ndarray] = field(default_factory=list)
    samples: int = 0
    #: Samples the VAD actually marked as speech. ``samples`` also counts the
    #: pre-roll and the trailing silence that closes the utterance, so only
    #: this counter can decide whether there was enough real speech to keep.
    speech_samples: int = 0
    continued: bool = False
    last_partial_samples: int = 0

    def add(self, frame: np.ndarray, is_speech: bool = False) -> None:
        self.chunks.append(frame)
        self.samples += frame.size
        if is_speech:
            self.speech_samples += frame.size

    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks)

    @property
    def duration(self) -> float:
        return self.samples / SAMPLE_RATE


class Segmenter:
    """Feeds audio blocks in, gets :class:`Segment` events out."""

    def __init__(self, vad: Vad, vad_cfg: VadConfig, seg_cfg: SegmenterConfig) -> None:
        self.vad = vad
        self.vad_cfg = vad_cfg
        self.cfg = seg_cfg

        self._leftover = np.zeros(0, dtype=np.float32)
        self._preroll: list[np.ndarray] = []
        self._preroll_frames = max(1, vad_cfg.speech_pad_ms // FRAME_MS)
        self._silence_frames_needed = max(1, vad_cfg.min_silence_ms // FRAME_MS)
        self._partial_interval_samples = int(
            SAMPLE_RATE * seg_cfg.partial_interval_ms / 1000
        )
        self._min_partial_samples = int(SAMPLE_RATE * seg_cfg.min_partial_ms / 1000)
        self._min_speech_samples = int(SAMPLE_RATE * vad_cfg.min_speech_ms / 1000)
        self._max_samples = int(SAMPLE_RATE * seg_cfg.max_segment_sec)
        self._carry_samples = int(SAMPLE_RATE * seg_cfg.carry_over_ms / 1000)

        self._current: _Utterance | None = None
        self._silence_run = 0
        self._elapsed_samples = 0
        #: Audio carried over from the previous forced cut.
        self._carry: np.ndarray | None = None

        # Exposed for the control panel's level meter.
        self.last_probability = 0.0

    # ------------------------------------------------------------------
    @property
    def in_speech(self) -> bool:
        return self._current is not None

    def feed(self, block: np.ndarray) -> list[Segment]:
        """Consume one capture block, returning any events it produced."""
        events: list[Segment] = []
        buf = (
            np.concatenate([self._leftover, block])
            if self._leftover.size
            else np.asarray(block, dtype=np.float32)
        )

        n_frames = buf.size // FRAME_SAMPLES
        for i in range(n_frames):
            frame = buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            events.extend(self._process_frame(frame))

        self._leftover = buf[n_frames * FRAME_SAMPLES :].copy()
        return events

    def flush(self) -> list[Segment]:
        """Close any in-progress utterance, e.g. when the user hits stop."""
        if self._current is None:
            return []
        seg = self._finalize(forced=False)
        return [seg] if seg is not None else []

    def reset(self) -> None:
        self._current = None
        self._silence_run = 0
        self._leftover = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._carry = None
        self.vad.reset()

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> list[Segment]:
        events: list[Segment] = []
        prob = self.vad.probability(frame)
        self.last_probability = prob
        is_speech = prob >= self.vad_cfg.threshold
        frame_start_time = self._elapsed_samples / SAMPLE_RATE
        self._elapsed_samples += frame.size

        if self._current is None:
            # Keep a rolling pre-roll so onsets are not clipped.
            self._preroll.append(frame)
            if len(self._preroll) > self._preroll_frames:
                self._preroll.pop(0)

            if is_speech:
                self._current = self._begin_utterance(frame_start_time)
            return events

        self._current.add(frame, is_speech)

        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._silence_frames_needed:
                seg = self._finalize(forced=False)
                if seg is not None:
                    events.append(seg)
                return events

        if self._current.samples >= self._max_samples:
            log.debug("max segment length reached, forcing commit")
            seg = self._finalize(forced=True)
            if seg is not None:
                events.append(seg)
            return events

        # Live line update.
        grown = self._current.samples - self._current.last_partial_samples
        if (
            self._current.samples >= self._min_partial_samples
            and grown >= self._partial_interval_samples
        ):
            self._current.last_partial_samples = self._current.samples
            events.append(
                Segment(
                    utterance_id=self._current.utterance_id,
                    audio=self._current.audio(),
                    is_final=False,
                    start_time=self._current.start_time,
                    duration=self._current.duration,
                    continued=self._current.continued,
                )
            )
        return events

    def _begin_utterance(self, frame_start_time: float) -> _Utterance:
        preroll_samples = sum(f.size for f in self._preroll)
        start_time = max(0.0, frame_start_time - preroll_samples / SAMPLE_RATE)

        utt = _Utterance(
            utterance_id=next(_ids),
            start_time=start_time,
            continued=self._carry is not None,
        )
        if self._carry is not None:
            utt.add(self._carry)
            self._carry = None
        # The last pre-roll frame is the one that triggered the onset, so it
        # counts as voiced; the earlier ones are the padding before speech.
        for index, f in enumerate(self._preroll):
            utt.add(f, is_speech=index == len(self._preroll) - 1)
        self._preroll.clear()
        return utt

    def _finalize(self, forced: bool) -> Segment | None:
        utt = self._current
        self._current = None
        self._silence_run = 0
        if utt is None:
            return None

        audio = utt.audio()
        if not forced:
            self.vad.reset()
            # Discard blips that are almost certainly not speech. Measured on
            # voiced samples only -- a 50 ms cough surrounded by pre-roll and
            # trailing silence still buffers ~800 ms, and feeding that to the
            # ASR model wastes GPU time and invites hallucinated text.
            if utt.speech_samples < self._min_speech_samples:
                log.debug(
                    "dropping blip: %.0f ms voiced in %.0f ms buffered",
                    utt.speech_samples / SAMPLE_RATE * 1000,
                    utt.duration * 1000,
                )
                return None
        elif self._carry_samples > 0:
            self._carry = audio[-self._carry_samples :].copy()

        return Segment(
            utterance_id=utt.utterance_id,
            audio=audio,
            is_final=True,
            start_time=utt.start_time,
            duration=utt.duration,
            forced=forced,
            continued=utt.continued,
        )
