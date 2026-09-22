"""Voice activity detection.

Two engines with the same interface:

``SileroVad``
    ONNX Silero VAD v5. Far better at ignoring music and background noise,
    which matters a lot when the source is system loopback audio. Needs
    ``models/silero_vad.onnx`` (~2 MB) and ``onnxruntime``.

``EnergyVad``
    RMS threshold with hysteresis. No model file, no extra dependency. Good
    enough for clean speech, poor on noisy or musical sources.

Both consume fixed 32 ms frames (512 samples at 16 kHz) and return a speech
probability in [0, 1].
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE, PROJECT_ROOT, VadConfig
from . import dsp

log = logging.getLogger(__name__)

#: Silero v5 requires exactly 512 samples per call at 16 kHz.
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32 ms
#: Trailing samples of the previous frame that Silero v5 expects prepended.
_CONTEXT_SAMPLES = 64

SILERO_MODEL_PATH = PROJECT_ROOT / "models" / "silero_vad.onnx"


class Vad(ABC):
    name = "vad"

    @abstractmethod
    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one ``FRAME_SAMPLES``-long frame."""

    def reset(self) -> None:
        """Clear any recurrent state between utterances."""


class EnergyVad(Vad):
    """RMS threshold with a hysteresis band to avoid chattering."""

    name = "energy"

    def __init__(self, threshold_dbfs: float = -42.0, hysteresis_db: float = 6.0):
        self.threshold_dbfs = threshold_dbfs
        self.hysteresis_db = hysteresis_db
        self._active = False
        # Tracks the noise floor so the VAD adapts to quiet or loud sources.
        self._floor_dbfs = threshold_dbfs - 10.0

    def probability(self, frame: np.ndarray) -> float:
        level = dsp.dbfs(frame)

        # Slowly track the floor upward, quickly downward.
        if level < self._floor_dbfs:
            self._floor_dbfs += (level - self._floor_dbfs) * 0.25
        else:
            self._floor_dbfs += (level - self._floor_dbfs) * 0.002

        gate = max(self.threshold_dbfs, self._floor_dbfs + 8.0)
        on_threshold = gate
        off_threshold = gate - self.hysteresis_db

        if self._active:
            self._active = level > off_threshold
        else:
            self._active = level > on_threshold
        return 1.0 if self._active else 0.0

    def reset(self) -> None:
        self._active = False


class SileroVad(Vad):
    """Silero VAD v5 through onnxruntime (CPU -- the model is tiny)."""

    name = "silero"

    def __init__(self, model_path: Path = SILERO_MODEL_PATH):
        import onnxruntime as ort  # type: ignore

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                "Run scripts/download_models.ps1 -VadOnly, or set vad.engine "
                'to "energy".'
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        # A 2 MB model on the GPU would just add PCIe round-trips.
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        log.info("Silero VAD loaded from %s", model_path)

    def probability(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size != FRAME_SAMPLES:
            # Pad or trim so a partial trailing frame is still usable.
            padded = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            n = min(FRAME_SAMPLES, x.size)
            padded[:n] = x[:n]
            x = padded

        # Silero v5 must see the last 64 samples of the previous frame in
        # front of the current one (576 samples in total), exactly as the
        # reference OnnxWrapper does. Without it the model still runs, but
        # scores real speech at ~0.1 -- so nothing is ever detected.
        model_input = np.concatenate([self._context, x]).reshape(1, -1)
        self._context = x[-_CONTEXT_SAMPLES:].copy()

        feeds: dict[str, np.ndarray] = {"input": model_input, "sr": self._sr}
        if "state" in self._input_names:
            feeds["state"] = self._state

        outputs = self._session.run(None, feeds)
        if len(outputs) > 1:
            self._state = outputs[1]
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)


def create_vad(cfg: VadConfig) -> Vad:
    """Build the configured VAD, falling back to energy when Silero is absent."""
    if cfg.engine in ("silero", "auto"):
        try:
            return SileroVad()
        except Exception as exc:
            if cfg.engine == "silero":
                raise
            log.warning("Silero VAD unavailable (%s); using energy VAD", exc)
    return EnergyVad(threshold_dbfs=cfg.energy_dbfs)
