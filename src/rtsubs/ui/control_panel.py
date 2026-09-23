"""Control panel: device/language/backend selection, plus a live debug log.

Everything the pipeline needs to be reconfigured lives here. Settings that only
affect rendering (font size, opacity, lock) are applied to the overlay live;
settings that affect capture or models require a restart of the pipeline and
are disabled while it is running.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..asr.registry import BACKEND_IDS as ASR_BACKENDS
from ..audio.devices import DeviceInfo, available_backends, list_devices
from ..config import AppConfig
from ..languages import ui_choices
from ..translate.registry import BACKEND_IDS as TRANSLATE_BACKENDS

log = logging.getLogger(__name__)

_LEVEL_COLORS = {
    "error": "#ff6b6b",
    "warn": "#ffd166",
    "info": "#8fb9ff",
    "final": "#b9f6ca",
    "partial": "#9aa5b1",
}


class QtLogBridge(logging.Handler):
    """Routes ``logging`` records into the panel's text view."""

    def __init__(self, emit_line) -> None:
        super().__init__()
        self._emit_line = emit_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = "error" if record.levelno >= logging.WARNING else "info"
            self._emit_line(level, f"{record.name}: {record.getMessage()}")
        except Exception:
            pass


class ControlPanel(QWidget):
    """Main window. Owns config editing and start/stop, not the pipeline."""

    start_requested = Signal()
    stop_requested = Signal()
    overlay_settings_changed = Signal()
    clear_requested = Signal()
    subtitle_file_requested = Signal()
    closed = Signal()

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._devices: list[DeviceInfo] = []
        self._running = False
        # Guards against widgets writing back to the config while they are
        # still being populated from it: each setValue() fires valueChanged,
        # which would otherwise commit the *other* widgets' unloaded defaults.
        self._loading = True

        self.setWindowTitle("Realtime Subtitles - Control Panel")
        self.resize(720, 760)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_audio_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_overlay_group())
        layout.addWidget(self._build_export_group())
        layout.addWidget(self._build_controls())
        layout.addWidget(self._build_log(), stretch=1)

        self.refresh_devices()
        self._load_from_config()
        self._loading = False
        self._on_asr_backend_changed()

        # Drives the level meter and latency readout.
        self._tick = QTimer(self)
        self._tick.setInterval(120)
        self._tick.timeout.connect(self._refresh_stats)
        self._stats_provider = None

    # -- construction ----------------------------------------------------
    def _build_audio_group(self) -> QGroupBox:
        box = QGroupBox("Audio source")
        form = QFormLayout(box)

        self.source_combo = QComboBox()
        self.source_combo.addItem("System audio (loopback)", "loopback")
        self.source_combo.addItem("Microphone", "input")
        self.source_combo.currentIndexChanged.connect(self.refresh_devices)
        form.addRow("Capture", self.source_combo)

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(380)
        device_row.addWidget(self.device_combo, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_devices)
        device_row.addWidget(refresh)
        device_widget = QWidget()
        device_widget.setLayout(device_row)
        form.addRow("Device", device_widget)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(12)
        form.addRow("Speech", self.level_bar)

        self.vad_combo = QComboBox()
        self.vad_combo.addItem("Auto (Silero if available)", "auto")
        self.vad_combo.addItem("Silero (ONNX)", "silero")
        self.vad_combo.addItem("Energy (no model)", "energy")
        form.addRow("VAD", self.vad_combo)

        self.audio_group = box
        return box

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Recognition and translation")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("ASR backend"), 0, 0)
        self.asr_combo = QComboBox()
        for backend in ASR_BACKENDS:
            self.asr_combo.addItem(
                {
                    "llamacpp": "Confucius4-R2T2 (llama.cpp)",
                    "whispercpp": "Whisper (whisper.cpp)",
                    "mock": "Mock (no model)",
                }.get(backend, backend),
                backend,
            )
        self.asr_combo.currentIndexChanged.connect(self._on_asr_backend_changed)
        grid.addWidget(self.asr_combo, 0, 1)

        grid.addWidget(QLabel("ASR server"), 0, 2)
        self.asr_url = QLineEdit()
        grid.addWidget(self.asr_url, 0, 3)

        grid.addWidget(QLabel("Spoken language"), 1, 0)
        self.language_combo = QComboBox()
        for label, value in ui_choices():
            self.language_combo.addItem(label, value)
        grid.addWidget(self.language_combo, 1, 1)

        grid.addWidget(QLabel("Hotwords / context"), 1, 2)
        self.context_edit = QLineEdit()
        self.context_edit.setPlaceholderText("names, jargon, topic hint")
        grid.addWidget(self.context_edit, 1, 3)

        grid.addWidget(QLabel("Translator"), 2, 0)
        self.translate_combo = QComboBox()
        for backend in TRANSLATE_BACKENDS:
            self.translate_combo.addItem(
                {
                    "llamacpp": "LLM via llama.cpp (best quality)",
                    "nllb": "NLLB-200 on CPU (fastest)",
                    "passthrough": "None - show original text",
                }.get(backend, backend),
                backend,
            )
        grid.addWidget(self.translate_combo, 2, 1)

        grid.addWidget(QLabel("Translator server"), 2, 2)
        self.translate_url = QLineEdit()
        grid.addWidget(self.translate_url, 2, 3)

        grid.addWidget(QLabel("Subtitle language"), 3, 0)
        self.target_combo = QComboBox()
        for label, value in ui_choices()[1:]:  # no "auto" target
            self.target_combo.addItem(label, value)
        grid.addWidget(self.target_combo, 3, 1)


        self.whisper_translate_check = QCheckBox(
            "Whisper: translate to English in a single pass (skips the translator)"
        )
        self.whisper_translate_check.setToolTip(
            "Whisper can transcribe and translate to English in one model "
            "pass.\nFaster and uses less VRAM, but Whisper may revise words "
            "it has already shown, and it can only target English."
        )
        self.whisper_translate_check.stateChanged.connect(
            self._on_asr_backend_changed
        )
        grid.addWidget(self.whisper_translate_check, 4, 0, 1, 4)

        self.model_group = box
        return box

    def _build_overlay_group(self) -> QGroupBox:
        box = QGroupBox("Overlay appearance")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Font size"), 0, 0)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(12, 96)
        self.font_spin.valueChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.font_spin, 0, 1)

        grid.addWidget(QLabel("Background"), 0, 2)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.valueChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.opacity_spin, 0, 3)

        grid.addWidget(QLabel("Max lines"), 1, 0)
        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 6)
        self.lines_spin.valueChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.lines_spin, 1, 1)

        grid.addWidget(QLabel("Height on screen"), 1, 2)
        self.anchor_spin = QDoubleSpinBox()
        self.anchor_spin.setRange(0.05, 0.95)
        self.anchor_spin.setSingleStep(0.02)
        self.anchor_spin.valueChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.anchor_spin, 1, 3)

        self.lock_check = QCheckBox("Click-through (locked)")
        self.lock_check.setToolTip(
            "When checked, mouse clicks pass through the overlay.\n"
            "Uncheck to drag it to a new position."
        )
        self.lock_check.stateChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.lock_check, 2, 0, 1, 2)

        self.source_text_check = QCheckBox("Show original text below")
        self.source_text_check.stateChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.source_text_check, 2, 2, 1, 2)

        grid.addWidget(QLabel("Live line"), 3, 0)
        self.live_line_combo = QComboBox()
        self.live_line_combo.addItem("Off - show each sentence once, when finished", "off")
        self.live_line_combo.addItem("Original language, as they speak", "original")
        self.live_line_combo.addItem("Translated, as they speak (can change)", "translated")
        self.live_line_combo.setToolTip(
            "What to show while someone is still mid-sentence.\n"
            "Languages that put the verb last (Japanese, Korean...) make a live\n"
            "translation rewrite itself several times, so Off is steadier."
        )
        self.live_line_combo.currentIndexChanged.connect(self._push_overlay_settings)
        grid.addWidget(self.live_line_combo, 3, 1, 1, 3)

        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Save transcript")
        grid = QGridLayout(box)

        self.export_check = QCheckBox("Save finished lines to disk")
        self.export_check.setToolTip(
            "Records every finished subtitle line while captioning runs.\n"
            "A new set of files is started each time you press Start."
        )
        self.export_check.stateChanged.connect(self._on_export_toggled)
        grid.addWidget(self.export_check, 0, 0, 1, 2)

        self.export_format_checks = {}
        formats = QHBoxLayout()
        for fmt, label, tip in (
            ("srt", "Subtitles (.srt)", "Standard subtitle file for VLC, MPC and video editors"),
            ("vtt", "Web subtitles (.vtt)", "For browsers and video platforms"),
            ("txt", "Text transcript (.txt)", "Timestamped plain text, easy to read or summarize"),
        ):
            check = QCheckBox(label)
            check.setToolTip(tip)
            self.export_format_checks[fmt] = check
            formats.addWidget(check)
        formats.addStretch(1)
        formats_widget = QWidget()
        formats_widget.setLayout(formats)
        grid.addWidget(formats_widget, 1, 0, 1, 4)

        self.export_original_check = QCheckBox("Include original-language text in .txt")
        grid.addWidget(self.export_original_check, 2, 0, 1, 2)

        grid.addWidget(QLabel("Folder"), 3, 0)
        self.export_dir_edit = QLineEdit()
        grid.addWidget(self.export_dir_edit, 3, 1, 1, 2)
        open_button = QPushButton("Open folder")
        open_button.clicked.connect(self._open_export_folder)
        grid.addWidget(open_button, 3, 3)

        self.export_group = box
        return box

    def _on_export_toggled(self) -> None:
        enabled = self.export_check.isChecked()
        for widget in (
            *self.export_format_checks.values(),
            self.export_original_check,
            self.export_dir_edit,
        ):
            widget.setEnabled(enabled)

    def _export_folder(self):
        from pathlib import Path

        from ..config import PROJECT_ROOT

        folder = Path(self.export_dir_edit.text().strip() or self.cfg.export.directory)
        return folder if folder.is_absolute() else PROJECT_ROOT / folder

    def _open_export_folder(self) -> None:
        import os

        folder = self._export_folder()
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)  # Windows: opens it in Explorer

    def _build_controls(self) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self._on_start_clicked)
        row.addWidget(self.start_button, stretch=2)

        clear = QPushButton("Clear overlay")
        clear.clicked.connect(self.clear_requested.emit)
        row.addWidget(clear, stretch=1)

        subtitle_file = QPushButton("Subtitle a file...")
        subtitle_file.setToolTip(
            "Transcribe and translate an existing video or audio file, much "
            "faster than real time, and write .srt/.vtt/.txt next to it."
        )
        subtitle_file.clicked.connect(self.subtitle_file_requested.emit)
        row.addWidget(subtitle_file, stretch=1)

        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.status_label, stretch=2)
        return widget

    def _build_log(self) -> QGroupBox:
        box = QGroupBox("Live transcript and diagnostics")
        layout = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_view)
        return box

    # -- config binding ---------------------------------------------------
    def _load_from_config(self) -> None:
        cfg = self.cfg
        self._select_data(self.source_combo, cfg.audio.source)
        self._select_data(self.vad_combo, cfg.vad.engine)
        self._select_data(self.asr_combo, cfg.asr.backend)
        self.asr_url.setText(self._url_for(cfg.asr.backend))
        self.whisper_translate_check.setChecked(cfg.asr.whisper_translate)
        self._select_data(self.language_combo, cfg.asr.language)
        self.context_edit.setText(cfg.asr.context_prompt)
        self._select_data(self.translate_combo, cfg.translate.backend)
        self.translate_url.setText(cfg.translate.base_url)
        self._select_data(self.target_combo, cfg.translate.target_language)
        self._select_data(self.live_line_combo, cfg.overlay.live_line)

        self.font_spin.setValue(cfg.overlay.font_size)
        self.opacity_spin.setValue(cfg.overlay.background_opacity)
        self.lines_spin.setValue(cfg.overlay.max_lines)
        self.anchor_spin.setValue(cfg.overlay.vertical_anchor)
        self.lock_check.setChecked(cfg.overlay.locked)
        self.source_text_check.setChecked(cfg.overlay.show_source_text)
        self.export_check.setChecked(cfg.export.enabled)
        for fmt, check in self.export_format_checks.items():
            check.setChecked(fmt in cfg.export.formats)
        self.export_original_check.setChecked(cfg.export.include_original)
        self.export_dir_edit.setText(cfg.export.directory)
        self._on_export_toggled()

    def commit_to_config(self) -> None:
        """Copy every widget value back into ``self.cfg``."""
        cfg = self.cfg
        cfg.audio.source = self.source_combo.currentData()
        device = self.device_combo.currentData()
        cfg.audio.device_name = device.name if isinstance(device, DeviceInfo) else ""
        cfg.vad.engine = self.vad_combo.currentData()

        cfg.asr.backend = self.asr_combo.currentData()
        url = self.asr_url.text().strip()
        if cfg.asr.backend == "whispercpp":
            cfg.asr.whisper_base_url = url or cfg.asr.whisper_base_url
        else:
            cfg.asr.base_url = url or cfg.asr.base_url
        cfg.asr.whisper_translate = self.whisper_translate_check.isChecked()
        cfg.asr.language = self.language_combo.currentData()
        cfg.asr.context_prompt = self.context_edit.text().strip()

        cfg.translate.backend = self.translate_combo.currentData()
        cfg.translate.base_url = (
            self.translate_url.text().strip() or cfg.translate.base_url
        )
        cfg.translate.target_language = self.target_combo.currentData()

        self._commit_overlay_config()
        self._commit_export_config()

    def _commit_export_config(self) -> None:
        export = self.cfg.export
        export.enabled = self.export_check.isChecked()
        export.formats = [f for f, c in self.export_format_checks.items() if c.isChecked()]
        export.include_original = self.export_original_check.isChecked()
        export.directory = self.export_dir_edit.text().strip() or export.directory

    def _commit_overlay_config(self) -> None:
        overlay = self.cfg.overlay
        overlay.font_size = self.font_spin.value()
        overlay.background_opacity = self.opacity_spin.value()
        overlay.max_lines = self.lines_spin.value()
        overlay.vertical_anchor = self.anchor_spin.value()
        overlay.locked = self.lock_check.isChecked()
        overlay.show_source_text = self.source_text_check.isChecked()
        overlay.live_line = self.live_line_combo.currentData() or "off"

    def _push_overlay_settings(self) -> None:
        if self._loading:
            return
        self._commit_overlay_config()
        self.overlay_settings_changed.emit()

    def _url_for(self, backend: str) -> str:
        if backend == "whispercpp":
            return self.cfg.asr.whisper_base_url
        return self.cfg.asr.base_url

    def _on_asr_backend_changed(self) -> None:
        """Swap the server URL and grey out controls the backend overrides."""
        backend = self.asr_combo.currentData()
        if not self._loading:
            self.asr_url.setText(self._url_for(backend))

        is_whisper = backend == "whispercpp"
        self.whisper_translate_check.setVisible(is_whisper)

        # In one-pass mode Whisper emits English itself, so the translation
        # stage is bypassed entirely -- say so instead of leaving live-looking
        # controls that do nothing.
        single_pass = is_whisper and self.whisper_translate_check.isChecked()
        for widget in (
            self.translate_combo,
            self.translate_url,
            self.target_combo,
        ):
            widget.setEnabled(not single_pass)
        self.translate_combo.setToolTip(
            "Disabled: Whisper is translating to English in a single pass."
            if single_pass
            else ""
        )

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    # -- devices -----------------------------------------------------------
    def refresh_devices(self) -> None:
        self.device_combo.clear()
        # Selecting entries below must not be mistaken for a user edit.
        source = self.source_combo.currentData() or "loopback"
        self._devices = list_devices(self.cfg.audio.backend)

        matching = [d for d in self._devices if d.is_loopback == (source == "loopback")]
        if not matching:
            backends = available_backends()
            hint = (
                "no capture backend installed - pip install PyAudioWPatch"
                if not backends
                else "no matching device found"
            )
            self.device_combo.addItem(f"({hint})", None)
            self.device_combo.setEnabled(False)
            return

        self.device_combo.setEnabled(True)
        for device in matching:
            self.device_combo.addItem(device.label, device)
        # Preselect whatever the config named, else the system default.
        for index, device in enumerate(matching):
            if self.cfg.audio.device_name and self.cfg.audio.device_name in device.name:
                self.device_combo.setCurrentIndex(index)
                return
        for index, device in enumerate(matching):
            if device.is_default:
                self.device_combo.setCurrentIndex(index)
                return

    # -- runtime -----------------------------------------------------------
    def bind_stats(self, provider) -> None:
        """``provider()`` should return (speech_probability, stats dict)."""
        self._stats_provider = provider

    def set_running(self, running: bool) -> None:
        self._running = running
        self.start_button.setText("Stop" if running else "Start")
        self.status_label.setText("Running" if running else "Idle")
        # Capture and model settings cannot change mid-run.
        self.audio_group.setEnabled(not running)
        self.model_group.setEnabled(not running)
        self.export_group.setEnabled(not running)
        if running:
            self._tick.start()
        else:
            self._tick.stop()
            self.level_bar.setValue(0)

    def _on_start_clicked(self) -> None:
        if self._running:
            self.stop_requested.emit()
        else:
            self.commit_to_config()
            self.start_requested.emit()

    def _refresh_stats(self) -> None:
        if self._stats_provider is None:
            return
        try:
            probability, stats = self._stats_provider()
        except Exception:
            return
        self.level_bar.setValue(int(max(0.0, min(1.0, probability)) * 100))
        self.status_label.setText(
            "Running | ASR {asr:.0f} ms | MT {mt:.0f} ms | e2e {e2e:.0f} ms".format(
                asr=stats.get("last_asr_latency", 0.0) * 1000,
                mt=stats.get("last_translate_latency", 0.0) * 1000,
                e2e=stats.get("last_total_latency", 0.0) * 1000,
            )
        )

    # -- log ----------------------------------------------------------------
    def append_log(self, level: str, message: str) -> None:
        color = _LEVEL_COLORS.get(level, "#d0d0d0")
        self.log_view.appendHtml(
            f'<span style="color:{color}">{_escape(message)}</span>'
        )
        self.log_view.moveCursor(QTextCursor.End)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.closed.emit()
        super().closeEvent(event)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
