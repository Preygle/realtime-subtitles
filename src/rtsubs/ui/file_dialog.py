"""The "Subtitle a file" window.

Runs :func:`rtsubs.offline.subtitle_file` on a worker thread and reports
progress through Qt signals, so the window stays responsive and the job can be
cancelled part-way.
"""

from __future__ import annotations

import logging
import os
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..languages import ui_choices
from ..offline import MEDIA_SUFFIXES, MediaError, subtitle_file

log = logging.getLogger(__name__)

_FILTER = (
    "Video and audio ("
    + " ".join(f"*{suffix}" for suffix in MEDIA_SUFFIXES)
    + ");;All files (*)"
)


class _Bridge(QObject):
    progress = Signal(object)
    finished = Signal(object, str)


def _human(seconds: float) -> str:
    minutes, secs = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class FileSubtitleDialog(QDialog):
    """Pick a video or audio file and write subtitle files for it."""

    def __init__(self, cfg: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._result = None

        self.setWindowTitle("Subtitle a file")
        self.setMinimumWidth(640)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_options_group())
        layout.addWidget(self._build_progress())
        layout.addWidget(self._build_buttons())
        self._load_from_config()
        self._update_ready()

        self._bridge = _Bridge()
        self._bridge.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._bridge.finished.connect(self._on_finished, Qt.QueuedConnection)

    # -- construction ----------------------------------------------------
    def _build_input_group(self) -> QGroupBox:
        box = QGroupBox("File")
        row = QHBoxLayout(box)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Drop a video or audio file here, or browse...")
        self.path_edit.textChanged.connect(self._update_ready)
        row.addWidget(self.path_edit, stretch=1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        return box

    def _build_options_group(self) -> QGroupBox:
        box = QGroupBox("Options")
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Spoken language"), 0, 0)
        self.language_combo = QComboBox()
        for label, value in ui_choices():
            self.language_combo.addItem(label, value)
        grid.addWidget(self.language_combo, 0, 1)

        grid.addWidget(QLabel("Subtitle language"), 0, 2)
        self.target_combo = QComboBox()
        for label, value in ui_choices()[1:]:
            self.target_combo.addItem(label, value)
        grid.addWidget(self.target_combo, 0, 3)

        self.format_checks: dict[str, QCheckBox] = {}
        formats = QHBoxLayout()
        for fmt, label in (
            ("srt", "Subtitles (.srt)"),
            ("vtt", "Web subtitles (.vtt)"),
            ("txt", "Text transcript (.txt)"),
        ):
            check = QCheckBox(label)
            self.format_checks[fmt] = check
            check.stateChanged.connect(self._update_ready)
            formats.addWidget(check)
        formats.addStretch(1)
        holder = QWidget()
        holder.setLayout(formats)
        grid.addWidget(holder, 1, 0, 1, 4)

        self.beside_check = QCheckBox("Save next to the file (named after it)")
        self.beside_check.setChecked(True)
        self.beside_check.stateChanged.connect(self._on_beside_toggled)
        grid.addWidget(self.beside_check, 2, 0, 1, 2)

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output folder")
        self.out_edit.setEnabled(False)
        grid.addWidget(self.out_edit, 2, 2, 1, 1)
        self.out_browse = QPushButton("Choose...")
        self.out_browse.setEnabled(False)
        self.out_browse.clicked.connect(self._browse_output)
        grid.addWidget(self.out_browse, 2, 3)

        self.options_group = box
        return box

    def _build_progress(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("")
        layout.addWidget(self.progress)

        self.status_label = QLabel("Pick a file to begin.")
        layout.addWidget(self.status_label)

        self.lines_view = QListWidget()
        self.lines_view.setFont(QFont("Segoe UI", 9))
        self.lines_view.setMinimumHeight(150)
        layout.addWidget(self.lines_view)
        return holder

    def _build_buttons(self) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(34)
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button, stretch=2)

        self.open_button = QPushButton("Open folder")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        row.addWidget(self.open_button, stretch=1)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        row.addWidget(self.close_button, stretch=1)
        return holder

    # -- settings --------------------------------------------------------
    def _load_from_config(self) -> None:
        index = self.language_combo.findData(self.cfg.asr.language)
        self.language_combo.setCurrentIndex(max(0, index))
        index = self.target_combo.findData(self.cfg.translate.target_language)
        self.target_combo.setCurrentIndex(max(0, index))
        wanted = self.cfg.export.formats or ["srt"]
        for fmt, check in self.format_checks.items():
            check.setChecked(fmt in wanted)
        if not any(c.isChecked() for c in self.format_checks.values()):
            self.format_checks["srt"].setChecked(True)

    def _on_beside_toggled(self) -> None:
        beside = self.beside_check.isChecked()
        self.out_edit.setEnabled(not beside)
        self.out_browse.setEnabled(not beside)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video or audio file", "", _FILTER)
        if path:
            self.path_edit.setText(path)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose an output folder")
        if folder:
            self.out_edit.setText(folder)

    def _update_ready(self) -> None:
        ready = bool(self.path_edit.text().strip()) and any(
            c.isChecked() for c in self.format_checks.values()
        )
        if self._worker is None:
            self.start_button.setEnabled(ready)

    # -- drag and drop ---------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.path_edit.setText(urls[0].toLocalFile())
            event.acceptProposedAction()

    # -- running ---------------------------------------------------------
    def _start(self) -> None:
        if self._worker is not None:  # button doubles as Cancel
            self._cancel.set()
            self.start_button.setEnabled(False)
            self.status_label.setText("Cancelling...")
            return

        media = self.path_edit.text().strip().strip('"')
        formats = [f for f, c in self.format_checks.items() if c.isChecked()]
        # The dialog's choices win over the saved config for this run.
        cfg = AppConfig()
        cfg.apply(self.cfg.to_dict())
        cfg.asr.language = self.language_combo.currentData()
        cfg.translate.target_language = self.target_combo.currentData()
        output = None if self.beside_check.isChecked() else (self.out_edit.text().strip() or None)

        self._cancel.clear()
        self._result = None
        self.lines_view.clear()
        self.progress.setValue(0)
        self.status_label.setText("Decoding audio...")
        self.start_button.setText("Cancel")
        self.open_button.setEnabled(False)
        self.options_group.setEnabled(False)

        def run() -> None:
            error = ""
            result = None
            try:
                result = subtitle_file(
                    media,
                    cfg,
                    output_dir=output,
                    formats=formats,
                    on_progress=self._bridge.progress.emit,
                    cancel=self._cancel,
                )
            except MediaError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                log.exception("subtitling failed")
                error = f"{exc.__class__.__name__}: {exc}"
            self._bridge.finished.emit(result, error)

        self._worker = threading.Thread(target=run, name="rtsubs-file", daemon=True)
        self._worker.start()

    def _on_progress(self, progress) -> None:
        self.progress.setValue(int(progress.fraction * 1000))
        eta = f" - about {_human(progress.eta_seconds)} left" if progress.eta_seconds > 1 else ""
        self.status_label.setText(
            f"{progress.fraction * 100:.0f}%  "
            f"({_human(progress.media_position)} of {_human(progress.media_seconds)}){eta}"
        )
        if progress.line:
            self.lines_view.addItem(f"[{_human(progress.media_position)}] {progress.line}")
            self.lines_view.scrollToBottom()

    def _on_finished(self, result, error: str) -> None:
        self._worker = None
        self._result = result
        self.start_button.setText("Start")
        self.options_group.setEnabled(True)
        self._update_ready()
        self.start_button.setEnabled(True)

        if error:
            self.progress.setValue(0)
            self.status_label.setText(error.splitlines()[0])
            self.lines_view.addItem(error)
            return
        if result is None:
            return
        if result.cancelled:
            self.status_label.setText("Cancelled. Lines written so far were kept.")
        elif not result.lines:
            self.status_label.setText("No speech found in that file.")
        else:
            self.progress.setValue(1000)
            self.status_label.setText(
                f"Done: {result.lines} lines from {_human(result.media_seconds)} "
                f"in {_human(result.elapsed_seconds)} ({result.speed:.1f}x real time). "
                f"Saved {', '.join(p.name for p in result.outputs.values())}"
            )
        self.open_button.setEnabled(bool(result.outputs))

    def _open_output(self) -> None:
        if self._result and self._result.outputs:
            folder = next(iter(self._result.outputs.values())).parent
            try:
                os.startfile(folder)  # Windows
            except OSError:
                log.warning("could not open %s", folder)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._cancel.set()
            self._worker.join(timeout=5.0)
        super().closeEvent(event)
