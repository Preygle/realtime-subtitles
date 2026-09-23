"""Qt application glue: control panel + overlay + pipeline.

The pipeline emits from worker threads, so every callback is bounced through a
Qt signal. Qt queues cross-thread signal deliveries onto the GUI thread, which
is the only place widgets may be touched.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from ..config import DEFAULT_CONFIG_PATH, AppConfig
from ..pipeline import SubtitleEvent, SubtitlePipeline
from .control_panel import ControlPanel, QtLogBridge
from .overlay import SubtitleOverlay

log = logging.getLogger(__name__)


class _Bridge(QObject):
    """Thread-safe hop from pipeline worker threads to the GUI thread."""

    event = Signal(object)
    status = Signal(str, str)


class SubtitleApp:
    def __init__(self, cfg: AppConfig, config_path=DEFAULT_CONFIG_PATH) -> None:
        self.cfg = cfg
        self.config_path = config_path

        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        self.qt_app.setApplicationName("Realtime Subtitles")
        self.qt_app.setQuitOnLastWindowClosed(False)

        self.panel = ControlPanel(cfg)
        self.overlay = SubtitleOverlay(cfg.overlay)
        self.pipeline: SubtitlePipeline | None = None

        self._bridge = _Bridge()
        self._bridge.event.connect(self._on_event, Qt.QueuedConnection)
        self._bridge.status.connect(self._on_status, Qt.QueuedConnection)

        self.panel.start_requested.connect(self.start)
        self.panel.stop_requested.connect(self.stop)
        self.panel.overlay_settings_changed.connect(self._apply_overlay_settings)
        self.panel.clear_requested.connect(self.overlay.clear)
        self.panel.subtitle_file_requested.connect(self._open_file_dialog)
        self.panel.bind_stats(self._stats)
        # closeEvent fires while the C++ objects are still alive; `destroyed`
        # fires after deletion, which makes touching the overlay a crash.
        self.panel.closed.connect(self._on_panel_closed)

        # Ctrl+Shift+S hides/shows captions without stopping recognition.
        self._toggle = QShortcut(QKeySequence("Ctrl+Shift+S"), self.panel)
        self._toggle.setContext(Qt.ApplicationShortcut)
        self._toggle.activated.connect(self._toggle_overlay)

        logging.getLogger().addHandler(QtLogBridge(self._bridge.status.emit))

    # ------------------------------------------------------------------
    def run(self) -> int:
        self.panel.show()
        self.overlay.show()
        self.panel.append_log("info", "Ready. Press Start to begin captioning.")
        try:
            return self.qt_app.exec()
        finally:
            self.stop()
            self._save_config()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.pipeline is not None:
            return
        self.overlay.clear()
        self.panel.append_log("info", "Starting pipeline...")

        pipeline = SubtitlePipeline(
            self.cfg,
            on_event=self._bridge.event.emit,
            on_status=self._bridge.status.emit,
        )
        try:
            pipeline.start()
        except Exception as exc:
            log.exception("failed to start pipeline")
            self.panel.append_log("error", f"Start failed: {exc}")
            self.panel.show_error("Could not start", str(exc))
            try:
                pipeline.stop()
            except Exception:
                pass
            return

        self.pipeline = pipeline
        self.panel.set_running(True)
        self._save_config()

    def stop(self) -> None:
        if self.pipeline is None:
            return
        pipeline, self.pipeline = self.pipeline, None
        try:
            pipeline.stop()
        except Exception:
            log.exception("error while stopping pipeline")
        self.panel.set_running(False)

    # ------------------------------------------------------------------
    def _on_event(self, event: SubtitleEvent) -> None:
        if event.is_final:
            self.overlay.push_final(event.text, event.source_text)
            detail = (
                f"[{event.language or '?'}] {event.text}"
                f"   ({event.total_latency * 1000:.0f} ms)"
            )
            self.panel.append_log("final", detail)
        else:
            self.overlay.push_partial(event.text, event.source_text)

    def _on_status(self, level: str, message: str) -> None:
        self.panel.append_log(level, message)

    def _open_file_dialog(self) -> None:
        """Open (or re-focus) the window that subtitles an existing file."""
        from .file_dialog import FileSubtitleDialog

        if getattr(self, "_file_dialog", None) is None:
            self._file_dialog = FileSubtitleDialog(self.cfg, self.panel)
            self._file_dialog.finished.connect(lambda *_: setattr(self, "_file_dialog", None))
        if self.pipeline is not None:
            self.panel.append_log(
                "warn",
                "Live captioning is running: it and the file job share the GPU, "
                "so both will be slower.",
            )
        self._file_dialog.show()
        self._file_dialog.raise_()
        self._file_dialog.activateWindow()

    def _apply_overlay_settings(self) -> None:
        self.overlay.apply_config(self.cfg.overlay)

    def _toggle_overlay(self) -> None:
        self.overlay.setVisible(not self.overlay.isVisible())

    def _stats(self):
        if self.pipeline is None:
            return 0.0, {}
        segmenter = getattr(self.pipeline, "_segmenter", None)
        probability = getattr(segmenter, "last_probability", 0.0) if segmenter else 0.0
        return probability, self.pipeline.stats.snapshot()

    def _on_panel_closed(self) -> None:
        self.stop()
        self._save_config()
        try:
            self.overlay.close()
        except RuntimeError:
            # Qt may already have torn the overlay down during shutdown.
            pass
        self.qt_app.quit()

    def _save_config(self) -> None:
        # config_path is None when the app is driven programmatically (tests).
        if self.config_path is None:
            return
        try:
            self.cfg.save(self.config_path)
        except (OSError, TypeError):
            log.warning("could not save config to %s", self.config_path)


def run_app(cfg: AppConfig, config_path=DEFAULT_CONFIG_PATH) -> int:
    return SubtitleApp(cfg, config_path).run()
