"""The always-on-top subtitle overlay.

Two behaviours matter for this to feel like real captioning:

*Click-through.* When locked, the window sets ``WS_EX_TRANSPARENT`` so mouse
input passes straight to whatever is underneath. You can keep watching a video
and click its controls with subtitles floating on top.

*Stable text.* Committed lines never change. Only the trailing live line is
redrawn as the current utterance grows, which avoids the flicker that makes
re-transcribing captioners unpleasant to read.

Text is drawn as a stroked ``QPainterPath`` rather than a styled label, so it
stays legible over bright and dark video alike.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QGuiApplication,
)
from PySide6.QtWidgets import QWidget

from ..config import OverlayConfig
from .timing import display_seconds

log = logging.getLogger(__name__)

# --- Win32 extended window styles -------------------------------------
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080


def _set_click_through(widget: QWidget, enabled: bool) -> None:
    """Toggle mouse transparency at the OS level (Windows only)."""
    if sys.platform != "win32":
        widget.setAttribute(Qt.WA_TransparentForMouseEvents, enabled)
        return

    try:
        import ctypes

        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32

        # SetWindowLongPtrW is the 64-bit-safe form; fall back on 32-bit Python.
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_long.restype = ctypes.c_longlong
        set_long.restype = ctypes.c_longlong
        set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]

        style = int(get_long(ctypes.c_void_p(hwnd), _GWL_EXSTYLE))
        base = style | _WS_EX_LAYERED | _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE
        style = base | _WS_EX_TRANSPARENT if enabled else base & ~_WS_EX_TRANSPARENT
        set_long(ctypes.c_void_p(hwnd), _GWL_EXSTYLE, style)
    except Exception:
        log.exception("could not toggle click-through; falling back to Qt attribute")
        widget.setAttribute(Qt.WA_TransparentForMouseEvents, enabled)


class SubtitleOverlay(QWidget):
    """Frameless, translucent, always-on-top caption window."""

    #: Emitted when the user drags the window, so the position can be saved.
    moved = Signal(int, int)

    def __init__(self, cfg: OverlayConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # (text, monotonic time it expires). Each line gets its own reading
        # time, so a short line doesn't linger just because a long one did.
        self._committed: deque[tuple[str, float]] = deque(maxlen=max(1, cfg.max_lines))
        self._live = ""
        self._live_expires = 0.0
        self._source_line = ""
        self._drag_origin = None

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            if cfg.locked
            else Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self._font = QFont(cfg.font_family, cfg.font_size, QFont.Bold)
        self._source_font = QFont(cfg.font_family, max(10, int(cfg.font_size * 0.6)))

        # Removes lines whose reading time is up. Only runs while something
        # is on screen.
        self._expiry_timer = QTimer(self)
        self._expiry_timer.setInterval(200)
        self._expiry_timer.timeout.connect(self._prune_expired)

        self._apply_geometry()

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        _set_click_through(self, self.cfg.locked)

    def _apply_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()

        width = int(area.width() * self.cfg.width_fraction)
        metrics = QFontMetrics(self._font)
        line_height = metrics.height() + 8
        height = line_height * max(1, self.cfg.max_lines) + 40
        if self.cfg.show_source_text:
            height += QFontMetrics(self._source_font).height() + 8

        x = area.x() + (area.width() - width) // 2
        y = area.y() + int(area.height() * self.cfg.vertical_anchor) - height // 2
        y = max(area.y(), min(y, area.y() + area.height() - height))
        self.setGeometry(x, y, width, height)

    # -- public API ------------------------------------------------------
    def apply_config(self, cfg: OverlayConfig) -> None:
        """Re-read settings changed from the control panel."""
        was_locked = self.cfg.locked
        self.cfg = cfg
        self._font = QFont(cfg.font_family, cfg.font_size, QFont.Bold)
        self._source_font = QFont(cfg.font_family, max(10, int(cfg.font_size * 0.6)))
        self._committed = deque(self._committed, maxlen=max(1, cfg.max_lines))
        self._apply_geometry()
        if cfg.locked != was_locked:
            self.set_locked(cfg.locked)
        self.update()

    def set_locked(self, locked: bool) -> None:
        self.cfg.locked = locked
        _set_click_through(self, locked)
        self.update()

    def push_partial(self, text: str, source_text: str = "") -> None:
        self._live = text
        # A live line normally gets replaced by its finished version, but if
        # that never arrives (the utterance turned out to be noise), don't
        # leave it stranded on screen.
        self._live_expires = time.monotonic() + self.cfg.line_max_sec
        if self.cfg.show_source_text:
            self._source_line = source_text
        self._expiry_timer.start()
        self.update()

    def push_final(self, text: str, source_text: str = "") -> None:
        if text:
            seconds = display_seconds(
                text, self.cfg.reading_cps, self.cfg.line_min_sec, self.cfg.line_max_sec
            )
            self._committed.append((text, time.monotonic() + seconds))
        self._live = ""
        if self.cfg.show_source_text:
            self._source_line = source_text
        self._expiry_timer.start()
        self.update()

    def clear(self) -> None:
        self._committed.clear()
        self._live = ""
        self._source_line = ""
        self._expiry_timer.stop()
        self.update()

    # ------------------------------------------------------------------
    def _prune_expired(self) -> None:
        now = time.monotonic()
        before = (len(self._committed), bool(self._live))
        while self._committed and self._committed[0][1] <= now:
            self._committed.popleft()
        if self._live and self._live_expires <= now:
            self._live = ""
        if not self._committed and not self._live:
            self._source_line = ""
            self._expiry_timer.stop()
        if (len(self._committed), bool(self._live)) != before:
            self.update()

    # -- painting --------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        lines = [text for text, _ in self._committed]
        if self._live:
            lines.append(self._live)
        if not lines and not self._source_line:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        metrics = QFontMetrics(self._font)
        padding = 14
        max_text_width = self.width() - 2 * padding

        # Wrap each logical line to the window width.
        wrapped: list[tuple[str, bool]] = []
        for index, line in enumerate(lines):
            is_live = self._live != "" and index == len(lines) - 1
            for piece in self._wrap(line, metrics, max_text_width):
                wrapped.append((piece, is_live))
        # Rolling window, like live TV captions: never more than max_lines on
        # screen, live line included. When new text needs another line, the
        # oldest line scrolls off the top.
        wrapped = wrapped[-max(1, self.cfg.max_lines) :]

        line_height = metrics.height() + 6
        block_height = line_height * len(wrapped)
        source_height = 0
        source_metrics = QFontMetrics(self._source_font)
        if self.cfg.show_source_text and self._source_line:
            source_height = source_metrics.height() + 6

        total = block_height + source_height
        # Bottom-anchored: the newest line always sits in the same place and
        # older lines rise above it, instead of the block re-centring (and
        # jumping) every time the line count changes.
        top = self.height() - total - padding

        # Caption background.
        if self.cfg.background_opacity > 0 and wrapped:
            widest = max(metrics.horizontalAdvance(t) for t, _ in wrapped)
            box_width = min(self.width(), widest + 2 * padding)
            box_x = (self.width() - box_width) // 2
            bg = QColor(0, 0, 0)
            bg.setAlphaF(max(0.0, min(1.0, self.cfg.background_opacity)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(
                box_x, top - padding // 2, box_width, total + padding, 10, 10
            )

        painter.setFont(self._font)
        y = top + metrics.ascent()
        for text, is_live in wrapped:
            color = QColor(self.cfg.partial_color if is_live else self.cfg.text_color)
            width = metrics.horizontalAdvance(text)
            x = (self.width() - width) / 2
            self._draw_outlined(painter, text, x, y, color, self._font)
            y += line_height

        if source_height:
            painter.setFont(self._source_font)
            text = self._elide(self._source_line, source_metrics, max_text_width)
            width = source_metrics.horizontalAdvance(text)
            x = (self.width() - width) / 2
            faded = QColor(self.cfg.partial_color)
            faded.setAlpha(170)
            self._draw_outlined(
                painter, text, x, y + source_metrics.ascent(), faded, self._source_font
            )
        painter.end()

    def _draw_outlined(
        self,
        painter: QPainter,
        text: str,
        x: float,
        y: float,
        color: QColor,
        font: QFont,
    ) -> None:
        path = QPainterPath()
        path.addText(x, y, font, text)
        if self.cfg.outline_width > 0:
            pen = QPen(QColor(0, 0, 0, 220))
            pen.setWidthF(float(self.cfg.outline_width))
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(path)

    @staticmethod
    def _wrap(text: str, metrics: QFontMetrics, max_width: int) -> list[str]:
        if max_width <= 0 or metrics.horizontalAdvance(text) <= max_width:
            return [text]

        # Word wrap where there are spaces; character wrap for CJK, which has
        # none. Mixed text falls out of this correctly because the character
        # path only runs on segments that are themselves too long.
        lines: list[str] = []
        current = ""
        for word in text.split(" "):
            candidate = f"{current} {word}".strip()
            if metrics.horizontalAdvance(candidate) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            if metrics.horizontalAdvance(word) <= max_width:
                current = word
                continue
            chunk = ""
            for char in word:
                if metrics.horizontalAdvance(chunk + char) <= max_width:
                    chunk += char
                else:
                    lines.append(chunk)
                    chunk = char
            current = chunk
        if current:
            lines.append(current)
        return lines or [text]

    @staticmethod
    def _elide(text: str, metrics: QFontMetrics, max_width: int) -> str:
        return metrics.elidedText(text, Qt.ElideLeft, max_width)

    # -- dragging (only reachable when unlocked) -------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and not self.cfg.locked:
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            self.moved.emit(self.x(), self.y())
            event.accept()
