"""How long a subtitle line stays on screen.

Kept free of Qt so it can be unit-tested on machines without PySide6.
"""

from __future__ import annotations


def display_seconds(
    text: str, reading_cps: float, min_sec: float, max_sec: float
) -> float:
    """Seconds to show ``text``: its reading time, clamped to a sane range.

    Short lines still get ``min_sec`` so they don't flash past; long lines are
    capped at ``max_sec`` so a paragraph doesn't linger after the speaker has
    moved on.
    """
    cps = reading_cps if reading_cps > 0 else 15.0
    seconds = len(text.strip()) / cps
    return max(min_sec, min(max_sec, seconds))
