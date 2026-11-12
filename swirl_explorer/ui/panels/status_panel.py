"""Live status: globality, class, and a compact line per pole (details in the tooltip)."""
from __future__ import annotations

import math

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox
from PySide6.QtCore import Qt

from ...core.poles import Summary
from ...viz import style
from ...viz.status_text import pole_line, head_html, _col  # noqa: F401


class StatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Current solution")
        bv = QVBoxLayout(box)
        self.head = QLabel("—")
        self.head.setTextFormat(Qt.RichText)
        self.north = QLabel("")
        self.south = QLabel("")
        self.head.setWordWrap(True)
        for w in (self.north, self.south):
            w.setTextFormat(Qt.RichText)
            w.setWordWrap(True)
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bv.addWidget(self.head)
        bv.addWidget(self.north)
        bv.addWidget(self.south)
        bv.addStretch(1)
        for w in (self.head, self.north, self.south):
            f = w.font()
            f.setPointSizeF(f.pointSizeF() + 1.5)
            w.setFont(f)
        v.addWidget(box)

    def show_busy(self):
        self.head.setText(f"<span style='color:{_col('muted')}'>computing…</span>")

    def show_summary(self, sm: Summary, solve_ms: float) -> None:
        self.head.setText(head_html(sm))
        for w, (html, tip) in ((self.north, pole_line("North x = +1", sm.north, +1)),
                               (self.south, pole_line("South x = −1", sm.south, -1))):
            w.setText(html)
            w.setToolTip(tip)
