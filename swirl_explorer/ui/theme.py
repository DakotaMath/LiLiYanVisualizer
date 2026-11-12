"""Application theme: Fusion style with the palette of :mod:`swirl_explorer.viz.style` (dark by default)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from ..viz import style


def apply_theme(app: QApplication, mode: str | None = None) -> None:
    if mode is not None:
        style.set_mode(mode)
    p = style.c()
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    pal = QPalette()
    text = QColor(p["text"])
    pal.setColor(QPalette.Window, QColor(p["window"]))
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, QColor(p["base"]))
    pal.setColor(QPalette.AlternateBase, QColor(p["panel"]))
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, QColor(p["button"]))
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.ToolTipBase, QColor(p["panel"]))
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Highlight, QColor(p["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.PlaceholderText, QColor(p["faint"]))
    pal.setColor(QPalette.Mid, QColor(p["border"]))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(p["faint"]))
    app.setPalette(pal)
