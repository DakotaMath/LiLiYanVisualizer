"""ODE profile plots: quantity selection, horizontal variable, mode, pins, export."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
                               QCheckBox, QLabel, QFileDialog, QGridLayout, QMenu, QToolButton)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...viz.plots2d import QUANTITIES, XVARS, Curve, draw_profiles, export_csv, PIN_COLORS, CURRENT

QUANTITY_BUTTONS = [("U", "U"), ("f", "f"), ("Q", "Q"), ("z", "z"), ("W", "W = 1+Q/2"), ("h", "h = z′/z"),
                    ("u_r", "u_r"), ("u_theta", "u_θ"), ("u_phi", "u_φ")]
TIPS = {"U": "U = u_θ sin θ, meridional circulation", "f": "f = u_φ sin θ, swirl",
        "Q": "potential Q: (1−x²)U_x + 2xU + U²/2 = Q", "z": "positive solution of z'' = W z (z(0) = 1)",
        "W": "Schrödinger potential W = 1 + Q/2", "h": "h = z'/z = U/2 + x",
        "u_r": "radial velocity profile u_r = U_x", "u_theta": "u_θ = U / sin θ", "u_phi": "u_φ = f / sin θ"}
MODE_LABELS = {"linear": "linear", "log": "log |·|", "loglog": "log–log near pole"}


class PlotsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current: Curve | None = None
        self.pins: list[Curve] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("x-axis"))
        self.xvar = QComboBox()
        for k, lab in XVARS.items():
            self.xvar.addItem(lab.replace("$", "").replace("\\", ""), k)
        self.xvar.setItemText(0, "s = −ln(1−x)  (north)")
        self.xvar.setItemText(1, "s = −ln(1+x)  (south)")
        self.xvar.setItemText(2, "x = cos θ")
        self.xvar.setItemText(3, "θ (degrees)")
        self.xvar.setItemText(4, "ξ = artanh x")
        row.addWidget(self.xvar)
        row.addWidget(QLabel("mode"))
        self.mode = QComboBox()
        for k, lab in MODE_LABELS.items():
            self.mode.addItem(lab, k)
        row.addWidget(self.mode)
        self.pole = QComboBox()
        self.pole.addItem("north pole", "north")
        self.pole.addItem("south pole", "south")
        row.addWidget(self.pole)
        row.addStretch(1)
        v.addLayout(row)

        row2 = QHBoxLayout()
        self.checks: dict[str, QCheckBox] = {}
        for key, lab in QUANTITY_BUTTONS:
            c = QCheckBox(lab)
            c.setChecked(key in ("U", "f", "Q"))
            c.setToolTip(TIPS[key])
            c.toggled.connect(self.redraw)
            self.checks[key] = c
            row2.addWidget(c)
        row2.addSpacing(10)
        self.overlay = QCheckBox("predictions")
        self.overlay.setChecked(True)
        self.overlay.setToolTip("overlay predicted pole limits / rates (dashed)")
        self.overlay.toggled.connect(self.redraw)
        row2.addWidget(self.overlay)
        row2.addStretch(1)
        v.addLayout(row2)

        row3 = QHBoxLayout()
        self.pin_btn = QPushButton("Pin current")
        self.pin_btn.setToolTip("keep the current solution as a comparison curve")
        self.pin_btn.clicked.connect(self.pin_current)
        self.clear_btn = QPushButton("Clear pins")
        self.clear_btn.clicked.connect(self.clear_pins)
        self.export_btn = QToolButton()
        self.export_btn.setText("Export ▾")
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self.export_btn)
        m.addAction("PNG…", lambda: self.export("png"))
        m.addAction("SVG…", lambda: self.export("svg"))
        m.addAction("CSV…", lambda: self.export("csv"))
        self.export_btn.setMenu(m)
        for w in (self.pin_btn, self.clear_btn, self.export_btn):
            row3.addWidget(w)
        self.pin_info = QLabel("")
        self.pin_info.setStyleSheet("color:#52514e;")
        row3.addWidget(self.pin_info)
        row3.addStretch(1)
        v.addLayout(row3)

        self.fig = Figure(figsize=(6, 7), layout="constrained")
        self.fig.set_facecolor("#fcfcfb")
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 1)

        self.xvar.setCurrentIndex(self.xvar.findData("xi"))
        for w in (self.xvar, self.mode, self.pole):
            w.currentIndexChanged.connect(self.redraw)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    # ---------------------------------------------------------------- state
    def quantities(self) -> list[str]:
        return [k for k, _ in QUANTITY_BUTTONS if self.checks[k].isChecked()]

    def _mode_changed(self, *_):
        ll = self.mode.currentData() == "loglog"
        self.pole.setEnabled(ll)
        self.xvar.setEnabled(not ll)

    def set_current(self, sol, summary, label: str = "current") -> None:
        self.current = Curve(label, sol, summary, CURRENT)
        self.redraw()

    def pin_current(self):
        if self.current is None:
            return
        col = PIN_COLORS[len(self.pins) % len(PIN_COLORS)]
        p = self.current.sol.params
        names = {"c1": "c₁", "c2": "c₂", "c3": "c₃", "gamma": "γ", "A": "A", "kappa": "κ"}
        parts = [f"{names[k]}={v:.4g}" for k, v in zip(names, p.as_tuple()) if v != 0.0]
        lab = (f"pin {len(self.pins) + 1}: " + (", ".join(parts) or "rest state")).replace("-", "\u2212")
        self.pins.append(Curve(lab, self.current.sol, self.current.summary, col, pinned=True))
        self.pin_info.setText(f"{len(self.pins)} pinned")
        self.redraw()

    def clear_pins(self):
        self.pins.clear()
        self.pin_info.setText("")
        self.redraw()

    def curves(self) -> list[Curve]:
        return self.pins + ([self.current] if self.current else [])

    def redraw(self, *_):
        if not self.curves():
            return
        draw_profiles(self.fig, self.curves(), self.quantities(), xvar=self.xvar.currentData(),
                      mode=self.mode.currentData(), pole=self.pole.currentData(),
                      overlay=self.overlay.isChecked())
        self.canvas.draw_idle()

    def export(self, kind: str, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f"Export {kind.upper()}", f"profiles.{kind}",
                                                  f"{kind.upper()} (*.{kind})")
        if not path:
            return None
        if kind == "csv":
            export_csv(path, self.curves(), self.quantities(), self.xvar.currentData())
        else:
            self.fig.savefig(path, dpi=200 if kind == "png" else None)
        return path
