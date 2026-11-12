"""Static layout mock-up of the main window (rendered offscreen to a PNG for approval).

Layout proposal:
  left dock   : parameter panel (6 sliders + fields + ranges), presets, live status
  centre      : tabbed views [3D view | Slice plane (2D) | Σ-map]
  right dock  : 2D ODE plots (stacked), with axis/mode selectors
  bottom bar  : path timeline (keyframes, events) + transport + export
Run: QT_QPA_PLATFORM=offscreen python scripts/mockup_layout.py
"""
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGridLayout, QLabel, QSlider, QDoubleSpinBox, QGroupBox,
                               QTabWidget, QDockWidget, QComboBox, QPushButton, QCheckBox,
                               QFrame, QSizePolicy)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")


def placeholder(text, img=None, h=None):
    lab = QLabel()
    lab.setAlignment(Qt.AlignCenter)
    lab.setFrameShape(QFrame.StyledPanel)
    if img and os.path.exists(img):
        pm = QPixmap(img)
        lab.setPixmap(pm.scaled(560 if h is None else 2000, h or 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    else:
        lab.setText(text)
        lab.setStyleSheet("background:#1d1f24;color:#c3c2b7;font-size:15px;")
    lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return lab


def param_panel():
    w = QWidget()
    v = QVBoxLayout(w)
    g = QGroupBox("Cauchy data at x₀ = 0")
    grid = QGridLayout(g)
    vals = [("c₁", 0.0), ("c₂", 0.0), ("c₃", 0.0), ("γ = U(0)", -1.6806), ("A = f(0)", 0.0), ("κ = f_x(0)", 1.0)]
    for i, (n, val) in enumerate(vals):
        grid.addWidget(QLabel(n), i, 0)
        s = QSlider(Qt.Horizontal)
        s.setValue(50 + int(val * 10))
        grid.addWidget(s, i, 1)
        sp = QDoubleSpinBox()
        sp.setDecimals(5)
        sp.setRange(-50, 50)
        sp.setValue(val)
        grid.addWidget(sp, i, 2)
        grid.addWidget(QLabel("[−3, 3]"), i, 3)
    v.addWidget(g)
    pr = QGroupBox("Presets")
    h = QHBoxLayout(pr)
    for t in ("Rest", "Landau…", "Obs 4.28 G", "Lens edge L1", "Edge R"):
        h.addWidget(QPushButton(t))
    v.addWidget(pr)
    st = QGroupBox("Status")
    sv = QVBoxLayout(st)
    for line, col in [("● GLOBAL   class Z", "#1baf7a"),
                      ("North: G   μ₊ = 0.6191  U(1) = −0.762  conf 1.00", None),
                      ("        Li–Yan 3 · BP Type III   (u_r ~ d^−0.76)", None),
                      ("South: G   μ₋ = 0.9881  U(−1) = 0.024  conf 1.00", None),
                      ("        Li–Yan 3 · BP Type II.5", None)]:
        l = QLabel(line)
        f = QFont("Consolas", 9)
        l.setFont(f)
        if col:
            l.setStyleSheet(f"color:{col};font-weight:bold;")
        sv.addWidget(l)
    v.addWidget(st)
    v.addStretch(1)
    return w


def plots_panel():
    w = QWidget()
    v = QVBoxLayout(w)
    row = QHBoxLayout()
    row.addWidget(QLabel("x-axis"))
    cb = QComboBox(); cb.addItems(["s = −ln(1−x)", "x", "θ", "ξ"]); row.addWidget(cb)
    row.addWidget(QLabel("mode"))
    cb2 = QComboBox(); cb2.addItems(["linear", "log", "log–log near pole"]); row.addWidget(cb2)
    row.addWidget(QPushButton("Pin"))
    row.addWidget(QPushButton("Export…"))
    v.addLayout(row)
    chk = QHBoxLayout()
    for n in ("U", "f", "Q", "z", "W", "h", "u_r", "u_θ", "u_φ"):
        c = QCheckBox(n); c.setChecked(n in ("U", "u_r")); chk.addWidget(c)
    chk.addWidget(QCheckBox("overlay predictions"))
    v.addLayout(chk)
    v.addWidget(placeholder("2D plots (stacked vertically)", os.path.join(FIG, "fig_pole_types.png"), 820))
    return w


def main():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Swirl Explorer — layout mock-up")
    tabs = QTabWidget()
    tabs.addTab(placeholder("3D", os.path.join(FIG, "mock_3d.png"), 860), "3D view")
    tabs.addTab(placeholder("Slice plane (true 2D)"), "Slice plane")
    tabs.addTab(placeholder("Σ-map", os.path.join(FIG, "fig_lens_map.png"), 520), "Σ-map")
    win.setCentralWidget(tabs)
    d1 = QDockWidget("Parameters"); d1.setWidget(param_panel())
    win.addDockWidget(Qt.LeftDockWidgetArea, d1)
    d2 = QDockWidget("ODE profiles"); d2.setWidget(plots_panel())
    win.addDockWidget(Qt.RightDockWidgetArea, d2)
    bar = QWidget(); bl = QHBoxLayout(bar)
    for t in ("⏮", "▶", "⏭"):
        bl.addWidget(QPushButton(t))
    tl = QLabel("Path: ◆────◆──────⚠ leaves Σ (t=0.71)──◆     frame 34/120   linear ▾   [Export MP4…]")
    tl.setFont(QFont("Consolas", 9))
    bl.addWidget(tl, 1)
    d3 = QDockWidget("Path / movie"); d3.setWidget(bar)
    win.addDockWidget(Qt.BottomDockWidgetArea, d3)
    win.statusBar().showMessage("global · class Z · N: G μ=0.619 (BP III) · S: G μ=0.988 (BP II.5) · solve 17 ms")
    win.resize(1680, 980)
    win.show()
    app.processEvents()
    out = os.path.join(FIG, "mockup_layout.png")
    win.grab().save(out)
    print(out)


if __name__ == "__main__":
    main()
