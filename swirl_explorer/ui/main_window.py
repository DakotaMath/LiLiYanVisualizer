"""Main window: parameters + status (left), views (centre tabs), ODE profiles (right)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QMainWindow, QDockWidget, QTabWidget, QLabel, QWidget, QVBoxLayout,
                               QMessageBox)

from ..core.params import Params
from .panels.params_panel import ParamsPanel
from .panels.status_panel import StatusPanel
from .panels.plots_panel import PlotsPanel
from .worker import SolveThread


def _placeholder(title: str, text: str) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    lab = QLabel(f"<h3>{title}</h3><p style='color:#52514e'>{text}</p>")
    lab.setAlignment(Qt.AlignCenter)
    lab.setWordWrap(True)
    v.addWidget(lab)
    return w


class MainWindow(QMainWindow):
    def __init__(self, initial: Params | None = None):
        super().__init__()
        self.setWindowTitle("Swirl Explorer — axisymmetric (−1)-homogeneous Navier–Stokes with swirl")
        self.solver = SolveThread()
        self.solver.worker.finished.connect(self._on_solved)
        self.solver.worker.failed.connect(self._on_failed)
        self.last = None

        # centre: views (3D, slice, Σ-map arrive in milestones 3-5)
        self.tabs = QTabWidget()
        self.tabs.addTab(_placeholder("3D view", "Streamlines, isosurfaces and glyphs (milestone 3)."), "3D view")
        self.tabs.addTab(_placeholder("Slice plane", "u·n colouring and projected streamlines (milestone 4)."),
                         "Slice plane")
        self.tabs.addTab(_placeholder("Σ-map", "Globality and pole strata on 2D slices of R⁶ (milestone 5)."),
                         "Σ-map")
        # left: parameters + status
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 0, 6, 0)
        self.params = ParamsPanel()
        self.status = StatusPanel()
        lv.addWidget(self.params)
        lv.addWidget(self.status)
        lv.addStretch(1)
        dl = QDockWidget("Parameters")
        dl.setObjectName("dock_params")
        dl.setWidget(left)
        dl.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dl)
        # right: stacked ODE plots
        self.plots = PlotsPanel()
        dr = QDockWidget("ODE profiles")
        dr.setObjectName("dock_plots")
        dr.setWidget(self.plots)
        dr.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dr)
        self.setCentralWidget(self.tabs)
        self.resizeDocks([dl, dr], [500, 900], Qt.Horizontal)

        m = self.menuBar().addMenu("&File")
        for text, kind in (("Export plots as PNG…", "png"), ("Export plots as SVG…", "svg"),
                           ("Export curves as CSV…", "csv")):
            a = QAction(text, self)
            a.triggered.connect(lambda _=False, k=kind: self.plots.export(k))
            m.addAction(a)
        m.addSeparator()
        q = QAction("Quit", self)
        q.triggered.connect(self.close)
        m.addAction(q)
        h = self.menuBar().addMenu("&Help")
        about = QAction("Conventions…", self)
        about.triggered.connect(self._about)
        h.addAction(about)

        self.params.paramsChanged.connect(self.request)
        self.params.set_params(initial or Params(gamma=-1.6806045, kappa=1.0))

    # ---------------------------------------------------------------- solving
    def request(self, p: Params) -> None:
        self.status.show_busy()
        self.solver.worker.submit(p)

    def _on_solved(self, p, sol, sm, ms):
        self.last = (p, sol, sm)
        self.status.show_summary(sm, ms)
        self.plots.set_current(sol, sm)
        self.statusBar().showMessage(
            f"(c₁, c₂, c₃, γ, A, κ) = ({', '.join(f'{v:.7g}' for v in p.as_tuple())}) at x₀ = {p.xb:g}"
            f"   ·   solve {ms:.0f} ms")

    def _on_failed(self, p, msg):
        self.statusBar().showMessage(f"solver error: {msg}")

    def _about(self):
        QMessageBox.information(
            self, "Conventions",
            "Data at the base point x_b (default the equator):\n"
            "c = Taylor constants of Q, γ = U(x_b), A = f(x_b), κ = f_x(x_b).\n"
            "The task brief's Γ is γ here, and its γ is κ.\n\n"
            "Pole types G / R / L1 / L2 and labels follow the monograph (Table 4.1) and "
            "docs/conventions.md.")

    def closeEvent(self, ev):
        self.solver.stop()
        super().closeEvent(ev)
