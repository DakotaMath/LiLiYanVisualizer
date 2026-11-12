"""Main window: parameters + status (left), views (centre tabs), ODE profiles (right)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QMainWindow, QDockWidget, QTabWidget, QLabel, QWidget, QVBoxLayout,
                               QHBoxLayout, QMessageBox)

from ..core.params import Params
from .panels.params_panel import ParamsPanel
from .panels.status_panel import StatusPanel
from .panels.plots_panel import PlotsPanel
from .panels.view3d_panel import View3DPanel
from .panels.slice_panel import SlicePanel, SliceController
from .worker import SolveThread


def _placeholder(title: str, text: str) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    from ..viz import style
    lab = QLabel(f"<h3>{title}</h3><p style='color:{style.c()['muted']}'>{text}</p>")
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
        self.exporting = False               # a movie export owns the views
        self.session_path = None

        # centre: views (3D, slice, Σ-map arrive in milestones 3-5)
        self.tabs = QTabWidget()
        self.view3d = View3DPanel()
        self.slice_ctrl = SliceController()
        self.slice = SlicePanel(self.slice_ctrl)
        self.view3d.attach_slice(self.slice_ctrl)
        self.slice.sliceReady.connect(self.view3d.set_slice)
        self.view3d.domainChanged.connect(self.slice.set_domain)
        self.slice.view_n_btn.clicked.connect(self.view3d.view_along_normal)
        from PySide6.QtWidgets import QSplitter
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.view3d)
        self.split.addWidget(self.slice)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 1)
        self.split.setChildrenCollapsible(False)
        self.tabs.addTab(self.split, "3D + slice")
        from .panels.sigma_panel import SigmaMapPanel
        self.sigma = SigmaMapPanel()
        self.tabs.addTab(self.sigma, "Σ-map")
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
        dl.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                       | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dl)
        # right: stacked ODE plots
        self.plots = PlotsPanel()
        dr = QDockWidget("ODE profiles")
        dr.setObjectName("dock_plots")
        dr.setWidget(self.plots)
        dr.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                       | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.RightDockWidgetArea, dr)
        # the view controls share the right dock area with the profiles (tabs)
        self.dock3d = QDockWidget("3D controls")
        self.dock3d.setObjectName("dock_3d")
        self.dock3d.setWidget(self.view3d.controls)
        self.dock_slice = QDockWidget("Slice controls")
        self.dock_slice.setObjectName("dock_slice")
        self.dock_slice.setWidget(self.slice.controls)
        self.dock_sigma = QDockWidget("Σ-map controls")
        self.dock_sigma.setObjectName("dock_sigma")
        self.dock_sigma.setWidget(self.sigma.controls)
        for d in (self.dock3d, self.dock_slice, self.dock_sigma):
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                          | QDockWidget.DockWidgetClosable)
            self.addDockWidget(Qt.RightDockWidgetArea, d)
        self.tabifyDockWidget(dr, self.dock3d)
        self.tabifyDockWidget(self.dock3d, self.dock_slice)
        self.tabifyDockWidget(self.dock_slice, self.dock_sigma)
        self.dock3d.raise_()
        self.setCentralWidget(self.tabs)
        self._docks = (dl, dr, self.dock3d, self.dock_slice, self.dock_sigma)
        # Σ-map: click/drag jumps the parameters; the global lock applies to it too
        self.sigma.jumpRequested.connect(lambda q: self.params.set_params(q))
        self.sigma.lock_check = self._lock_allows
        self.tabs.currentChanged.connect(self._tab_changed)
        self._docks_sized = False
        # bottom: the path editor (hidden until asked for, the 3D view keeps the height)
        from .panels.path_panel import PathPanel
        from . import panel_state as ps
        self.path_panel = PathPanel()
        self.path_panel.current_params = lambda: self.params.params()
        self.path_panel.capture = lambda keys: ps.capture_extras(self, keys)
        self.path_panel.applyFrame.connect(self._apply_frame)
        self.path_panel.pathChanged.connect(lambda: self.sigma.set_path_overlay(self.path_panel.overlay()))
        self.path_panel.exportRequested.connect(self.export_movie)
        self.sigma.keyframeRequested.connect(self._sigma_keyframe)
        self.dock_path = QDockWidget("Path")
        self.dock_path.setObjectName("dock_path")
        self.dock_path.setWidget(self.path_panel)
        self.dock_path.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                                   | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_path)
        self.dock_path.hide()
        self._docks = self._docks + (self.dock_path,)

        m = self.menuBar().addMenu("&File")
        for text, fn, key in (("Open session…", self.open_session, "Ctrl+O"),
                              ("Save session", self.save_session, "Ctrl+S"),
                              ("Save session as…", lambda: self.save_session(ask=True), "Ctrl+Shift+S")):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(lambda _=False, f=fn: f())
            m.addAction(a)
        m.addSeparator()
        a = QAction("Export movie of the path…", self)
        a.triggered.connect(self.export_movie)
        m.addAction(a)
        m.addSeparator()
        for text, kind in (("Export plots as PNG…", "png"), ("Export plots as SVG…", "svg"),
                           ("Export curves as CSV…", "csv")):
            a = QAction(text, self)
            a.triggered.connect(lambda _=False, k=kind: self.plots.export(k))
            m.addAction(a)
        m.addSeparator()
        q = QAction("Quit", self)
        q.triggered.connect(self.close)
        m.addAction(q)
        vm = self.menuBar().addMenu("&View")
        # the 2D slice pane is hidden until the user asks for it (button at the top right of the view tabs)
        self.act_slice = QAction("Show 2D slice", self, checkable=True, checked=False)
        self.act_slice.setToolTip("show the 2D slice plane beside the 3D view, with its controls")
        self.act_slice.toggled.connect(self._show_slice)
        vm.addAction(self.act_slice)
        from PySide6.QtWidgets import QToolButton
        self.slice_btn = QToolButton()
        self.slice_btn.setDefaultAction(self.act_slice)
        self.slice_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.slice_btn.setStyleSheet("QToolButton{padding:3px 10px;}")
        self.act_path = self.dock_path.toggleViewAction()
        self.act_path.setText("Path && movie")
        self.act_path.setToolTip("keyframes in parameter space, preview and MP4 export (bottom panel)")
        self.path_btn = QToolButton()
        self.path_btn.setDefaultAction(self.act_path)
        self.path_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.path_btn.setStyleSheet("QToolButton{padding:3px 10px;}")
        corner = QWidget()
        ch = QHBoxLayout(corner)
        ch.setContentsMargins(0, 0, 4, 0)
        ch.setSpacing(4)
        ch.addWidget(self.path_btn)
        ch.addWidget(self.slice_btn)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        self._show_slice(False)
        for d in self._docks:
            if d is not self.dock_path:
                vm.addAction(d.toggleViewAction())
        vm.addAction(self.act_path)
        vm.addSeparator()
        for text, mode in (("Dark theme", "dark"), ("Light theme", "light")):
            a = QAction(text, self)
            a.triggered.connect(lambda _=False, m=mode: self.set_theme(m))
            vm.addAction(a)
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
        if self.exporting:                   # the movie export draws the views itself
            return
        self.last = (p, sol, sm)
        self.status.show_summary(sm, ms)
        self.plots.set_current(sol, sm)
        self.view3d.set_solution(sol, sm)
        self.slice.set_solution(sol, sm)
        self.sigma.set_current(p)
        self.statusBar().showMessage(
            f"(c₁, c₂, c₃, γ, A, κ) = ({', '.join(f'{v:.7g}' for v in p.as_tuple())}) at x₀ = {p.xb:g}"
            f"   ·   solve {ms:.0f} ms")

    def _on_failed(self, p, msg):
        self.statusBar().showMessage(f"solver error: {msg}")

    def set_theme(self, mode: str) -> None:
        from PySide6.QtWidgets import QApplication
        from .theme import apply_theme
        apply_theme(QApplication.instance(), mode)
        self.params.refresh_style()
        self.plots.refresh_style()
        self.view3d.refresh_style()
        self.slice.refresh_style()
        self.sigma.refresh_style()
        self.path_panel.refresh_style()
        if self.last is not None:
            p, sol, sm = self.last
            self.status.show_summary(sm, 0.0)

    def _lock_allows(self, q: Params) -> bool:
        if not self.params.lock.isChecked():
            return True
        from ..core.solver import is_global_fast
        return is_global_fast(q)

    def _tab_changed(self, i: int) -> None:
        sigma = self.tabs.widget(i) is self.sigma
        self.slice_btn.setVisible(not sigma)          # the slice button belongs to the 3D tab
        if sigma:
            self.dock_sigma.show()
            self.dock_sigma.raise_()
            self.sigma.ensure_computed()              # the first visit computes a map right away
        else:
            self.dock3d.raise_()

    def _show_slice(self, on: bool) -> None:
        self.slice.setVisible(on)
        self.dock_slice.setVisible(on)
        self.act_slice.setText("Hide 2D slice" if on else "Show 2D slice")
        # Showing the slice clears the 3D view down to the slice plane (streamlines, arrows, surfaces ...
        # off); hiding it turns the plane off and puts those layers back as they were.  Every checkbox
        # in 3D controls can still be changed by hand in between.
        layers = self.view3d.chk
        if on:
            self._layers_before_slice = {k: c.isChecked() for k, c in layers.items()}
            for c in layers.values():
                c.setChecked(False)
        elif getattr(self, "_layers_before_slice", None) is not None:
            for k, was in self._layers_before_slice.items():
                layers[k].setChecked(was)
            self._layers_before_slice = None
        self.view3d.chk_slice.setChecked(on)
        if on:
            self.split.setSizes([10000, 10000])         # 3D and slice equally wide
            self.dock_slice.raise_()

    # ---------------------------------------------------------------- path, movie, sessions
    def _apply_frame(self, p, extras):
        if self.exporting:
            return
        from . import panel_state as ps
        self.params.set_params(p)            # the path may go where the lock would refuse: shown in red
        ps.apply_extras(self, extras)

    def _sigma_keyframe(self, q):
        self.dock_path.show()
        self.path_panel.add_keyframe(q)
        self.statusBar().showMessage(f"keyframe {len(self.path_panel.path)} added from the Σ-map", 4000)

    def export_movie(self):
        if len(self.path_panel.path) < 2:
            self.dock_path.show()
            QMessageBox.information(self, "Export movie", "Make a path first: it needs at least two keyframes.")
            return
        from .movie_export import ExportDialog, MovieExporter
        pp = self.path_panel
        dlg = ExportDialog(self, pp.n_frames(), pp.fps.value(), self.act_slice.isChecked(),
                           getattr(self, "_movie_path", ""), pp.policy.currentData())
        if dlg.exec() != ExportDialog.Accepted:
            return
        o = dlg.options()
        self._movie_path = o["path"]
        self._exporter = MovieExporter(self, o)
        self._exporter.finished.connect(self._movie_done)
        self._exporter.start()

    def _movie_done(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 15000)
        if not getattr(self, "quiet", False):
            QMessageBox.information(self, "Export movie", msg)

    def collect_session(self):
        from ..io.session import Session
        from . import panel_state as ps
        from ..viz import style
        sec = {
            "lock": self.params.lock.isChecked(),
            "scene": ps.scene_state(self.view3d),
            "camera": ps.camera_state(self.view3d),
            "plane": ps.plane_state(self.slice_ctrl),
            "slice": ps.slice_options(self.slice),
            "plots": ps.plots_state(self.plots),
            "sigma": ps.sigma_state(self.sigma),
            "movie": self.path_panel.state(),
            "ui": {"theme": style.MODE, "tab": self.tabs.currentIndex(), "show_slice": self.act_slice.isChecked(),
                   "show_path": self.dock_path.isVisible()},
        }
        return Session(self.params.params(), sec, self.path_panel.path)

    def apply_session(self, s) -> None:
        from . import panel_state as ps
        sec = s.sections
        ui = sec.get("ui", {})
        if ui.get("theme") in ("dark", "light"):
            self.set_theme(ui["theme"])
        self.act_slice.setChecked(bool(ui.get("show_slice", False)))
        if "lock" in sec:
            self.params.lock.setChecked(bool(sec["lock"]))
        ps.apply_scene(self.view3d, sec.get("scene"))
        ps.apply_plane(self.slice, sec.get("plane"))
        ps.apply_slice_options(self.slice, sec.get("slice"))
        ps.apply_plots(self.plots, sec.get("plots"))
        ps.apply_sigma(self.sigma, sec.get("sigma"))
        mv = sec.get("movie", {})
        self.path_panel.set_path(s.path, mv.get("frames"), mv.get("fps"))
        self.path_panel.loop.setChecked(bool(mv.get("loop", False)))
        self.dock_path.setVisible(bool(ui.get("show_path", s.path is not None)))
        self.params.set_params(s.params)
        if sec.get("camera"):
            ps.apply_camera(self.view3d, sec["camera"])
        if isinstance(ui.get("tab"), int) and 0 <= ui["tab"] < self.tabs.count():
            self.tabs.setCurrentIndex(ui["tab"])

    def save_session(self, ask: bool = False, path: str | None = None) -> str | None:
        from PySide6.QtWidgets import QFileDialog
        from ..io.session import save_session
        path = path or self.session_path
        if ask or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save session", path or "session.swirl.json",
                                                  "Swirl Explorer session (*.json)")
            if not path:
                return None
        try:
            save_session(self.collect_session(), path)
        except Exception as exc:
            QMessageBox.warning(self, "Save session", f"Could not save:\n{exc}")
            return None
        self.session_path = path
        self.statusBar().showMessage(f"session saved to {path}", 6000)
        return path

    def open_session(self, path: str | None = None) -> bool:
        from PySide6.QtWidgets import QFileDialog
        from ..io.session import load_session, SessionError
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open session", self.session_path or "",
                                                  "Swirl Explorer session (*.json);;All files (*)")
            if not path:
                return False
        try:
            s = load_session(path)
        except (OSError, SessionError) as exc:
            QMessageBox.warning(self, "Open session", f"Could not open {path}:\n{exc}")
            return False
        self.apply_session(s)
        self.session_path = path
        self.statusBar().showMessage(f"session loaded from {path}", 6000)
        return True

    def _about(self):
        QMessageBox.information(
            self, "Conventions",
            "Data at the base point x_b (default the equator):\n"
            "c = Taylor constants of Q, γ = U(x_b), A = f(x_b), κ = f_x(x_b).\n"
            "The task brief's Γ is γ here, and its γ is κ.\n\n"
            "Pole types G / R / L1 / L2 and labels follow the monograph (Table 4.1) and "
            "docs/conventions.md.")

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._docks_sized:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.resizeDocks([self._docks[0], self._docks[1]], [330, 360],
                                                          Qt.Horizontal))
            QTimer.singleShot(50, lambda: self._show_slice(self.act_slice.isChecked()))
            self._docks_sized = True

    def closeEvent(self, ev):
        self.solver.stop()
        self.view3d.shutdown()
        self.slice.shutdown()
        self.sigma.shutdown()
        self.path_panel.shutdown()
        super().closeEvent(ev)
