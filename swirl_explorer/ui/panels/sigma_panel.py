"""Σ-slice map: globality and pole strata on a 2D slice of the data space R⁶.

Pick two of the six data, keep the other four at the current solution's values, and shoot to
both poles on a grid (process pool in a worker thread: the UI never blocks).  The map fills in
as rows arrive.  Click the map to jump there; drag to move the point continuously.
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QComboBox,
                               QDoubleSpinBox, QSpinBox, QPushButton, QProgressBar, QScrollArea, QCheckBox,
                               QToolButton, QMenu, QFileDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...core.params import Params, PARAM_NAMES
from ...core.sigma_map import (SigmaMapSpec, compute_sigma_map, BLOWUP, UNSAFE, G, R, L1, L2, CODE_NAMES)
from ...viz import style
from ...core.params import DEFAULT_RANGES
from ...viz.sigma_plot import (MODES, EXPLAIN, CLICK_HELP, draw_sigma, readout, _name, KAPPA_MAX_BOOK,  # noqa: F401
                               GLOBAL_COL, MU_LEVELS, _is_book_lens)

class _MapWorker(QObject):
    progress = Signal(float)
    started_res = Signal(object)
    finished = Signal(object, float)
    failed = Signal(str)
    _run_req = Signal(object, int)

    def __init__(self):
        super().__init__()
        self.cancel_evt = threading.Event()
        self._run_req.connect(self._run)

    def run(self, spec, workers):
        self.cancel_evt.clear()
        self._run_req.emit(spec, workers)

    @Slot(object, int)
    def _run(self, spec, workers):
        t0 = time.perf_counter()
        try:
            res = compute_sigma_map(spec, progress=self.progress.emit, cancelled=self.cancel_evt.is_set,
                                    workers=workers, on_start=self.started_res.emit)
        except Exception as exc:              # never crash the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(res, time.perf_counter() - t0)


class SigmaMapPanel(QWidget):
    jumpRequested = Signal(object)           # Params
    keyframeRequested = Signal(object)       # Params (shift-click adds a path keyframe)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.running = False
        self.current: Params | None = None
        self.lock_check = None               # callable(Params) -> bool; set by the main window
        self.path_overlay = None             # dict(samples, keys, bad, cur) from the path dock
        self._axes = []
        self._drag = False
        self._pending_jump = None
        self._last_draw = 0.0
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(7, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 1)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{style.c()['muted']};padding:2px 8px;")
        v.addWidget(self.note)
        self.hover = QLabel("")
        self.hover_hint = ("Hover the map to read a point; click to jump there, drag to move; "
                           "shift-click adds a path keyframe.")
        self.hover.setStyleSheet(f"color:{style.c()['text']};padding:2px 8px;")
        self.hover.setMinimumHeight(18)
        v.insertWidget(1, self.hover)
        self.hover.setText(self.hover_hint)
        self.controls = self._controls()
        self.thread = QThread()
        self.worker = _MapWorker()
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self._progress)
        self.worker.started_res.connect(self._started)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.thread.start()
        self._jump_timer = QTimer(self)
        self._jump_timer.setSingleShot(True)
        self._jump_timer.setInterval(40)
        self._jump_timer.timeout.connect(self._emit_jump)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(350)
        self._redraw_timer.timeout.connect(self.redraw)
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_release_event", self._release)
        self.redraw()

    # ---------------------------------------------------------------- controls
    def _controls(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 4, 6, 4)
        g = QGroupBox("Map axes")
        gl = QGridLayout(g)
        self.x_name, self.y_name = QComboBox(), QComboBox()
        for cb in (self.x_name, self.y_name):
            for n in PARAM_NAMES:
                cb.addItem(_name(n), n)
        self.x_lo, self.x_hi, self.y_lo, self.y_hi = (self._spin() for _ in range(4))
        gl.addWidget(QLabel("horizontal"), 0, 0)
        gl.addWidget(self.x_name, 0, 1, 1, 2)
        gl.addWidget(self.x_lo, 1, 1)
        gl.addWidget(self.x_hi, 1, 2)
        gl.addWidget(QLabel("from / to"), 1, 0)
        gl.addWidget(QLabel("vertical"), 2, 0)
        gl.addWidget(self.y_name, 2, 1, 1, 2)
        gl.addWidget(QLabel("from / to"), 3, 0)
        gl.addWidget(self.y_lo, 3, 1)
        gl.addWidget(self.y_hi, 3, 2)
        gl.addWidget(QLabel("grid"), 4, 0)
        self.res_spin = QSpinBox()
        self.res_spin.setRange(11, 401)
        self.res_spin.setSingleStep(10)
        self.res_spin.setValue(101)
        self.res_spin.setSuffix(" × same")
        self.res_spin.setToolTip("points per axis (101 × 101 takes a few seconds on all cores)")
        gl.addWidget(self.res_spin, 4, 1, 1, 2)
        self.fixed_lab = QLabel("")
        self.fixed_lab.setWordWrap(True)
        self.fixed_lab.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        gl.addWidget(self.fixed_lab, 5, 0, 1, 3)
        v.addWidget(g)
        self.x_name.setCurrentIndex(PARAM_NAMES.index("gamma"))
        self.y_name.setCurrentIndex(PARAM_NAMES.index("kappa"))
        self._set_ranges("gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6))
        self.x_name.currentIndexChanged.connect(lambda _: self._axis_changed(self.x_name, self.x_lo, self.x_hi))
        self.y_name.currentIndexChanged.connect(lambda _: self._axis_changed(self.y_name, self.y_lo, self.y_hi))

        row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute map")
        self.compute_btn.setToolTip("shoot to both poles at every grid point (all CPU cores, in the background)")
        self.compute_btn.clicked.connect(self.compute)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.compute_btn, 2)
        row.addWidget(self.cancel_btn, 1)
        v.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        v.addWidget(self.bar)
        self.lens_btn = QPushButton("Book's lens (Figure 1.3): γ × κ at c = 0, A = 0")
        self.lens_btn.setToolTip("x₀ = 0, c = 0, A = 0: the lens between γ₋(κ) and γ₊(κ), apex κ ≈ 4.1608")
        self.lens_btn.clicked.connect(self.preset_lens)
        v.addWidget(self.lens_btn)

        g = QGroupBox("Show")
        gv = QVBoxLayout(g)
        self.mode = QComboBox()
        for k, t in MODES.items():
            self.mode.addItem(t, k)
        self.mode.currentIndexChanged.connect(lambda _: self.redraw())
        gv.addWidget(self.mode)
        self.chk_edge = QCheckBox("edge of the global region")
        self.chk_edge.setChecked(True)
        self.chk_mu = QCheckBox("μ contours on this view (north white, south amber)")
        self.chk_mu.setToolTip("μ = ½ dotted, ⅔ dashed, 1 solid; always shown in the exponent view")
        self.chk_mu.setChecked(False)
        for w in (self.chk_edge, self.chk_mu):
            w.toggled.connect(lambda _: self.redraw())
            gv.addWidget(w)
        v.addWidget(g)
        g = QGroupBox("Reading the map")
        gv = QVBoxLayout(g)
        self.explain = QLabel("")
        self.explain.setWordWrap(True)
        self.explain.setTextFormat(Qt.RichText)
        self.explain.setStyleSheet(f"color:{style.c()['text']};font-size:12px;")
        gv.addWidget(self.explain)
        v.addWidget(g)
        self.export_btn = QToolButton()
        self.export_btn.setText("Export ▾")
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self.export_btn)
        m.addAction("PNG…", lambda: self.export("png"))
        m.addAction("SVG…", lambda: self.export("svg"))
        self.export_btn.setMenu(m)
        v.addWidget(self.export_btn)
        v.addStretch(1)
        sc = QScrollArea()
        sc.setWidget(box)
        sc.setWidgetResizable(True)
        sc.setMinimumWidth(260)
        return sc

    @staticmethod
    def _spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-1e3, 1e3)
        s.setDecimals(3)
        s.setSingleStep(0.1)
        s.setKeyboardTracking(False)
        return s

    def _set_ranges(self, xn, yn, xr, yr):
        for cb, n in ((self.x_name, xn), (self.y_name, yn)):
            cb.blockSignals(True)
            cb.setCurrentIndex(PARAM_NAMES.index(n))
            cb.blockSignals(False)
        for s, val in zip((self.x_lo, self.x_hi, self.y_lo, self.y_hi), (*xr, *yr)):
            s.setValue(val)

    def _axis_changed(self, cb, lo, hi):
        a, b = DEFAULT_RANGES[cb.currentData()]
        lo.setValue(a)
        hi.setValue(b)
        self._update_fixed()

    def _update_fixed(self):
        p = self.current or Params()
        others = [n for n in PARAM_NAMES if n not in (self.x_name.currentData(), self.y_name.currentData())]
        txt = ",  ".join(f"{_name(n)} = {getattr(p, n):.4g}" for n in others)
        self.fixed_lab.setText(f"Fixed at the current solution: {txt},  x₀ = {p.xb:g}".replace("-", "−"))

    # ---------------------------------------------------------------- state
    def set_current(self, p: Params) -> None:
        self.current = p
        self._update_fixed()
        if not self.running:
            self.redraw()

    def spec(self, base: Params | None = None) -> SigmaMapSpec:
        n = self.res_spin.value()
        return SigmaMapSpec(base or self.current or Params(), self.x_name.currentData(), self.y_name.currentData(),
                            (self.x_lo.value(), self.x_hi.value()), (self.y_lo.value(), self.y_hi.value()),
                            nx=n, ny=n)

    def ensure_computed(self):
        if self.res is None and not self.running:
            self.compute()

    def compute(self, base: Params | None = None):
        if self.x_name.currentData() == self.y_name.currentData():
            self.note.setText("Pick two different parameters for the two axes.")
            return
        if self.x_lo.value() >= self.x_hi.value() or self.y_lo.value() >= self.y_hi.value():
            self.note.setText("Each range needs 'from' < 'to'.")
            return
        if self.running:
            self.worker.cancel_evt.set()
            QTimer.singleShot(150, lambda: self.compute(base))
            return
        spec = self.spec(base if isinstance(base, Params) else None)
        workers = 1 if spec.nx * spec.ny < 600 else max(1, min((os.cpu_count() or 2) - 1, 32))
        self.running = True
        self.compute_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.bar.setValue(0)
        self.note.setText(f"computing {spec.nx} × {spec.ny} points on {workers} "
                          f"{'core' if workers == 1 else 'cores'}…")
        self.worker.run(spec, workers)
        self._redraw_timer.start()

    def cancel(self):
        if self.running:
            self.worker.cancel_evt.set()
            self.note.setText("cancelling…")

    def preset_lens(self):
        base = Params(gamma=0.0, kappa=1.0, xb=0.0)
        self._set_ranges("gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6))
        self.mode.setCurrentIndex(list(MODES).index("global"))
        self.jumpRequested.emit(base)
        self.current = base
        self._update_fixed()
        self.compute(base)

    @Slot(object)
    def _started(self, res):
        self.res = res

    @Slot(float)
    def _progress(self, f):
        self.bar.setValue(int(1000 * f))

    @Slot(object, float)
    def _finished(self, res, secs):
        self.res = res
        self.running = False
        self._redraw_timer.stop()
        self.compute_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        done = res.done.mean()
        g = res.is_global[res.done].mean() if res.done.any() else 0.0
        state = "done" if done >= 1 else f"cancelled at {100 * done:.0f}%"
        ng, nd = int(res.is_global.sum()), int(res.done.sum())
        self.note.setText(f"{state} in {secs:.1f} s:  {ng:,} of {nd:,} grid points are global "
                          f"({100 * g:.1f}%), on a {res.spec.nx} × {res.spec.ny} grid")
        self.redraw()

    @Slot(str)
    def _failed(self, msg):
        self.running = False
        self._redraw_timer.stop()
        self.compute_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.note.setText(f"Σ-map error: {msg}")

    # ---------------------------------------------------------------- drawing
    def redraw(self):
        mode = self.mode.currentData()
        self.chk_mu.setVisible(mode != "mu")       # the exponent view always shows them
        self.explain.setText(EXPLAIN[mode] + CLICK_HELP)
        self.fig.clear()
        self.fig.set_facecolor(style.c()["plot_bg"])
        self._axes = draw_sigma(self.fig, self.res, mode, edge=self.chk_edge.isChecked(),
                                mu_contours=self.chk_mu.isChecked(), current=self.current,
                                path_overlay=self.path_overlay)
        if self._axes:
            from matplotlib.widgets import MultiCursor
            self._cursor = MultiCursor(self._axes, useblit=True, horizOn=True, color=style.c()["plot_ink"],
                                       lw=0.6, alpha=0.6)
        self.canvas.draw_idle()

    def set_path_overlay(self, overlay) -> None:
        self.path_overlay = overlay
        if not self.running:
            self.redraw()

    def refresh_style(self):
        self.note.setStyleSheet(f"color:{style.c()['muted']};padding:2px 8px;")
        self.fixed_lab.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        self.explain.setStyleSheet(f"color:{style.c()['text']};font-size:12px;")
        self.hover.setStyleSheet(f"color:{style.c()['text']};padding:2px 8px;")
        self.redraw()

    # ---------------------------------------------------------------- click / drag
    def _point(self, ev):
        if self.res is None or ev.inaxes not in self._axes or ev.xdata is None:
            return None
        spec = self.res.spec
        return spec.params_at(float(ev.xdata), float(ev.ydata))

    def _press(self, ev):
        q = self._point(ev)
        if q is None or ev.button != 1:
            return
        if ev.key == "shift":                # a keyframe for the path dock, no jump
            self.keyframeRequested.emit(q)
            return
        self._drag = True
        self._request(q)

    def _motion(self, ev):
        if self._drag:
            q = self._point(ev)
            if q is not None:
                self._request(q)
        else:
            txt = self.readout(ev.xdata, ev.ydata) if ev.inaxes in self._axes else ""
            self.hover.setText(txt or self.hover_hint)

    def readout(self, x, y) -> str:
        """One line about the grid point nearest (x, y)."""
        return readout(self.res, x, y)

    def _release(self, ev):
        self._drag = False

    def _request(self, q: Params):
        if self.lock_check is not None and not self.lock_check(q):
            self.note.setText(f"Locked: ({_name(self.res.spec.x_name)}, {_name(self.res.spec.y_name)}) = "
                              f"({getattr(q, self.res.spec.x_name):.4g}, {getattr(q, self.res.spec.y_name):.4g}) "
                              "is not global. Untick 'lock to global solutions' to go there.")
            return
        self._pending_jump = q
        if not self._jump_timer.isActive():
            self._jump_timer.start()

    def _emit_jump(self):
        q, self._pending_jump = self._pending_jump, None
        if q is not None:
            self.jumpRequested.emit(q)

    def export(self, kind: str, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f"Export {kind.upper()}", f"sigma_map.{kind}",
                                                  f"{kind.upper()} (*.{kind})")
        if path:
            self.fig.savefig(path, dpi=200 if kind == "png" else None)
        return path

    def shutdown(self):
        self.worker.cancel_evt.set()
        self.thread.quit()
        self.thread.wait(5000)
