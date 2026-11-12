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
from ...viz.scene3d import STRATUM_COLORS
from .params_panel import LABELS, DEFAULT_RANGES

MODES = {"global": "global / not global",
         "strata": "pole strata (north | south)",
         "mu": "pole exponents μ₊ | μ₋"}
MU_LEVELS = (0.5, 2 / 3, 1.0)
MU_LABELS = {0.5: "½", 2 / 3: "⅔", 1.0: "1"}
MU_STYLES = (":", "--", "-.")
MU_COLORS = ("#ffffff", "#7fd3ff", "#ffb000")


def MU_NORM():
    """Colour scale for μ that gives 0..1 (where the classes change) 80 % of the colours."""
    import matplotlib.colors as mc
    f = lambda x: np.where(np.asarray(x) <= 1.0, 0.8 * np.asarray(x), 0.8 + 0.4 * (np.asarray(x) - 1.0))
    g = lambda y: np.where(np.asarray(y) <= 0.8, np.asarray(y) / 0.8, 1.0 + (np.asarray(y) - 0.8) / 0.4)
    return mc.FuncNorm((f, g), vmin=0.0, vmax=1.5)   # ½ white, ⅔ sky blue, 1 amber         # ½ dotted, ⅔ dashed, 1 dash-dot (the edge of Σ is solid)
MARK = "#ff4fd8"
_CLICK = ("<br><br>Hover to read a point (crosshair); <b>click</b> to jump there, <b>drag</b> to move. "
          "With 'lock to global solutions' on, non-global points are refused. "
          "<b>Shift-click</b> adds a keyframe to the path (Path dock).")
EXPLAIN = {
    "global": "<b>Green</b>: global, i.e. the solution exists on the whole sphere and both poles pass the "
              "book's terminal test.<br><b>Olive</b>: not global, the solution blows up at some x before "
              "reaching a pole.<br><b>Grey</b>: not global, it reaches a pole that fails the terminal test "
              "(usually a thin fringe along the edge, where the pole types R, L1, L2 live).",
    "strata": "Each panel shows the type of one pole. <b>G</b> (generic, green as in the global view) fills "
              "the interior of Σ; R, L1, L2 occur only on edges of the global region, so on a grid they "
              "appear as thin slivers if at all.<br><b>Hatched</b>: this pole is fine but the other pole "
              "fails, so the point is not global. Olive / grey: this pole itself blows up / fails.",
    "mu": "<b>μ</b> = √(1 + Q/2) at a generic pole sets the radial velocity near the axis, "
          "u<sub>r</sub> ~ d<sup>2μ−2</sup>: μ &lt; ½ beyond class III, ½–⅔ class III, ⅔–1 class II.5, "
          "μ &gt; 1 u<sub>r</sub> bounded. Contours: ½ white dotted, ⅔ blue dashed, 1 amber dash-dot. "
          "Cross-hatched: not global (switch to the global view to see why).",
}                     # current point (magenta, as the predictions in the profile plots)


PATH_FAIL = {"blowup": "#c9ad72", "unsafe": "#d9dde3"}   # the path where it is not global (timeline colours)


class _Labelled:
    """A legend handle made of overlaid lines, with a label (matplotlib draws tuples with HandlerTuple)."""

    def __init__(self, parts, label):
        self.parts, self._label = parts, label

    def get_label(self):
        return self._label


def _legend_map():
    from matplotlib.legend_handler import HandlerTuple, HandlerBase

    class _H(HandlerBase):
        def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
            return HandlerTuple(ndivide=1).create_artists(legend, orig.parts, xd, yd, w, h, fs, trans)
    return {_Labelled: _H()}


def _name(n: str) -> str:
    return LABELS[n][0]


def _fail(kind: str) -> str:
    """Colours of the two ways to be non-global; distinct from each other and from the background."""
    dark = style.MODE == "dark"
    return {"blow": "#6b5d45" if dark else "#b3a68c", "unsafe": "#b9bdc7" if dark else "#8e9097"}[kind]


GLOBAL_COL = "#3fa877"
NEUTRAL = "#141518"                   # not global in the μ view: off the colour ramp (near black)                   # not global, in the continuous (μ) view


def _stratum_col(name: str) -> str:
    """Pole-type colours on the Σ-map: G (the interior of Σ) in the same green as 'global'."""
    return GLOBAL_COL if name == "G" else STRATUM_COLORS[name]
KAPPA_MAX_BOOK = 4.1608               # lens apex at c = 0, A = 0, x₀ = 0 (monograph, Example 5.71)


def _is_book_lens(spec) -> bool:
    b = spec.base
    return ({spec.x_name, spec.y_name} == {"gamma", "kappa"} and spec.x_name == "gamma"
            and b.c1 == b.c2 == b.c3 == b.A == 0.0 and b.xb == 0.0)


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
    def _fill(self, ax, xs, ys, mask, color=None, hatch=None, z=1):
        """Smooth-edged fill of a boolean mask (marching squares, not grid steps)."""
        if not mask.any():
            return
        kw = dict(hatches=[hatch]) if hatch else {}
        ax.contourf(xs, ys, mask.astype(float), levels=[0.5, 1.5], colors=[color] if color else "none",
                    antialiased=True, zorder=z, **kw)

    def _fill_failures(self, ax, res, xs, ys, code=None):
        """Not global: brown = blow-up before a pole, grey = a pole fails the terminal test.
        With ``code`` (one pole) only that pole's own failure is shown."""
        if code is None:
            blow = res.done & ((res.code_n == BLOWUP) | (res.code_s == BLOWUP))
            unsafe = res.done & ~res.is_global & ~blow
        else:
            blow = res.done & (code == BLOWUP)
            unsafe = res.done & (code == UNSAFE)
        self._fill(ax, xs, ys, blow, _fail("blow"))
        self._fill(ax, xs, ys, unsafe, _fail("unsafe"))

    def _mu_lines(self, ax, xs, ys, m, color, halo, labels=True, extra_ls=None):
        from matplotlib import patheffects as pe
        if np.isfinite(m).sum() <= 4:
            return
        ymid = 0.5 * (ys[0] + ys[-1])
        for lev, ls, lw, lc in zip(MU_LEVELS, MU_STYLES, (1.6, 1.6, 2.0), MU_COLORS):
            if color != "#ffffff":           # overlay of the second pole: one colour for all levels
                lc = color
            cs = ax.contour(xs, ys, m, levels=[lev], colors=[lc], linewidths=lw,
                            linestyles=[extra_ls or ls], zorder=4)
            cs.set_path_effects([pe.Stroke(linewidth=lw + 1.8, foreground=halo, alpha=0.6), pe.Normal()])
            segs = sorted((sg for sg in cs.allsegs[0] if len(sg) > 3), key=len, reverse=True)[:2] \
                if cs.allsegs else []
            if labels and segs:
                # one label per piece, at 55 % of its height span: away from the tips and from the
                # places where several contours meet the edge
                spots = []
                for sg in segs:
                    y0, y1 = sg[:, 1].min(), sg[:, 1].max()
                    fr = {0.5: 0.3, 2 / 3: 0.6, 1.0: 0.85}[lev]      # stagger the labels of nearby levels
                    target = y0 + fr * (y1 - y0) if abs(y1 - ymid) >= abs(y0 - ymid) else y1 - fr * (y1 - y0)
                    spots.append(tuple(sg[np.argmin(abs(sg[:, 1] - target))]))
                for t in ax.clabel(cs, fmt={lev: MU_LABELS[lev]}, fontsize=10, inline=True, inline_spacing=4,
                                   manual=spots):
                    t.set_rotation(0)
                    t.set_color(lc)
                    t.set_path_effects([pe.Stroke(linewidth=2.5, foreground=halo), pe.Normal()])

    def redraw(self):
        import matplotlib
        import matplotlib.colors as mc
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        p = style.c()
        halo = "#101114" if style.MODE == "dark" else "#ffffff"
        self.fig.clear()
        self.fig.set_facecolor(p["plot_bg"])
        self._axes = []
        res = self.res
        mode = self.mode.currentData()
        self.chk_mu.setVisible(mode != "mu")       # the exponent view always shows them
        self.explain.setText(EXPLAIN[mode] + _CLICK)
        rc = {**style.mpl_rc(), "hatch.color": "#d9dde3", "hatch.linewidth": 0.8}
        with matplotlib.rc_context(rc):
            if res is None:
                ax = self.fig.add_subplot()
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No map yet: choose two parameters and press 'Compute map'.",
                        ha="center", va="center", color=p["muted"], transform=ax.transAxes)
                self.canvas.draw_idle()
                return
            spec = res.spec
            xs, ys = spec.axes()
            two = mode in ("strata", "mu")
            axs = self.fig.subplots(1, 2, sharex=True, sharey=True) if two else [self.fig.add_subplot()]
            # the same shape in every view (the lens must not stretch between views)
            box = float(np.clip((ys[-1] - ys[0]) / (xs[-1] - xs[0]), 0.6, 1.15)) if not two else None
            for ax in (axs if not two else []):
                ax.set_box_aspect(box)
            if two:
                self.fig.get_layout_engine().set(wspace=0.08)
            self._axes = list(axs)
            handles = []
            for ax in axs:                          # not-yet-computed cells show the plain background
                ax.set_facecolor(p["plot_bg"])
            fail_handles = [Patch(color=_fail("unsafe"), label="not global: a pole fails the terminal test"),
                            Patch(color=_fail("blow"), label="not global: blows up before a pole")]
            if mode == "global":
                self._fill_failures(axs[0], res, xs, ys)
                self._fill(axs[0], xs, ys, res.is_global, GLOBAL_COL)
                unsafe = res.done & ~res.is_global & (res.code_n != BLOWUP) & (res.code_s != BLOWUP)
                self._fill(axs[0], xs, ys, unsafe, _fail("unsafe"), z=6)      # on top of the edge line
                handles += [Patch(color=GLOBAL_COL, label="global (in Σ)")] + fail_handles
            elif mode == "strata":
                for ax, code, title in ((axs[0], res.code_n, "north pole x = +1"),
                                        (axs[1], res.code_s, "south pole x = −1")):
                    self._fill_failures(ax, res, xs, ys, code)
                    for k, name in ((G, "G"), (R, "R"), (L1, "L1"), (L2, "L2")):
                        self._fill(ax, xs, ys, res.done & (code == k), _stratum_col(name))
                    other = res.done & ~res.is_global & (code >= G)
                    self._fill(ax, xs, ys, other, (0, 0, 0, 0.45), z=2)          # dimmed ...
                    self._fill(ax, xs, ys, other, None, "////", z=2)             # ... with pale stripes
                    ax.set_title(title, fontsize=10)
                present = set(np.unique(res.code_n[res.done])) | set(np.unique(res.code_s[res.done]))
                absent = []
                for k, name, lab in ((G, "G", "G: generic"), (R, "R", "R: recessive"),
                                     (L1, "L1", "L1: logarithmic"), (L2, "L2", "L2: logarithmic, on the wall")):
                    if k in present:
                        handles.append(Patch(color=_stratum_col(name), label=lab))
                    else:
                        absent.append(name)
                if absent:
                    handles.append(Line2D([], [], ls="", label=", ".join(absent)
                                          + ": edge only, not resolved on this grid"))
                import matplotlib.colors as mc2
                dim = 0.55 * np.array(mc2.to_rgb(_stratum_col("G")))
                handles.append(Patch(facecolor=dim, hatch="////", edgecolor="#d9dde3",
                                     label="hatched: this pole is fine, the other pole fails"))
                codes = np.concatenate([res.code_n[res.done], res.code_s[res.done]])
                if (codes == UNSAFE).any():
                    handles.append(Patch(color=_fail("unsafe"), label="this pole fails the terminal test"))
                if (codes == BLOWUP).any():
                    handles.append(Patch(color=_fail("blow"), label="blows up before reaching this pole"))
            else:
                import matplotlib
                SEQUENTIAL = matplotlib.colormaps["cividis"]      # no green: green means 'global' elsewhere
                vmax = 1.5
                lv = np.linspace(0.0, vmax, 151)
                for ax, mu, title in ((axs[0], res.mu_n, "μ₊ at the north pole"), (axs[1], res.mu_s, "μ₋ at the south pole")):
                    self._fill(ax, xs, ys, res.done & ~res.is_global, NEUTRAL)
                    self._fill(ax, xs, ys, res.done & ~res.is_global, None, "x", z=1)     # sparse cross-hatch
                    m = np.where(res.is_global, mu, np.nan)
                    if np.isfinite(m).sum() > 4:
                        cf = ax.contourf(xs, ys, np.clip(m, 0, vmax), levels=lv, cmap=SEQUENTIAL, norm=MU_NORM(),
                                         zorder=1)
                    self._mu_lines(ax, xs, ys, m, "#ffffff", "#101114")
                    ax.set_title(title, fontsize=10)
                sm = matplotlib.cm.ScalarMappable(norm=MU_NORM(), cmap=SEQUENTIAL)
                cb = self.fig.colorbar(sm, ax=list(axs), orientation="horizontal", fraction=0.05, pad=0.03,
                                       aspect=30, shrink=0.8, extend="max", ticks=[0, 0.5, 2 / 3, 1.0, 1.5])
                cb.ax.set_xticklabels(["0", "½", "⅔", "1", "1.5"])
                gm_all = np.concatenate([res.mu_n[res.is_global], res.mu_s[res.is_global]])
                top = float(np.nanmax(gm_all)) if gm_all.size else vmax
                top = min(vmax, max(0.75, top * 1.03))
                cb.ax.set_xlim(0, top)                     # only the range present on this map
                for x0, x1, lab in ((0, 0.5, r"beyond III: $u_r$ worse than $1/d$"), (0.5, 2 / 3, "III"),
                                    (2 / 3, 1.0, "II.5"), (1.0, 1.5, r"$u_r$ bounded")):
                    if x0 >= top:
                        continue
                    x1 = min(x1, top)
                    cb.ax.text(0.5 * (x0 + x1), 1.25, lab, transform=cb.ax.get_xaxis_transform(), ha="center",
                               va="bottom", fontsize=8.5, color=p["plot_ink"])
                    if x0 > 0:
                        cb.ax.axvline(x0, color="#101114", lw=1.2)
                cb.set_label(r"pole exponent $\mu = \sqrt{1 + Q/2}$ at a generic (G) pole: "
                             r"$u_r \sim d^{\,2\mu-2}$ near the axis",
                             fontsize=9)
                gmu = np.concatenate([res.mu_n[res.is_global], res.mu_s[res.is_global]])
                for lev, ls, lc in zip(MU_LEVELS, MU_STYLES, MU_COLORS):
                    crosses = gmu.size and np.nanmin(gmu) < lev < np.nanmax(gmu)
                    if crosses:
                        handles.append(Line2D([], [], color=lc, lw=1.6, ls=ls, label=f"μ = {MU_LABELS[lev]}"))
                handles.append(Patch(facecolor=NEUTRAL, hatch="x", edgecolor="#d9dde3",
                                     label="not global (see the global view for why)"))
            for ax in axs:
                if self.chk_edge.isChecked() and res.done.sum() > 4:
                    gm = np.where(res.done, res.is_global.astype(float), np.nan)
                    ax.contour(xs, ys, gm, levels=[0.5], colors=[p["plot_ink"]], linewidths=1.4, zorder=5)
                if self.chk_mu.isChecked() and mode != "mu":
                    self._mu_lines(ax, xs, ys, np.where(res.is_global, res.mu_n, np.nan), "#ffffff", "#101114")
                    self._mu_lines(ax, xs, ys, np.where(res.is_global, res.mu_s, np.nan), "#f2c14e", "#101114",
                                   labels=False)
                px, py = 0.015 * (xs[-1] - xs[0]), 0.015 * (ys[-1] - ys[0])    # keep edge marks off the frame
                ax.set_xlim(xs[0] - px, xs[-1] + px)
                ax.set_ylim(ys[0] - py, ys[-1] + py)
                ax.set_xlabel(_name(spec.x_name), fontsize=11)
                ax.axhline(0, color=p["plot_ink2"], lw=0.5, alpha=0.5, zorder=3)
                ax.axvline(0, color=p["plot_ink2"], lw=0.5, alpha=0.5, zorder=3)
            axs[0].set_ylabel(_name(spec.y_name), fontsize=11)
            if _is_book_lens(spec) and mode == "global":   # the book's landmarks (Figure 1.3), to check by eye
                for ax in axs:
                    for yk in (KAPPA_MAX_BOOK, -KAPPA_MAX_BOOK):
                        ax.plot([0], [yk], marker="_", ms=16, mew=2.2, color="#f2c14e", zorder=6)
                    ax.plot([-2, 2], [0, 0], ls="", marker="|", ms=16, mew=2.2, color="#f2c14e", zorder=6)
                    box = dict(boxstyle="round,pad=0.2", fc="#101114", ec="none", alpha=0.75)
                    for yk, va, dy in ((KAPPA_MAX_BOOK, "bottom", 6), (-KAPPA_MAX_BOOK, "top", -6)):
                        ax.annotate(f"κ = {yk:+.4f}".replace("-", "−"), (0, yk), xytext=(-12, dy),
                                    textcoords="offset points", fontsize=8.5, color="#f2c14e", va=va, ha="right",
                                    bbox=box)
                    for xk, ha, dx in ((2.0, "left", 8), (-2.0, "right", -8)):
                        ax.annotate(f"γ = {xk:+g}".replace("-", "−"), (xk, 0), xytext=(dx, -12),
                                    textcoords="offset points", fontsize=8.5, color="#f2c14e", ha=ha, va="top",
                                    bbox=box)
                handles.append(Line2D([], [], ls="", marker="|", ms=10, mew=2, color="#f2c14e",
                                      label="book's values (Fig. 1.3): apex κ_max, Landau window"))
            if self.chk_mu.isChecked() and mode != "mu":
                handles += [Line2D([], [], color="#ffffff", lw=1.4, label="μ₊ contours (north): ½ dotted, ⅔ dashed, 1 dash-dot"),
                            Line2D([], [], color="#f2c14e", lw=1.4, label="μ₋ contours (south), same styles")]
            if self.chk_edge.isChecked():
                handles.append(Line2D([], [], color=p["plot_ink"], lw=1.4, label="edge of the global region"))
            self._draw_path(axs, spec, handles)
            # the current solution
            cur = self.current
            if cur is not None:
                cx, cy = getattr(cur, spec.x_name), getattr(cur, spec.y_name)
                same = all(abs(getattr(cur, n) - getattr(spec.base, n)) < 1e-12
                           for n in PARAM_NAMES if n not in (spec.x_name, spec.y_name)) \
                    and abs(cur.xb - spec.base.xb) < 1e-12
                for ax in axs:
                    ring = bool(self.path_overlay and self.path_overlay.get("keys"))
                    if ring and same:          # a ring, so keyframe badges under it stay readable
                        ax.plot([cx], [cy], marker="o", ms=17, mfc="none", mec=MARK, mew=2.6, zorder=12,
                                clip_on=False)
                    else:
                        ax.plot([cx], [cy], marker="o", ms=11, mfc=MARK if same else "none",
                                mec="white" if same else MARK, mew=2.0, zorder=12, clip_on=False)
                if self.path_overlay and self.path_overlay.get("keys"):
                    handles.append(Line2D([], [], ls="", marker="o", ms=12, mfc="none", mec=MARK, mew=2.2,
                                          label="current solution (on the path: the preview position)"))
                else:
                    handles.append(Line2D([], [], ls="", marker="o", ms=9, mfc=MARK, mec="white", mew=1.5,
                                          label="current solution"))
                if not same:
                    handles.append(Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=MARK, mew=1.8,
                                          label="current solution (other fixed values: recompute)"))
            others = [n for n in PARAM_NAMES if n not in (spec.x_name, spec.y_name)]
            fixed = ",  ".join(f"{_name(n)} = {getattr(spec.base, n):.4g}" for n in others).replace("-", "−")
            self.fig.suptitle(f"Σ-slice in ({_name(spec.x_name)}, {_name(spec.y_name)})   at   {fixed},  "
                              f"x₀ = {spec.base.xb:g}", fontsize=10, color=p["plot_ink"])
            self.fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="outside lower center",
                            ncol=2, fontsize=8.5, frameon=False, handler_map=_legend_map())
            from matplotlib.widgets import MultiCursor
            self._cursor = MultiCursor(list(axs), useblit=True, horizOn=True, color=p["plot_ink"],
                                       lw=0.6, alpha=0.6)
        self.canvas.draw_idle()

    def set_path_overlay(self, overlay) -> None:
        self.path_overlay = overlay
        if not self.running:
            self.redraw()

    def _draw_path(self, axs, spec, handles):
        """The path dock's path projected on the map axes: samples, red where not global, numbered keys."""
        ov = self.path_overlay
        if not ov or not ov.get("keys"):
            return
        from matplotlib.lines import Line2D
        xn, yn = spec.x_name, spec.y_name
        xs = np.array([getattr(q, xn) for q in ov["samples"]])
        ys = np.array([getattr(q, yn) for q in ov["samples"]])
        kind = ov.get("kind")
        kind = np.asarray(kind) if kind is not None and len(kind) == len(xs) else np.array(["ok"] * len(xs))
        bad = kind != "ok"
        kx = [getattr(q, xn) for q in ov["keys"]]
        ky = [getattr(q, yn) for q in ov["keys"]]
        halo = "#101114" if style.MODE == "dark" else "#ffffff"
        for ax in axs:
            ax.plot(xs, ys, color=halo, lw=4.2, zorder=7, solid_capstyle="round")
            ax.plot(xs, ys, color="#ffffff" if style.MODE == "dark" else "#222222", lw=2.0, zorder=8)
            for k, ls in (("blowup", "-"), ("unsafe", "-")):
                m = kind == k
                if m.any():   # include the neighbours so short runs still show as segments
                    mm = m | np.r_[m[1:], False] | np.r_[False, m[:-1]]
                    yb = np.where(mm, ys, np.nan)
                    ax.plot(xs, yb, color="#000000", lw=7.5, zorder=8.4, solid_capstyle="butt")
                    ax.plot(xs, yb, color=PATH_FAIL[k], lw=4.5, ls=ls, zorder=8.5, solid_capstyle="butt")
            for q in ov.get("events", []):            # the timeline's event triangles, on the map
                ax.plot([getattr(q, xn)], [getattr(q, yn)], marker="v", ms=11, mfc="#f2c14e", mec=halo,
                        mew=1.0, zorder=10.5, ls="")
            ax.plot(kx, ky, ls="", marker="o", ms=13, mfc="#f2c14e", mec=halo, mew=1.2, zorder=11,
                    clip_on=False)
            for i, (a, b) in enumerate(zip(kx, ky)):
                ax.annotate(str(i + 1), (a, b), ha="center", va="center", fontsize=7.5, color="#111111",
                            fontweight="bold", zorder=11.5, annotation_clip=False)
        handles.append(Line2D([], [], color="#ffffff" if style.MODE == "dark" else "#222222", lw=2,
                              marker="o", ms=8, mfc="#f2c14e", mec=halo, label="path, global part (numbered keyframes)"))
        labels_extra = []
        from matplotlib.legend_handler import HandlerTuple   # noqa: F401  (tuples drawn as overlays)
        if (kind == "blowup").any():
            handles.append((Line2D([], [], color="#000000", lw=7.5),
                            Line2D([], [], color=PATH_FAIL["blowup"], lw=4.5)))
            labels_extra.append("path: not global, blows up (as on the timeline)")
        if (kind == "unsafe").any():
            handles.append((Line2D([], [], color="#000000", lw=7.5),
                            Line2D([], [], color=PATH_FAIL["unsafe"], lw=4.5)))
            labels_extra.append("path: not global, a pole fails the terminal test")
        # tuple handles carry no label of their own: wrap them so the legend can name them
        for i, h in enumerate(handles):
            if isinstance(h, tuple):
                handles[i] = _Labelled(h, labels_extra.pop(0))
        if ov.get("events"):
            handles.append(Line2D([], [], ls="", marker="v", ms=9, mfc="#f2c14e", mec=halo,
                                  label="path event (as on the timeline)"))
        other = [n for n in ov.get("varying", []) if n not in (xn, yn)]
        if other:
            handles.append(Line2D([], [], ls="", label="path also varies " + ", ".join(_name(n) for n in other)
                                  + " (not shown on this map)"))

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
        res = self.res
        if res is None or x is None:
            return ""
        xs, ys = res.spec.axes()
        i, j = int(np.argmin(abs(xs - x))), int(np.argmin(abs(ys - y)))
        sx, sy = _name(res.spec.x_name), _name(res.spec.y_name)
        head = f"{sx} = {xs[i]:.4g},  {sy} = {ys[j]:.4g}:  ".replace("-", "−")
        if not res.done[j, i]:
            return head + "not computed yet"

        def pole(code, mu):
            if code == BLOWUP:
                return "blows up before it"
            if code == UNSAFE:
                return "fails the terminal test"
            return f"{CODE_NAMES[code]}, μ = {mu:.3f}"
        state = "GLOBAL" if res.is_global[j, i] else "not global"
        extra = ""
        if math.isfinite(res.blow_x[j, i]):
            extra = f" (blow-up at x ≈ {res.blow_x[j, i]:.3f})".replace("-", "−")
        return (head + f"{state}{extra}.   North: {pole(res.code_n[j, i], res.mu_n[j, i])}.   "
                f"South: {pole(res.code_s[j, i], res.mu_s[j, i])}.")

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
