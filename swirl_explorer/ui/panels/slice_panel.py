"""Slice plane: a true 2D view of u·n and of the in-plane flow, synced with the 3D view."""
from __future__ import annotations

import math
import threading

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QSlider, QLabel,
                               QCheckBox, QDoubleSpinBox, QGridLayout, QScrollArea, QToolButton, QMenu,
                               QFileDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...core.fields import FieldInterpolator
from ...viz import style
from ...viz.slices import PlaneSpec, PRESETS, sample_plane, lic, projected_streamlines
from ...viz.streamlines import Domain


class SliceController(QObject):
    """Single source of truth for the slice plane (shared by the 2D panel and the 3D view)."""
    planeChanged = Signal(object)

    def __init__(self):
        super().__init__()
        self.spec = PlaneSpec()

    def set_spec(self, spec: PlaneSpec, source=None) -> None:
        self.spec = spec
        self.planeChanged.emit((spec, source))


class _SliceWorker(QObject):
    done = Signal(object, int)
    _kick = Signal()

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._job = None
        self._kick.connect(self._run)

    def submit(self, job):
        with self._lock:
            self._job = job
        self._kick.emit()

    @Slot()
    def _run(self):
        while True:
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                return
            sol, spec, domain, opts, token = job
            try:
                fi = FieldInterpolator(sol, dxi=0.002, fast=True)
                data = sample_plane(fi, spec, domain)
                if opts["lic"]:
                    data.lic = lic(data.us, data.ut)
                lines = projected_streamlines(data, density=opts["density"]) if opts["lines"] else []
                out = (data, lines)
            except Exception as exc:  # never crash the UI
                out = exc
            with self._lock:
                stale = self._job is not None
            if not stale:
                self.done.emit(out, token)


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{c:+.2f}".replace("-", "−") for c in v) + ")"


class SlicePanel(QWidget):
    sliceReady = Signal(object, object)          # SliceData, list of (k,2) polylines in (s, t)

    def __init__(self, controller: SliceController, parent=None):
        super().__init__(parent)
        self.ctrl = controller
        self.sol = None
        self.domain = Domain()
        self._token = 0
        self._block = False
        self.last = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        h.addWidget(self.canvas, 1)
        h.addWidget(self._controls())
        self.thread = QThread()
        self.worker = _SliceWorker()
        self.worker.moveToThread(self.thread)
        self.worker.done.connect(self._draw)
        self.thread.start()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(90)
        self._debounce.timeout.connect(self._submit)
        self.ctrl.planeChanged.connect(self._plane_changed)
        self._load_spec(self.ctrl.spec)

    # ---------------------------------------------------------------- controls
    def _controls(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 4, 6, 4)
        g = QGroupBox("Plane orientation")
        gv = QVBoxLayout(g)
        pres = QGridLayout()
        for k, (name, (az, el)) in enumerate(PRESETS.items()):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, a=az, e=el: self._preset(a, e))
            pres.addWidget(b, k, 0)
        gv.addLayout(pres)
        grid = QGridLayout()
        self.az = self._slider(-180, 180, "azimuth of the normal (degrees, from +x toward +y)")
        self.el = self._slider(-90, 90, "elevation of the normal above the equator (degrees)")
        self.off = self._slider(-95, 95, "offset of the plane from the origin along its normal (×0.01)")
        self.az_lab, self.el_lab, self.off_lab = QLabel(), QLabel(), QLabel()
        for r, (name, s, lab) in enumerate((("azimuth", self.az, self.az_lab), ("elevation", self.el, self.el_lab),
                                            ("offset", self.off, self.off_lab))):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(s, r, 1)
            grid.addWidget(lab, r, 2)
        gv.addLayout(grid)
        self.normal_lab = QLabel("")
        self.normal_lab.setWordWrap(True)
        gv.addWidget(self.normal_lab)
        v.addWidget(g)

        g = QGroupBox("Show on the plane")
        gv = QVBoxLayout(g)
        self.chk_un = QCheckBox("colour: normal velocity u·n")
        self.chk_lines = QCheckBox("streamlines of the in-plane flow")
        self.chk_lic = QCheckBox("texture (line-integral convolution)")
        self.chk_un.setChecked(True)
        self.chk_lines.setChecked(True)
        self.chk_lic.setChecked(False)
        self.chk_lic.setToolTip("streaks along the in-plane flow direction (LIC)")
        dens = QHBoxLayout()
        dens.addWidget(QLabel("line density"))
        self.density = QDoubleSpinBox()
        self.density.setRange(0.3, 4.0)
        self.density.setSingleStep(0.2)
        self.density.setValue(1.4)
        dens.addWidget(self.density)
        for w in (self.chk_un, self.chk_lines, self.chk_lic):
            gv.addWidget(w)
            w.toggled.connect(self._options_changed)
        gv.addLayout(dens)
        self.density.valueChanged.connect(self._options_changed)
        v.addWidget(g)

        self.export_btn = QToolButton()
        self.export_btn.setText("Export ▾")
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self.export_btn)
        m.addAction("PNG…", lambda: self.export("png"))
        m.addAction("SVG…", lambda: self.export("svg"))
        self.export_btn.setMenu(m)
        v.addWidget(self.export_btn)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        v.addWidget(self.info)
        v.addStretch(1)
        sc = QScrollArea()
        sc.setWidget(box)
        sc.setWidgetResizable(True)
        sc.setFixedWidth(270)
        return sc

    def _slider(self, lo, hi, tip) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setToolTip(tip)
        s.valueChanged.connect(self._sliders_moved)
        return s

    # ---------------------------------------------------------------- state sync
    def _load_spec(self, spec: PlaneSpec):
        self._block = True
        self.az.setValue(int(round(spec.azimuth)))
        self.el.setValue(int(round(spec.elevation)))
        self.off.setValue(int(round(spec.offset * 100)))
        self._block = False
        self._update_labels(spec)

    def _update_labels(self, spec: PlaneSpec):
        self.az_lab.setText(f"{spec.azimuth:.0f}°")
        self.el_lab.setText(f"{spec.elevation:.0f}°")
        self.off_lab.setText(f"{spec.offset:+.2f}".replace("-", "−"))
        e1, e2, n = spec.basis()
        self.normal_lab.setText(f"normal n = {_fmt_vec(n)} points toward you<br>"
                                f"right = {_fmt_vec(e1)}, up = {_fmt_vec(e2)}")

    def _preset(self, az, el):
        spec = PlaneSpec(azimuth=az, elevation=el, offset=0.0, half_size=self.ctrl.spec.half_size,
                         n=self.ctrl.spec.n)
        self.ctrl.set_spec(spec, source=self)

    def _sliders_moved(self, *_):
        if self._block:
            return
        old = self.ctrl.spec
        spec = PlaneSpec(azimuth=self.az.value(), elevation=self.el.value(), offset=self.off.value() / 100,
                         half_size=old.half_size, n=old.n)
        self.ctrl.set_spec(spec, source=self)

    def _plane_changed(self, payload):
        spec, source = payload
        if source is not self:
            self._load_spec(spec)
        else:
            self._update_labels(spec)
        self._debounce.start()

    def _options_changed(self, *_):
        self._debounce.start()

    # ---------------------------------------------------------------- compute + draw
    def set_solution(self, sol, summary=None, domain: Domain | None = None):
        self.sol = sol
        if domain is not None:
            self.domain = domain
        self._debounce.start()

    def set_domain(self, domain: Domain):
        self.domain = domain
        self._debounce.start()

    def _submit(self):
        if self.sol is None:
            return
        self._token += 1
        opts = dict(lic=self.chk_lic.isChecked(), lines=self.chk_lines.isChecked(), density=self.density.value())
        self.worker.submit((self.sol, self.ctrl.spec, self.domain, opts, self._token))
        self.info.setText("computing…")

    def _draw(self, out, token):
        if token != self._token:
            return
        if isinstance(out, Exception):
            self.info.setText(f"slice error: {out}")
            return
        data, lines = out
        self.last = (data, lines)
        self.redraw()
        self.sliceReady.emit(data, lines)

    def redraw(self):
        if self.last is None:
            return
        import matplotlib
        data, lines = self.last
        p = style.c()
        self.fig.clear()
        self.fig.set_facecolor(p["plot_bg"])
        with matplotlib.rc_context(style.mpl_rc()):
            ax = self.fig.add_subplot()
            L = data.spec.half_size
            ext = (-L, L, -L, L)
            if data.lic is not None:
                ax.imshow(np.ma.masked_invalid(data.lic), origin="lower", extent=ext, cmap="gray",
                          vmin=0.2, vmax=0.8, interpolation="bilinear")
            if self.chk_un.isChecked():
                v = data.un[np.isfinite(data.un)]
                m = float(np.percentile(np.abs(v), 85)) if v.size else 1.0   # the axis spike would wash it out
                m = m if m > 0 else 1.0
                cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad=(0, 0, 0, 0))
                im = ax.imshow(np.ma.masked_invalid(data.un), origin="lower", extent=ext, cmap=cmap, vmin=-m,
                               vmax=m, interpolation="bilinear", alpha=0.65 if data.lic is not None else 1.0)
                cb = self.fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend="both")
                cb.set_label("u·n  (red: toward you, blue: away)")
            # excluded regions
            bad = ~np.isfinite(data.un)
            ax.contourf(data.s, data.t, bad.astype(float), levels=[0.5, 1.5], colors=[p["plot_grid"]])
            line_col = "#151515"                         # dark on the light colour map, in both themes
            for poly in lines:
                ax.plot(poly[:, 0], poly[:, 1], color=line_col, lw=0.9)
                if len(poly) > 6:                       # one arrowhead per line, mid-way
                    k = len(poly) // 2
                    ax.annotate("", xy=poly[k + 1], xytext=poly[k],
                                arrowprops=dict(arrowstyle="-|>", color=line_col, lw=0.9, mutation_scale=10))
            # region boundary (|X| = R intersected with the plane)
            R = self.domain.R
            d = data.spec.offset
            if abs(d) < R:
                rr = math.sqrt(R * R - d * d)
                th = np.linspace(0, 2 * np.pi, 200)
                ax.plot(rr * np.cos(th), rr * np.sin(th), color=p["plot_ink2"], lw=0.8, ls="--")
            for kind, a, b in data.axis_hits:
                if kind == "point" and abs(a) <= L and abs(b) <= L:
                    ax.plot([a], [b], marker="o", ms=7, mfc=p["plot_bg"], mec=p["plot_ink"], mew=1.5)
                    ax.annotate("axis", (a, b), xytext=(6, 6), textcoords="offset points", fontsize=9)
                elif kind == "line":
                    ax.plot([-a * L, a * L], [-b * L, b * L], color=p["plot_ink"], lw=1.0, ls=":")
                    ax.annotate("symmetry axis", (a * 0.95 * L, b * 0.95 * L), fontsize=9, ha="center",
                                xytext=(8, 0), textcoords="offset points")
            e1, e2, n = data.spec.basis()
            ax.set_xlabel(f"s   (right = {_fmt_vec(e1)})")
            ax.set_ylabel(f"t   (up = {_fmt_vec(e2)})")
            ax.set_title(f"normal n = {_fmt_vec(n)} (toward you), offset {data.spec.offset:+.2f}".replace("-", "−")
                         + "\nlines: in-plane flow (u projected onto the plane)"
                         + ("; streaks: LIC texture" if data.lic is not None else ""), fontsize=10, loc="left")
            ax.set_aspect("equal")
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
        self.canvas.draw_idle()
        self.info.setText(f"{len(lines)} in-plane streamlines; grey = cut-out (origin ball, axis cylinder) "
                          f"or outside the region; dashed circle = |X| = {R:g}")

    def refresh_style(self):
        self.info.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        self.redraw()

    def export(self, kind: str, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f"Export {kind.upper()}", f"slice.{kind}",
                                                  f"{kind.upper()} (*.{kind})")
        if path:
            self.fig.savefig(path, dpi=200 if kind == "png" else None)
        return path

    def shutdown(self):
        self.thread.quit()
        self.thread.wait(2000)
