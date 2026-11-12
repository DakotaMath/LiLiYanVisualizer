"""Slice plane: a true 2D view of u·n and of the in-plane flow, synced with the 3D view."""
from __future__ import annotations

import math
import threading
import warnings

# harmless: the hidden (zero-size) slice canvas cannot be laid out until it is shown
warnings.filterwarnings("ignore", message="constrained_layout not applied")

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
                dens = opts["density"] * (0.5 if opts["lic"] else 1.0)
                lines = projected_streamlines(data, density=dens) if opts["lines"] else []
                out = (data, lines)
            except Exception as exc:  # never crash the UI
                out = exc
            with self._lock:
                stale = self._job is not None
            if not stale:
                self.done.emit(out, token)


def _axis_name(v) -> str:
    """'x', '−z', ... for (anti)parallel coordinate directions, else the vector."""
    for k, name in enumerate("xyz"):
        if abs(abs(v[k]) - 1.0) < 1e-6:
            return name if v[k] > 0 else "−" + name
    return _fmt_vec(v)


def _plane_name(spec) -> str:
    n = spec.normal
    off = f"{spec.offset:+.2f}".replace("-", "−")
    if abs(n[2]) > 1 - 1e-6:
        return f"Horizontal slice z = {spec.offset * np.sign(n[2]):+.2f}".replace("-", "−")
    if abs(n[2]) < 1e-6 and abs(spec.offset) < 1e-9:
        return "Meridional slice through the axis"
    return f"Slice at offset {off}"


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
        self.controls = self._controls()       # placed in a dock by the main window
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
        self.az_spin = self._spin(-180, 180, 2, 1.0, "°")
        self.el_spin = self._spin(-90, 90, 2, 1.0, "°")
        self.off_spin = self._spin(-0.95, 0.95, 3, 0.01, "")
        for r, (name, s, sp) in enumerate((("azimuth", self.az, self.az_spin), ("elevation", self.el, self.el_spin),
                                           ("offset", self.off, self.off_spin))):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(s, r, 1)
            grid.addWidget(sp, r, 2)
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
        self.chk_rho = QCheckBox("scale by ρ (distance to the axis)")
        self.chk_rho.setToolTip("plot ρ·(u·n): removes the 1/ρ blow-up at the axis so the rest is visible")
        self.chk_rho.toggled.connect(self._display_changed)
        self.chk_log = QCheckBox("signed-log colour scale")
        self.chk_log.setToolTip("colour by sign(v)·log(1 + |v|/v₀): shows small and very large values together")
        self.chk_log.toggled.connect(self._display_changed)
        self.chk_lic.setToolTip("streaks along the in-plane flow direction (LIC)")
        dens = QHBoxLayout()
        dens.addWidget(QLabel("line density"))
        self.density = QDoubleSpinBox()
        self.density.setRange(0.3, 4.0)
        self.density.setSingleStep(0.2)
        self.density.setValue(1.4)
        dens.addWidget(self.density)
        for w in (self.chk_un, self.chk_rho, self.chk_log, self.chk_lines, self.chk_lic):
            gv.addWidget(w)
            w.toggled.connect(self._options_changed)
        gv.addLayout(dens)
        self.density.valueChanged.connect(self._options_changed)
        v.addWidget(g)

        self.view_n_btn = QPushButton("Point the 3D camera along n")
        self.view_n_btn.setToolTip("look at the 3D view from the same side as this slice")
        v.addWidget(self.view_n_btn)
        self.follow = QCheckBox("3D camera follows the slice")
        self.follow.setToolTip("re-aim the 3D camera along n whenever the plane changes")
        v.addWidget(self.follow)
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
        sc.setMinimumWidth(260)
        return sc

    def _slider(self, lo, hi, tip) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setToolTip(tip)
        s.valueChanged.connect(self._sliders_moved)
        return s

    def _spin(self, lo, hi, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)          # apply on Enter / focus-out, not on every keystroke
        s.setMinimumWidth(78)
        s.valueChanged.connect(self._spins_changed)
        return s

    # ---------------------------------------------------------------- state sync
    def _load_spec(self, spec: PlaneSpec):
        """Show `spec` in the sliders and the typed fields without emitting anything."""
        self._block = True
        self.az.setValue(int(round(spec.azimuth)))
        self.el.setValue(int(round(spec.elevation)))
        self.off.setValue(int(round(spec.offset * 100)))
        self.az_spin.setValue(spec.azimuth)
        self.el_spin.setValue(spec.elevation)
        self.off_spin.setValue(spec.offset)
        self._block = False
        self._update_labels(spec)

    def _update_labels(self, spec: PlaneSpec):
        e1, e2, n = spec.basis()
        self.normal_lab.setText(f"normal n = {_fmt_vec(n)} points toward you<br>"
                                f"right = {_fmt_vec(e1)}, up = {_fmt_vec(e2)}")

    def _preset(self, az, el):
        spec = PlaneSpec(azimuth=az, elevation=el, offset=0.0, half_size=self.ctrl.spec.half_size,
                         n=self.ctrl.spec.n)
        self._load_spec(spec)                 # the sliders and fields must show the preset
        self.ctrl.set_spec(spec, source=self)

    def _emit(self, az, el, off):
        old = self.ctrl.spec
        spec = PlaneSpec(azimuth=az, elevation=el, offset=off, half_size=old.half_size, n=old.n)
        self._load_spec(spec)
        self.ctrl.set_spec(spec, source=self)

    def _sliders_moved(self, *_):
        if self._block:
            return
        self._emit(self.az.value(), self.el.value(), self.off.value() / 100)

    def _spins_changed(self, *_):
        if self._block:
            return
        self._emit(self.az_spin.value(), self.el_spin.value(), self.off_spin.value())

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
        data.set_display(self.chk_rho.isChecked(), self.chk_log.isChecked())
        self.last = (data, lines)
        if self.follow.isChecked():
            self.view_n_btn.click()
        self.redraw()
        self.sliceReady.emit(data, lines)

    def _display_changed(self, *_):
        if self.last is not None:
            self.last[0].set_display(self.chk_rho.isChecked(), self.chk_log.isChecked())
            self.redraw()
            self.sliceReady.emit(*self.last)

    def redraw(self):
        if self.last is None:
            return
        import matplotlib
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        data, lines = self.last
        p = style.c()
        self.fig.clear()
        self.fig.set_facecolor(p["plot_bg"])
        with matplotlib.rc_context(style.mpl_rc()):
            ax = self.fig.add_subplot()
            L = data.spec.half_size
            ext = (-L, L, -L, L)
            AXIS_COL = "#e09a2d"
            m = data.clim
            norm = matplotlib.colors.Normalize(-m, m)
            cmap = matplotlib.colormaps["RdBu_r"]
            if self.chk_un.isChecked():
                rgba = cmap(norm(np.nan_to_num(data.show)))
                if data.lic is not None:       # LIC only modulates brightness: the colour stays readable
                    g = np.nan_to_num(data.lic, nan=0.5)
                    lo, hi = np.percentile(g, [2, 98])          # contrast-stretch the texture
                    g = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)
                    f = 0.55 + 0.6 * g
                    rgba[..., :3] = np.clip(rgba[..., :3] * f[..., None], 0, 1)
                rgba[..., 3] = np.isfinite(data.show)
                ax.imshow(rgba, origin="lower", extent=ext, interpolation="bilinear")
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                tk = [-m, -m / 2, 0, m / 2, m]
                if data.symlog:                                 # ticks at decades of the real value
                    dec = [10.0 ** k for k in range(-6, 12) if data.lin_thresh <= 10.0 ** k <= data.vmax]
                    pos = [float(np.log10(1.0 + v / data.lin_thresh)) for v in dec[-3:]]
                    if pos:
                        tk = [-t for t in pos[::-1]] + [0.0] + pos
                cb = self.fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09,
                                       extend="neither" if data.symlog else "both", aspect=35, ticks=tk)
                cb.ax.set_xticklabels([f"{v:.3g}".replace("-", "−") for v in
                                       (np.round(data.to_value(tk), 6) if data.symlog else tk)])
                if data.symlog:
                    note = f"(signed-log scale, linear below {data.lin_thresh:.2g}; max |{data.show_label}| = {data.vmax:.3g})"
                else:
                    note = f"(values beyond ±{m:.3g} saturate; max |{data.show_label}| = {data.vmax:.3g})"
                cb.set_label(f"{data.show_label}:  red = toward you,  blue = away" + "\n" + note, fontsize=8.5)
                sat = np.abs(np.nan_to_num(data.show)) > m
                if sat.any():
                    ax.contour(data.s, data.t, sat.astype(float), levels=[0.5], colors=[p["plot_ink"]],
                               linewidths=0.8, linestyles=":")
            elif data.lic is not None:
                ax.imshow(np.ma.masked_invalid(data.lic), origin="lower", extent=ext, cmap="gray",
                          interpolation="bilinear")
            # geometric cut-outs as smooth hatched shapes
            from matplotlib.patches import Circle, Rectangle
            R = self.domain.R
            d = data.spec.offset
            if abs(d) < self.domain.r_min:
                rc = math.sqrt(self.domain.r_min ** 2 - d * d)
                ax.add_patch(Circle((0, 0), rc, facecolor=p["plot_bg"], edgecolor=p["plot_ink2"], hatch="////",
                                    lw=0.8, zorder=4))
            line_col = "#151515"                         # dark on the light colour map, in both themes
            shown = lines if data.lic is None else sorted(lines, key=len, reverse=True)[:8]   # 8 longest
            from matplotlib import patheffects as _pe
            halo = [_pe.Stroke(linewidth=2.6, foreground="#f4f4f4", alpha=0.55), _pe.Normal()]
            for poly in shown:
                ax.plot(poly[:, 0], poly[:, 1], color=line_col, lw=0.9 if data.lic is None else 1.0, zorder=5,
                        path_effects=halo)      # light halo keeps lines visible on saturated colour
                if len(poly) > 6:                       # one arrowhead per line, mid-way
                    k = len(poly) // 2
                    ax.annotate("", xy=poly[k + 1], xytext=poly[k], zorder=5,
                                arrowprops=dict(arrowstyle="-|>", color=line_col, lw=0.9, mutation_scale=10))
            if abs(d) < R:
                rr = math.sqrt(R * R - d * d)
                th = np.linspace(0, 2 * np.pi, 200)
                ax.plot(rr * np.cos(th), rr * np.sin(th), color=p["plot_ink2"], lw=0.9, ls="--")
            flow_lab = "in-plane flow (u projected)" if data.lic is None else "in-plane flow: texture + a few arrows"
            handles = [Line2D([], [], color=p["plot_ink"], lw=0.9, marker=">", ms=4, label=flow_lab),
                       Line2D([], [], color=p["plot_ink2"], lw=0.9, ls="--", label=f"edge of the ball |X| = {R:g}")]
            if abs(d) < self.domain.r_min:
                handles.append(Patch(facecolor="none", edgecolor=p["plot_ink2"], hatch="////",
                                     label="cut out around the origin"))
            if self.chk_un.isChecked() and (np.abs(np.nan_to_num(data.show)) > m).any():
                handles.append(Line2D([], [], color=p["plot_ink"], lw=1.0, ls=":",
                                      label=f"|{data.show_label}| beyond the colour range"))
            for kind, a, b in data.axis_hits:
                if kind == "point" and abs(a) <= L and abs(b) <= L:
                    ax.plot([a], [b], marker="o", ms=9, mfc="white", mec=AXIS_COL, mew=2.2, zorder=6)
                    handles.append(Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=AXIS_COL, mew=2,
                                          label="symmetry axis crosses here"))
                elif kind == "line":
                    w = 2 * self.domain.axis_a
                    ax.add_patch(Rectangle((-w / 2, -L), w, 2 * L, facecolor=p["plot_bg"], edgecolor="none",
                                           zorder=4))
                    ax.plot([-a * L, a * L], [-b * L, b * L], color=AXIS_COL, lw=1.6, ls="-.", zorder=6)
                    handles.append(Line2D([], [], color=AXIS_COL, lw=1.6, ls="-.", label="symmetry axis (z)"))
            e1, e2, n = data.spec.basis()
            ax.set_xlabel(_axis_name(e1) + "  →", fontsize=11)
            ax.set_ylabel(_axis_name(e2) + "  →", fontsize=11)
            where = ""
            if abs(d) > self.domain.r_min:
                where = f"\norigin {abs(d):.2f} " + ("below" if d > 0 else "above") + " the plane"
            ax.set_title(f"{_plane_name(data.spec)}\nn = {_axis_name(n)} points toward you{where}", fontsize=10,
                         loc="left")
            ticks = np.arange(-L, L + 1e-9, 0.5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_aspect("equal")
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=1, fontsize=8.5,
                      frameon=False)
        self.canvas.draw_idle()
        self.info.setText("")

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
