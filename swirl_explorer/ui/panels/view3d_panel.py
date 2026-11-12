"""3D view: PyVista interactor with draggable seed widgets, streamlines and field visualisations."""
from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QComboBox, QSpinBox,
                               QDoubleSpinBox, QCheckBox, QLabel, QSlider, QScrollArea, QFormLayout,
                               QPushButton)

from ...viz.scene_compute import SceneSettings, compute_numeric, build_scene
from ...viz.scene3d import COLOR_BY, STRATUM_COLORS, VTK_TITLES
from ...viz import style

try:                                    # the 3D view needs an OpenGL context
    from pyvistaqt import QtInteractor
    HAVE_PV = True
except Exception:                       # pragma: no cover
    HAVE_PV = False

ISO_FIELDS = {"speed": "|u|", "swirl": "swirl f / sin θ", "helicity": "helicity u·ω"}


class _SceneWorker(QObject):
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
            sol, sm, settings, token = job
            try:
                data = (compute_numeric(sol, settings), sm)     # NumPy only: VTK stays on the GUI thread
            except Exception as exc:  # never crash the UI
                data = exc
            with self._lock:
                stale = self._job is not None
            if not stale:
                self.done.emit(data, token)


class View3DPanel(QWidget):
    domainChanged = Signal(object)           # Domain, so the slice panel uses the same cut-outs

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = SceneSettings()
        self.slice_ctrl = None
        self._slice = None                   # (SliceData, lines) from the slice panel
        self._plane_widget = None
        self.sol = None
        self.summary = None
        self._token = 0
        self._last = None
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        if HAVE_PV:
            self.plotter = QtInteractor(self)
            self.plotter.set_background(style.c()["view_bg"], top=style.c()["view_bg_top"])
            self.plotter.enable_parallel_projection()
            self.plotter.add_axes(color=style.c()["view_text"], xlabel="x", ylabel="y", zlabel="z", line_width=5,
                                  viewport=(0.0, 0.0, 0.3, 0.36))
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)
            col.addWidget(self.plotter.interactor, 1)
            self.legend = QLabel("")
            self.legend.setTextFormat(Qt.RichText)
            self.legend.setWordWrap(True)
            self._style_legend()
            col.addWidget(self.legend)
            h.addLayout(col, 1)
        else:  # pragma: no cover
            self.plotter = None
            h.addWidget(QLabel("PyVista / VTK not available"), 1)
        h.addWidget(self._controls())
        self.thread = QThread()
        self.worker = _SceneWorker()
        self.worker.moveToThread(self.thread)
        self.worker.done.connect(self._render)
        self.thread.start()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._submit)
        self._widgets_mode = None
        self._camera_set = False

    # ---------------------------------------------------------------- controls
    def _controls(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 4, 6, 4)

        g = QGroupBox("Seeds (drag the widget in the view)")
        f = QFormLayout(g)
        self.seed_mode = QComboBox()
        for k, t in (("rings", "rings at 5 heights"), ("ring", "one ring around the axis"),
                     ("sphere", "points in a ball"), ("disc", "disc rake"), ("line", "line rake")):
            self.seed_mode.addItem(t, k)
        self.n_seeds = QSpinBox()
        self.n_seeds.setRange(1, 600)
        self.n_seeds.setValue(self.settings.n_seeds)
        f.addRow("seed shape", self.seed_mode)
        f.addRow("number", self.n_seeds)
        self.show_handle = QCheckBox("show seed handle (drag to move seeds)")
        self.show_handle.setChecked(False)
        self.show_handle.setToolTip("off by default; tick it to place the seeds by dragging in the view")
        self.show_handle.toggled.connect(self._handle_toggled)
        f.addRow(self.show_handle)
        v.addWidget(g)

        g = QGroupBox("Colour lines and surfaces by")
        gv = QVBoxLayout(g)
        self.color_by = QComboBox()
        for k, t in COLOR_BY.items():
            self.color_by.addItem(t, k)
        gv.addWidget(self.color_by)
        v.addWidget(g)

        g = QGroupBox("Show")
        gv = QVBoxLayout(g)
        self.chk = {}
        for key, text, tip in (
                ("show_streamlines", "streamlines", "integral curves of u from the seeds"),
                ("show_surface", "stream surface", "surface swept by streamlines from a seed line"),
                ("show_vortex", "vortex lines", "integral curves of the vorticity ω from the same seeds"),
                ("show_glyphs", "arrows", "velocity arrows on a coarse grid (length ∝ log speed)"),
                ("show_iso", "isosurface", "level set of the field chosen below"),
                ("show_selfsim", "self-similar copies", "the seed-centre streamline and its rescalings λX "
                                                         "(also streamlines, since u is (−1)-homogeneous)"),
                ("clip", "cut away the half y > 0", "clip lines and surfaces by the plane y = 0")):
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.setChecked(getattr(self.settings, key))
            self.chk[key] = c
            gv.addWidget(c)
        iso = QHBoxLayout()
        self.iso_field = QComboBox()
        for k, t in ISO_FIELDS.items():
            self.iso_field.addItem(t, k)
        self.iso_level = QSlider(Qt.Horizontal)
        self.iso_level.setRange(1, 99)
        self.iso_level.setValue(int(self.settings.iso_quantile * 100))
        self.iso_level.setToolTip("isosurface level, as a percentile of the sampled values")
        iso.addWidget(self.iso_field)
        iso.addWidget(self.iso_level)
        gv.addLayout(iso)
        self.iso_info = QLabel("")
        self.iso_info.setStyleSheet(f"color:{style.c()['faint']};font-size:10px;")
        gv.addWidget(self.iso_info)
        v.addWidget(g)

        g = QGroupBox("Slice plane (see the Slice plane tab)")
        gv = QVBoxLayout(g)
        self.chk_slice = QCheckBox("show the slice plane here")
        self.chk_slice.setToolTip("the plane coloured by u·n with its in-plane streamlines")
        self.chk_plane_widget = QCheckBox("drag the plane in 3D")
        self.chk_plane_widget.setToolTip("adds a plane handle: drag it to move, the arrow to tilt")
        self.chk_slice.toggled.connect(self._slice_toggled)
        self.chk_plane_widget.toggled.connect(self._plane_widget_toggled)
        gv.addWidget(self.chk_slice)
        gv.addWidget(self.chk_plane_widget)
        v.addWidget(g)

        g = QGroupBox("Region")
        f = QFormLayout(g)
        self.shape = QComboBox()
        self.shape.addItem("ball |X| ≤ 1", "ball")
        self.shape.addItem("box |X|∞ ≤ 1", "box")
        self.r_min = QDoubleSpinBox()
        self.r_min.setRange(0.01, 0.8)
        self.r_min.setSingleStep(0.02)
        self.r_min.setValue(self.settings.r_min)
        self.r_min.setToolTip("excluded ball around the origin (u ~ 1/r)")
        self.axis_a = QDoubleSpinBox()
        self.axis_a.setDecimals(3)
        self.axis_a.setRange(0.002, 0.3)
        self.axis_a.setSingleStep(0.005)
        self.axis_a.setValue(self.settings.axis_a)
        self.axis_a.setToolTip("excluded cylinder around the axis (u singular there)")
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 100)
        self.opacity.setValue(int(self.settings.opacity * 100))
        f.addRow("shape", self.shape)
        f.addRow("origin cut-out r", self.r_min)
        f.addRow("axis cut-out ρ", self.axis_a)
        f.addRow("surface opacity", self.opacity)
        v.addWidget(g)

        cam = QHBoxLayout()
        reset = QPushButton("Oblique view")
        reset.clicked.connect(self._reset_camera)
        top = QPushButton("View from N pole")
        top.clicked.connect(self._view_from_pole)
        cam.addWidget(reset)
        cam.addWidget(top)
        v.addLayout(cam)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        v.addWidget(self.info)
        v.addStretch(1)

        for w in (self.seed_mode, self.color_by, self.iso_field, self.shape):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.n_seeds, self.r_min, self.axis_a):
            w.valueChanged.connect(self._changed)
        for w in (self.iso_level, self.opacity):
            w.valueChanged.connect(self._changed)
        for c in self.chk.values():
            c.toggled.connect(self._changed)
        sc = QScrollArea()
        sc.setWidget(box)
        sc.setWidgetResizable(True)
        sc.setFixedWidth(258)
        return sc

    def _read_controls(self):
        s = self.settings
        s.seed_mode = self.seed_mode.currentData()
        s.n_seeds = self.n_seeds.value()
        s.color_by = self.color_by.currentData()
        for k, c in self.chk.items():
            setattr(s, k, c.isChecked())
        s.iso_field = self.iso_field.currentData()
        s.iso_quantile = self.iso_level.value() / 100
        s.domain_shape = self.shape.currentData()
        s.r_min = self.r_min.value()
        s.axis_a = self.axis_a.value()
        s.opacity = self.opacity.value() / 100
        from ...viz.streamlines import Domain
        self.domainChanged.emit(Domain(R=1.0, r_min=s.r_min, axis_a=s.axis_a, shape=s.domain_shape))

    def _changed(self, *_):
        self._read_controls()
        self._sync_widgets()
        self._debounce.start()

    # ---------------------------------------------------------------- seed widgets
    def _handle_toggled(self, on: bool):
        self._widgets_mode = None
        self._sync_widgets()
        self._debounce.start()

    def _sync_widgets(self):
        want = self.settings.seed_mode if self.show_handle.isChecked() else "off"
        if self.plotter is None or self._widgets_mode == want:
            return
        if want != "off" and getattr(self.plotter, "iren", None) is None:
            # interactor not ready (being (re)initialised): try again on the next update
            self._widgets_mode = None
            QTimer.singleShot(200, self._sync_widgets)
            return
        self.plotter.clear_sphere_widgets()
        self.plotter.clear_line_widgets()
        self._widgets_mode = want
        if want == "off":
            return
        s = self.settings
        if s.seed_mode in ("ring", "rings"):
            # a small handle: only its distance from the axis (and height for one ring) matters
            w = self.plotter.add_sphere_widget(self._sphere_moved, center=s.sphere_center, radius=0.045,
                                               color="#d81b60", style="surface", pass_widget=True,
                                               selected_color="#ff8800")
        elif s.seed_mode in ("sphere", "disc"):
            w = self.plotter.add_sphere_widget(self._sphere_moved, center=s.sphere_center, radius=s.sphere_radius,
                                               color="#d81b60", style="wireframe", pass_widget=True,
                                               theta_resolution=12, phi_resolution=8, selected_color="#ff8800")
            try:
                w.GetSphereProperty().SetLineWidth(2.5)
            except Exception:
                pass
        else:
            self.plotter.add_line_widget(self._line_moved, use_vertices=True, color="#e0a800")
            w = (self.plotter.widgets.line_widgets if hasattr(self.plotter, "widgets") else self.plotter.line_widgets)[-1]
            w.SetPoint1(*s.line_p0)
            w.SetPoint2(*s.line_p1)
        self._widgets_mode = s.seed_mode

    def _sphere_moved(self, center, widget=None):
        self.settings.sphere_center = tuple(float(c) for c in center)
        if widget is not None and self.settings.seed_mode not in ("ring", "rings"):
            self.settings.sphere_radius = float(widget.GetRadius())
        self._debounce.start()

    def _line_moved(self, a, b):
        self.settings.line_p0 = tuple(float(c) for c in a)
        self.settings.line_p1 = tuple(float(c) for c in b)
        self._debounce.start()

    # ---------------------------------------------------------------- solution
    def set_solution(self, sol, summary):
        self.sol, self.summary = sol, summary
        self._sync_widgets()
        self._debounce.start()

    def _submit(self):
        if self.sol is None:
            return
        self._token += 1
        from dataclasses import replace
        self.worker.submit((self.sol, self.summary, replace(self.settings), self._token))
        self.info.setText("computing…")

    def _view_from_pole(self):
        if self.plotter is not None:
            self.plotter.camera_position = [(0, 0, 6), (0, 0, 0), (0, 1, 0)]
            self.plotter.reset_camera(bounds=(-1.1, 1.1, -1.1, 1.1, -1.3, 1.3))
            self.plotter.render()

    def _reset_camera(self):
        if self.plotter is not None:
            self.plotter.camera_position = [(2.7, -2.9, 2.4), (0, 0, 0), (0, 0, 1)]
            self.plotter.reset_camera(bounds=(-1.1, 1.1, -1.1, 1.1, -1.3, 1.3))
            self.plotter.camera.zoom(1.85)
            self.plotter.render()

    # ---------------------------------------------------------------- rendering
    def _render(self, data, token):
        import pyvista as pv
        if token != self._token or self.plotter is None:
            return
        pl = self.plotter
        if isinstance(data, Exception):
            self.info.setText(f"3D error: {data}")
            return
        try:
            data = build_scene(*data)
        except Exception as exc:
            self.info.setText(f"3D error: {exc}")
            return
        self._last = data
        s = self.settings
        for n in list(pl.actors):
            if n.startswith("sx_"):
                pl.remove_actor(n, render=False)
        if pl.scalar_bars:
            pl.remove_scalar_bar()
        P = style.c()
        pl.add_mesh(data.outline, color=P["outline"], line_width=1, name="sx_outline", pickable=False)
        for (mesh, col, lab), nm in zip(data.axis, ("sx_axisN", "sx_axisS")):
            pl.add_mesh(mesh, color=col, name=nm, smooth_shading=True)
        for (mesh, col, lab), z, nm in zip(data.axis, (1.1, -1.1), ("sx_labN", "sx_labS")):
            pl.add_point_labels(np.array([[0, 0, z]]), [lab], name=nm, text_color="white", font_size=13,
                                shape_color=col, shape_opacity=0.95, show_points=False, always_visible=True)
        if s.clip:
            disc = pv.Disc(center=(0, 0, 0), normal=(0, 1, 0), inner=0.0, outer=data.domain.R, c_res=72)
            pl.add_mesh(disc, color=P["cut_plane"], opacity=0.07, name="sx_cutplane")
        cs = data.color_spec
        hi = max(abs(cs["clim"][0]), abs(cs["clim"][1]))
        fmt = "%.0f" if hi >= 100 else "%.3g"
        coloured = (s.show_streamlines and not ((s.show_selfsim and data.selfsim)
                                               or (s.show_vortex and data.vortex is not None))) \
            or (s.show_surface and data.surface is not None)
        self.color_by.setEnabled(bool(coloured))
        self.color_by.setToolTip("" if coloured else "nothing on screen is colour-mapped right now "
                                 "(streamlines are dimmed while overlays are shown)")
        if coloured:
            pl.add_text(VTK_TITLES[s.color_by], position=(0.70, 0.86), viewport=True, font_size=8,
                        color=P["view_text"], name="sx_bartitle")
        bar = dict(title="", color=P["view_text"], vertical=True, position_x=0.88,
                   position_y=0.12, height=0.72, width=0.05, title_font_size=1, label_font_size=13,
                   n_labels=5, fmt=fmt)
        first = True

        def add_coloured(mesh, name, **kw):
            nonlocal first
            if mesh is None or mesh.n_points == 0:
                return
            pl.add_mesh(mesh, scalars="value", cmap=cs["cmap"], clim=cs["clim"], log_scale=cs["log_scale"],
                        name=name, show_scalar_bar=first, scalar_bar_args=bar, nan_opacity=0.0, **kw)
            first = False

        overlays = bool((s.show_selfsim and data.selfsim) or (s.show_vortex and data.vortex is not None))
        if data.streamlines is not None:
            if overlays:
                pl.add_mesh(data.streamlines, color=P["outline"], line_width=1, opacity=0.5, name="sx_streamlines")
            else:
                add_coloured(data.streamlines.tube(radius=0.007, n_sides=10), "sx_streamlines",
                             smooth_shading=True)
                if data.arrows is not None:
                    add_coloured(data.arrows, "sx_arrows", smooth_shading=True)
        if data.origin_ball is not None:
            pl.add_mesh(data.origin_ball, color="#8a8c92", ambient=0.3, name="sx_origin")
            pl.add_point_labels(np.array([[-0.75, 0.0, -0.3]]), ["origin (cut out)"],
                                name="sx_origin_lab", font_size=11, text_color=P["view_text"], shape_opacity=0.0,
                                show_points=False, always_visible=True)
        for i, (arc, sense) in enumerate(data.swirl_arcs):
            pl.add_mesh(arc, color="#e65100", name=f"sx_arc{i}", smooth_shading=True)
        if s.seed_mode in ("ring", "rings") and self.show_handle.isChecked():
            c = np.asarray(s.sphere_center, float)
            guide = pv.Line((0.0, 0.0, c[2]), tuple(c))
            pl.add_mesh(guide, color="#d81b60", line_width=2, name="sx_guide")

        add_coloured(data.surface, "sx_surface", opacity=s.opacity, smooth_shading=True)
        if data.iso is not None and data.iso.n_points:   # one neutral colour: does not compete with the lines
            pl.add_mesh(data.iso, color="#9fb3c8", opacity=s.opacity, smooth_shading=True, name="sx_iso")
        if data.vortex is not None and data.vortex.n_points:
            pl.add_mesh(data.vortex.tube(radius=0.006, n_sides=8), color="#2e9d3a", name="sx_vortex",
                        smooth_shading=True)
        if data.glyphs is not None:
            pl.add_mesh(data.glyphs, color=P["view_text"], name="sx_glyphs", smooth_shading=True)
        lam = (1.0, 0.6, 0.8, 1.25)
        for i, L in enumerate(data.selfsim[:4]):
            tube = pv.lines_from_points(L).tube(radius=(0.016, 0.006, 0.008, 0.011)[i], n_sides=10)
            pl.add_mesh(tube, color=P["view_text"], name=f"sx_selfsim{i}",
                        smooth_shading=True)
        if data.selfsim:
            fr = (0.85, 0.15, 0.4, 0.65)
            ends = np.array([L[int(fr[i] * (len(L) - 1))] for i, L in enumerate(data.selfsim[:4])])
            pl.add_point_labels(ends, ["X(s)"] + [f"{l}·X" for l in lam[1:len(ends)]], name="sx_selfsim_lab",
                                font_size=12, text_color=P["view_text"], shape_opacity=0.6,
                                shape_color=P["view_bg"],
                                show_points=False, always_visible=True)
        sw = "<span style='color:{c};font-size:15px'>■</span>&nbsp;"
        import matplotlib
        cm = matplotlib.colormaps[cs["cmap"]] if isinstance(cs["cmap"], str) else cs["cmap"]
        ramp = "".join(f"<span style='color:{matplotlib.colors.to_hex(cm(t))};font-size:15px'>■</span>"
                       for t in (0.05, 0.35, 0.65, 0.95)) + "&nbsp;"
        items = []
        if s.show_streamlines:
            if overlays:
                items.append(sw.format(c=style.c()["outline"]) + f"streamlines of u ({data.n_lines}), dimmed "
                             "while vortex lines / rescalings are shown")
            else:
                items.append(ramp + f"streamlines of u ({data.n_lines}), coloured by "
                             f"{COLOR_BY[s.color_by]} (colour bar); small cones point downstream")
        if s.show_surface and data.surface is not None:
            items.append(ramp + f"stream surface (translucent), coloured by {COLOR_BY[s.color_by]}")
        if s.show_iso and data.iso is not None:
            items.append(sw.format(c="#9fb3c8") + f"isosurface {ISO_FIELDS[s.iso_field]} = {data.iso_level:.3g}")
        if s.show_vortex and data.vortex is not None:
            items.append(sw.format(c="#2e9d3a") + "vortex lines (curves along ω)")
        if s.show_glyphs and data.glyphs is not None:
            items.append(sw.format(c=style.c()["view_text"]) + "arrows: direction of u; length grows with log |u|")
        if s.show_selfsim and data.selfsim:
            items.append(sw.format(c=style.c()["view_text"]) + "thick: one streamline X(s); thin: its rescalings λX "
                         "(labelled), also streamlines since u is (−1)-homogeneous")
        if self.show_handle.isChecked():
            items.append(sw.format(c="#d81b60") + ("magenta handle: drag it; its distance from the axis is the "
                                                     "ring radius" if s.seed_mode in ("rings", "ring") else
                                                     "magenta ball: drag to move the seeds, handles resize"))
        items.append(sw.format(c=STRATUM_COLORS["G"]) + "axis coloured by pole type (blue G, red R, amber L1, "
                     "purple L2); big cone = direction of the flow along the axis · " + sw.format(c="#9a9ca3") + f"cut-outs: ball r &lt; {s.r_min:g} around 0, "
                     f"cylinder ρ &lt; {s.axis_a:g} around the axis"
                     + (" · half y > 0 cut away" if s.clip else ""))
        if data.swirl_arcs:
            items.append(sw.format(c="#e65100") + "orange arcs: sense of the swirl u_φ near each pole")
        items.append(sw.format(c=style.c()["origin"]) + "grey ball: origin cut out (u ~ 1/|X| is singular there)")
        if self.chk_slice.isChecked():
            items.append("<span style='color:#b2182b;font-size:15px'>■</span><span style='color:#2166ac;"
                         "font-size:15px'>■</span>&nbsp;slice plane: red = flow through it toward +n, blue = "
                         "against n; dark lines = in-plane streamlines")
        items.append(sw.format(c=style.c()["outline"]) + "thin ellipses: equator and cut-plane meridian of |X| = 1")
        half = (len(items) + 1) // 2
        cols = ["<br>".join(items[:half]), "<br>".join(items[half:])]
        self.legend.setText(f"<table width='100%'><tr><td valign='top' width='50%'>{cols[0]}</td>"
                            f"<td valign='top'>{cols[1]}</td></tr></table>")
        shown = []
        if s.show_streamlines:
            shown.append(f"Streamlines of u from {data.n_lines} seeds; cones show the flow direction")
        if s.show_surface and data.surface is not None:
            shown.append("Stream surface swept from a seed line")
        if s.show_vortex and data.vortex is not None:
            shown.append("Magenta: vortex lines (curves along the vorticity)")
        if s.show_iso and data.iso is not None:
            fld = {"speed": "speed", "swirl": "swirl f/sin(theta)", "helicity": "helicity"}[s.iso_field]
            shown.append(f"Translucent surface: {fld} = {data.iso_level:.3g}")
        if s.show_glyphs and data.glyphs is not None:
            shown.append("Grey arrows: direction of u (darker = faster)")
        if s.show_selfsim and data.selfsim:
            shown.append("Black: one streamline X(s).  Orange: its rescalings 0.6X, 0.8X, 1.25X,\n"
                         "  which are streamlines too because u is (-1)-homogeneous")
        shown.append("Yellow ball: drag to move the seeds (handles resize it)")
        shown.append("Axis colour = pole type: blue G, red R, amber L1, purple L2")

        if not self._camera_set:
            self._reset_camera()
            self._camera_set = True
        self._draw_slice(render=False)
        pl.render()
        tm = data.timings.get("total", 0) * 1e3
        self.iso_info.setText(f"level = {data.iso_level:.3g}" if s.show_iso else "")
        why = ", ".join(f"{v}× {k}" for k, v in sorted(data.stop_reasons.items(), key=lambda t: -t[1]))
        self.info.setText((f"streamline ends (both directions): {why}<br>" if why else "")
                          + f"computed in {tm:.0f} ms")

    def _style_legend(self):
        p = style.c()
        self.legend.setStyleSheet(f"background:{p['panel']};border-top:1px solid {p['border']};padding:6px 8px;"
                                  f"font-size:12px;color:{p['text']};")

    # ---------------------------------------------------------------- slice plane
    def attach_slice(self, controller):
        self.slice_ctrl = controller
        controller.planeChanged.connect(self._plane_changed_elsewhere)

    def set_slice(self, data, lines):
        self._slice = (data, lines)
        if self.chk_slice.isChecked():
            self._draw_slice(render=True)

    def _slice_toggled(self, on):
        if not on and self.plotter is not None:
            for n in ("sl_plane", "sl_lines", "sl_normal"):
                self.plotter.remove_actor(n, render=False)
            self.plotter.render()
        else:
            self._draw_slice(render=True)
        self._debounce.start()            # legend

    def _draw_slice(self, render=True):
        pl = self.plotter
        if pl is None:
            return
        for n in ("sl_plane", "sl_lines", "sl_normal"):
            pl.remove_actor(n, render=False)
        if not self.chk_slice.isChecked() or self._slice is None:
            if render:
                pl.render()
            return
        import pyvista as pv
        data, lines = self._slice
        X = data.X
        grid = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2])
        un = data.un
        v = un[np.isfinite(un)]
        m = float(np.percentile(np.abs(v), 95)) if v.size else 1.0
        grid["un"] = un.ravel(order="F")
        pl.add_mesh(grid, scalars="un", cmap="RdBu_r", clim=(-m, m), nan_opacity=0.0, opacity=0.85,
                    show_scalar_bar=False, name="sl_plane", smooth_shading=False)
        lift = 0.006 * data.spec.normal                # draw the lines just above the plane (no z-fighting)
        polys = [data.spec.to_world(L[:, 0], L[:, 1]) + lift for L in lines if len(L) > 1]
        if polys:
            from ...viz.scene3d import _polylines
            pl.add_mesh(_polylines(polys), color="#111111", line_width=1.5, name="sl_lines")
        c, n = data.spec.center, data.spec.normal
        pl.add_mesh(pv.Arrow(start=c, direction=n, scale=0.35, tip_radius=0.08, shaft_radius=0.025),
                    color="#f2c14e", name="sl_normal")
        if render:
            pl.render()

    def _plane_widget_toggled(self, on):
        pl = self.plotter
        if pl is None:
            return
        pl.clear_plane_widgets()
        self._plane_widget = None
        if on and self.slice_ctrl is not None and getattr(pl, "iren", None) is not None:
            spec = self.slice_ctrl.spec
            self._plane_widget = pl.add_plane_widget(self._plane_dragged, normal=tuple(spec.normal),
                                                     origin=tuple(spec.center), bounds=(-1.0, 1.0) * 3,
                                                     implicit=False, normal_rotation=True, color="#f2c14e",
                                                     outline_translation=False, origin_translation=True,
                                                     interaction_event="end")
            if not self.chk_slice.isChecked():
                self.chk_slice.setChecked(True)

    def _plane_dragged(self, normal, origin):
        if self.slice_ctrl is None:
            return
        from ...viz.slices import PlaneSpec
        old = self.slice_ctrl.spec
        spec = PlaneSpec.from_normal_origin(normal, origin, half_size=old.half_size, n=old.n)
        self.slice_ctrl.set_spec(spec, source=self)

    def _plane_changed_elsewhere(self, payload):
        spec, source = payload
        w = self._plane_widget
        if source is self or w is None:
            return
        try:
            w.SetNormal(*spec.normal)
            w.SetOrigin(*spec.center)
            self.plotter.render()
        except Exception:
            pass

    def refresh_style(self):
        if self.plotter is None:
            return
        p = style.c()
        self.plotter.set_background(p["view_bg"], top=p["view_bg_top"])
        self._style_legend()
        self.iso_info.setStyleSheet(f"color:{p['faint']};font-size:10px;")
        self.info.setStyleSheet(f"color:{p['muted']};font-size:11px;")
        self._debounce.start()

    def shutdown(self):
        self.thread.quit()
        self.thread.wait(2000)
        if self.plotter is not None:
            self.plotter.close()
