"""3D view: PyVista interactor with draggable seed widgets, streamlines and field visualisations."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QComboBox, QSpinBox,
                               QDoubleSpinBox, QCheckBox, QLabel, QSlider, QScrollArea, QFormLayout,
                               QPushButton)

from ...viz.scene_compute import SceneSettings, compute_numeric, build_scene
from ...viz.scene3d import COLOR_BY
from ...viz.scene_draw import (ISO_FIELDS, draw_scene, draw_slice_plane, clear_slice_plane, oblique_camera,
                               pole_camera)
from ...viz import style

try:                                    # the 3D view needs an OpenGL context
    from pyvistaqt import QtInteractor
    HAVE_PV = True
except Exception:                       # pragma: no cover
    HAVE_PV = False



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
                                  viewport=(0.0, 0.08, 0.34, 0.44))
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)
            col.addWidget(self.plotter.interactor, 1)
            self.legend = QLabel("")
            self.legend.setTextFormat(Qt.RichText)
            self.legend.setWordWrap(True)
            self._style_legend()
            # the legend lives at the top of the controls (in their dock)
            h.addLayout(col, 1)
        else:  # pragma: no cover
            self.plotter = None
            h.addWidget(QLabel("PyVista / VTK not available"), 1)
        self.controls = self._controls()       # placed in a dock by the main window
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
        if getattr(self, "legend", None) is not None:
            lg = QGroupBox("What you see")
            lv = QVBoxLayout(lg)
            lv.setContentsMargins(2, 2, 2, 2)
            lv.addWidget(self.legend)
            v.addWidget(lg)
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
                ("clip", "cut away the near half (y < 0)", "clip lines and surfaces by the plane y = 0; the half facing the default camera goes")):
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
        self.chk_slice.setChecked(False)      # off by default: the user turns the plane on
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
        sc.setMinimumWidth(260)
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
            pole_camera(self.plotter)
            self.plotter.render()

    def _reset_camera(self):
        if self.plotter is not None:
            oblique_camera(self.plotter)
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
        drawn = draw_scene(pl, data, s, show_handle=self.show_handle.isChecked(),
                           slice_shown=self.chk_slice.isChecked())
        self.color_by.setEnabled(drawn.coloured)
        self.color_by.setToolTip("" if drawn.coloured else "nothing on screen is colour-mapped right now "
                                 "(streamlines are dimmed while overlays are shown)")
        items = drawn.legend
        half = (len(items) + 1) // 2
        cols = ["<br>".join(items[:half]), "<br>".join(items[half:])]
        self.legend.setText(f"<table width='100%'><tr><td valign='top' width='50%'>{cols[0]}</td>"
                            f"<td valign='top'>{cols[1]}</td></tr></table>")

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
            clear_slice_plane(self.plotter)
            self.plotter.render()
        else:
            self._draw_slice(render=True)
        self._debounce.start()            # legend

    def _draw_slice(self, render=True):
        pl = self.plotter
        if pl is None:
            return
        if not self.chk_slice.isChecked() or self._slice is None:
            clear_slice_plane(pl)
        else:
            draw_slice_plane(pl, *self._slice)
        if render:
            pl.render()

    def view_along_normal(self):
        """Camera on the +n side of the slice, looking at it, with the slice's 'up' up."""
        if self.plotter is None or self.slice_ctrl is None:
            return
        spec = self.slice_ctrl.spec
        e1, e2, n = spec.basis()
        c = spec.center
        self.plotter.camera_position = [tuple(c + 5 * n), tuple(c), tuple(e2)]
        self.plotter.reset_camera(bounds=(-1.1, 1.1, -1.1, 1.1, -1.2, 1.2))
        self.plotter.camera.zoom(1.35)
        self.plotter.render()

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
