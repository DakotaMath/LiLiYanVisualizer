"""Movie export: the options dialog and the frame loop.

The numbers for each frame (solve, pole status, streamlines, slice sampling) are computed
in a worker thread one frame ahead; the GUI thread only draws (VTK must stay on it),
grabs the panels, composites and encodes.  A progress dialog can cancel at any time.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (QDialog, QFormLayout, QComboBox, QSpinBox, QDialogButtonBox, QLineEdit,
                               QPushButton, QHBoxLayout, QLabel, QFileDialog, QProgressDialog, QWidget,
                               QMessageBox)

from ..core.paths import frame_status, _why
from ..core.solver import solve
from ..core.poles import summarize
from ..io.movie import (LAYOUTS, RESOLUTIONS, NON_GLOBAL_POLICIES, MovieWriter, Caption, compose, cells,
                        panels_needed)
from . import panel_state as ps


class ExportDialog(QDialog):
    def __init__(self, parent, n_frames: int, fps: int, slice_shown: bool, last_path: str = "",
                 policy: str = "hold"):
        super().__init__(parent)
        self.setWindowTitle("Export movie")
        f = QFormLayout(self)
        self.layout_cb = QComboBox()
        for k, (label, _) in LAYOUTS.items():
            self.layout_cb.addItem(label, k)
        self.layout_cb.setCurrentIndex(self.layout_cb.findData("3d+slice" if slice_shown else "3d+plots"))
        self.res_cb = QComboBox()
        for k in RESOLUTIONS:
            self.res_cb.addItem(k, RESOLUTIONS[k])
        self.res_cb.setCurrentIndex(1)
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(fps)
        self.fps.setSuffix(" fps")
        self.frames = QSpinBox()
        self.frames.setRange(2, 5000)
        self.frames.setValue(n_frames)
        self.policy = QComboBox()
        for k, t in NON_GLOBAL_POLICIES.items():
            self.policy.addItem(t, k)
        self.policy.setCurrentIndex(max(0, self.policy.findData(policy)))
        self.out = QLineEdit(last_path or str(Path.home() / "swirl_path.mp4"))
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.out, 1)
        row.addWidget(browse)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self.length = QLabel("")
        f.addRow("show", self.layout_cb)
        f.addRow("resolution", self.res_cb)
        f.addRow("frames", self.frames)
        f.addRow("frame rate", self.fps)
        f.addRow("", self.length)
        f.addRow("frames that are not global", self.policy)
        f.addRow("file", w)
        note = QLabel("The camera, plane and seeds follow the path where its keyframes store them, "
                      "and otherwise stay as they are now. Every frame carries a caption with t, the six "
                      "data and the pole status.")
        note.setWordWrap(True)
        note.setStyleSheet("color:gray;font-size:11px;")
        f.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Render")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        for s in (self.fps, self.frames):
            s.valueChanged.connect(self._len)
        self._len()

    def _len(self, *_):
        self.length.setText(f"{self.frames.value() / self.fps.value():.1f} s of video")

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Movie file", self.out.text(), "MP4 video (*.mp4)")
        if p:
            self.out.setText(p if p.lower().endswith(".mp4") else p + ".mp4")

    def options(self) -> dict:
        W, H = self.res_cb.currentData()
        return {"layout": self.layout_cb.currentData(), "size": (W, H), "fps": self.fps.value(),
                "frames": self.frames.value(), "policy": self.policy.currentData(), "path": self.out.text()}


class _FrameWorker(QObject):
    """Computes everything numeric for one frame; the GUI asks for the next when it has drawn."""
    ready = Signal(int, object)
    _req = Signal(int, object)

    def __init__(self):
        super().__init__()
        self.cancel = threading.Event()
        self._req.connect(self._run)

    @Slot(int, object)
    def _run(self, i, job):
        if self.cancel.is_set():
            return
        p, settings, spec, domain, slice_opts, need = job
        out = {"p": p}
        try:
            out["status"] = frame_status(p, i)
            sol = solve(p)
            out["sol"], out["sm"] = sol, summarize(sol)
            if "3d" in need:
                from ..viz.scene_compute import compute_numeric
                out["num"] = compute_numeric(sol, settings)
            if "slice" in need:
                from ..core.fields import FieldInterpolator
                from ..viz.slices import sample_plane, lic, projected_streamlines
                fi = FieldInterpolator(sol, dxi=0.002, fast=True)
                data = sample_plane(fi, spec, domain)
                if slice_opts["chk_lic"]:
                    data.lic = lic(data.us, data.ut)
                dens = slice_opts["density"] * (0.5 if slice_opts["chk_lic"] else 1.0)
                lines = projected_streamlines(data, density=dens) if slice_opts["chk_lines"] else []
                out["slice"] = (data, lines)
        except Exception as exc:          # report in the movie, never crash
            out["error"] = f"{type(exc).__name__}: {exc}"
        if not self.cancel.is_set():
            self.ready.emit(i, out)


class MovieExporter(QObject):
    """Runs one export.  ``win`` is the main window (panels, path dock)."""
    finished = Signal(str)            # message for the status bar

    def __init__(self, win, opts: dict):
        super().__init__(win)
        self.win, self.o = win, opts
        self.path_panel = win.path_panel
        self.n = opts["frames"]
        W, H = opts["size"]
        self.W, self.H = W - W % 2, H - H % 2
        need = set(panels_needed(opts["layout"]))
        if "3d" in need and win.view3d.chk_slice.isChecked():
            need.add("slice")            # the plane is drawn in the 3D view as well
        self.need = need
        self.cells = dict(cells(opts["layout"], self.W, self.H)[0])
        self.writer = None
        self.last_good = None            # images of the last global frame (for "hold")
        self.held = self.skipped = self.written = 0
        self.t_start = time.perf_counter()
        # what to restore afterwards
        self.saved_camera = ps.camera_state(win.view3d)
        self.saved_plane = ps.plane_state(win.slice_ctrl)
        self.saved_seeds = ps.seeds_state(win.view3d)
        self.thread = QThread()
        self.worker = _FrameWorker()
        self.worker.moveToThread(self.thread)
        self.worker.ready.connect(self._draw)
        self.thread.start()
        self.dlg = QProgressDialog("Rendering frames…", "Cancel", 0, self.n, win)
        self.dlg.setWindowTitle("Export movie")
        self.dlg.setWindowModality(Qt.WindowModal)
        self.dlg.setMinimumDuration(0)
        self.dlg.canceled.connect(self.cancel)
        self._cancelled = False

    # ------------------------------------------------------------------ run
    def start(self):
        try:
            self.writer = MovieWriter(self.o["path"], self.o["fps"], (self.W, self.H))
        except Exception as exc:
            QMessageBox.warning(self.win, "Export movie", f"Cannot write {self.o['path']}:\n{exc}")
            self._cleanup()
            return
        self.win.exporting = True
        # the whole path's status, for the timeline strip in every caption
        from ..core.paths import scan_path, failure_kind
        path = self.path_panel.path
        self.status, evs = scan_path(path, self.n, refine=6)
        pal = {"ok": (63, 168, 119), "blowup": (138, 118, 80), "unsafe": (185, 189, 199)}
        self.strip = [pal[failure_kind(f)] if not f.is_global or (f.north, f.south) == ("G", "G")
                      else (47, 143, 157) for f in self.status]
        t0, t1 = path.duration
        self.keyfracs = tuple((k.t - t0) / ((t1 - t0) or 1.0) for k in path.keyframes)
        self.eventfracs = tuple((e.t - t0) / ((t1 - t0) or 1.0) for e in evs)
        kinds = {failure_kind(f) for f in self.status}
        self.key = tuple((pal[k], lab) for k, lab in (("ok", "global"), ("unsafe", "pole fails test"),
                                                      ("blowup", "blows up")) if k in kinds)
        self._tab = self.win.tabs.currentIndex()
        if "3d" in self.need:                     # the 3D view must be on screen to be grabbed
            self.win.tabs.setCurrentIndex(0)
        self.path_panel.play_btn.setChecked(False)
        self._request(0)

    def _t(self, i):
        t0, t1 = self.path_panel.path.duration
        return t0 + (t1 - t0) * i / max(1, self.n - 1)

    def _request(self, i):
        path = self.path_panel.path
        t = self._t(i)
        p = path.at(t)[0]
        settings = replace(self.win.view3d.settings)
        seeds = path.extra_at(t, "seeds") if path.keyed("seeds") else None
        if seeds:
            for k in ps.SEED_KEYS:
                if k in seeds:
                    setattr(settings, k, tuple(seeds[k]) if isinstance(seeds[k], (list, tuple)) else seeds[k])
        plane = path.extra_at(t, "plane") if path.keyed("plane") else None
        from ..viz.slices import PlaneSpec
        spec = PlaneSpec.from_dict(plane) if plane else self.win.slice_ctrl.spec
        self._frame_extras = {"camera": path.extra_at(t, "camera") if path.keyed("camera") else None,
                              "plane": plane, "seeds": seeds}
        job = (p, settings, spec, self.win.slice.domain, ps.slice_options(self.win.slice), self.need)
        self.worker._req.emit(i, job)

    @Slot(int, object)
    def _draw(self, i, out):
        if self._cancelled:
            return
        try:
            self._draw_frame(i, out)
        except Exception as exc:                  # stop cleanly, keep nothing half-made
            import traceback
            traceback.print_exc()
            self._fail(f"Movie export stopped at frame {i + 1}: {type(exc).__name__}: {exc}")

    def _fail(self, msg):
        self._cancelled = True
        path = self.o["path"]
        self._cleanup()
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        self.finished.emit(msg)

    def _draw_frame(self, i, out):
        win = self.win
        st = out.get("status")
        ok = st is not None and st.is_global and "error" not in out
        policy = self.o["policy"]
        if not ok and policy == "skip":
            self.skipped += 1
            return self._next(i)
        caption = self._caption(i, out, ok)
        self._last_status = caption.status
        why = out.get("error") or (_why(st) if st is not None else "")
        if not ok and policy == "hold" and self.last_good is not None:
            images, tg, pg = self.last_good
            caption.banner = f"HELD · from path t = {tg:.4g}".replace("-", "−")
            caption.status += f" · showing t = {tg:.3g}".replace("-", "−")
            pic = "(" + ", ".join(f"{v:.4g}" for v in pg.as_tuple()) + ")"
            caption.data += f"   ·   picture: path t = {tg:.4g}, {pic}".replace("-", "−")
            t0, t1 = self.path_panel.path.duration
            caption.held_at = (tg - t0) / ((t1 - t0) or 1.0)
            self.held += 1
        else:
            images = self._render(out)
            if ok:
                self.last_good = (images, self._t(i), out["p"])
            else:
                caption.banner = "NOT GLOBAL"
        frame = compose(images, self.o["layout"], self.W, self.H, caption)
        self.writer.append(frame)
        self.written += 1
        self._next(i)

    def _render(self, out) -> dict:
        win, ex = self.win, self._frame_extras
        imgs = {}
        if "sol" not in out:
            return imgs
        if ex.get("camera"):
            ps.apply_camera(win.view3d, ex["camera"], render=False)
        if "slice" in out:
            data, lines = out["slice"]
            data.set_display(win.slice.chk_rho.isChecked(), win.slice.chk_log.isChecked())
            win.slice.last = (data, lines)
            if "slice" in self.cells:
                win.slice.redraw()
                x, y, w, h = self.cells["slice"]
                imgs["slice"] = ps.fig_to_array(win.slice.fig, w, h)
            win.view3d._slice = (data, lines)
        if "num" in out:
            v = win.view3d
            v._token += 1
            v._render((out["num"], out["sm"]), v._token)
            if ex.get("camera"):
                ps.apply_camera(v, ex["camera"], render=True)
            x, y, w, h = self.cells["3d"]
            imgs["3d"] = ps.grab_3d(v, w, h)
        if "plots" in self.cells:
            win.plots.current = None
            win.plots.set_current(out["sol"], out["sm"])
            x, y, w, h = self.cells["plots"]
            imgs["plots"] = ps.fig_to_array(win.plots.fig, w, h)
        return imgs

    def _caption(self, i, out, ok) -> Caption:
        p, st = out["p"], out.get("status")
        names = {"c1": "c₁", "c2": "c₂", "c3": "c₃", "gamma": "γ", "A": "A", "kappa": "κ"}
        vary = self.path_panel.path.varying()
        from ..core.params import PARAM_NAMES
        moving = ",  ".join(f"{names[n]} = {getattr(p, n):.4g}" for n in PARAM_NAMES if n in vary)
        fixed = ", ".join(f"{names[n]} = {getattr(p, n):.4g}" for n in PARAM_NAMES if n not in vary)
        data = (moving + (f"   (fixed: {fixed})" if fixed else "")) if moving else \
            "(c₁, c₂, c₃, γ, A, κ) = (" + ", ".join(f"{v:.5g}" for v in p.as_tuple()) + ")"
        data = data.replace("-", "−") + (f"   x₀ = {p.xb:g}" if p.xb else "")
        title = f"path t = {self._t(i):.4g}    frame {i + 1} / {self.n}".replace("-", "−")
        if st is None:
            status = "error"
        elif st.is_global:
            status = f"global · N {st.north} μ₊={st.mu_n:.3f} · S {st.south} μ₋={st.mu_s:.3f}"
        else:
            from ..core.paths import failure_kind
            status = "NOT GLOBAL · " + ("blows up" if failure_kind(st) == "blowup"
                                       else "a pole fails the terminal test")
        return Caption(title, data, status, ok, strip=self.strip, pos=i, keys=self.keyfracs,
                       events=self.eventfracs, key=self.key)

    def _next(self, i):
        self.dlg.setValue(i + 1)
        el = time.perf_counter() - self.t_start
        left = el / (i + 1) * (self.n - i - 1)
        self.dlg.setLabelText(f"Rendering frame {i + 1} of {self.n}  (about {left:.0f} s left)")
        if i + 1 >= self.n:
            return self._done()
        self._request(i + 1)

    def cancel(self):
        if self._cancelled:
            return
        self._cancelled = True
        self.worker.cancel.set()
        path = self.o["path"]
        self._cleanup()
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        self.finished.emit("Movie export cancelled; no file written.")

    def _done(self):
        msg = (f"Wrote {self.written} frames ({self.written / self.o['fps']:.1f} s) to {self.o['path']}")
        notes = []
        if self.held:
            notes.append(f"{self.held} not-global frames held at the last global frame")
        if self.skipped:
            notes.append(f"{self.skipped} not-global frames left out")
        if notes:
            msg += " — " + "; ".join(notes)
        self._cleanup()
        self.finished.emit(msg)

    def _cleanup(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None
        self.worker.cancel.set()
        self.thread.quit()
        self.thread.wait(5000)
        try:
            self.dlg.canceled.disconnect(self.cancel)     # closing the dialog emits canceled
        except (RuntimeError, TypeError):
            pass
        self.dlg.reset()
        self.dlg.close()
        win = self.win
        win.exporting = False
        if getattr(self, "_tab", None) is not None:
            win.tabs.setCurrentIndex(self._tab)
        # put the live views back as they were
        ps.apply_seeds(win.view3d, self.saved_seeds)
        ps.apply_plane(win.slice, self.saved_plane)
        ps.apply_camera(win.view3d, self.saved_camera)
        win.request(win.params.params())
