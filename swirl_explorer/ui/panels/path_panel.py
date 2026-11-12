"""Path dock: keyframes in parameter space, a coloured timeline, preview playback and movie export.

Left: the keyframe table (typed, or added from the current solution / shift-click on the
Σ-map).  Right: transport, the timeline and the events.  The timeline is coloured per
frame from a background scan (green: global with generic poles at both ends, teal:
global with an R / L1 / L2 pole, red: not global) with a tick at every event, so a
movie never silently runs through solutions that do not exist.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF, QFont
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox,
                               QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QToolButton,
                               QAbstractItemView, QSizePolicy, QFrame)

from ...core.params import Params, PARAM_NAMES
from ...core.paths import ParamPath, Keyframe, scan_path, EXTRA_KEYS, failure_kind
from ...viz import style

NAMES = {"c1": "c₁", "c2": "c₂", "c3": "c₃", "gamma": "γ", "A": "A", "kappa": "κ"}
COL_OK = "#3fa877"          # the Σ-map's global green
COL_OTHER = "#2f8f9d"       # global, but a pole of type R / L1 / L2
COL_BAD = "#d64045"          # text: not global
COL_BLOW = "#8a7650"         # timeline: blows up before a pole (the Σ-map's olive, a little lighter)
COL_UNSAFE = "#b9bdc7"       # timeline: a pole fails the terminal test (the Σ-map's grey)
COL_EVENT = "#f2c14e"


def _m(x: float) -> str:
    return f"{x:.6g}".replace("-", "−")


# ----------------------------------------------------------------------------- scan worker
class _ScanWorker(QObject):
    done = Signal(object, object, int)
    _req = Signal(object, int, int)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._gen = 0
        self._req.connect(self._run)

    def submit(self, path_dict: dict, n: int, gen: int):
        with self._lock:
            self._gen = gen
        self._req.emit(path_dict, n, gen)

    @Slot(object, int, int)
    def _run(self, path_dict, n, gen):
        with self._lock:
            if gen != self._gen:
                return
        path = ParamPath.from_dict(path_dict)
        try:
            frames, events = scan_path(path, n, cancelled=lambda: gen != self._gen)
        except Exception as exc:          # never crash the UI
            frames, events = exc, []
        if gen == self._gen:
            self.done.emit(frames, events, gen)


# ----------------------------------------------------------------------------- timeline
class Timeline(QWidget):
    """Per-frame status strip with event ticks and a draggable cursor."""
    scrubbed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n = 0
        self.frames = []
        self.events = []
        self.t_range = (0.0, 1.0)
        self.keys = []
        self.index = 0
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, n, frames, events, t_range, keys=()):
        self.n, self.frames, self.events, self.t_range = n, frames, events, t_range
        self.keys = list(keys)
        self.update()

    def set_index(self, i):
        self.index = i
        self.update()

    def _x(self, f: float) -> float:
        return 8 + f * (self.width() - 16)

    def _frame_at(self, x: float) -> int:
        if self.n < 2:
            return 0
        f = (x - 8) / max(1.0, self.width() - 16)
        return int(round(min(1.0, max(0.0, f)) * (self.n - 1)))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        P = style.c()
        p.fillRect(self.rect(), QColor(P["panel"]))
        top, h = 18, 20
        if self.n < 2:
            p.setPen(QColor(P["faint"]))
            p.drawText(self.rect(), Qt.AlignCenter, "add two keyframes to make a path")
            return
        w = (self.width() - 16) / self.n
        if self.frames:
            for fs in self.frames:
                k = failure_kind(fs)
                col = {"blowup": COL_BLOW, "unsafe": COL_UNSAFE}.get(k) or \
                    (COL_OK if (fs.north, fs.south) == ("G", "G") else COL_OTHER)
                p.fillRect(QRectF(8 + fs.index * w, top, w + 0.6, h), QColor(col))
            if len(self.frames) < self.n:      # still scanning
                p.fillRect(QRectF(8 + len(self.frames) * w, top, (self.n - len(self.frames)) * w, h),
                           QColor(P["border"]))
        else:
            p.fillRect(QRectF(8, top, self.width() - 16, h), QColor(P["border"]))
        t0, t1 = self.t_range
        span = (t1 - t0) or 1.0
        f = QFont(self.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1.5))
        p.setFont(f)
        for e in self.events:
            x = self._x((e.t - t0) / span)
            p.setPen(QPen(QColor(COL_EVENT), 2))
            p.drawLine(QPointF(x, top - 2), QPointF(x, top + h + 2))
            p.setBrush(QColor(COL_EVENT))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x - 5, 4), QPointF(x + 5, 4), QPointF(x, 12)]))
        # cursor
        x = self._x(self.index / (self.n - 1))
        p.setPen(QPen(QColor(P["text"]), 2))
        p.drawLine(QPointF(x, top - 6), QPointF(x, top + h + 6))
        p.setBrush(QColor(P["text"]))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(x - 5, top - 9), QPointF(x + 5, top - 9), QPointF(x, top - 3)]))
        # light ticks at whole (and, for short paths, half) units of the path parameter
        step = 0.5 if span <= 4 else (1.0 if span <= 12 else 10 ** math.floor(math.log10(span / 4)))
        p.setPen(QPen(QColor(P["faint"]), 1))
        tk = math.ceil(t0 / step) * step
        while tk <= t1 + 1e-9:
            xk = self._x((tk - t0) / span)
            p.drawLine(QPointF(xk, top + h), QPointF(xk, top + h + 4))
            tk += step
        # keyframes: numbered badges under the bar (the same numbers as on the Σ-map)
        for j, tk in enumerate(self.keys):
            xk = self._x((tk - t0) / span)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(COL_EVENT))
            p.drawEllipse(QPointF(xk, top + h + 12), 7, 7)
            p.setPen(QColor("#111111"))
            p.drawText(QRectF(xk - 7, top + h + 5, 14, 14), Qt.AlignCenter, str(j + 1))
        p.setPen(QColor(P["faint"]))
        p.drawText(QRectF(8, top + h + 22, 200, 14), Qt.AlignLeft, f"path t = {_m(t0)}")
        p.drawText(QRectF(self.width() - 208, top + h + 22, 200, 14), Qt.AlignRight, f"path t = {_m(t1)}")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.n >= 2:
            self.scrubbed.emit(self._frame_at(ev.position().x()))

    def mouseMoveEvent(self, ev):
        if self.n < 2:
            return
        i = self._frame_at(ev.position().x())
        if ev.buttons() & Qt.LeftButton:
            self.scrubbed.emit(i)
        x = ev.position().x()
        t0, t1 = self.t_range
        span = (t1 - t0) or 1.0
        names = {"leaves_global": "leaves the global set", "enters_global": "back in the global set",
                 "stratum_change": "a pole changes type", "failure_change": "not global, now"}
        for e in self.events:
            if abs(self._x((e.t - t0) / span) - x) < 6:
                self.setToolTip(f"event at t = {_m(e.t)}: {names.get(e.kind, e.kind)} {e.detail.replace('->', '→')}")
                return
        for j, tk in enumerate(self.keys):
            if abs(self._x((tk - t0) / span) - x) < 8 and ev.position().y() > 38:
                self.setToolTip(f"keyframe {j + 1} at t = {_m(tk)}")
                return
        if self.frames and i < len(self.frames):
            fs = self.frames[i]
            state = "global" if fs.is_global else "NOT global"
            self.setToolTip(f"frame {i + 1}, t = {_m(fs.t)}: {state}; north {fs.north}, south {fs.south}")


# ----------------------------------------------------------------------------- panel
class PathPanel(QWidget):
    applyFrame = Signal(object, object)       # Params, extras {camera/plane/seeds: dict}
    pathChanged = Signal()
    exportRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = ParamPath()
        self.frames, self.events = [], []
        self.index = 0
        self.current_params = None            # callable() -> Params, set by the main window
        self.capture = None                   # callable(keys) -> extras dict, set by the main window
        self._gen = 0
        self._filling = False
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 2, 6, 4)

        # ---- left: keyframes
        left = QVBoxLayout()
        row = QHBoxLayout()
        self.add_btn = QPushButton("+ Add current")
        self.add_btn.setToolTip("append the current solution as a keyframe (shift-click on the Σ-map does the same "
                                "for the clicked point)")
        self.add_btn.clicked.connect(self.add_current)
        self.upd_btn = QPushButton("Overwrite with current")
        self.upd_btn.setToolTip("overwrite the selected keyframe's values with the current solution "
                                "(and the stored view, if ticked); selecting a row moves to that keyframe")
        self.upd_btn.clicked.connect(self.update_selected)
        self.del_btn = QPushButton("Remove")
        self.del_btn.clicked.connect(self.remove_selected)
        self.clr_btn = QPushButton("Clear")
        self.clr_btn.clicked.connect(self.clear)
        for b in (self.add_btn, self.upd_btn, self.del_btn, self.clr_btn):
            row.addWidget(b)
        row.addSpacing(8)
        row.addWidget(QLabel("between keyframes:"))
        self.interp = QComboBox()
        self.interp.addItem("linear", "linear")
        self.interp.addItem("spline", "spline")
        self.interp.setToolTip("how the six data move between keyframes (spline needs three or more)")
        self.interp.currentIndexChanged.connect(self._interp_changed)
        row.addWidget(self.interp)
        row.addStretch(1)
        left.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("keyframes also store:"))
        self.keep = {}
        for k, t, tip in (("camera", "camera", "the 3D camera moves between keyframes"),
                          ("plane", "slice plane", "the slice plane turns / moves between keyframes"),
                          ("seeds", "seeds", "the streamline seeds move between keyframes")):
            c = QCheckBox(t)
            c.setToolTip(tip + "; when off it stays where it is")
            self.keep[k] = c
            row.addWidget(c)
        row.addStretch(1)
        left.addLayout(row)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["path t"] + [NAMES[n] for n in PARAM_NAMES] + ["restores view"])
        tips = ["the path parameter at this keyframe (not physical time: the solutions are stationary)"] + \
            [f"{NAMES[n]} at this keyframe; bold '(varies)' = changes along the path" for n in PARAM_NAMES] + \
            ["the view this keyframe restores (camera, slice plane, seeds), chosen by the boxes above"]
        for c, tip in enumerate(tips):
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setToolTip("double-click a value to type it; selecting a row moves to that keyframe; "
                              "● and bold: the data that change along the path")
        self.table.itemChanged.connect(self._cell_edited)
        self.table.itemSelectionChanged.connect(self._row_selected)
        left.addWidget(self.table, 1)
        h.addLayout(left, 5)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{style.c()['border']};")
        h.addWidget(sep)

        # ---- right: transport, timeline, events
        right = QVBoxLayout()
        row = QHBoxLayout()
        self.first_btn = QToolButton()
        self.first_btn.setText("⏮")
        self.first_btn.setToolTip("first frame")
        self.first_btn.clicked.connect(lambda: self.goto(0))
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.setToolTip("preview: every view follows the path (frames are dropped if solving lags)")
        self.play_btn.toggled.connect(self._play_toggled)
        self.last_btn = QToolButton()
        self.last_btn.setText("⏭")
        self.last_btn.setToolTip("last frame")
        self.last_btn.clicked.connect(lambda: self.goto(self.n_frames() - 1))
        self.loop = QCheckBox("loop")
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(2, 5000)
        self.frames_spin.setValue(120)
        self.frames_spin.setSuffix(" frames")
        self.frames_spin.setKeyboardTracking(False)
        self.frames_spin.valueChanged.connect(self._schedule_scan)
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(24)
        self.fps.setSuffix(" fps")
        self.fps.valueChanged.connect(self._fps_changed)
        self.length_lab = QLabel("")
        for w in (self.first_btn, self.play_btn, self.last_btn, self.loop, self.frames_spin, self.fps,
                  self.length_lab):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(QLabel("not-global frames in the MP4:"))
        from ...io.movie import NON_GLOBAL_POLICIES
        self.policy = QComboBox()
        for k, t in (("hold", "hold the last global frame"), ("skip", "leave out"), ("mark", "show, labelled")):
            self.policy.addItem(t, k)
        self.policy.setToolTip("\n".join(NON_GLOBAL_POLICIES.values()))
        self.policy.currentIndexChanged.connect(lambda _: self._update_events())
        row.addWidget(self.policy)
        self.export_btn = QPushButton("Export MP4…")
        self.export_btn.setToolTip("render the path to a movie: 3D view, slice, profiles or a combination")
        self.export_btn.clicked.connect(self.exportRequested.emit)
        row.addWidget(self.export_btn)
        right.addLayout(row)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 119)
        self.slider.valueChanged.connect(self.goto)
        self.timeline = Timeline()
        self.timeline.scrubbed.connect(self.goto)
        right.addWidget(self.timeline)
        self.frame_lab = QLabel("")
        self.frame_lab.setTextFormat(Qt.RichText)
        right.addWidget(self.frame_lab)
        self.events_lab = QLabel("")
        self.events_lab.setTextFormat(Qt.RichText)
        self.events_lab.setWordWrap(True)
        self.events_lab.setOpenExternalLinks(False)
        self.events_lab.linkActivated.connect(lambda s: self.goto(int(s)))
        self.events_lab.setToolTip("click a time to jump to that frame")
        self.events_lab.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right.addWidget(self.events_lab, 1)
        h.addLayout(right, 6)

        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.setInterval(250)
        self._scan_timer.timeout.connect(self._scan)
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(30)
        self._apply_timer.timeout.connect(self._apply_now)
        self._play = QTimer(self)
        self._play.timeout.connect(self._tick)
        self.thread = QThread()
        self.worker = _ScanWorker()
        self.worker.moveToThread(self.thread)
        self.worker.done.connect(self._scanned)
        self.thread.start()
        self._fps_changed()
        self._refresh()

    # ---------------------------------------------------------------- keyframes
    def n_frames(self) -> int:
        return self.frames_spin.value()

    def keyed(self) -> list[str]:
        return [k for k, c in self.keep.items() if c.isChecked()]

    def add_keyframe(self, p: Params, extras: dict | None = None, t: float | None = None) -> None:
        if t is None:
            t = (self.path.keyframes[-1].t + 1.0) if len(self.path) else 0.0
        if extras is None:
            extras = self.capture(self.keyed()) if self.capture else {}
        if len(self.path) and abs(self.path.keyframes[0].params.xb - p.xb) > 1e-12:
            p = Params(*p.as_tuple(), xb=self.path.keyframes[0].params.xb)   # one base point per path
        self.path.add(t, p, **extras)
        self._changed(select=t)

    def add_current(self):
        if self.current_params is not None:
            self.add_keyframe(self.current_params())

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def update_selected(self):
        r = self._selected_row()
        if r < 0 or self.current_params is None:
            return
        k = self.path.keyframes[r]
        k.params = self.current_params()
        k.extras = self.capture(self.keyed()) if self.capture else {}
        self._changed(select=k.t)

    def remove_selected(self):
        r = self._selected_row()
        if r >= 0:
            del self.path.keyframes[r]
            self._changed()

    def clear(self):
        self.path.keyframes.clear()
        self._changed()

    def set_path(self, path: ParamPath | None, n_frames: int | None = None, fps: int | None = None) -> None:
        self.path = path or ParamPath()
        self.interp.blockSignals(True)
        self.interp.setCurrentIndex(max(0, self.interp.findData(self.path.interpolation)))
        self.interp.blockSignals(False)
        for k, c in self.keep.items():
            c.setChecked(self.path.keyed(k))
        if n_frames:
            self.frames_spin.blockSignals(True)
            self.frames_spin.setValue(int(n_frames))
            self.frames_spin.blockSignals(False)
        if fps:
            self.fps.setValue(int(fps))
        self._changed()

    def _interp_changed(self, *_):
        self.path.interpolation = self.interp.currentData()
        self._changed()

    def _changed(self, select: float | None = None):
        self._refresh(select)
        self.frames, self.events = [], []
        self._schedule_scan()
        self.pathChanged.emit()

    def _refresh(self, select: float | None = None):
        """Fill the table from the path."""
        self._filling = True
        t = self.table
        t.setRowCount(len(self.path))
        for r, k in enumerate(self.path.keyframes):
            vals = [k.t, *k.params.as_tuple()]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(f"{v:.10g}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
            also = ", ".join(x for x in EXTRA_KEYS if x in k.extras) or "—"
            it = QTableWidgetItem(also)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setForeground(QColor(style.c()["faint"]))
            t.setItem(r, 7, it)
            if select is not None and k.t == select:
                t.selectRow(r)
        # the data that change along the path stand out; the constant ones are greyed
        vary = set(self.path.varying())
        for c, name in enumerate(PARAM_NAMES, start=1):
            hi = name in vary
            for r in range(t.rowCount()):
                it = t.item(r, c)
                if it is not None:
                    it.setForeground(QColor(style.c()["text"] if hi else style.c()["faint"]))
                    f = it.font()
                    f.setBold(hi)
                    it.setFont(f)
            hd = t.horizontalHeaderItem(c)
            if hd is not None:
                hd.setText(NAMES[name] + (" (varies)" if hi else ""))
        self._filling = False
        n = len(self.path)
        self.upd_btn.setEnabled(n > 0)
        self.del_btn.setEnabled(n > 0)
        self.clr_btn.setEnabled(n > 0)
        ok = n >= 2
        for w in (self.play_btn, self.first_btn, self.last_btn, self.slider, self.export_btn):
            w.setEnabled(ok)
        if not ok and self.play_btn.isChecked():
            self.play_btn.setChecked(False)
        if n < 2:
            self.events_lab.setText(
                f"<span style='color:{style.c()['muted']}'>A path needs two keyframes. Add the current solution "
                "with <b>+ Add current</b>, or <b>shift-click</b> points on the Σ-map. Type values in the table "
                "to make them exact. The timeline then shows where along the path the solution is global "
                "(green), global with a special pole (teal) or not global (red).</span>")
        self._update_length()

    def _cell_edited(self, item: QTableWidgetItem):
        if self._filling:
            return
        r, c = item.row(), item.column()
        try:
            v = float(item.text().replace("−", "-"))
        except ValueError:
            self._refresh()
            return
        k = self.path.keyframes[r]
        if c == 0:
            k.t = v
            self.path.sort()
            self._changed(select=v)
        else:
            vals = list(k.params.as_tuple())
            vals[c - 1] = v
            k.params = Params(*vals, xb=k.params.xb)
            self._changed(select=k.t)

    def _row_selected(self):
        if self._filling:
            return
        r = self._selected_row()
        if r < 0 or len(self.path) < 2:
            return
        t0, t1 = self.path.duration
        f = (self.path.keyframes[r].t - t0) / ((t1 - t0) or 1.0)
        self.goto(int(round(f * (self.n_frames() - 1))))

    # ---------------------------------------------------------------- scanning
    def _schedule_scan(self, *_):
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_frames() - 1)
        self.slider.blockSignals(False)
        self.index = min(self.index, self.n_frames() - 1)
        self._update_length()
        self.frames, self.events = [], []
        self._update_timeline()
        self._scan_timer.start()

    def _scan(self):
        self._gen += 1
        if len(self.path) < 2:
            self._update_timeline()
            return
        self.frame_lab.setText(f"<span style='color:{style.c()['muted']}'>checking every frame…</span>")
        self.worker.submit(self.path.to_dict(), self.n_frames(), self._gen)

    def _scanned(self, frames, events, gen):
        if gen != self._gen:
            return
        if isinstance(frames, Exception):
            self.frame_lab.setText(f"scan error: {frames}")
            return
        self.frames, self.events = frames, events
        self._update_timeline()
        self._update_events()
        self._update_frame_label()
        self.pathChanged.emit()

    def _update_timeline(self):
        rng = self.path.duration if len(self.path) else (0.0, 1.0)
        self.timeline.set_data(self.n_frames() if len(self.path) >= 2 else 0, self.frames, self.events, rng,
                               [k.t for k in self.path.keyframes])
        self.timeline.set_index(self.index)

    def _update_events(self):
        P = style.c()
        if len(self.path) < 2:
            return
        if not self.frames:
            self.events_lab.setText("")
            return
        bad = sum(not f.is_global for f in self.frames)
        n = len(self.frames)
        kinds = {failure_kind(f) for f in self.frames}
        other = any(f.is_global and (f.north, f.south) != ("G", "G") for f in self.frames)
        sw = "<span style='color:{}'>■</span> {} &nbsp; "
        head = sw.format(COL_OK, "global")
        if other:
            head += sw.format(COL_OTHER, "global, an R / L1 / L2 pole")
        if "unsafe" in kinds:
            head += sw.format(COL_UNSAFE, "not global: a pole fails the terminal test")
        if "blowup" in kinds:
            head += sw.format(COL_BLOW, "not global: blows up before a pole")
        if self.events:
            head += f"<span style='color:{COL_EVENT}'>▼</span> event &nbsp; "
        head += f"<span style='color:{COL_EVENT}'>●</span> keyframe (numbers as on the Σ-map)<br>"
        if bad == 0:
            summary = f"<b style='color:{COL_OK}'>All {n} frames are global.</b>"
        else:
            what = {"hold": "show the last global frame instead (labelled HELD)",
                    "skip": "leave them out", "mark": "show them, labelled NOT GLOBAL"}[self.policy.currentData()]
            summary = (f"<b style='color:{COL_BAD}'>{bad} of {n} frames are not global.</b> "
                       f"In the MP4 they {what}; the preview shows them as they are.")
        lines = []
        names = {"leaves_global": "leaves the global set", "enters_global": "comes back into the global set",
                 "stratum_change": "a pole changes type", "failure_change": "still not global, now"}
        link = P["accent"]
        for e in self.events[:12]:
            detail = e.detail.replace("->", "→")
            sep = " " if e.kind == "failure_change" else " — "
            extra = ""
            if e.kind == "leaves_global" and e.index > 0:
                extra = (f" <span style='color:{P['faint']}'>(last global frame: {e.index}, "
                         f"t = {self.t_of(e.index - 1):.4g}; the MP4 holds it)</span>")
            lines.append(f"<a href='{e.index}' style='color:{link}'>t = {e.t:.4g}</a> (frame {e.index + 1}): "
                         f"{names[e.kind]}" + (f"{sep}{detail}" if detail else "") + extra)
        if len(self.events) > 12:
            lines.append(f"… and {len(self.events) - 12} more")
        self.events_lab.setText(head + summary + ("<br>" + "<br>".join(lines) if lines else ""))

    def _update_length(self):
        secs = self.n_frames() / max(1, self.fps.value())
        self.length_lab.setText(f"= {secs:.1f} s")

    def _fps_changed(self, *_):
        self._play.setInterval(int(1000 / max(1, self.fps.value())))
        self._update_length()

    # ---------------------------------------------------------------- frames
    def t_of(self, i: int) -> float:
        t0, t1 = self.path.duration
        return t0 + (t1 - t0) * i / max(1, self.n_frames() - 1)

    def frame(self, i: int) -> tuple[Params, dict]:
        t = self.t_of(i)
        p = self.path.at(t)[0]
        extras = {k: self.path.extra_at(t, k) for k in EXTRA_KEYS if self.path.keyed(k)}
        return p, extras

    def goto(self, i: int):
        if len(self.path) < 2:
            return
        i = int(max(0, min(self.n_frames() - 1, i)))
        self.index = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.timeline.set_index(i)
        self._update_frame_label()
        self._apply_timer.start()

    def _apply_now(self):
        if len(self.path) < 2:
            return
        p, extras = self.frame(self.index)
        self.applyFrame.emit(p, extras)
        self.pathChanged.emit()

    def _update_frame_label(self):
        if len(self.path) < 2:
            self.frame_lab.setText("")
            return
        i = self.index
        txt = f"frame {i + 1} / {self.n_frames()}, path t = {_m(self.t_of(i))}"
        if i < len(self.frames):
            fs = self.frames[i]
            if fs.is_global:
                txt += (f" &nbsp; <b style='color:{COL_OK}'>global</b> · north {fs.north}, μ₊ = {fs.mu_n:.3f}"
                        f" · south {fs.south}, μ₋ = {fs.mu_s:.3f}")
            else:
                from ...core.paths import _why
                txt += f" &nbsp; <b style='color:{COL_BAD}'>NOT GLOBAL</b>: {_why(fs)}"
        self.frame_lab.setText(txt)

    def _play_toggled(self, on: bool):
        self.play_btn.setText("⏸ Pause" if on else "▶ Play")
        if on:
            if self.index >= self.n_frames() - 1:
                self.goto(0)
            self._play.start()
        else:
            self._play.stop()

    def _tick(self):
        if self.index >= self.n_frames() - 1:
            if self.loop.isChecked():
                self.goto(0)
            else:
                self.play_btn.setChecked(False)
            return
        self.goto(self.index + 1)

    # ---------------------------------------------------------------- Σ-map overlay
    def overlay(self) -> dict | None:
        if len(self.path) < 2:
            return {"keys": [k.params for k in self.path.keyframes], "samples": [k.params for k in self.path.keyframes],
                    "kind": None, "cur": None, "varying": []} if len(self.path) else None
        n = self.n_frames()
        samples = self.path.at(np.linspace(*self.path.duration, n))
        kind = [failure_kind(f) for f in self.frames] if len(self.frames) == n else None
        cur = samples[self.index] if self.index < n else None
        evs = [self.path.at(e.t)[0] for e in self.events] if kind is not None else []
        return {"keys": [k.params for k in self.path.keyframes], "samples": samples, "kind": kind, "cur": cur,
                "events": evs,
                "varying": self.path.varying()}

    def state(self) -> dict:
        return {"frames": self.n_frames(), "fps": self.fps.value(), "loop": self.loop.isChecked()}

    def refresh_style(self):
        self._refresh()
        self._update_events()
        self._update_frame_label()
        self.timeline.update()

    def shutdown(self):
        self._play.stop()
        self._gen += 1
        self.thread.quit()
        self.thread.wait(3000)
