"""Parameter panel: six Cauchy data with sliders, numeric fields and editable ranges."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QGridLayout, QLabel, QSlider, QDoubleSpinBox, QGroupBox,
                               QVBoxLayout, QHBoxLayout, QPushButton, QToolButton, QMenu,
                               QInputDialog, QSizePolicy)

from ...core.params import PARAM_UI, DEFAULT_RANGES, Params, PARAM_NAMES
from ...core.edges import gamma_minus
from ...viz import style

LABELS = PARAM_UI          # (label, tooltip) per datum; shared with the browser version
SLIDER_STEPS = 2000


class CompactSpin(QDoubleSpinBox):
    """Shows ~6 significant digits; keeps full precision internally."""

    def textFromValue(self, v: float) -> str:
        return f"{v:.7g}"


class ParamRow:
    """One parameter: name | slider | value | [min, max]."""

    def __init__(self, grid: QGridLayout, row: int, name: str, on_change, value_filter=None):
        self.name = name
        self._on_change = on_change
        self._filter = value_filter          # (name, old, new) -> allowed new value (the global lock)
        self._prev = 0.0                     # last accepted value
        self._block = False
        text, tip = LABELS[name]
        lab = QLabel(text)
        lab.setToolTip(tip)
        lo, hi = DEFAULT_RANGES[name]
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setMinimumWidth(100)
        self.spin = self._spin(-1e6, 1e6, 13)    # full precision (edges need it); shown with 7 digits
        self.spin.setToolTip(tip)
        self.lo = self._spin(-1e6, 1e6, 3, lo)
        self.hi = self._spin(-1e6, 1e6, 3, hi)
        for w in (self.lo, self.hi):
            w.setFixedWidth(62)
            w.setToolTip("slider range (editable)")
            f = w.font()
            f.setPointSizeF(max(f.pointSizeF() - 1.0, 7.0))
            w.setFont(f)

        self.ranges = (self.lo, self.hi)
        self.range_lab = QLabel()
        self.range_lab.setStyleSheet(f"color:{style.c()['faint']};font-size:10px;")
        self._update_range_label()
        self.lo.valueChanged.connect(lambda _: self._update_range_label())
        self.hi.valueChanged.connect(lambda _: self._update_range_label())
        grid.addWidget(lab, row, 0)
        grid.addWidget(self.range_lab, row, 5)
        grid.addWidget(self.slider, row, 1)
        grid.addWidget(self.spin, row, 2)
        grid.addWidget(self.lo, row, 3)
        grid.addWidget(self.hi, row, 4)
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)
        self.lo.valueChanged.connect(lambda _: self._sync_slider())
        self.hi.valueChanged.connect(lambda _: self._sync_slider())

    @staticmethod
    def _spin(lo, hi, dec, val=0.0):
        s = CompactSpin()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(10 ** (-min(dec, 3)))
        s.setValue(val)
        s.setKeyboardTracking(False)
        s.setButtonSymbols(QDoubleSpinBox.NoButtons)
        return s

    def _update_range_label(self):
        self.range_lab.setText(f"{self.lo.value():g} … {self.hi.value():g}".replace("-", "−"))
        self.range_lab.setToolTip("slider range (tick 'edit slider ranges' to change it)")

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, v: float, emit: bool = False) -> None:
        self._block = not emit
        if v < self.lo.value():
            self.lo.setValue(v)
        if v > self.hi.value():
            self.hi.setValue(v)
        self.spin.setValue(v)
        self._prev = self.spin.value()
        self._sync_slider()
        self._block = False

    def _accept(self, v: float) -> None:
        """Pass a user edit through the lock, show the accepted value and notify."""
        if self._filter is not None:
            v = self._filter(self.name, self._prev, v)
        old = self._block
        self._block = True
        self.spin.setValue(v)
        self._block = old
        self._prev = self.spin.value()
        self._sync_slider()
        self._on_change()

    def _sync_slider(self):
        lo, hi = self.lo.value(), self.hi.value()
        if hi <= lo:
            return
        pos = int(round((self.spin.value() - lo) / (hi - lo) * SLIDER_STEPS))
        old = self._block
        self._block = True
        self.slider.setValue(max(0, min(SLIDER_STEPS, pos)))
        self._block = old

    def _slider_moved(self, pos: int):
        if self._block:
            return
        lo, hi = self.lo.value(), self.hi.value()
        self._accept(lo + (hi - lo) * pos / SLIDER_STEPS)

    def _spin_changed(self, v):
        if self._block:
            return
        self._accept(v)


class ParamsPanel(QWidget):
    """Emits ``paramsChanged(Params)`` whenever a value changes."""

    paramsChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        box = QGroupBox("Data at the base point")
        grid = QGridLayout(box)
        self._range_heads = []
        for col, t in enumerate(("", "", "value", "slider min", "max")):
            if t:
                h = QLabel(t)
                h.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
                grid.addWidget(h, 0, col)
                if col >= 3:
                    self._range_heads.append(h)
        self.rows = {n: ParamRow(grid, i + 1, n, self._changed, self._lock_filter) for i, n in enumerate(PARAM_NAMES)}
        grid.addWidget(QLabel("base point x₀"), len(PARAM_NAMES) + 1, 0)
        self.xb = ParamRow._spin(-0.99, 0.99, 4, 0.0)
        self.xb.setToolTip("base point x_b = cos θ_b of the Cauchy data (default: equator)")
        self.xb.valueChanged.connect(lambda _: self._changed())
        grid.addWidget(self.xb, len(PARAM_NAMES) + 1, 2)
        grid.setColumnStretch(1, 1)
        from PySide6.QtWidgets import QCheckBox
        self.show_ranges = QCheckBox("edit slider ranges")
        self.show_ranges.setToolTip("show the min/max of each slider")
        self.show_ranges.toggled.connect(self._toggle_ranges)

        grid.addWidget(self.show_ranges, len(PARAM_NAMES) + 2, 2, 1, 3)
        self.lock = QCheckBox("lock to global solutions")
        self.lock.setChecked(True)
        self.lock.setToolTip("a slider or typed value that would leave the global set stops at its edge "
                             "(the last global value in that direction)")
        grid.addWidget(self.lock, len(PARAM_NAMES) + 2, 0, 1, 2)
        self.lock_note = QLabel("")
        self.lock_note.setWordWrap(True)
        self.lock_note.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        grid.addWidget(self.lock_note, len(PARAM_NAMES) + 3, 0, 1, 6)
        self.lock_note.hide()
        self._toggle_ranges(False)
        v.addWidget(box)

        pres = QGroupBox("Presets")
        pg = QGridLayout(pres)
        items = (("Rest state", self.preset_rest, "c = 0, γ = A = κ = 0: U ≡ 0, no flow"),
                 ("Landau jet…", self.preset_landau, "no swirl: U = 2(1−x²)/(λ_L + x), global iff |λ_L| ≥ 1"),
                 ("Generic pole (type G)", self.preset_G,
                  "inside the lens: c = 0, κ = 1, γ = γ₋ + 0.1 (monograph, Numerical observation 4.28)"),
                 ("Lens edge (type L1)", self.preset_L1,
                  "on the lower lens edge: c = 0, κ = 1, γ = γ₋ (north pole logarithmic, Obs. 4.28)"),
                 ("Recessive edge (type R)", self.preset_R,
                  "lens edge for c = (0, −0.9, 0), κ = 0.3: north pole recessive, μ ≈ 0.22 (Obs. 4.28)"))
        for i, (text, fn, tip) in enumerate(items):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            pg.addWidget(b, i // 3, i % 3)
        v.addWidget(pres)
        note = QLabel(
            "<span><b>Notation.</b> x = cos θ; the poles x = ±1 are the symmetry axis, "
            "d = sin θ is the distance to it. c<sub>1</sub>, c<sub>2</sub>, c<sub>3</sub>: Taylor constants of "
            "the potential Q at the base point, Q ≈ c<sub>1</sub>(1−x) + c<sub>2</sub>(1+x) + c<sub>3</sub>(1−x<sup>2</sup>). "
            "γ = U, A = f, κ = f<sub>x</sub> at the base point, where U = u<sub>θ</sub> sin θ and "
            "f = u<sub>φ</sub> sin θ. z solves z′′ = W z with W = 1 + Q/2, and h = z′/z.</span>")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _toggle_ranges(self, on: bool):
        for r in self.rows.values():
            for w in r.ranges:
                w.setVisible(on)
        for h in self._range_heads:
            h.setVisible(on)
        for r in self.rows.values():
            r.range_lab.setVisible(not on)

    def refresh_style(self):
        for r in self.rows.values():
            r.range_lab.setStyleSheet(f"color:{style.c()['faint']};font-size:10px;")
        for h in self._range_heads:
            h.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")
        self.lock_note.setStyleSheet(f"color:{style.c()['muted']};font-size:11px;")

    # ---------------------------------------------------------------- state
    def params(self) -> Params:
        return Params(*(self.rows[n].value() for n in PARAM_NAMES), xb=self.xb.value())

    def set_params(self, p: Params, emit: bool = True) -> None:
        for n, val in zip(PARAM_NAMES, p.as_tuple()):
            self.rows[n].set_value(val, emit=False)
        self.xb.blockSignals(True)
        self.xb.setValue(p.xb)
        self.xb.blockSignals(False)
        if emit:
            self._changed()

    def _changed(self):
        self.paramsChanged.emit(self.params())

    # ---------------------------------------------------------------- global lock
    def _lock_filter(self, name: str, old: float, new: float) -> float:
        """With the lock on, a move from a global point stops at the last global value toward `new`."""
        self.lock_note.hide()
        if not self.lock.isChecked() or new == old:
            return new
        from ...core.solver import is_global_fast
        base = self.params()                        # the other parameters as they are now
        at = lambda v: base.with_(**{name: v})
        if is_global_fast(at(new)):
            return new
        if not is_global_fast(at(old)):              # already outside: nothing to hold on to
            return new
        g, b = old, new                              # g global, b not: bisect for the edge
        tol = 1e-9 * max(1.0, abs(old), abs(new))
        while abs(b - g) > tol:
            mid = 0.5 * (g + b)
            if is_global_fast(at(mid)):
                g = mid
            else:
                b = mid
        side = "maximum" if new > old else "minimum"
        self.lock_note.setText(f"Locked: {LABELS[name][0]} held at its global {side} {g:.7g} "
                               f"(untick 'lock to global solutions' to go past it)")
        self.lock_note.show()
        return g

    # ---------------------------------------------------------------- presets
    def preset_rest(self):
        self.set_params(Params())

    def preset_landau(self):
        L, ok = QInputDialog.getDouble(self, "Landau family",
                                       "λ_L (|λ_L| ≥ 1 is global; U = 2(1−x²)/(λ_L+x)):",
                                       1.5, -100.0, 100.0, 4)
        if ok and L != 0:
            self.set_params(Params(gamma=2.0 / L))

    def preset_G(self):
        gm = gamma_minus(Params(kappa=1.0))
        self.set_params(Params(gamma=gm + 0.1, kappa=1.0))

    def preset_L1(self):
        gm = gamma_minus(Params(kappa=1.0))
        self.set_params(Params(gamma=gm + 1e-11, kappa=1.0))

    def preset_R(self):
        p = Params(c2=-0.9, kappa=0.3)
        gm = gamma_minus(p)
        self.set_params(p.with_(gamma=gm + 1e-11))
