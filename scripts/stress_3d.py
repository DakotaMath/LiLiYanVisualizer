"""UI stress test for the 3D view (needs a real OpenGL window; one appears for ~1 min).
Exits non-zero if any exception is raised inside a Qt slot."""
import os, sys, time, faulthandler, itertools
faulthandler.enable(); faulthandler.dump_traceback_later(240, exit=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERRORS = []
_hook = sys.excepthook
def _record(t, v, tb):
    ERRORS.append(v); _hook(t, v, tb)
sys.excepthook = _record
from PySide6.QtWidgets import QApplication
from swirl_explorer.core.params import Params
from swirl_explorer.ui.main_window import MainWindow
from swirl_explorer.ui.theme import apply_theme
app = QApplication(sys.argv); apply_theme(app)
w = MainWindow(); w.resize(1500, 900); w.show(); v = w.view3d
def settle(t=0.6):
    e = time.time()
    while time.time() - e < t:
        app.processEvents(); time.sleep(0.01)
settle(2)
step = 0
def log(msg):
    global step; step += 1; print(step, msg, flush=True)
for mode in range(v.seed_mode.count()):
    v.seed_mode.setCurrentIndex(mode); log(f"seed mode {v.seed_mode.currentData()}"); settle()
    v.show_handle.setChecked(True); settle(0.4); v.show_handle.setChecked(False); settle(0.3)
for k, c in v.chk.items():
    c.setChecked(not c.isChecked()); log(f"toggle {k}"); settle(0.5)
for i in range(v.color_by.count()):
    v.color_by.setCurrentIndex(i); log(f"colour {v.color_by.currentData()}"); settle(0.5)
for i in range(v.iso_field.count()):
    v.iso_field.setCurrentIndex(i); v.chk["show_iso"].setChecked(True); log(f"iso {v.iso_field.currentData()}"); settle(0.5)
v.shape.setCurrentIndex(1); log("box"); settle()
v.r_min.setValue(0.3); v.axis_a.setValue(0.1); log("cutouts"); settle()
for p in (Params(gamma=-1.9, kappa=1.0), Params(gamma=5.0), Params(), Params(c2=-0.9, gamma=-1.0186, kappa=0.3),
          Params(gamma=-1.780604544215862 + 1e-11, kappa=1.0), Params(c1=2, c2=2, c3=3, gamma=0.5, A=2, kappa=4)):
    w.params.set_params(p); log(f"params {p.as_tuple()}"); settle(1.0)
for g in [x / 10 for x in range(-30, 31, 3)]:
    w.params.rows["gamma"].set_value(g, emit=True); app.processEvents()
log("rapid slider"); settle(2)
v._view_from_pole(); v._reset_camera(); log("cameras"); settle()
from swirl_explorer.viz.slices import PlaneSpec
settle(1)
for spec in (PlaneSpec(), PlaneSpec(elevation=90, offset=0.4), PlaneSpec(azimuth=40, elevation=30, offset=-0.3)):
    w.slice_ctrl.set_spec(spec, source="stress"); log(f"slice {spec.azimuth},{spec.elevation},{spec.offset}"); settle(1)
w.slice.chk_lic.setChecked(True); w.slice.density.setValue(2.5); log("lic + density"); settle(1.5)
for v_ in range(-60, 61, 15):
    w.slice.el.setValue(v_); app.processEvents()
log("rapid slice slider"); settle(2)
v.chk_slice.setChecked(True); v.chk_plane_widget.setChecked(True); log("plane widget"); settle(1)
w.slice_ctrl.set_spec(PlaneSpec(azimuth=10, elevation=50, offset=0.2), source="stress"); settle(1)
v.chk_plane_widget.setChecked(False); v.chk_slice.setChecked(False); log("slice off"); settle(0.5)
w.set_theme("light"); log("light theme"); settle(1); w.set_theme("dark"); log("dark theme"); settle(1)
w.close()
print("STRESS OK" if not ERRORS else f"STRESS FAILED: {len(ERRORS)} exception(s)")
sys.exit(1 if ERRORS else 0)
