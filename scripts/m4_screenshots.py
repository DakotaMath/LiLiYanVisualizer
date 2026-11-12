"""Milestone-4 screenshots (needs a real OpenGL window; one appears for a few seconds)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication
from swirl_explorer.ui.main_window import MainWindow
from swirl_explorer.ui.theme import apply_theme
from swirl_explorer.viz.slices import PlaneSpec

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
ERRORS = []
_hook = sys.excepthook
sys.excepthook = lambda t, v, tb: (ERRORS.append(v), _hook(t, v, tb))
app = QApplication(sys.argv)
apply_theme(app)
w = MainWindow()
w.resize(1680, 980)
w.move(20, 20)
w.show()


def settle(sec=2.5):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.02)


settle(3)
w.slice.chk_log.setChecked(True)
settle(3)
w.view3d.view_along_normal()
settle(1)
w.grab().save(os.path.join(OUT, "m4_slice_meridional.png"))
w.slice.chk_log.setChecked(False)
w.slice_ctrl.set_spec(PlaneSpec(azimuth=0, elevation=90, offset=0.45), source="script")
w.slice.chk_lic.setChecked(True)
settle(3)
w.view3d.view_along_normal()
settle(1)
w.grab().save(os.path.join(OUT, "m4_slice_horizontal_lic.png"))
w.view3d.chk_slice.setChecked(True)
w.view3d.chk["show_streamlines"].setChecked(False)
settle(3)
w.view3d._reset_camera()
settle(1)
w.grab().save(os.path.join(OUT, "m4_slice_in_3d.png"))
print("saved" if not ERRORS else f"FAILED with {len(ERRORS)} exception(s)")
w.close()
sys.exit(1 if ERRORS else 0)
