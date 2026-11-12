"""Milestone-3 screenshots. Needs a real OpenGL context: run WITHOUT QT_QPA_PLATFORM=offscreen
(a window appears for a few seconds)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from swirl_explorer.core.params import Params
from swirl_explorer.ui.main_window import MainWindow

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
app = QApplication(sys.argv)
from swirl_explorer.ui.theme import apply_theme
apply_theme(app)
w = MainWindow()
w.resize(1680, 980)
w.move(20, 20)
w.show()
v = w.view3d


def settle(sec=2.5):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.02)


def shot(name):
    settle()
    v._reset_camera()                      # same view in every screenshot
    settle(0.8)
    w.grab().save(os.path.join(OUT, name))


settle(3)
shot("m3_streamlines.png")
v.chk["show_iso"].setChecked(True)
v.color_by.setCurrentIndex(v.color_by.findData("u_phi"))
shot("m3_iso_glyphs.png")
v.chk["show_iso"].setChecked(False)
v.chk["show_selfsim"].setChecked(True)
v.chk["show_vortex"].setChecked(True)
v.color_by.setCurrentIndex(v.color_by.findData("helicity"))
shot("m3_selfsim_vortex.png")
print("saved")
w.close()
