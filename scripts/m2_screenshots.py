"""Milestone-2 screenshots (offscreen): default view, log-log view, non-global + pinned comparison."""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
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
w.show()


def settle(cond=lambda: True, n=25):
    t = time.time()
    while (w.last is None or not cond()) and time.time() - t < 10:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(n):
        app.processEvents()
        time.sleep(0.02)


settle()
w.grab().save(os.path.join(OUT, "m2_main.png"))
pp = w.plots
for k, c in pp.checks.items():
    c.setChecked(k in ("U", "u_r", "u_phi"))
pp.mode.setCurrentIndex(pp.mode.findData("loglog"))
settle()
w.grab().save(os.path.join(OUT, "m2_loglog.png"))
pp.mode.setCurrentIndex(pp.mode.findData("linear"))
for k, c in pp.checks.items():
    c.setChecked(k in ("U", "f"))
pp.pin_current()
w.params.set_params(Params(gamma=-1.9, kappa=1.0))
settle(lambda: w.last[0].gamma == -1.9)
w.grab().save(os.path.join(OUT, "m2_nonglobal.png"))
print("saved")
w.close()
