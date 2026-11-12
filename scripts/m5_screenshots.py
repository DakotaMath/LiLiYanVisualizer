"""Milestone-5 screenshots of the Σ-map tab (a window appears for a few seconds)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def main():
    from PySide6.QtWidgets import QApplication
    from swirl_explorer.core.params import Params
    from swirl_explorer.ui.main_window import MainWindow
    from swirl_explorer.ui.theme import apply_theme
    errors = []
    hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: (errors.append(v), hook(t, v, tb))
    app = QApplication(sys.argv)
    apply_theme(app)
    w = MainWindow()
    w.resize(1680, 980)
    w.move(20, 20)
    w.show()

    def settle(sec=1.0, until=None, timeout=120):
        t = time.time()
        while time.time() - t < sec or (until is not None and not until() and time.time() - t < timeout):
            app.processEvents()
            time.sleep(0.02)

    settle(2)
    s = w.sigma
    w.tabs.setCurrentWidget(s)               # first visit computes the default map (the lens)
    settle(1, until=lambda: s.res is not None and not s.running)
    settle(1)
    w.grab().save(os.path.join(OUT, "m5_sigma_lens.png"))
    s.mode.setCurrentIndex(2)                 # exponents
    settle(1.5)
    w.grab().save(os.path.join(OUT, "m5_sigma_mu.png"))
    w.params.set_params(Params(c2=-0.9, kappa=0.3))
    settle(2)
    s._set_ranges("c2", "gamma", (-1.0, 3.0), (-3.0, 3.0))
    s.mode.setCurrentIndex(1)                 # strata
    s.compute()
    settle(1, until=lambda: not s.running)
    settle(1)
    w.grab().save(os.path.join(OUT, "m5_sigma_strata.png"))
    print("saved" if not errors else f"FAILED with {len(errors)} exception(s)")
    w.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
