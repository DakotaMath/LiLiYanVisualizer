"""Milestone-6 screenshots: the path dock, the path on the Σ-map, and frames of an exported movie.

A window appears for a while (the 3D view needs a real OpenGL context).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "figures")


def main():
    from PySide6.QtWidgets import QApplication
    from swirl_explorer.core.params import Params
    from swirl_explorer.ui.main_window import MainWindow
    from swirl_explorer.ui.theme import apply_theme
    from swirl_explorer.ui.movie_export import MovieExporter
    from swirl_explorer.ui import panel_state as ps
    errors = []
    hook = sys.excepthook
    sys.excepthook = lambda t, v, tb: (errors.append(v), hook(t, v, tb))
    app = QApplication(sys.argv)
    apply_theme(app)
    w = MainWindow()
    w.quiet = True
    w.resize(1680, 1020)
    w.move(20, 20)
    w.show()

    def settle(sec=1.0, until=None, timeout=180):
        t = time.time()
        while time.time() - t < sec or (until is not None and not until() and time.time() - t < timeout):
            app.processEvents()
            time.sleep(0.02)

    settle(2)
    pp = w.path_panel
    w.path_btn.click()
    pp.keep["camera"].setChecked(True)
    # a path through the lens at c = 0, A = 0 that leaves it near the end
    for g, k in ((-1.2, 0.5), (0.0, 2.5), (0.45, 3.3), (0.3, 4.4)):
        w.params.lock.setChecked(False)
        w.params.set_params(Params(gamma=g, kappa=k))
        settle(0.3)
        w.view3d.plotter.camera.azimuth = 0
        w.view3d.plotter.camera.Azimuth(25)
        pp.add_current()
    pp.interp.setCurrentIndex(1)             # spline
    pp.frames_spin.setValue(48)
    settle(1, until=lambda: len(pp.frames) == 48)
    pp.goto(20)
    settle(3)
    w.grab().save(os.path.join(OUT, "m6_path_dock.png"))
    pp.goto(47)                               # outside the lens
    settle(3)
    w.grab().save(os.path.join(OUT, "m6_path_not_global.png"))
    s = w.sigma
    w.tabs.setCurrentWidget(s)
    settle(1, until=lambda: s.res is not None and not s.running)
    pp.goto(30)
    settle(2)
    w.grab().save(os.path.join(OUT, "m6_path_on_sigma.png"))
    w.tabs.setCurrentIndex(0)
    settle(1)
    # a short movie; keep two frames as pictures
    msgs = []
    mp4 = os.path.join(OUT, "m6_demo.mp4")
    ex = MovieExporter(w, {"layout": "all", "size": (1280, 720), "fps": 24, "frames": 48, "policy": "hold",
                           "path": mp4})
    ex.finished.connect(msgs.append)
    ex.start()
    settle(1, until=lambda: bool(msgs), timeout=600)
    print(msgs)
    import imageio.v2 as imageio
    r = imageio.get_reader(mp4)
    for i, fr in enumerate(r):
        if i in (16, 47):        # file names count from 1, like the captions
            imageio.imwrite(os.path.join(OUT, f"m6_movie_frame{i + 1}.png"), fr)
    print("saved" if not errors else f"FAILED with {len(errors)} exception(s)")
    w.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
