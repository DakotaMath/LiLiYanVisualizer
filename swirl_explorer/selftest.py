"""End-to-end self-test, mainly for the frozen Windows build:  SwirlExplorer.exe --selftest DIR

Exercises every part a missing file in the bundle would break: the solver, a Σ-map on a
process pool, a path scan, the Qt window with the 3D view (VTK / OpenGL), the matplotlib
panels, an MP4 export (ffmpeg) and a session round trip.  Writes DIR/selftest.txt and
DIR/selftest_window.png and returns 0 when everything passed.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


class _Report:
    def __init__(self, out: Path):
        self.out = out
        self.lines: list[str] = []
        self.failed = 0

    def step(self, name, fn):
        t = time.perf_counter()
        try:
            note = fn() or ""
            self.lines.append(f"PASS  {name:<28} {1e3 * (time.perf_counter() - t):7.0f} ms  {note}")
        except Exception as exc:
            self.failed += 1
            self.lines.append(f"FAIL  {name:<28} {type(exc).__name__}: {exc}")
            self.lines.append("      " + traceback.format_exc().replace("\n", "\n      "))
        self.write()

    def write(self):
        from . import __version__
        head = [f"Swirl Explorer {__version__} self-test", f"python {sys.version.split()[0]}, "
                f"frozen = {bool(getattr(sys, 'frozen', False))}, executable = {sys.executable}", ""]
        tail = ["", "RESULT: " + ("all passed" if not self.failed else f"{self.failed} step(s) FAILED")]
        (self.out / "selftest.txt").write_text("\n".join(head + self.lines + tail), encoding="utf-8")


def run(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = _Report(out)

    from .core.params import Params

    def core():
        from .core.solver import solve
        from .core.poles import summarize
        sol = solve(Params(gamma=2 / 1.5))                 # Landau, λ_L = 1.5: global
        sm = summarize(sol)
        assert sol.is_global, "Landau λ_L = 1.5 should be global"
        bad = solve(Params(gamma=5.0))                      # Landau |λ_L| < 1: blows up
        assert not bad.is_global
        return sm.status_line()[:60]

    def sigma():
        from .core.sigma_map import SigmaMapSpec, compute_sigma_map
        spec = SigmaMapSpec(Params(), "gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6), nx=21, ny=21)
        res = compute_sigma_map(spec, workers=2)            # a real process pool
        g = float(res.is_global.mean())
        assert 0.2 < g < 0.7, f"lens fraction {g:.2f}"
        return f"21 × 21 on 2 processes, {100 * g:.0f}% global"

    def path():
        from .core.paths import ParamPath, scan_path
        p = ParamPath()
        p.add(0, Params(gamma=-0.5, kappa=1.0))
        p.add(1, Params(gamma=-2.5, kappa=1.0))
        _, ev = scan_path(p, 21)
        t = next(e.t for e in ev if e.kind == "leaves_global")
        assert abs(t - 0.6403) < 2e-3, t
        return f"leaves the lens at t = {t:.4f}"

    rep.step("solver", core)
    rep.step("Σ-map (process pool)", sigma)
    rep.step("path scan", path)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from .ui.theme import apply_theme
    apply_theme(app)
    from .ui.main_window import MainWindow
    win = MainWindow()
    win.quiet = True
    win.resize(1400, 900)
    win.show()

    def wait(cond, timeout=60.0):
        t = time.time()
        while not cond() and time.time() - t < timeout:
            app.processEvents()
            time.sleep(0.01)
        return cond()

    def gui():
        assert wait(lambda: win.last is not None), "no solution shown"
        assert wait(lambda: win.view3d._last is not None), "3D scene never rendered"
        wait(lambda: False, 1.0)
        win.grab().save(str(out / "selftest_window.png"))
        from .ui import panel_state as ps
        img = ps.grab_3d(win.view3d, 400, 300)
        assert img is not None and img.std() > 5, "3D view is blank"
        return f"3D: {win.view3d._last.n_lines} streamlines"

    def movie():
        import imageio.v2 as imageio
        from .ui.movie_export import MovieExporter
        pp = win.path_panel
        pp.add_keyframe(Params(gamma=-1.0, kappa=1.0), extras={})
        pp.add_keyframe(Params(gamma=-2.4, kappa=1.0), extras={})
        msgs = []
        mp4 = out / "selftest.mp4"
        ex = MovieExporter(win, {"layout": "all", "size": (640, 360), "fps": 8, "frames": 6, "policy": "hold",
                                 "path": str(mp4)})
        ex.finished.connect(msgs.append)
        ex.start()
        assert wait(lambda: bool(msgs), 180), "export did not finish"
        assert msgs[0].startswith("Wrote 6 frames"), msgs[0]
        n = sum(1 for _ in imageio.get_reader(str(mp4)))
        assert n == 6, f"{n} frames in the file"
        return f"{mp4.stat().st_size // 1024} kB, 6 frames"

    def session():
        f = win.save_session(path=str(out / "selftest_session.json"))
        win.params.set_params(Params())
        assert win.open_session(f)
        assert len(win.path_panel.path) == 2, "the path did not come back"
        return "saved and reopened"

    rep.step("window + 3D view", gui)
    rep.step("MP4 export (ffmpeg)", movie)
    rep.step("session round trip", session)
    win.close()
    app.processEvents()
    rep.write()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()))
