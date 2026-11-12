"""Smoke tests of the Qt shell and the 2D plots (offscreen)."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import solve
from swirl_explorer.core.poles import summarize
from swirl_explorer.viz.plots2d import Curve, draw_profiles, export_csv, QUANTITIES, XVARS, MODES


@pytest.fixture(scope="module")
def curve():
    sol = solve(Params(gamma=-1.6806, kappa=1.0))
    return Curve("cur", sol, summarize(sol))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("xvar", list(XVARS))
def test_draw_all_modes(curve, mode, xvar):
    fig = Figure()
    axes = draw_profiles(fig, [curve], list(QUANTITIES), xvar=xvar, mode=mode, pole="south")
    assert len(axes) == len(QUANTITIES)


def test_draw_blowup_solution():
    sol = solve(Params(gamma=5.0))            # Landau |L| < 1: blows up at x = -0.4
    c = Curve("blow", sol, summarize(sol))
    axes = draw_profiles(Figure(), [c], ["U", "u_r"], xvar="x", mode="linear")
    lo, hi = axes[0].get_ylim()
    assert hi - lo < 1e4                      # robust limits, not the blow-up spike


def test_export_csv(tmp_path, curve):
    p = tmp_path / "c.csv"
    export_csv(str(p), [curve], ["U", "f"], "xi")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("cur:xi") and len(lines) > 100


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=10.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_main_window_solves_and_updates(app, tmp_path):
    from swirl_explorer.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    assert _wait(app, lambda: w.last is not None)
    assert "GLOBAL" in w.status.head.text() and "type" in w.status.north.text()
    # the 2D slice is hidden until the button is pressed
    app.processEvents()
    assert not w.slice.isVisible() and not w.dock_slice.isVisible()
    w.slice_btn.click()
    assert w.slice.isVisible() and w.dock_slice.isVisible() and w.act_slice.text() == "Hide 2D slice"
    w.slice_btn.click()
    assert not w.slice.isVisible()
    # a non-global choice updates the status with the blow-up point
    w.params.set_params(Params(gamma=5.0))
    assert _wait(app, lambda: w.last is not None and w.last[0].gamma == 5.0)
    assert "NOT GLOBAL" in w.status.head.text()
    # rapid slider drags: only the latest request is shown
    for g in (0.1, 0.2, 0.3, 0.4, 0.5):
        w.params.rows["gamma"].set_value(g, emit=True)
    assert _wait(app, lambda: w.last[0].gamma == pytest.approx(0.5))
    # pin + exports
    w.plots.pin_current()
    assert len(w.plots.pins) == 1
    for kind in ("png", "svg", "csv"):
        out = w.plots.export(kind, str(tmp_path / f"p.{kind}"))
        assert os.path.getsize(out) > 0
    w.close()


def test_slider_and_spin_stay_in_sync(app):
    from swirl_explorer.ui.panels.params_panel import ParamsPanel
    pp = ParamsPanel()
    seen = []
    pp.paramsChanged.connect(seen.append)
    row = pp.rows["kappa"]
    row.slider.setValue(1500)                 # 3/4 of [-5, 5] -> 2.5
    assert row.value() == pytest.approx(2.5, abs=1e-9)
    assert seen and seen[-1].kappa == pytest.approx(2.5)
    row.spin.setValue(8.0)                    # outside the range: slider pins to the end
    assert row.slider.value() == 2000
