"""Σ-map: the book's lens (Figure 1.3) as a sanity check, and the panel's compute / click-to-jump."""
import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from swirl_explorer.core.params import Params
from swirl_explorer.core.edges import gamma_minus, gamma_plus
from swirl_explorer.core.sigma_map import SigmaMapSpec, compute_sigma_map


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def lens():
    spec = SigmaMapSpec(Params(xb=0.0), "gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6), nx=27, ny=24)
    return spec, compute_sigma_map(spec)


def test_lens_matches_the_edges_gamma_minus_plus(lens):
    """Figure 1.3: at c = 0, A = 0, x₀ = 0 the global set is γ₋(κ) ≤ γ ≤ γ₊(κ), mirror-symmetric in κ.
    Every grid point at least one cell away from an edge is classified as the edges say."""
    spec, res = lens
    xs, ys = spec.axes()
    dx = xs[1] - xs[0]
    checked = 0
    for j, k in enumerate(ys):
        gm = gamma_minus(Params(kappa=k), tol=1e-8)
        gp = gamma_plus(Params(kappa=k), tol=1e-8)
        for i, g in enumerate(xs):
            inside = math.isfinite(gm) and math.isfinite(gp) and gm < g < gp
            far = (not math.isfinite(gm) or abs(g - gm) > dx) and (not math.isfinite(gp) or abs(g - gp) > dx)
            if far:
                assert res.is_global[j, i] == inside, (g, k, gm, gp)
                checked += 1
    assert checked > 0.8 * res.done.size


def test_lens_shape(lens):
    spec, res = lens
    xs, ys = spec.axes()
    gl = res.is_global
    assert np.array_equal(gl, gl[::-1, :])                      # κ → −κ mirror image
    width = gl.sum(axis=1)
    assert width[np.argmin(abs(ys))] >= width.max() - 1          # widest near κ = 0 (Landau window)
    assert not gl[abs(ys) > 4.25].any()                          # apex κ_max = 4.1608
    assert gl[abs(ys) < 4.0].any(axis=1).all()


def _wait(app, cond, timeout=30):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_panel_computes_and_click_jumps(app, tmp_path):
    from matplotlib.backend_bases import MouseEvent
    from swirl_explorer.ui.panels.sigma_panel import SigmaMapPanel
    pan = SigmaMapPanel()
    pan.resize(700, 600)
    pan.res_spin.setValue(15)
    pan.set_current(Params(kappa=1.0))
    pan.compute()
    assert _wait(app, lambda: not pan.running)
    assert pan.res is not None and pan.res.done.all()
    for k in range(pan.mode.count()):                           # every display mode draws
        pan.mode.setCurrentIndex(k)
    pan.chk_mu.setChecked(True)
    pan.mode.setCurrentIndex(0)
    pan.canvas.draw()
    jumps = []
    pan.jumpRequested.connect(jumps.append)
    ax = pan._axes[0]
    x, y = ax.transData.transform((0.3, 1.2))
    pan._press(MouseEvent("button_press_event", pan.canvas, x, y, button=1))
    pan._release(None)
    assert _wait(app, lambda: bool(jumps), 5)
    assert jumps[-1].gamma == pytest.approx(0.3, abs=0.05) and jumps[-1].kappa == pytest.approx(1.2, abs=0.05)
    # the lock refuses non-global points
    pan.lock_check = lambda q: False
    jumps.clear()
    pan._press(MouseEvent("button_press_event", pan.canvas, x, y, button=1))
    app.processEvents()
    time.sleep(0.1)
    app.processEvents()
    assert not jumps and pan.note.text().startswith("Locked")
    assert pan.export("png", str(tmp_path / "m.png"))
    pan.shutdown()


def test_panel_cancel(app):
    from swirl_explorer.ui.panels.sigma_panel import SigmaMapPanel
    pan = SigmaMapPanel()
    pan.res_spin.setValue(41)
    pan.compute()
    pan.cancel()
    assert _wait(app, lambda: not pan.running)
    assert "cancel" in pan.note.text() or pan.res.done.all()
    pan.shutdown()


def test_hover_readout(app):
    from swirl_explorer.ui.panels.sigma_panel import SigmaMapPanel
    pan = SigmaMapPanel()
    pan.res_spin.setValue(15)
    pan.set_current(Params(kappa=1.0))
    pan.compute()
    assert _wait(app, lambda: not pan.running)
    inside = pan.readout(0.0, 1.0)
    outside = pan.readout(2.5, 4.5)
    assert "GLOBAL" in inside and "North: G" in inside
    assert "not global" in outside
    pan.shutdown()


@pytest.mark.parametrize("mode", ["global", "strata", "mu"])
@pytest.mark.parametrize("mu_contours", [False, True])
def test_every_view_draws(mode, mu_contours):
    """The shared drawing (viz.sigma_plot.draw_sigma) in every view, with and without μ contours."""
    from matplotlib.figure import Figure
    from swirl_explorer.core.sigma_map import SigmaMapSpec, compute_sigma_map
    from swirl_explorer.viz.sigma_plot import draw_sigma
    res = compute_sigma_map(SigmaMapSpec(Params(), "gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6), nx=15, ny=15))
    fig = Figure(figsize=(6, 6), layout="constrained")
    axs = draw_sigma(fig, res, mode, edge=True, mu_contours=mu_contours, current=Params(gamma=-1.6, kappa=1.0))
    assert len(axs) == (1 if mode == "global" else 2)
    fig.canvas.draw()
