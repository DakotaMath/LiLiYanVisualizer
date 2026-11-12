"""Quick off-screen PyVista render of streamlines for the layout mock-up (prototype of milestone 3)."""
import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import solve
from swirl_explorer.core.fields import FieldInterpolator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def main():
    sol = solve(Params(gamma=-1.6806, kappa=1.0))
    fi = FieldInterpolator(sol)
    n = 72
    ax = np.linspace(-1.2, 1.2, n)
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(ax[1] - ax[0],) * 3, origin=(ax[0],) * 3)
    P = grid.points
    U = fi.velocity(P)
    r = np.linalg.norm(P, axis=1)
    rho = np.hypot(P[:, 0], P[:, 1])
    bad = ~np.isfinite(U).all(axis=1) | (r < 0.18) | (rho < 0.06) | (r > 1.2)
    U[bad] = 0.0
    grid["u"] = U
    grid["|u|"] = np.where(bad, np.nan, np.linalg.norm(U, axis=1))
    seeds = pv.Disc(center=(0, 0, 0.35), inner=0.15, outer=0.7, normal=(0, 0, 1), r_res=2, c_res=10)
    sl = grid.streamlines_from_source(seeds, vectors="u", max_length=12.0, integration_direction="both",
                                      initial_step_length=0.2, terminal_speed=1e-6)
    pl = pv.Plotter(off_screen=True, window_size=(1000, 860))
    pl.set_background("#1d1f24")
    if sl.n_points:
        tube = sl.tube(radius=0.008)
        pl.add_mesh(tube, scalars="|u|", cmap="viridis", log_scale=True, clim=(0.3, 30),
                    nan_opacity=0.0, scalar_bar_args=dict(title="|u| (log scale)", color="white"))
    north = pv.Line((0, 0, 0.18), (0, 0, 1.3))
    south = pv.Line((0, 0, -1.3), (0, 0, -0.18))
    pl.add_mesh(north.tube(radius=0.02), color="#2a78d6")
    pl.add_mesh(south.tube(radius=0.02), color="#86b6ef")
    pl.add_point_labels([(0, 0, 1.35), (0, 0, -1.4)], ["N: G, μ₊=0.62 (III)", "S: G, μ₋=0.99 (II.5)"],
                        text_color="white", font_size=14, shape=None, show_points=False)
    pl.add_mesh(pv.Sphere(radius=1.2, theta_resolution=48, phi_resolution=48), style="wireframe",
                color="#55575c", opacity=0.15)
    pl.camera_position = [(4.6, -3.8, 2.0), (0, 0, 0), (0, 0, 1)]
    out = os.path.join(OUT, "mock_3d.png")
    pl.screenshot(out)
    print(out, sl.n_points)


if __name__ == "__main__":
    main()
