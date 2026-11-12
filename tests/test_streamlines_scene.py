"""Tests of the streamline tracer and the 3D scene builders (no window needed)."""
import numpy as np
import pytest

from swirl_explorer.core.params import Params, landau_params
from swirl_explorer.core.solver import solve
from swirl_explorer.core.poles import summarize
from swirl_explorer.core.fields import FieldInterpolator
from swirl_explorer.viz.streamlines import (Domain, trace, seeds_sphere, seeds_line, seeds_disc,
                                            stream_surface, rescaled_copies)
from swirl_explorer.viz.scene_compute import compute_scene, SceneSettings


@pytest.fixture(scope="module")
def sol():
    return solve(Params(gamma=-1.6806, kappa=1.0))


def test_fast_interpolator_close_to_spline(sol):
    X = seeds_sphere((0.2, 0.3, 0.1), 0.5, 200)
    a = FieldInterpolator(sol).velocity(X)
    b = FieldInterpolator(sol, dxi=0.002, fast=True).velocity(X)
    ok = np.isfinite(a).all(axis=1)
    rel = np.linalg.norm(a[ok] - b[ok], axis=1) / np.linalg.norm(a[ok], axis=1)
    assert rel.max() < 1e-4


def test_streamlines_are_tangent(sol):
    fi = FieldInterpolator(sol, dxi=0.002, fast=True)
    tl = trace(fi.velocity, seeds_sphere((0.3, 0.2, 0.4), 0.15, 20), Domain())
    assert len(tl.lines) == 20
    ref = FieldInterpolator(sol)
    for L in tl.lines:
        d = np.diff(L, axis=0)
        v = ref.velocity(0.5 * (L[1:] + L[:-1]))
        c = np.sum(d * v, 1) / np.linalg.norm(d, axis=1) / np.linalg.norm(v, axis=1)
        assert np.nanmin(c) > 1 - 1e-5


def test_stop_reasons_and_exclusions(sol):
    dom = Domain(R=1.0, r_min=0.2, axis_a=0.05)
    fi = FieldInterpolator(sol, dxi=0.002, fast=True)
    tl = trace(fi.velocity, seeds_sphere((0.0, 0.3, 0.0), 0.25, 40), dom)
    for L in tl.lines:
        r = np.linalg.norm(L, axis=1)
        rho = np.hypot(L[:, 0], L[:, 1])
        assert r.max() <= 1.0 + 1e-9 and r.min() >= 0.2 - 1e-9 and rho.min() >= 0.05 - 1e-9
    assert all(r for r in tl.reasons)
    # seeds outside the domain are dropped
    assert len(trace(fi.velocity, np.array([[0, 0, 0.5], [3, 0, 0]]), dom).lines) == 0


def test_no_swirl_streamlines_stay_meridional():
    """The Landau jet has u_φ = 0, so its streamlines stay in their meridional plane."""
    s = solve(landau_params(3.0))
    fi = FieldInterpolator(s, dxi=0.002, fast=True)
    tl = trace(fi.velocity, seeds_line((0.2, 0.0, 0.3), (0.6, 0.0, -0.2), 8), Domain())
    for L in tl.lines:
        assert np.max(np.abs(L[:, 1])) < 1e-9          # u_phi = 0: no motion out of y = 0


def test_self_similarity_of_streamlines(sol):
    """λX(s) of a streamline is again a streamline (u is (−1)-homogeneous)."""
    fi = FieldInterpolator(sol, dxi=0.002, fast=True)
    L = trace(fi.velocity, np.array([[0.3, 0.2, 0.4]]), Domain(R=0.8)).lines[0]
    ref = FieldInterpolator(sol)
    for M in rescaled_copies(L):
        d = np.diff(M, axis=0)
        v = ref.velocity(0.5 * (M[1:] + M[:-1]))
        c = np.sum(d * v, 1) / np.linalg.norm(d, axis=1) / np.linalg.norm(v, axis=1)
        assert np.nanmin(c) > 1 - 1e-5


def test_seed_generators():
    assert seeds_sphere((0, 0, 0), 0.3, 50).shape == (50, 3)
    assert np.all(np.linalg.norm(seeds_sphere((0, 0, 0), 0.3, 50), axis=1) <= 0.3 + 1e-12)
    d = seeds_disc((0, 0, 0.5), (0, 0, 1), 0.2, 30)
    assert np.allclose(d[:, 2], 0.5) and np.all(np.hypot(d[:, 0], d[:, 1]) <= 0.2 + 1e-12)
    assert np.allclose(seeds_line((0, 0, 0), (1, 0, 0), 3)[:, 0], [0, 0.5, 1])


def test_stream_surface_quads():
    lines = [np.column_stack([np.linspace(0, 1, 10), np.full(10, y), np.zeros(10)]) for y in (0, 0.1, 0.2)]
    P, Q = stream_surface(lines, m=20)
    assert P.shape == (60, 3) and Q.shape == (2 * 19, 4)


@pytest.mark.parametrize("kw", [dict(), dict(seed_mode="line", show_surface=True),
                                dict(seed_mode="disc", show_iso=True, iso_field="helicity"),
                                dict(show_vortex=True, show_glyphs=True, show_selfsim=True, clip=True,
                                     color_by="pressure", domain_shape="box")])
def test_compute_scene_variants(sol, kw):
    data = compute_scene(sol, summarize(sol), SceneSettings(**kw))
    assert data.outline.n_points > 0 and len(data.axis) == 2
    if kw.get("show_iso"):
        assert data.iso is not None and data.iso.n_points > 0
    if kw.get("show_surface"):
        assert data.surface is not None
    assert data.timings["total"] < 3.0


def test_scene_for_nonglobal_solution():
    s = solve(Params(gamma=-1.9, kappa=1.0))       # blows up near the north pole
    data = compute_scene(s, summarize(s), SceneSettings(show_iso=True))
    assert "not reached" in data.axis[0][2] or "fails the terminal test" in data.axis[0][2]
    assert data.n_lines > 0


def test_settings_roundtrip():
    s = SceneSettings(seed_mode="line", n_seeds=17, line_p0=(0.1, 0.2, 0.3))
    assert SceneSettings.from_dict(s.to_dict()) == s


@pytest.mark.parametrize("field", ["speed", "helicity", "swirl"])
def test_isosurface_points_lie_on_the_level(sol, field):
    """The worker samples in meshgrid 'ij' order and the main thread builds a VTK grid (x fastest);
    a wrong reshape would put the isosurface in the wrong place."""
    d = compute_scene(sol, summarize(sol), SceneSettings(show_iso=True, iso_field=field, clip=False))
    val = FieldInterpolator(sol).scalar(d.iso.points, field)
    assert np.nanmedian(np.abs(val - d.iso_level)) / abs(d.iso_level) < 0.01


def test_numeric_stage_has_no_vtk_objects(sol):
    """VTK is not thread-safe: the worker stage must return NumPy data only."""
    import pyvista as pv
    from swirl_explorer.viz.scene_compute import compute_numeric
    num = compute_numeric(sol, SceneSettings(show_iso=True, show_vortex=True, show_surface=True,
                                             show_selfsim=True))
    for v in vars(num).values():
        assert not isinstance(v, pv.DataSet)
