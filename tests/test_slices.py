"""Tests of slice planes: geometry, sampled components, projected streamlines, LIC, panel sync."""
import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import solve
from swirl_explorer.core.fields import FieldInterpolator
from swirl_explorer.viz.slices import PlaneSpec, PRESETS, sample_plane, lic, projected_streamlines
from swirl_explorer.viz.streamlines import Domain


@pytest.fixture(scope="module")
def fi():
    return FieldInterpolator(solve(Params(gamma=-1.6806, kappa=1.0)), dxi=0.002, fast=True)


@pytest.mark.parametrize("az,el", [(0, 0), (90, 0), (0, 90), (0, -90), (37, 21), (-120, -45)])
def test_basis_orthonormal_right_handed(az, el):
    e1, e2, n = PlaneSpec(azimuth=az, elevation=el).basis()
    M = np.array([e1, e2, n])
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert np.allclose(np.cross(e1, e2), n, atol=1e-12)          # n points at the viewer
    if abs(el) < 89:
        assert e2[2] > 0                                         # "up" is up


def test_presets_and_roundtrip():
    sp = PlaneSpec(*PRESETS["horizontal x–y"], offset=0.4)
    assert np.allclose(sp.normal, [0, 0, 1], atol=1e-12) and np.allclose(sp.center, [0, 0, 0.4])
    for az, el, d in ((30, 20, 0.3), (-100, -60, -0.5), (90, 0, 0.0)):
        a = PlaneSpec(azimuth=az, elevation=el, offset=d)
        b = PlaneSpec.from_normal_origin(a.normal, a.center + 0.2 * a.basis()[0])
        assert np.allclose(a.normal, b.normal, atol=1e-12) and b.offset == pytest.approx(d, abs=1e-12)


def test_sampled_components_match_velocity(fi):
    sp = PlaneSpec(azimuth=37, elevation=21, offset=0.25, n=41)
    d = sample_plane(fi, sp, Domain())
    e1, e2, n = sp.basis()
    j, i = 25, 13
    X = d.X[j, i]
    assert X @ n == pytest.approx(0.25, abs=1e-12)
    u = fi.velocity(X[None])[0]
    assert d.un[j, i] == pytest.approx(u @ n, rel=1e-12)
    assert d.us[j, i] == pytest.approx(u @ e1, rel=1e-12)
    assert d.ut[j, i] == pytest.approx(u @ e2, rel=1e-12)


def test_exclusions_are_nan(fi):
    dom = Domain(r_min=0.2, axis_a=0.05)
    d = sample_plane(fi, PlaneSpec(n=101), dom)                    # meridional plane x–z through the axis
    r = np.linalg.norm(d.X, axis=-1)
    rho = np.hypot(d.X[..., 0], d.X[..., 1])
    assert np.all(np.isnan(d.un[r < 0.2])) and np.all(np.isnan(d.un[rho < 0.05]))
    assert np.all(np.isnan(d.un[r > 1.0]))
    assert np.isfinite(d.un[(r > 0.3) & (r < 0.9) & (rho > 0.1)]).all()


def test_axis_hits():
    from swirl_explorer.viz.slices import _axis_hits
    h = _axis_hits(PlaneSpec(elevation=90, offset=0.5))
    assert h[0][0] == "point" and abs(h[0][1]) < 1e-12 and abs(h[0][2]) < 1e-12
    assert _axis_hits(PlaneSpec(azimuth=90, elevation=0, offset=0.0))[0][0] == "line"
    assert _axis_hits(PlaneSpec(azimuth=90, elevation=0, offset=0.3)) == []


def test_projected_streamlines_follow_tangential_field(fi):
    sp = PlaneSpec(elevation=90, offset=0.4, n=121)
    d = sample_plane(fi, sp, Domain())
    lines = projected_streamlines(d, density=1.0)
    assert len(lines) > 10
    from scipy.interpolate import RegularGridInterpolator
    fs = RegularGridInterpolator((d.t, d.s), d.us, bounds_error=False)
    ft = RegularGridInterpolator((d.t, d.s), d.ut, bounds_error=False)
    cos = []
    for L in lines:
        seg = np.diff(L, axis=0)
        mid = 0.5 * (L[1:] + L[:-1])
        v = np.column_stack([fs(mid[:, ::-1]), ft(mid[:, ::-1])])
        ok = np.isfinite(v).all(axis=1) & (np.linalg.norm(seg, axis=1) > 0)
        c = np.sum(seg[ok] * v[ok], 1) / np.linalg.norm(seg[ok], axis=1) / np.linalg.norm(v[ok], axis=1)
        cos.extend(np.abs(c))
    assert np.median(cos) > 0.99


def test_lic_shape_and_mask(fi):
    d = sample_plane(fi, PlaneSpec(n=81), Domain())
    out = lic(d.us, d.ut, length=10)
    assert out.shape == d.un.shape
    assert np.all(np.isnan(out[~d.valid])) and np.isfinite(out[d.valid]).all()


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_slice_panel_sync_and_export(app, tmp_path):
    from swirl_explorer.ui.panels.slice_panel import SlicePanel, SliceController
    ctrl = SliceController()
    panel = SlicePanel(ctrl)
    got = []
    panel.sliceReady.connect(lambda d, l: got.append((d, l)))
    panel.set_solution(solve(Params(gamma=-1.6806, kappa=1.0)))
    t = time.time()
    while not got and time.time() - t < 15:
        app.processEvents()
        time.sleep(0.01)
    assert got and len(got[-1][1]) > 5
    # an external change (e.g. the 3D plane widget) moves the sliders
    ctrl.set_spec(PlaneSpec(azimuth=10, elevation=30, offset=0.2), source="3d")
    assert panel.az.value() == 10 and panel.el.value() == 30 and panel.off.value() == 20
    # moving a slider updates the shared spec
    panel.el.setValue(45)
    assert ctrl.spec.elevation == pytest.approx(45)
    t = time.time()
    while (not got or got[-1][0].spec.elevation != 45) and time.time() - t < 15:
        app.processEvents()
        time.sleep(0.01)
    assert got[-1][0].spec.elevation == pytest.approx(45)
    # a preset moves the sliders, and a later slider move starts from the preset
    panel._preset(0, 0)
    assert panel.az.value() == 0 and panel.el.value() == 0 and panel.off.value() == 0
    panel.off.setValue(30)
    assert ctrl.spec.azimuth == pytest.approx(0) and ctrl.spec.offset == pytest.approx(0.3)
    # typed values are exact (finer than the slider steps)
    panel.az_spin.setValue(12.34)
    assert ctrl.spec.azimuth == pytest.approx(12.34) and panel.az.value() == 12
    panel.off_spin.setValue(0.123)
    assert ctrl.spec.offset == pytest.approx(0.123) and ctrl.spec.azimuth == pytest.approx(12.34)
    out = panel.export("png", str(tmp_path / "s.png"))
    assert os.path.getsize(out) > 1000
    panel.shutdown()


def test_signed_log_display_inverts(fi):
    d = sample_plane(fi, PlaneSpec(n=61), Domain())
    d.set_display(False, symlog=True)
    ok = np.isfinite(d.un)
    assert np.allclose(d.to_value(d.show[ok]), d.un[ok], rtol=1e-10, atol=1e-12)
    assert np.nanmax(np.abs(d.show)) <= d.clim + 1e-12           # nothing saturates on the log scale
    d.set_display(True)                                            # rho-scaled, linear
    rho = np.hypot(d.X[..., 0], d.X[..., 1])
    assert np.allclose(d.show[ok], (d.un * rho)[ok]) and d.to_value(2.0) == 2.0
