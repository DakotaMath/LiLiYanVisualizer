"""Unit tests of the core API: blow-up, velocity evaluation, Sigma-maps, paths, speed."""
import math
import time

import numpy as np
import pytest

from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import solve, shoot, s_from_xi, xi_from_s
from swirl_explorer.core.poles import diagnose_pole, summarize, fit_exponent
from swirl_explorer.core.fields import evaluate_velocity
from swirl_explorer.core.sigma_map import SigmaMapSpec, compute_sigma_map
from swirl_explorer.core.paths import ParamPath, scan_path


def test_s_xi_roundtrip():
    for s in (0.5, 14.0, 30.0):
        assert s_from_xi(xi_from_s(s)) == pytest.approx(s, rel=1e-12)


def test_blowup_detected_and_located():
    # Landau family with |L| < 1: U = 2(1-x^2)/(L + x) blows up at x = -L
    L = 0.4
    sol = solve(Params(gamma=2 / L))
    assert sol.blowup_south and not sol.is_global
    assert sol.x_blowup_south == pytest.approx(-L, abs=1e-4)
    sm = summarize(sol)
    assert not sm.is_global and sm.blowup_x == pytest.approx(-L, abs=1e-4)
    assert "NOT global" in sm.status_line()


def test_shoot_matches_solve():
    p = Params(gamma=-1.6806, kappa=1.0)
    a = shoot(p, +1, 12.0, rtol=1e-11, atol=1e-13)
    b = solve(p).state(12.0)
    assert np.allclose(a.y_end, b, rtol=1e-7, atol=1e-8)


def test_evaluate_velocity_homogeneity_and_nan():
    sol = solve(Params(gamma=0.3, A=0.2, kappa=0.8))
    X = np.array([[0.3, 0.4, 0.2], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1e-30, 0, 1]])
    u = evaluate_velocity(sol, X)
    assert np.all(np.isfinite(u[0]))
    assert np.all(np.isnan(u[1])) and np.all(np.isnan(u[2])) and np.all(np.isnan(u[3]))
    u2 = evaluate_velocity(sol, 2.5 * X[:1])
    assert np.allclose(u2, u[:1] / 2.5, rtol=1e-12)
    # axisymmetry: rotating X about the axis rotates u
    c, s = math.cos(0.7), math.sin(0.7)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.allclose(evaluate_velocity(sol, (R @ X[0])[None]), (R @ u[0])[None], atol=1e-12)


def test_evaluate_velocity_matches_profiles():
    sol = solve(Params(c1=0.2, gamma=-0.1, A=0.3, kappa=0.5))
    theta = 1.1
    X = np.array([[math.sin(theta), 0.0, math.cos(theta)]])
    u = evaluate_velocity(sol, X)[0]
    pr = sol.profiles(math.atanh(math.cos(theta)))
    ur, ut, up = pr["u_r"][0], pr["u_theta"][0], pr["u_phi"][0]
    er = np.array([math.sin(theta), 0, math.cos(theta)])
    et = np.array([math.cos(theta), 0, -math.sin(theta)])
    ep = np.array([0, 1.0, 0])
    assert np.allclose(u, ur * er + ut * et + up * ep, atol=1e-9)


def test_exponent_fit_on_power_law():
    d = np.logspace(-6, -1, 30)
    slope, spread = fit_exponent(d, 3 * d ** -0.37)
    assert slope == pytest.approx(-0.37, abs=1e-10)


def test_fitted_exponents_type_G():
    """For a type-G pole with 1/2 < mu < 1, u_r ~ d^(2 mu - 2) (brief's hypothesis)."""
    sol = solve(Params(gamma=-1.6806, kappa=1.0))
    d = diagnose_pole(sol, "north")
    assert d.stratum == "G"
    assert d.exponents["u_r"]["slope"] == pytest.approx(2 * d.mu - 2, abs=0.02)
    assert d.exponents["u_theta"]["slope"] == pytest.approx(-1.0, abs=0.02)
    assert d.exponents["u_phi"]["slope"] == pytest.approx(-1.0, abs=0.02)
    assert d.bp == "III"     # mu = 0.619 lies in [1/2, 2/3]


def test_solve_speed():
    p = Params(c1=0.13, c2=-0.07, gamma=0.21, A=0.1, kappa=0.77)
    solve(Params(gamma=0.11))   # warm-up
    t = time.perf_counter()
    solve(p)
    assert time.perf_counter() - t < 0.05
    t = time.perf_counter()
    solve(p)                    # cached
    assert time.perf_counter() - t < 1e-3


def test_sigma_map_lens_small():
    spec = SigmaMapSpec(Params(), "gamma", "kappa", (-2.4, 2.4), (0.05, 5.0), nx=13, ny=9)
    seen = []
    res = compute_sigma_map(spec, progress=seen.append)
    assert res.done.all() and seen[-1] == pytest.approx(1.0)
    g = res.is_global
    xs, ys = spec.axes()
    # at kappa ~ 0.05 the lens is close to the Landau window [-2, 2]
    assert g[0, np.argmin(abs(xs - 0.0))] and not g[0, 0] and not g[0, -1]
    # above the apex (kappa_max = 4.16) nothing is global
    assert not g[ys > 4.3].any()
    assert res.class_z[g].all()          # A = 0 at the base point: class Z


def test_sigma_map_cancel():
    spec = SigmaMapSpec(Params(), "gamma", "kappa", (-2, 2), (0.1, 3), nx=5, ny=20)
    calls = {"n": 0}
    def cancelled():
        calls["n"] += 1
        return calls["n"] > 3
    res = compute_sigma_map(spec, cancelled=cancelled)
    assert 0 < res.done.sum() < res.done.size


def test_path_events():
    path = ParamPath()
    path.add(0.0, Params(gamma=0.0, kappa=1.0))
    path.add(1.0, Params(gamma=-2.5, kappa=1.0))
    frames, events = scan_path(path, 26)
    assert frames[0].is_global and not frames[-1].is_global
    kinds = [e.kind for e in events]
    assert "leaves_global" in kinds
    t_leave = next(e.t for e in events if e.kind == "leaves_global")
    # the lens edge at kappa = 1 is gamma_- = -1.7806, i.e. t = 0.712
    assert 0.68 < t_leave < 0.76
    d = ParamPath.from_dict(path.to_dict())
    assert d.at(0.5)[0].gamma == pytest.approx(-1.25)
