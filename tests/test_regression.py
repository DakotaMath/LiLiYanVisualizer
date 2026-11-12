"""Regression tests of the task brief, section 6, against the monograph.

Reference values: Example 1.2 / 2.29 (Landau), Numerical observation 4.28 and
Example 4.23 (pole types), Example 5.71 / Appendix C.2 (kappa_max = 4.1608149).
Tolerances are those of the brief; see docs/conventions.md for the normalisations.
"""
import math

import numpy as np
import pytest

from swirl_explorer.core.params import Params, landau_params
from swirl_explorer.core.solver import solve, is_global_fast
from swirl_explorer.core.poles import diagnose_pole, summarize
from swirl_explorer.core.edges import gamma_minus, kappa_max
from swirl_explorer.core.fields import FieldInterpolator


# ----------------------------------------------------------------- 1. Landau
@pytest.mark.parametrize("L", [1.5, -1.5, 3.0, -7.0, 1.05])
def test_landau_profile(L):
    sol = solve(landau_params(L, A=0.7))
    assert sol.is_global
    xi = sol.nodes()
    pr = sol.profiles(xi)
    x = pr["x"]
    U_exact = 2.0 * (1.0 - x * x) / (L + x)
    assert np.max(np.abs(pr["U"] - U_exact)) < 1e-9
    assert np.allclose(pr["f"], 0.7)            # trivial swirl is invisible in U
    assert np.allclose(pr["Q"], 0.0, atol=1e-12)


@pytest.mark.parametrize("L,glob", [(1.0, True), (-1.0, True), (1.001, True),
                                    (0.999, False), (-0.999, False), (0.5, False)])
def test_landau_globality(L, glob):
    p = landau_params(L)
    assert solve(p).is_global == glob
    assert is_global_fast(p) == glob


# ------------------------------------------------------- 2-4. Numerical obs 4.28
@pytest.fixture(scope="module")
def gm_c0():
    return gamma_minus(Params(kappa=1.0))


def test_lens_edge_value(gm_c0):
    assert gm_c0 == pytest.approx(-1.7806, abs=5e-5)


def test_obs428_interior(gm_c0):
    sol = solve(Params(gamma=gm_c0 + 0.1, kappa=1.0))
    assert sol.is_global
    d = diagnose_pole(sol, "north", s_eval=14)
    assert d.stratum == "G"
    assert d.Q_end == pytest.approx(-1.2334, abs=1e-4)
    assert d.mu == pytest.approx(0.6191, abs=1e-4)
    assert d.U_end == pytest.approx(-0.7609, abs=1e-4)


def test_obs428_edge_L1(gm_c0):
    sol = solve(Params(gamma=gm_c0 + 1e-11, kappa=1.0))
    d = diagnose_pole(sol, "north", s_eval=14)
    assert d.stratum == "L1"
    assert d.Q_end + 2 == pytest.approx(7e-5, rel=0.1)
    assert d.U_end + 2 == pytest.approx(0.323, abs=1e-3)
    # the brief's L1 law U + 2 ~ 4/s holds only to O(1/s) relative error
    assert abs((d.U_end + 2) / (4 / d.s_eval) - 1) < 0.2


def test_obs428_recessive_edge():
    p = Params(c2=-0.9, kappa=0.3)
    gm = gamma_minus(p)
    assert gm == pytest.approx(-1.0186, abs=5e-5)
    sol = solve(p.with_(gamma=gm + 1e-11))
    d = diagnose_pole(sol, "north", s_eval=14)
    assert d.stratum == "R"
    assert d.Q_end == pytest.approx(-1.9002, abs=1e-4)
    assert math.sqrt(1 + d.Q_end / 2) == pytest.approx(0.2234, abs=1e-4)
    assert d.U_end == pytest.approx(-2.4461, abs=1e-4)
    # the extrapolated exponent stays below 1/2 (Theorem 4.9)
    assert d.mu < 0.5


# ----------------------------------------------------------- 5. lens apex
@pytest.mark.slow
def test_lens_apex():
    assert kappa_max(Params(), rtol=1e-6) == pytest.approx(4.1608149, abs=2e-5)


def test_rest_state_edge():
    # gamma_-(0, 0, 0) = -2 at kappa -> 0 (Appendix C.2 validation point)
    assert gamma_minus(Params(kappa=0.0)) == pytest.approx(-2.0, abs=1e-8)


# ---------------------------------------------- 6. divergence and momentum
def _fd_jacobian(fn, X, h):
    """4th-order central differences: J[n, i, j] = d f_i / d x_j at X[n]."""
    J = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        d = (-fn(X + 2 * e) + 8 * fn(X + e) - 8 * fn(X - e) + fn(X - 2 * e)) / (12 * h)
        J.append(d)
    return np.stack(J, axis=-1)


def _fd_laplacian(fn, X, h):
    out = 0.0
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        out = out + (-fn(X + 2 * e) + 16 * fn(X + e) - 30 * fn(X) + 16 * fn(X - e)
                     - fn(X - 2 * e)) / (12 * h * h)
    return out


@pytest.mark.parametrize("params", [
    Params(gamma=-1.6806, kappa=1.0),
    Params(c1=0.3, c2=-0.2, c3=0.5, gamma=0.4, A=0.8, kappa=0.6),
    landau_params(2.5, A=1.0),
])
def test_divergence_and_momentum(params):
    sol = solve(params)
    assert sol.is_global
    fi = FieldInterpolator(sol, dxi=0.002)
    rng = np.random.default_rng(1)
    pts = []
    while len(pts) < 25:
        X = rng.normal(size=3)
        X *= rng.uniform(0.6, 2.0) / np.linalg.norm(X)
        if np.hypot(X[0], X[1]) / np.linalg.norm(X) > 0.2:   # stay away from the axis
            pts.append(X)
    X = np.array(pts)
    h = 2e-3
    J = _fd_jacobian(fi.velocity, X, h)
    div = np.trace(J, axis1=1, axis2=2)
    u = fi.velocity(X)
    lap = _fd_laplacian(fi.velocity, X, h)
    adv = np.einsum("nj,nij->ni", u, J)
    gp = _fd_jacobian(lambda Y: fi.pressure(Y)[:, None], X, h)[:, 0, :]
    res = -lap + adv + gp
    scale = np.abs(lap) + np.abs(adv) + np.abs(gp) + 1e-12
    gscale = np.abs(J).sum(axis=(1, 2))
    assert np.max(np.abs(div) / gscale) < 1e-6
    assert np.max(np.linalg.norm(res, axis=1) / np.linalg.norm(scale, axis=1)) < 1e-5


def test_ur_sign_from_divergence():
    """u_r = +U_x: with the opposite sign the divergence would be O(1)."""
    sol = solve(Params(gamma=0.3, A=0.2, kappa=0.8))
    fi = FieldInterpolator(sol)
    X = np.array([[0.5, 0.3, 0.4], [0.7, -0.2, -0.3]])
    div = np.trace(_fd_jacobian(fi.velocity, X, 1e-3), axis1=1, axis2=2)
    assert np.all(np.abs(div) < 1e-7)


def test_q_system_x_form():
    """The xi-integration satisfies the original x-form Q-system (brief section 1)."""
    sol = solve(Params(c1=0.2, c2=-0.1, c3=0.3, gamma=0.2, A=0.5, kappa=0.9))
    x = np.linspace(-0.9, 0.9, 41)
    xi = np.arctanh(x)
    pr = sol.profiles(xi)
    r1 = (1 - x ** 2) * pr["U_x"] + 2 * x * pr["U"] + 0.5 * pr["U"] ** 2 - pr["Q"]
    assert np.max(np.abs(r1)) < 1e-9
    hx = 1e-4
    def fx(xx):
        return sol.profiles(np.arctanh(xx))["f_x"]
    def Qxx(xx):
        return sol.profiles(np.arctanh(xx))["Q_xx"]
    f_xx = (fx(x + hx) - fx(x - hx)) / (2 * hx)
    r2 = (1 - x ** 2) * f_xx + pr["U"] * pr["f_x"]
    assert np.max(np.abs(r2)) < 1e-6
    Q_xxx = (Qxx(x + hx) - Qxx(x - hx)) / (2 * hx)
    r3 = (1 - x ** 2) * Q_xxx + 2 * pr["f"] * pr["f_x"]
    assert np.max(np.abs(r3)) < 1e-5
    # jet at the base point (Remark 2.6)
    p0 = sol.profiles(0.0)
    assert p0["Q"][0] == pytest.approx(0.2 - 0.1 + 0.3)
    assert p0["Q_x"][0] == pytest.approx(-0.1 - 0.2)
    assert p0["Q_xx"][0] == pytest.approx(-2 * 0.3)
    assert p0["U"][0] == pytest.approx(0.2)
    assert p0["f"][0] == pytest.approx(0.5)
    assert p0["f_x"][0] == pytest.approx(0.9)


# ----------------------------------------------------------- 7. symmetries
def test_symmetry_sigma2():
    p = Params(c1=0.4, c2=-0.3, c3=0.2, gamma=0.35, A=0.6, kappa=0.7)
    s1, s2 = solve(p), solve(p.sigma2())
    x = np.linspace(-0.95, 0.95, 39)
    a = s1.profiles(np.arctanh(x))
    b = s2.profiles(np.arctanh(-x))
    assert np.allclose(a["U"], -b["U"], atol=1e-9)
    assert np.allclose(a["f"], b["f"], atol=1e-9)
    assert np.allclose(a["Q"], b["Q"], atol=1e-9)
    n1, s_1 = diagnose_pole(s1, "north"), diagnose_pole(s1, "south")
    n2, s_2 = diagnose_pole(s2, "north"), diagnose_pole(s2, "south")
    assert n1.stratum == s_2.stratum and s_1.stratum == n2.stratum
    assert n1.mu == pytest.approx(s_2.mu, abs=1e-8)
    assert s_1.mu == pytest.approx(n2.mu, abs=1e-8)


def test_symmetry_sigma1():
    p = Params(c1=0.1, c2=0.2, c3=0.1, gamma=-0.2, A=0.4, kappa=1.1)
    s1, s2 = solve(p), solve(p.sigma1())
    xi = np.linspace(-6, 6, 31)
    a, b = s1.profiles(xi), s2.profiles(xi)
    assert np.allclose(a["U"], b["U"], atol=1e-10)
    assert np.allclose(a["f"], -b["f"], atol=1e-10)
    assert np.allclose(a["Q"], b["Q"], atol=1e-10)
    assert summarize(s1).north.stratum == summarize(s2).north.stratum
