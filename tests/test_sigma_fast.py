"""The batched Σ-map kernel (core.sigma_fast) against quick_pole, one point at a time.

Two kinds of check:
  * agreement: same stratum codes as quick_pole at rtol 1e-9, values equal to the tolerance;
  * accuracy: against quick_pole at rtol 1e-13 as the reference, the batched kernel at rtol 1e-9
    is no less accurate than quick_pole at rtol 1e-9 (up to a factor 3 either way: the two use
    different step-size controllers, so their errors differ point by point).
"""
from __future__ import annotations

import numpy as np
import pytest

from swirl_explorer.core.params import Params, PARAM_NAMES
from swirl_explorer.core.sigma_map import quick_pole, SigmaMapSpec, compute_sigma_map, rows_poles
from swirl_explorer.core import sigma_fast as F


def _points(n=300, seed=7):
    rng = np.random.default_rng(seed)
    P = {"c1": rng.uniform(-1.5, 1.5, n), "c2": rng.uniform(-1.5, 1.5, n), "c3": rng.uniform(-1, 1, n),
         "gamma": rng.uniform(-3, 3, n), "A": rng.uniform(-2, 2, n), "kappa": rng.uniform(-3, 3, n)}
    P["kappa"][: n // 10] = 0.0
    return P


def _reference(P, rtol):
    n = len(P["c1"])
    out = np.empty((n, 2, 4))
    for i in range(n):
        p = Params(*(float(P[k][i]) for k in PARAM_NAMES))
        out[i, 0] = quick_pole(p, +1, 14.0, rtol)
        out[i, 1] = quick_pole(p, -1, 14.0, rtol)
    return out


def _fast(P, rtol=1e-9):
    n, s = F.poles_at(P, 0.0, 14.0, rtol)
    return np.stack([np.stack(n, 1), np.stack(s, 1)], 1)


def _err(X, R, k):
    ok = np.isfinite(X[:, :, k]) & np.isfinite(R[:, :, k])
    assert (np.isfinite(X[:, :, k]) == np.isfinite(R[:, :, k])).all()
    return np.abs(X[:, :, k] - R[:, :, k])[ok]


@pytest.fixture(scope="module")
def sample():
    P = _points()
    return P, _reference(P, 1e-13), _reference(P, 1e-9), _fast(P)


def test_codes_agree_with_quick_pole(sample):
    P, ref, old, new = sample
    assert (new[:, :, 0] == old[:, :, 0]).all()
    assert (new[:, :, 0] == ref[:, :, 0]).all()
    assert len(np.unique(ref[:, :, 0])) >= 3            # the sample has blow-ups, unsafe and G poles


@pytest.mark.parametrize("k,name", [(1, "mu"), (2, "f_end"), (3, "x_blowup")])
def test_no_less_accurate_than_quick_pole(sample, k, name):
    P, ref, old, new = sample
    e_new, e_old = _err(new, ref, k), _err(old, ref, k)
    assert e_new.max() <= 3 * max(e_old.max(), 1e-12), name
    assert np.median(e_new) <= 3 * max(np.median(e_old), 1e-13), name


def test_lane_independence():
    """A point's result does not depend on which other points share its block."""
    P = _points(64, seed=3)
    a = _fast(P)
    one = {k: v[5:6] for k, v in P.items()}
    b = _fast(one)
    np.testing.assert_array_equal(a[5:6], b)


def test_rows_poles_matches_quick_pole_on_a_map():
    spec = SigmaMapSpec(Params(A=0.3, kappa=0.4), "gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6), nx=15, ny=11)
    rows, n, s = rows_poles(spec, range(spec.ny))
    xs, ys = spec.axes()
    for j in range(0, spec.ny, 3):
        for i in range(0, spec.nx, 2):
            p = spec.params_at(xs[i], ys[j])
            assert n[0][j, i] == quick_pole(p, +1, spec.xi_max, spec.rtol)[0]
            assert s[0][j, i] == quick_pole(p, -1, spec.xi_max, spec.rtol)[0]


def test_compute_sigma_map_partial_rows_in_centre_first_order():
    spec = SigmaMapSpec(Params(), "gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6), nx=9, ny=17)
    seen = []
    res = compute_sigma_map(spec, on_start=lambda r: None,
                            progress=lambda f: seen.append(f))
    assert res.done.all() and seen[-1] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(seen, seen[1:]))
