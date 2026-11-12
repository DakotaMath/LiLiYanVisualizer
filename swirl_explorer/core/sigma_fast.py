"""Batched shooting for Σ-maps: many parameter points integrated at once with numpy.

:func:`swirl_explorer.core.sigma_map.quick_pole` shoots one pole of one point with scipy's
Fortran DOP853, which calls a Python right-hand side ~650 times per shot; that call overhead,
not the arithmetic, is what a Σ-map pays for.  Here every (grid point, pole) is one *lane* of
an explicit DOP853 written with numpy arrays: each lane keeps its own position, step size and
stopping rule, so lanes step independently; numpy does the per-stage arithmetic for all live
lanes in one call.  The method is the same Dormand–Prince 8(5,3) pair (coefficients and error
norm of ``scipy.integrate.DOP853``), with the same tolerances, the same blow-up test
(|h| > ``H_BLOW`` or a non-finite state) and the same pole classification as ``quick_pole``.

Results agree with ``quick_pole`` to the integration tolerance; the stratum codes agree except,
in principle, at a point whose pole sits exactly on a stratum boundary (none was seen in the
tests: tests/test_sigma_fast.py).  Pure numpy, so it also runs in Pyodide (the web version).
"""
from __future__ import annotations

import math

import numpy as np

from .solver import H_BLOW  # noqa: F401  (the single-shot threshold, 1e5)

# Blow-up threshold for map lanes.  Past |h| = 1e4 the Riccati equation h' = W - h^2 is
# h' = -h^2 (1 + O(W/h^2)), so the lane is certain to blow up and the estimate
# xi* = xi - 1/h is off by O(W/|h|^3) ~ 1e-12, far below the integration error (~1e-10);
# stopping here instead of at 1e5 saves ~10-15% of a map with the same codes.
MAP_H_BLOW = 1.0e4

# --------------------------------------------------------------- DOP853 tableau (Hairer)
# the same numbers as scipy.integrate._ivp.dop853_coefficients (12 stages + FSAL)
from scipy.integrate._ivp import dop853_coefficients as _C

NS = _C.N_STAGES                                   # 12
_A = np.array(_C.A[:NS, :NS], float)
_B = np.array(_C.B, float)
_CN = np.array(_C.C[:NS], float)
_E3 = np.array(_C.E3, float)                       # length NS + 1
_E5 = np.array(_C.E5, float)

SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10.0
ERR_EXP = -1.0 / 8.0
BLOCK = 2048                                       # lanes per block: keeps the stage arrays in cache
MAX_STEPS = 50000

# stratum codes (the same as core.sigma_map)
BLOWUP, UNSAFE, G, R, L1, L2 = 0, 1, 2, 3, 4, 5


def _rhs_into(out, t, Y, lam, kap):
    """out[:] = F(t, Y) for the line form of the Q-system (core.solver.make_rhs), lane-wise.

    sech^2 and tanh come from one exp(-2|t|), which is accurate for every t."""
    e = np.exp(-2.0 * np.abs(t))
    ip = 1.0 / (1.0 + e)
    th = np.copysign((1.0 - e) * ip, t)
    s2 = 4.0 * e * ip * ip
    h, lz, f, Q, Qp = Y
    np.multiply(h, h, out=out[0]); np.subtract(1.0 + 0.5 * Q, out[0], out=out[0])
    out[1] = h
    np.multiply(kap, np.exp(-2.0 * lz), out=out[2])
    out[3] = Qp
    out[4] = (lam - f * f + 2.0 * Q) * s2 - 4.0 * th * Qp


def _initial_step(t, y, f0, d, lam, kap, rtol, atol, span):
    """Hairer's starting step (scipy.integrate._ivp.common.select_initial_step, order 7), lane-wise."""
    sc = atol + np.abs(y) * rtol
    d0 = np.sqrt(np.mean((y / sc) ** 2, 0))
    d1 = np.sqrt(np.mean((f0 / sc) ** 2, 0))
    h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / np.where(d1 > 0, d1, 1.0))
    h0 = np.minimum(h0, span)
    f1 = np.empty_like(f0)
    _rhs_into(f1, t + h0 * d, y + h0 * d * f0, lam, kap)
    d2 = np.sqrt(np.mean(((f1 - f0) / sc) ** 2, 0)) / h0
    m = np.maximum(d1, d2)
    h1 = np.where(m <= 1e-15, np.maximum(1e-6, h0 * 1e-3), (0.01 / np.where(m > 0, m, 1.0)) ** (1.0 / 8.0))
    return np.minimum(np.minimum(100 * h0, h1), span)


def _shoot_block(y0, t0, target, lam, kap, rtol, atol, h_blow):
    """Integrate lanes y0 (5, n) from t0 (n,) to target (n,).  Returns (t_end, y_end, blow)."""
    n = y0.shape[1]
    y = np.array(y0, float)
    t = np.array(t0, float)
    d = np.sign(target - t)
    d[d == 0] = 1.0
    blow = np.zeros(n, bool)
    active = np.abs(target - t) > 0
    fcur = np.empty_like(y)
    _rhs_into(fcur, t, y, lam, kap)
    hs = _initial_step(t, y, fcur, d, lam, kap, rtol, atol, np.abs(target - t))
    K = np.empty((NS + 1, 5, n))
    for _ in range(MAX_STEPS):
        idx = np.flatnonzero(active)
        m = idx.size
        if m == 0:
            break
        full = m == n
        ta = t if full else t[idx]
        ya = y if full else y[:, idx]
        la = lam if full else lam[idx]
        ka = kap if full else kap[idx]
        rem = np.abs((target if full else target[idx]) - ta)
        hh = np.minimum(hs if full else hs[idx], rem) * (d if full else d[idx])
        Ka = K[:, :, :m]
        Ka[0] = fcur if full else fcur[:, idx]
        for s in range(1, NS):
            a = _A[s, :s]
            dy = a[0] * Ka[0]
            for j in range(1, s):
                if a[j] != 0.0:
                    dy += a[j] * Ka[j]
            dy *= hh
            dy += ya
            _rhs_into(Ka[s], ta + _CN[s] * hh, dy, la, ka)
        acc_y = _B[0] * Ka[0]
        for j in range(1, NS):
            if _B[j] != 0.0:
                acc_y += _B[j] * Ka[j]
        yn = ya + hh * acc_y
        tn = ta + hh
        _rhs_into(Ka[NS], tn, yn, la, ka)
        scale = atol + np.maximum(np.abs(ya), np.abs(yn)) * rtol
        e5 = _E5[0] * Ka[0]
        e3 = _E3[0] * Ka[0]
        for j in range(1, NS + 1):
            if _E5[j] != 0.0:
                e5 += _E5[j] * Ka[j]
            if _E3[j] != 0.0:
                e3 += _E3[j] * Ka[j]
        e5 /= scale; e3 /= scale
        e5n = np.einsum("ij,ij->j", e5, e5)
        e3n = np.einsum("ij,ij->j", e3, e3)
        den = e5n + 0.01 * e3n
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            err = np.where(den > 0, np.abs(hh) * e5n / np.sqrt(np.where(den > 0, den, 1.0) * 5.0), 0.0)
            ok = np.isfinite(err)
            acc = ok & (err <= 1.0)
            fac = np.where(err == 0, MAX_FACTOR,
                           np.clip(SAFETY * np.where(err > 0, err, 1.0) ** ERR_EXP, MIN_FACTOR, MAX_FACTOR))
        fac = np.where(acc, fac, np.minimum(fac, 1.0))
        fac[~ok] = MIN_FACTOR
        new_h = np.abs(hh) * fac
        # accepted lanes move on
        ia = idx[acc]
        t[ia] = tn[acc]
        y[:, ia] = yn[:, acc]
        fcur[:, ia] = Ka[NS][:, acc]
        hs[idx] = new_h
        ya_ = y[:, ia]
        with np.errstate(invalid="ignore"):
            b = (np.abs(ya_[0]) > h_blow) | ~np.isfinite(ya_).all(0)
        blow[ia[b]] = True
        fin = b | (np.abs(target[ia] - t[ia]) <= 1e-12 * np.maximum(1.0, np.abs(target[ia])))
        active[ia[fin]] = False
        # a collapsing step is an integration failure: the Fortran code reports it, quick_pole
        # then calls it a blow-up (r.successful() is False short of the target)
        # (only lanes still running: the last step to the target may legitimately be tiny)
        tiny = active[idx] & (new_h < 1e-14 * np.maximum(1.0, np.abs(t[idx])))
        if tiny.any():
            blow[idx[tiny]] = True
            active[idx[tiny]] = False
    blow |= active
    return t, y, blow


def shoot_lanes(y0, t0, target, lam, kap, rtol=1e-9, atol=None, h_blow=MAP_H_BLOW, block=BLOCK):
    """Integrate every lane of y0 (5, n) from t0 to target[lane]; blocks of ``block`` lanes."""
    atol = rtol * 1e-2 if atol is None else atol
    y0 = np.asarray(y0, float)
    n = y0.shape[1]
    t0 = np.broadcast_to(np.asarray(t0, float), (n,))
    target = np.broadcast_to(np.asarray(target, float), (n,))
    lam = np.broadcast_to(np.asarray(lam, float), (n,))
    kap = np.broadcast_to(np.asarray(kap, float), (n,))
    T = np.empty(n); Y = np.empty((5, n)); BL = np.empty(n, bool)
    for a in range(0, n, block):
        s = slice(a, min(a + block, n))
        T[s], Y[:, s], BL[s] = _shoot_block(y0[:, s], t0[s], target[s], lam[s], kap[s], rtol, atol, h_blow)
    return T, Y, BL


# --------------------------------------------------------------- grid points -> lanes
def lane_data(P: dict, xb: float):
    """Initial states and constants for parameter arrays P[name] (same as solver.initial_state)."""
    c1, c2, c3, gamma, A, kappa = (np.asarray(P[k], float) for k in ("c1", "c2", "c3", "gamma", "A", "kappa"))
    lam = A ** 2 - 2 * c1 - 2 * c2 - 4 * c3
    kap = (1.0 - xb ** 2) * kappa
    Q0 = c1 * (1 - xb) + c2 * (1 + xb) + c3 * (1 - xb * xb)
    Qx0 = -c1 + c2 - 2 * c3 * xb
    y0 = np.stack([0.5 * gamma + xb, np.zeros_like(Q0), A, Q0, (1 - xb ** 2) * Qx0])
    return y0, lam, kap


def classify(side, t_end, y_end, blow, kappa):
    """quick_pole's rules, lane-wise.  Returns (code, mu, f_end, x_blowup) arrays."""
    n = t_end.size
    side = np.broadcast_to(side, (n,))
    kappa = np.broadcast_to(kappa, (n,))
    code = np.full(n, UNSAFE, int)
    mu = np.full(n, np.nan); fe = np.full(n, np.nan); bx = np.full(n, np.nan)
    h, f, Q = y_end[0], y_end[2], y_end[3]
    with np.errstate(all="ignore"):
        xib = t_end - 1.0 / h
        okb = blow & np.isfinite(h) & (h != 0) & np.isfinite(xib)
    bx[okb] = np.tanh(xib[okb])
    code[blow] = BLOWUP
    nb = ~blow
    hm = np.where(side > 0, h, -h)
    D = 1.0 + 0.5 * Q
    with np.errstate(invalid="ignore"):
        safe = nb & (D >= -1e-7) & (hm > -np.sqrt(np.maximum(D, 0.0)) - 1e-7)
    fe[nb] = f[nb]
    m = np.sqrt(np.maximum(D, 0.0))
    xi_e = np.abs(t_end)
    mu[safe] = m[safe]
    k0 = safe & (kappa == 0.0)
    code[k0] = np.where(hm[k0] >= 0, G, R)
    kn = safe & (kappa != 0.0)
    log_like = kn & ((m < 0.5 / np.maximum(xi_e, 1.0)) | ((m < 0.05) & (hm > 3 * m)))
    code[log_like] = np.where(xi_e[log_like] * hm[log_like] > 0.4, L1, L2)
    rest = kn & ~log_like
    code[rest] = np.where(hm[rest] > 0, G, R)
    return code, mu, fe, bx


def poles_at(P: dict, xb: float, xi_max: float, rtol: float, block: int = BLOCK):
    """Both poles at the points P[name] (equal-length 1-D arrays).

    Returns two tuples (code, mu, f_end, x_blowup), north then south, as quick_pole would."""
    y0, lam, kap = lane_data(P, xb)
    n = y0.shape[1]
    xi0 = math.atanh(xb)
    tgt = np.concatenate([np.full(n, xi_max), np.full(n, -xi_max)])
    t, y, bl = shoot_lanes(np.concatenate([y0, y0], 1), xi0, tgt,
                           np.concatenate([lam, lam]), np.concatenate([kap, kap]), rtol, block=block)
    side = np.concatenate([np.ones(n), -np.ones(n)])
    kk = np.asarray(P["kappa"], float)
    code, mu, fe, bx = classify(side, t, y, bl, np.concatenate([kk, kk]))
    return (code[:n], mu[:n], fe[:n], bx[:n]), (code[n:], mu[n:], fe[n:], bx[n:])
