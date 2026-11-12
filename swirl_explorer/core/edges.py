"""Bisection for the edges of Omega (Appendix C.2, method B).

* ``gamma_minus``: lower lens edge gamma_-(kappa) = smallest gamma = U(x_b) for which
  the north pole is safe (the north-pole safety is monotone in gamma).
* ``gamma_plus``: the mirror quantity at the south pole.
* ``kappa_max``: tip of the lens at a class-Z base point (A = 0), by bisection on the
  sign of gamma_+ - gamma_-.
All use the book's terminal test; without it the tips come out wrong (Atlas 5.73).
"""
from __future__ import annotations

import math

from .params import Params
from .solver import shoot, DEFAULT_XI_MAX


def north_safe(p: Params, xi_max: float = DEFAULT_XI_MAX, rtol: float = 1e-12) -> bool:
    return shoot(p, +1, xi_max, rtol=rtol, atol=1e-14).safe


def south_safe(p: Params, xi_max: float = DEFAULT_XI_MAX, rtol: float = 1e-12) -> bool:
    return shoot(p, -1, xi_max, rtol=rtol, atol=1e-14).safe


def _bisect(pred, lo: float, hi: float, tol: float) -> float:
    """pred(lo) False, pred(hi) True -> boundary."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


def gamma_minus(p: Params, lo: float = -6.0, hi: float = 6.0, tol: float = 1e-13,
                xi_max: float = DEFAULT_XI_MAX) -> float:
    """Lower edge gamma_- for the other data of ``p`` (gamma of p is ignored)."""
    pred = lambda g: north_safe(p.with_(gamma=g), xi_max)
    if pred(lo):
        return -math.inf
    if not pred(hi):
        return math.nan
    return _bisect(pred, lo, hi, tol)


def gamma_plus(p: Params, lo: float = -6.0, hi: float = 6.0, tol: float = 1e-13,
               xi_max: float = DEFAULT_XI_MAX) -> float:
    """Upper edge gamma_+: largest gamma for which the south pole is safe."""
    pred = lambda g: not south_safe(p.with_(gamma=g), xi_max)
    if not pred(hi):
        return math.inf
    if pred(lo):
        return math.nan
    return _bisect(pred, lo, hi, tol)


def kappa_max(p: Params, lo: float = 0.0, hi: float = 20.0, rtol: float = 1e-9,
              xi_max: float = DEFAULT_XI_MAX) -> float:
    """Tip of the lens: sup{kappa : gamma_-(kappa) <= gamma_+(kappa)} at the data of p."""
    def lens_nonempty(k: float) -> bool:
        q = p.with_(kappa=k)
        gm = gamma_minus(q, tol=1e-10, xi_max=xi_max)
        gp = gamma_plus(q, tol=1e-10, xi_max=xi_max)
        return math.isfinite(gm) and math.isfinite(gp) and gm <= gp
    if not lens_nonempty(max(lo, 1e-6)):
        return math.nan
    while lens_nonempty(hi):
        hi *= 2
    lo = max(lo, 1e-6)
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if lens_nonempty(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
