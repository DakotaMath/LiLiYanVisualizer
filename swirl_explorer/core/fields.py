"""Cartesian fields of a solution: velocity, vorticity, helicity, pressure.

With r = |X|, rho = sqrt(X1^2 + X2^2) and the spherical frame (e_r, e_theta, e_phi):

    u     = r^-1 ( u_r e_r + u_theta e_theta + u_phi e_phi ),
            u_r = U_x,  u_theta = U / sin(theta),  u_phi = f / sin(theta)
    omega = curl u = r^-2 ( -f_x e_r + sin(theta) U_xx e_phi )
    u . omega = r^-3 ( -U_x f_x + f U_xx )
    p     = r^-2 ( -Q_xx/2 + U_x - (U^2 + f^2) / (2 (1 - x^2)) )      (eq. (2.15))

Profiles are interpolated in xi = artanh(cos theta) by cubic splines on a fine
uniform grid built from the dense output; for a point X we compute
xi = sign(X3) * ln((r + |X3|)/rho), which has no cancellation near the axis.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline

from .solver import Solution

_SPLINE_KEYS = ("U", "f", "u_r", "f_x", "U_xx", "p", "hel", "Q")


class _Linear:
    """Piecewise-linear interpolation on a uniform grid (much cheaper than a spline call)."""

    def __init__(self, x0: float, dx: float, y: np.ndarray):
        self.x0, self.dx, self.y = x0, dx, np.asarray(y, float)
        self.n = len(y)

    def __call__(self, x):
        t = (np.asarray(x, float) - self.x0) / self.dx
        i = np.clip(np.floor(t).astype(int), 0, self.n - 2)
        w = t - i
        return self.y[i] * (1.0 - w) + self.y[i + 1] * w


class FieldInterpolator:
    """Fast vectorised evaluation of the 3D fields of a :class:`Solution`.

    ``fast=True`` uses piecewise-linear interpolation (error ~ dxi^2, ~1e-6 relative at the
    default spacing): meant for streamline tracing and 3D sampling, not for the PDE checks.
    """

    def __init__(self, sol: Solution, dxi: float = 0.004, margin: float = 0.0, fast: bool = False):
        self.sol = sol
        lo, hi = sol.xi_lo + margin, sol.xi_hi - margin
        if sol.blowup_north:
            hi = min(hi, sol.xi_hi - 0.05)
        if sol.blowup_south:
            lo = max(lo, sol.xi_lo + 0.05)
        n = max(int((hi - lo) / dxi) + 1, 16)
        xi = np.linspace(lo, hi, n)
        pr = sol.profiles(xi)
        self.xi_lo, self.xi_hi = float(lo), float(hi)
        if fast:
            step = xi[1] - xi[0]
            self._sp = {k: _Linear(lo, step, pr[k]) for k in _SPLINE_KEYS}
        else:
            self._sp = {k: CubicSpline(xi, pr[k]) for k in _SPLINE_KEYS}

    # ----------------------------------------------------------------- geometry
    @staticmethod
    def geometry(X: np.ndarray):
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        rho = np.hypot(x1, x2)
        r = np.sqrt(rho * rho + x3 * x3)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.sign(x3) * np.log((r + np.abs(x3)) / rho)
            cos_t = x3 / r
            sin_t = rho / r
            cphi = x1 / rho
            sphi = x2 / rho
        return X, r, rho, xi, cos_t, sin_t, cphi, sphi

    def _valid(self, r, rho, xi):
        return (r > 0) & (rho > 0) & np.isfinite(xi) & (xi >= self.xi_lo) & (xi <= self.xi_hi)

    def profile(self, key: str, xi) -> np.ndarray:
        return self._sp[key](xi)

    # ----------------------------------------------------------------- fields
    def velocity(self, X) -> np.ndarray:
        """Cartesian velocity at points X (N,3); NaN on the axis, at 0 and outside the window."""
        X, r, rho, xi, ct, st, cp, sp = self.geometry(X)
        ok = self._valid(r, rho, xi)
        out = np.full(X.shape, np.nan)
        if not ok.any():
            return out
        xo = xi[ok]
        U, f, ur = self._sp["U"](xo), self._sp["f"](xo), self._sp["u_r"](xo)
        s, c = st[ok], ct[ok]
        ut, uph = U / s, f / s
        horiz = ur * s + ut * c
        inv_r = 1.0 / r[ok]
        out[ok, 0] = inv_r * (horiz * cp[ok] - uph * sp[ok])
        out[ok, 1] = inv_r * (horiz * sp[ok] + uph * cp[ok])
        out[ok, 2] = inv_r * (ur * c - U)        # u_r cos - u_theta sin, u_theta sin = U
        return out

    def vorticity(self, X) -> np.ndarray:
        X, r, rho, xi, ct, st, cp, sp = self.geometry(X)
        ok = self._valid(r, rho, xi)
        out = np.full(X.shape, np.nan)
        if not ok.any():
            return out
        xo = xi[ok]
        wr = -self._sp["f_x"](xo)
        wph = st[ok] * self._sp["U_xx"](xo)
        s, c, inv = st[ok], ct[ok], 1.0 / r[ok] ** 2
        out[ok, 0] = inv * (wr * s * cp[ok] - wph * sp[ok])
        out[ok, 1] = inv * (wr * s * sp[ok] + wph * cp[ok])
        out[ok, 2] = inv * (wr * c)
        return out

    def scalar(self, X, which: str) -> np.ndarray:
        """Scalar fields: 'speed', 'u_phi', 'swirl' (f/sin), 'helicity', 'pressure', 'u_r'."""
        X, r, rho, xi, ct, st, cp, sp = self.geometry(X)
        ok = self._valid(r, rho, xi)
        out = np.full(r.shape, np.nan)
        if not ok.any():
            return out
        xo, ro, so = xi[ok], r[ok], st[ok]
        if which == "speed":
            U, f, ur = self._sp["U"](xo), self._sp["f"](xo), self._sp["u_r"](xo)
            out[ok] = np.sqrt(ur ** 2 + (U / so) ** 2 + (f / so) ** 2) / ro
        elif which in ("u_phi", "swirl"):
            out[ok] = self._sp["f"](xo) / so / ro
        elif which == "u_r":
            out[ok] = self._sp["u_r"](xo) / ro
        elif which == "helicity":
            out[ok] = self._sp["hel"](xo) / ro ** 3
        elif which == "pressure":
            out[ok] = self._sp["p"](xo) / ro ** 2
        else:
            raise KeyError(which)
        return out

    def pressure(self, X) -> np.ndarray:
        return self.scalar(X, "pressure")


_cache: dict = {}


def interpolator(sol: Solution) -> FieldInterpolator:
    key = id(sol)
    fi = _cache.get(key)
    if fi is None or fi.sol is not sol:
        if len(_cache) > 32:
            _cache.clear()
        fi = _cache[key] = FieldInterpolator(sol)
    return fi


def evaluate_velocity(sol: Solution, X: Iterable) -> np.ndarray:
    """Cartesian velocity u(X) for X of shape (N, 3), using u(X) = |X|^-1 u(X/|X|).

    Returns NaN on the axis, at the origin, and where the solution does not exist.
    """
    return interpolator(sol).velocity(np.asarray(X, float))
