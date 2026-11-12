"""Shooting solver for the axisymmetric (-1)-homogeneous Navier-Stokes ODE system.

We integrate the line form of the Q-system (monograph eqs. (2.27)-(2.29) and
Appendix C.2) in the variable xi = artanh x, for the state

    Y = (h, ln z, f, Q, Q')        (' = d/dxi)

    h'     = W - h^2,              W = 1 + Q/2        (Riccati form of z'' = W z)
    (ln z)' = h
    f'     = kappa_line * exp(-2 ln z)                 (f' = kappa z^-2)
    Q''    = (lambda - f^2 + 2 Q)/cosh^2(xi) - 4 tanh(xi) Q'   (first integral L[Q] + f^2 = lambda)

with U = 2 (h - tanh xi).  The third-order equation (1-x^2) Q_xxx = -2 f f_x is
replaced by its first integral, which is exact (Prop. 2.20) and lowers the order.
Initial data at xi_b = artanh(x_b), with z(xi_b) = 1:

    h = gamma/2 + x_b,  ln z = 0,  f = A,  Q = P_c(x_b),  Q' = (1 - x_b^2) P_c'(x_b).

Loss of globality is z -> 0 at a finite point, i.e. h -> -inf (forward) or
h -> +inf (backward) (Lemma 2.24).  We stop when |h| exceeds ``H_BLOW`` and
estimate the blow-up point from h ~ -1/(xi* - xi).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp, ode

from .params import Params

H_BLOW = 1.0e5          # |h| threshold for blow-up detection
DEFAULT_XI_MAX = 18.0   # s = -ln(1-x) ~ 35 at the end of the window
DEFAULT_RTOL = 1e-11
DEFAULT_ATOL = 1e-13


# --------------------------------------------------------------------------- RHS
def make_rhs(p: Params) -> Callable[[float, np.ndarray], list]:
    lam = p.lam
    kap = p.kappa_line

    cosh, tanh, exp = math.cosh, math.tanh, math.exp

    def rhs(t, y):
        # plain floats: arithmetic on numpy scalars is several times slower (same IEEE results)
        h, lz, f, Q, Qp = y.tolist() if hasattr(y, "tolist") else y
        ch = cosh(t)
        return [1.0 + 0.5 * Q - h * h,
                h,
                kap * exp(-2.0 * lz),
                Qp,
                (lam - f * f + 2.0 * Q) / (ch * ch) - 4.0 * tanh(t) * Qp]
    return rhs


def initial_state(p: Params) -> list[float]:
    Q0, Qx0, _ = p.jet()
    return [0.5 * p.gamma + p.xb, 0.0, p.A, Q0, (1.0 - p.xb ** 2) * Qx0]


def xi_from_s(s: float) -> float:
    """Inverse of s = -ln(1 - tanh xi) (north pole); use -xi for the south pole."""
    return 0.5 * math.log(2.0 * math.exp(s) - 1.0)


def s_from_xi(xi):
    """s = -ln(1 - x) for x = tanh(xi), computed without cancellation."""
    return np.logaddexp(0.0, 2.0 * np.asarray(xi, float)) - math.log(2.0)


# ------------------------------------------------------------------ terminal test
def terminal_safe(h_end: float, Q_end: float, side: int, tol: float = 0.0) -> bool:
    """Book's terminal test (Appendix C.2): safe iff D >= 0 and h_side > -sqrt(D).

    ``side`` = +1 (north, xi -> +inf) or -1 (south); at the south pole the test is
    applied to the mirrored Riccati variable -h.
    """
    D = 1.0 + 0.5 * Q_end
    if D < -tol:
        return False
    hm = h_end if side > 0 else -h_end
    return hm > -math.sqrt(max(D, 0.0)) - tol


# ------------------------------------------------------------------ fast shooting
@dataclass
class ShotResult:
    """Outcome of a one-sided shot (no dense output)."""
    side: int
    xi_end: float
    y_end: np.ndarray
    blowup: bool
    xi_blowup: float = math.nan

    @property
    def safe(self) -> bool:
        return (not self.blowup) and terminal_safe(self.y_end[0], self.y_end[3], self.side, 1e-9)


def shoot(p: Params, side: int, xi_max: float = DEFAULT_XI_MAX,
          rtol: float = 1e-10, atol: float = 1e-12) -> ShotResult:
    """Integrate from the base point to ``side*xi_max`` with Fortran DOP853 (fast path).

    Used by bisection, Sigma-maps and path scans.  ~1-2 ms per side.
    """
    rhs = make_rhs(p)
    xi0 = p.xi_b
    target = side * xi_max
    last = {"t": xi0, "y": np.array(initial_state(p)), "blow": False}

    isfinite = math.isfinite

    def solout(t, y):
        last["t"] = t
        last["y"] = y.copy()
        v = y.tolist()
        if abs(v[0]) > H_BLOW or not all(map(isfinite, v)):
            last["blow"] = True
            return -1
        return 0

    r = ode(rhs).set_integrator("dop853", rtol=rtol, atol=atol, nsteps=200000)
    r.set_solout(solout)
    r.set_initial_value(initial_state(p), xi0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.integrate(target)
    blow = last["blow"] or (not r.successful() and abs(last["t"] - target) > 1e-9)
    y = last["y"]
    xib = last["t"] - 1.0 / y[0] if blow and np.isfinite(y[0]) and y[0] != 0 else math.nan
    return ShotResult(side, float(last["t"]), y, bool(blow), float(xib))


def is_global_fast(p: Params, xi_max: float = DEFAULT_XI_MAX) -> bool:
    return shoot(p, +1, xi_max).safe and shoot(p, -1, xi_max).safe


# ------------------------------------------------------------------- full solve
def _blow_event(t, y):
    return H_BLOW - abs(y[0])


_blow_event.terminal = True  # type: ignore[attr-defined]


class Solution:
    """A solution of the Q-system with dense output on [xi_lo, xi_hi].

    All profile functions take xi (scalar or array) and return numpy arrays.
    Velocity profiles are those of u = r^{-1}(u_r e_r + u_theta e_theta + u_phi e_phi).
    """

    def __init__(self, p: Params, fwd, bwd, xi_max: float):
        self.params = p
        self.xi_max = xi_max
        self._fwd = fwd
        self._bwd = bwd
        self.xi_b = p.xi_b
        self.kappa_line = p.kappa_line
        self.lam = p.lam
        self.xi_hi = float(fwd.t[-1])
        self.xi_lo = float(bwd.t[-1])
        self.blowup_north = fwd.status == 1
        self.blowup_south = bwd.status == 1

        def _est(res):
            h = res.y[0, -1]
            return float(res.t[-1] - 1.0 / h) if h != 0 else math.nan
        self.xi_blowup_north = _est(fwd) if self.blowup_north else math.nan
        self.xi_blowup_south = _est(bwd) if self.blowup_south else math.nan

    # ------------------------------------------------------------ state access
    @property
    def blowup(self) -> bool:
        return self.blowup_north or self.blowup_south

    @property
    def x_blowup_north(self) -> float:
        return math.tanh(self.xi_blowup_north) if self.blowup_north else math.nan

    @property
    def x_blowup_south(self) -> float:
        return math.tanh(self.xi_blowup_south) if self.blowup_south else math.nan

    @property
    def is_global(self) -> bool:
        """No blow-up in the window and the terminal test passes at both ends."""
        if self.blowup:
            return False
        yN = self.state(self.xi_hi)
        yS = self.state(self.xi_lo)
        return terminal_safe(yN[0], yN[3], +1, 1e-9) and terminal_safe(yS[0], yS[3], -1, 1e-9)

    def state(self, xi) -> np.ndarray:
        """State (h, ln z, f, Q, Q') at xi; shape (5,) or (5, N). NaN outside the window."""
        xi = np.asarray(xi, dtype=float)
        scalar = xi.ndim == 0
        xi1 = np.atleast_1d(xi)
        out = np.full((5, xi1.size), np.nan)
        m_f = (xi1 >= self.xi_b) & (xi1 <= self.xi_hi)
        m_b = (xi1 < self.xi_b) & (xi1 >= self.xi_lo)
        if m_f.any():
            out[:, m_f] = self._fwd.sol(xi1[m_f])
        if m_b.any():
            out[:, m_b] = self._bwd.sol(xi1[m_b])
        return out[:, 0] if scalar else out

    def nodes(self) -> np.ndarray:
        """Integrator step points, sorted (south to north)."""
        return np.concatenate([self._bwd.t[::-1], self._fwd.t[1:]])

    def grid(self, n: int = 2001, margin: float = 0.0) -> np.ndarray:
        return np.linspace(self.xi_lo + margin, self.xi_hi - margin, n)

    # ----------------------------------------------------------------- profiles
    def profiles(self, xi) -> dict[str, np.ndarray]:
        """All scalar profiles at xi.  Keys:

        geometry : xi, x, theta, sin (= sin theta = sech xi), s_north, s_south, one_m_x, one_p_x
        unknowns : h, z, U, f, Q, W, Qp (= dQ/dxi)
        x-derivs : U_x, U_xx, f_x, Q_x, Q_xx
        velocity : u_r, u_theta, u_phi                  (profiles, u = r^-1 (...))
        vorticity: w_r, w_theta, w_phi                  (omega = r^-2 (...))
        helicity : hel                                  (u . omega = r^-3 hel)
        pressure : p                                    (p = r^-2 p(theta), eq. (2.5))
        """
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        h, lz, f, Q, Qp = self.state(xi)
        th = np.tanh(xi)
        ch = np.cosh(xi)
        sech = 1.0 / ch
        ch2 = ch * ch
        one_m_x = 2.0 / (1.0 + np.exp(np.clip(2.0 * xi, -700, 700)))
        one_p_x = 2.0 / (1.0 + np.exp(np.clip(-2.0 * xi, -700, 700)))
        omx2 = sech * sech                      # 1 - x^2
        W = 1.0 + 0.5 * Q
        U = 2.0 * (h - th)
        U_xi = 2.0 * (W - h * h - omx2)         # dU/dxi
        U_x = U_xi * ch2                        # = u_r (div u = 0, eq. (2.11))
        Q_x = Qp * ch2
        f_xi = self.kappa_line * np.exp(-2.0 * lz)
        f_x = f_xi * ch2
        U_xx = (Q_x - 2.0 * U - U * U_x) / omx2                 # from (2.14)
        Q_xx = (self.lam - f * f + 2.0 * Q - 2.0 * th * Q_x) / omx2   # first integral
        p = -0.5 * Q_xx + U_x - (U * U + f * f) / (2.0 * omx2)      # (2.15)
        u_theta = U / sech
        u_phi = f / sech
        return dict(
            xi=xi, x=th, theta=2.0 * np.arctan(np.exp(-xi)), sin=sech,
            s_north=s_from_xi(xi), s_south=s_from_xi(-xi),
            one_m_x=one_m_x, one_p_x=one_p_x,
            h=h, z=np.exp(lz), lnz=lz, U=U, f=f, Q=Q, W=W, Qp=Qp,
            U_x=U_x, U_xx=U_xx, f_x=f_x, Q_x=Q_x, Q_xx=Q_xx,
            u_r=U_x, u_theta=u_theta, u_phi=u_phi,
            w_r=-f_x, w_theta=np.zeros_like(xi), w_phi=sech * U_xx,
            hel=-U_x * f_x + f * U_xx,
            p=p,
        )

    def evaluate_velocity(self, X) -> np.ndarray:
        """Cartesian velocity at points X (N, 3); see :mod:`swirl_explorer.core.fields`."""
        from .fields import evaluate_velocity
        return evaluate_velocity(self, X)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Solution({self.params}, xi in [{self.xi_lo:.3f}, {self.xi_hi:.3f}], "
                f"global={self.is_global})")


@lru_cache(maxsize=256)
def _solve_cached(key: tuple, xi_max: float, rtol: float, atol: float) -> Solution:
    p = Params(*key[:6], xb=key[6])
    rhs = make_rhs(p)
    y0 = initial_state(p)
    xi0 = p.xi_b
    kw = dict(method="DOP853", rtol=rtol, atol=atol, events=_blow_event, dense_output=True)
    fwd = solve_ivp(rhs, (xi0, xi_max), y0, **kw)
    bwd = solve_ivp(rhs, (xi0, -xi_max), y0, **kw)
    return Solution(p, fwd, bwd, xi_max)


def solve(params: Params | tuple, base_point: Optional[float] = None,
          xi_max: float = DEFAULT_XI_MAX, rtol: float = DEFAULT_RTOL,
          atol: float = DEFAULT_ATOL) -> Solution:
    """Solve the Q-system from the Cauchy data ``params`` toward both poles.

    ``params`` may be a :class:`Params` or a 6-tuple (c1, c2, c3, gamma, A, kappa);
    ``base_point`` overrides ``params.xb``.  Results are cached by parameters.
    """
    if not isinstance(params, Params):
        params = Params.from_tuple(params, xb=base_point or 0.0)
    elif base_point is not None:
        params = Params(*params.as_tuple(), xb=float(base_point))
    if not -1.0 < params.xb < 1.0:
        raise ValueError("base point must lie in (-1, 1)")
    xi_max = max(float(xi_max), abs(params.xi_b) + 1.0)
    return _solve_cached(params.key(), xi_max, float(rtol), float(atol))
