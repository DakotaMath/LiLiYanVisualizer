"""Streamlines and vortex lines of the (−1)-homogeneous field (no GUI imports).

Lines are traced with an adaptive Dormand–Prince 5(4) method (RK45), vectorised over all
seeds, in the arclength parametrisation dX/ds = v/|v| (same curves as dX/dt = v, but the
step is not governed by the 1/r and 1/d singularities). A line stops when it

* leaves the domain (ball |X| ≤ R or box |X|∞ ≤ R),
* enters the small ball |X| < r_min around the origin,
* enters the thin cylinder ρ = √(X₁²+X₂²) < a around the axis,
* reaches a point where the field is not defined (blow-up region) or vanishes,
* exceeds ``max_steps`` or ``max_length``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.fields import FieldInterpolator

# Dormand–Prince 5(4) tableau
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

STOP_REASONS = {0: "running", 1: "left domain", 2: "origin", 3: "axis", 4: "undefined field",
                5: "max steps", 6: "max length", 7: "zero speed"}


@dataclass
class Domain:
    R: float = 1.0            # outer radius (ball) or half-width (box)
    r_min: float = 0.12       # excluded ball around the origin
    axis_a: float = 0.025     # excluded cylinder around the axis
    shape: str = "ball"       # "ball" | "box"

    def inside(self, X: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(X, axis=1)
        rho = np.hypot(X[:, 0], X[:, 1])
        outer = r <= self.R if self.shape == "ball" else np.max(np.abs(X), axis=1) <= self.R
        return outer & (r >= self.r_min) & (rho >= self.axis_a)

    def reason(self, X: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(X, axis=1)
        rho = np.hypot(X[:, 0], X[:, 1])
        outer = r <= self.R if self.shape == "ball" else np.max(np.abs(X), axis=1) <= self.R
        out = np.zeros(len(X), int)
        out[~outer] = 1
        out[(r < self.r_min) & (out == 0)] = 2
        out[(rho < self.axis_a) & (out == 0)] = 3
        return out


@dataclass
class TracedLines:
    lines: list[np.ndarray] = field(default_factory=list)      # each (n_i, 3)
    reasons: list[str] = field(default_factory=list)
    seeds: Optional[np.ndarray] = None

    @property
    def n_points(self) -> int:
        return int(sum(len(l) for l in self.lines))


def _direction(fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, sign: float):
    v = fn(X)
    s = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sign * v / s[:, None]
    return d, s


def trace(fn: Callable[[np.ndarray], np.ndarray], seeds: np.ndarray, domain: Domain,
          direction: str = "both", max_steps: int = 1500, max_length: float = 20.0,
          h0: float = 0.01, h_max: Optional[float] = None, tol: float = 1e-5,
          min_points: int = 2) -> TracedLines:
    """Trace integral curves of the vector field ``fn`` from ``seeds`` (N, 3)."""
    seeds = np.asarray(seeds, float).reshape(-1, 3)
    h_max = h_max if h_max is not None else 0.03 * domain.R
    keep = domain.inside(seeds)
    seeds = seeds[keep]
    halves = {}
    signs = {"forward": [1.0], "backward": [-1.0], "both": [1.0, -1.0]}[direction]
    for sign in signs:
        halves[sign] = _trace_one_way(fn, seeds, domain, sign, max_steps, max_length, h0, h_max, tol)
    out = TracedLines(seeds=seeds)
    for i in range(len(seeds)):
        parts = []
        reasons = []
        if -1.0 in halves:
            pts, rs = halves[-1.0][0][i], halves[-1.0][1][i]
            parts.append(pts[::-1])
            reasons.append(rs)
        if 1.0 in halves:
            pts, rs = halves[1.0][0][i], halves[1.0][1][i]
            parts.append(pts[1:] if parts else pts)
            reasons.append(rs)
        line = np.vstack(parts) if parts else np.empty((0, 3))
        if len(line) >= min_points:
            out.lines.append(line)
            out.reasons.append(" / ".join(STOP_REASONS[r] for r in reasons))
    return out


def _trace_one_way(fn, seeds, domain, sign, max_steps, max_length, h0, h_max, tol):
    n = len(seeds)
    X = seeds.copy()
    h = np.full(n, h0)
    length = np.zeros(n)
    alive = np.ones(n, bool)
    reason = np.zeros(n, int)
    paths = [[x.copy()] for x in X]
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        Xa, ha = X[idx], h[idx]
        K = []
        k1, s1 = _direction(fn, Xa, sign)
        bad = ~np.isfinite(k1).all(axis=1) | (s1 <= 1e-14)
        K.append(np.nan_to_num(k1))
        for j in range(1, 7):
            Y = Xa + ha[:, None] * sum(a * K[m] for m, a in enumerate(_A[j]))
            kj, _ = _direction(fn, Y, sign)
            K.append(np.nan_to_num(kj))
        X5 = Xa + ha[:, None] * sum(b * k for b, k in zip(_B5, K))
        X4 = Xa + ha[:, None] * sum(b * k for b, k in zip(_B4, K))
        err = np.linalg.norm(X5 - X4, axis=1) / (tol * (1.0 + np.linalg.norm(X5, axis=1)))
        accept = (err <= 1.0) & ~bad
        fac = np.clip(0.9 * np.where(err > 0, err, 1e-10) ** (-0.2), 0.2, 4.0)
        h[idx] = np.minimum(ha * fac, h_max)
        # field undefined or zero at the start point: stop
        for i_loc in np.flatnonzero(bad):
            g = idx[i_loc]
            alive[g] = False
            reason[g] = 7 if np.isfinite(k1[i_loc]).all() else 4
        acc = idx[accept]
        if acc.size:
            Xn = X5[accept]
            r_new = domain.reason(Xn)
            length[acc] += ha[accept]
            for j, g in enumerate(acc):
                if r_new[j]:
                    alive[g] = False
                    reason[g] = r_new[j]
                    continue
                X[g] = Xn[j]
                paths[g].append(Xn[j].copy())
                if length[g] >= max_length:
                    alive[g] = False
                    reason[g] = 6
        # steps that became absurdly small mean we are at a singularity: stop
        tiny = alive & (h < 1e-7)
        reason[tiny] = 4
        alive[tiny] = False
    reason[alive] = 5
    return [np.array(p) for p in paths], list(reason)


# ----------------------------------------------------------------- seeds
def seeds_sphere(center, radius: float, n: int) -> np.ndarray:
    """Quasi-uniform points inside a ball (Fibonacci shells), deterministic."""
    center = np.asarray(center, float)
    if n <= 1:
        return center[None, :]
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    rr = radius * (k / n) ** (1 / 3)
    pts = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
    return center + rr[:, None] * pts


def seeds_ring(through, n: int) -> np.ndarray:
    """n points on the circle around the axis through the point ``through`` (axisymmetric seeding)."""
    x, y, z = map(float, through)
    rho = max(np.hypot(x, y), 1e-3)
    ph = np.arctan2(y, x) + 2 * np.pi * np.arange(n) / max(n, 1)
    return np.column_stack([rho * np.cos(ph), rho * np.sin(ph), np.full(n, z)])


def seeds_rings(rho: float, n_per_ring: int, heights=(0.6, 0.3, 0.0, -0.3, -0.6)) -> np.ndarray:
    """Rings around the axis at several heights (covers both hemispheres)."""
    out = []
    for k, z in enumerate(heights):
        ph = 2 * np.pi * (np.arange(n_per_ring) + 0.5 * (k % 2)) / n_per_ring
        out.append(np.column_stack([rho * np.cos(ph), rho * np.sin(ph), np.full(n_per_ring, z)]))
    return np.vstack(out)


def seeds_line(p0, p1, n: int) -> np.ndarray:
    t = np.linspace(0, 1, max(n, 1))
    return np.asarray(p0, float) + t[:, None] * (np.asarray(p1, float) - np.asarray(p0, float))


def seeds_disc(center, normal, radius: float, n: int) -> np.ndarray:
    """Sunflower points on a disc."""
    center = np.asarray(center, float)
    nrm = np.asarray(normal, float)
    nrm = nrm / np.linalg.norm(nrm)
    a = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(nrm, a)
    k = np.arange(n) + 0.5
    rr = radius * np.sqrt(k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return center + rr[:, None] * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)


# ----------------------------------------------------------------- stream surface
def stream_surface(lines: list[np.ndarray], m: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Ruled surface through consecutive streamlines (resampled to m points by arclength fraction).

    Returns (points (k*m, 3), quads (n_quads, 4)).
    """
    res = []
    for L in lines:
        if len(L) < 2:
            continue
        seg = np.linalg.norm(np.diff(L, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        if s[-1] <= 0:
            continue
        t = np.linspace(0, s[-1], m)
        res.append(np.column_stack([np.interp(t, s, L[:, j]) for j in range(3)]))
    if len(res) < 2:
        return np.empty((0, 3)), np.empty((0, 4), int)
    P = np.vstack(res)
    quads = []
    for i in range(len(res) - 1):
        for j in range(m - 1):
            a = i * m + j
            quads.append([a, a + 1, a + m + 1, a + m])
    return P, np.array(quads, int)


def rescaled_copies(line: np.ndarray, factors=(0.6, 0.8, 1.25)) -> list[np.ndarray]:
    """Self-similarity: if X(s) is a streamline of a (−1)-homogeneous field, so is λX(s)."""
    return [lam * line for lam in factors]
