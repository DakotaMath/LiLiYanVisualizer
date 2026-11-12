"""Sigma-slice maps: globality and pole strata on a 2D slice of the parameter space R^6.

Pick two of the six data (c1, c2, c3, gamma, A, kappa), fix the others, and shoot
from the base point to both poles at every grid point.  Each pole is classified from its
end state by the same rules as :func:`swirl_explorer.core.poles.diagnose_pole`, without
extrapolation (:func:`quick_pole`).

A map integrates whole blocks of grid points at once (:mod:`swirl_explorer.core.sigma_fast`,
the same DOP853 method and tolerances as :func:`quick_pole`, ~10x faster per core).  The
computation is chunked so that a GUI worker can report progress and cancel; ``workers > 1``
uses a process pool, kept warm for later maps.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
from typing import Callable, Optional

import numpy as np

from .params import Params, PARAM_NAMES
from .solver import shoot, terminal_safe

# stratum codes
BLOWUP, UNSAFE, G, R, L1, L2 = 0, 1, 2, 3, 4, 5
CODE_NAMES = {BLOWUP: "blow-up", UNSAFE: "unsafe", G: "G", R: "R", L1: "L1", L2: "L2"}


def quick_pole(p: Params, side: int, xi_max: float, rtol: float) -> tuple[int, float, float, float]:
    """(code, mu, f_end, x_blowup) for one pole from a fast shot."""
    sh = shoot(p, side, xi_max, rtol=rtol, atol=rtol * 1e-2)
    if sh.blowup:
        return BLOWUP, math.nan, math.nan, math.tanh(sh.xi_blowup) if math.isfinite(sh.xi_blowup) else math.nan
    h, _, f, Q, _ = sh.y_end
    if not terminal_safe(h, Q, side, tol=1e-7):
        return UNSAFE, math.nan, f, math.nan
    hm = h if side > 0 else -h
    mu = math.sqrt(max(1.0 + 0.5 * Q, 0.0))
    xi_e = abs(sh.xi_end)
    if p.kappa == 0.0:
        return (G if hm >= 0 else R), mu, f, math.nan
    if mu < 0.5 / max(xi_e, 1.0) or (mu < 0.05 and hm > 3 * mu):
        return (L1 if xi_e * hm > 0.4 else L2), mu, f, math.nan
    return (G if hm > 0 else R), mu, f, math.nan


@dataclass
class SigmaMapSpec:
    base: Params
    x_name: str
    y_name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    nx: int = 81
    ny: int = 81
    xi_max: float = 14.0
    rtol: float = 1e-9

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.linspace(*self.x_range, self.nx), np.linspace(*self.y_range, self.ny))

    def params_at(self, xv: float, yv: float) -> Params:
        return self.base.with_(**{self.x_name: xv, self.y_name: yv})

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["base"] = self.base.to_dict()
        return d


@dataclass
class SigmaMapResult:
    spec: SigmaMapSpec
    code_n: np.ndarray            # (ny, nx) int
    code_s: np.ndarray
    mu_n: np.ndarray
    mu_s: np.ndarray
    f_n: np.ndarray
    f_s: np.ndarray
    blow_x: np.ndarray
    done: np.ndarray              # bool mask of computed points

    @property
    def is_global(self) -> np.ndarray:
        ok = lambda c: (c >= G)
        return ok(self.code_n) & ok(self.code_s) & self.done

    @property
    def class_z(self) -> np.ndarray:
        return self.is_global & (np.sign(self.f_n) * np.sign(self.f_s) < 0) & (self.spec_kappa() != 0)

    def spec_kappa(self) -> np.ndarray:
        X, Y = np.meshgrid(*self.spec.axes())
        s = self.spec
        if s.x_name == "kappa":
            return X
        if s.y_name == "kappa":
            return Y
        return np.full(X.shape, s.base.kappa)


def rows_poles(spec: SigmaMapSpec, rows) -> tuple[list, tuple, tuple]:
    """Both poles at every point of the given rows, all integrated together (core.sigma_fast).

    Returns (rows, north, south); north/south = (code, mu, f_end, x_blowup), each of shape
    (len(rows), nx).  Same results as quick_pole at each point, to the integration tolerance."""
    from .sigma_fast import poles_at
    rows = [int(j) for j in rows]
    xs, ys = spec.axes()
    X, Y = np.meshgrid(xs, ys[rows])
    P = {k: np.full(X.size, getattr(spec.base, k), float) for k in PARAM_NAMES}
    P[spec.x_name] = X.ravel()
    P[spec.y_name] = Y.ravel()
    n, s = poles_at(P, spec.base.xb, spec.xi_max, spec.rtol)
    sh = X.shape
    return rows, tuple(a.reshape(sh) for a in n), tuple(a.reshape(sh) for a in s)


def _rows_job(args):
    spec, rows = args
    return rows_poles(spec, rows)


# One pool for the life of the process: starting worker processes (and importing numpy/scipy
# in each) costs more than a whole small map, so later maps reuse the warm pool.
_POOL: Optional[ProcessPoolExecutor] = None
_POOL_N = 0


def _pool(workers: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_N
    if _POOL is None or _POOL_N != workers:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL, _POOL_N = ProcessPoolExecutor(max_workers=workers), workers
    return _POOL


def shutdown_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


atexit.register(shutdown_pool)

# a job of about this many points (two lanes each) keeps the batched kernel efficient
JOB_POINTS = 1024


def compute_sigma_map(spec: SigmaMapSpec,
                      progress: Optional[Callable[[float], None]] = None,
                      cancelled: Optional[Callable[[], bool]] = None,
                      workers: int = 1,
                      on_start: Optional[Callable[["SigmaMapResult"], None]] = None) -> SigmaMapResult:
    """``on_start(res)`` receives the result object before any row is computed; it is filled
    in place a few rows at a time (``res.done``), so a GUI can draw partial maps while it grows.

    Rows are integrated in blocks by the batched kernel (core.sigma_fast); ``workers > 1``
    spreads the blocks over a process pool that is kept for later maps."""
    if spec.x_name not in PARAM_NAMES or spec.y_name not in PARAM_NAMES:
        raise ValueError("unknown parameter name")
    shape = (spec.ny, spec.nx)
    res = SigmaMapResult(spec,
                         np.full(shape, -1, int), np.full(shape, -1, int),
                         np.full(shape, np.nan), np.full(shape, np.nan),
                         np.full(shape, np.nan), np.full(shape, np.nan),
                         np.full(shape, np.nan), np.zeros(shape, bool))

    def store(rows, n, s):
        res.code_n[rows], res.mu_n[rows], res.f_n[rows] = n[0], n[1], n[2]
        res.code_s[rows], res.mu_s[rows], res.f_s[rows] = s[0], s[1], s[2]
        res.blow_x[rows] = np.where(np.isfinite(n[3]), n[3], s[3])
        res.done[rows] = True

    if on_start:
        on_start(res)
    # centre rows first: the interesting part of a map is usually in the middle
    order = sorted(range(spec.ny), key=lambda j: abs(j - (spec.ny - 1) / 2))
    per = max(1, round(JOB_POINTS / spec.nx))
    if workers > 1:                     # enough jobs to keep every worker busy to the end
        per = max(1, min(per, math.ceil(spec.ny / (4 * workers))))
    else:                               # small jobs early on, so a partial map shows up soon
        per = max(1, min(per, math.ceil(spec.ny / 8)))
    jobs = [(spec, order[k:k + per]) for k in range(0, spec.ny, per)]
    done_rows = 0
    if workers <= 1:
        for job in jobs:
            if cancelled and cancelled():
                break
            store(*_rows_job(job))
            done_rows += len(job[1])
            if progress:
                progress(done_rows / spec.ny)
    else:
        from concurrent.futures import as_completed
        ex = _pool(workers)
        futs = [ex.submit(_rows_job, job) for job in jobs]
        try:
            for fu in as_completed(futs):
                if cancelled and cancelled():
                    for f2 in futs:
                        f2.cancel()
                    break
                rows, n, s = fu.result()
                store(rows, n, s)
                done_rows += len(rows)
                if progress:
                    progress(done_rows / spec.ny)
        except Exception:
            shutdown_pool()             # a broken pool is not reused
            raise
    return res
