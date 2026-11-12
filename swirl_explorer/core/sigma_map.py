"""Sigma-slice maps: globality and pole strata on a 2D slice of the parameter space R^6.

Pick two of the six data (c1, c2, c3, gamma, A, kappa), fix the others, and shoot
from the base point to both poles at every grid point (fast Fortran DOP853).  Each
pole is classified from its end state by the same rules as
:func:`swirl_explorer.core.poles.diagnose_pole`, without extrapolation.

The computation is chunked so that a GUI worker can report progress and cancel;
``workers > 1`` uses a process pool.
"""
from __future__ import annotations

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


def _row(args) -> tuple[int, list]:
    spec, j = args
    xs, ys = spec.axes()
    row = []
    for xv in xs:
        p = spec.params_at(xv, ys[j])
        n = quick_pole(p, +1, spec.xi_max, spec.rtol)
        s = quick_pole(p, -1, spec.xi_max, spec.rtol)
        row.append((n, s))
    return j, row


def compute_sigma_map(spec: SigmaMapSpec,
                      progress: Optional[Callable[[float], None]] = None,
                      cancelled: Optional[Callable[[], bool]] = None,
                      workers: int = 1,
                      on_start: Optional[Callable[["SigmaMapResult"], None]] = None) -> SigmaMapResult:
    """``on_start(res)`` receives the result object before any row is computed; it is filled
    in place row by row (``res.done``), so a GUI can draw partial maps while it grows."""
    if spec.x_name not in PARAM_NAMES or spec.y_name not in PARAM_NAMES:
        raise ValueError("unknown parameter name")
    shape = (spec.ny, spec.nx)
    res = SigmaMapResult(spec,
                         np.full(shape, -1, int), np.full(shape, -1, int),
                         np.full(shape, np.nan), np.full(shape, np.nan),
                         np.full(shape, np.nan), np.full(shape, np.nan),
                         np.full(shape, np.nan), np.zeros(shape, bool))

    def store(j, row):
        for i, ((cn, mn, fn, bn), (cs, ms, fs, bs)) in enumerate(row):
            res.code_n[j, i], res.mu_n[j, i], res.f_n[j, i] = cn, mn, fn
            res.code_s[j, i], res.mu_s[j, i], res.f_s[j, i] = cs, ms, fs
            res.blow_x[j, i] = bn if math.isfinite(bn) else bs
            res.done[j, i] = True

    if on_start:
        on_start(res)
    # centre rows first: the interesting part of a map is usually in the middle
    order = sorted(range(spec.ny), key=lambda j: abs(j - (spec.ny - 1) / 2))
    jobs = [(spec, j) for j in order]
    if workers <= 1:
        for k, job in enumerate(jobs):
            if cancelled and cancelled():
                break
            store(*_row(job))
            if progress:
                progress((k + 1) / spec.ny)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            from concurrent.futures import as_completed
            futs = [ex.submit(_row, job) for job in jobs]
            for k, fu in enumerate(as_completed(futs)):
                if cancelled and cancelled():
                    for f2 in futs:
                        f2.cancel()
                    break
                store(*fu.result())
                if progress:
                    progress((k + 1) / spec.ny)
    return res
