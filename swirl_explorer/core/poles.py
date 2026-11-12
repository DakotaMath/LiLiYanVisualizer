"""Pole diagnostics: limits, strata G/R/L1/L2, Type labels and fitted velocity exponents.

Book references: Theorem 4.5, Definition 4.16, Table 4.1, Proposition 4.19,
Corollaries 4.24-4.25.  All statements are made at the north pole x = 1
(xi -> +inf); the south pole is handled by the reflection sigma_2, i.e. by the
mirrored Riccati variable  h_m = -h  and  U_m(1) = -U(-1).

Strata (kappa != 0):
    G   q > -2,  h -> +mu,  U(1) = -2 + 2 mu,  f(1) finite
    R   q > -2,  h -> -mu,  0 < mu < 1/2,  U(1) = -2 - 2 mu,  f(1) = +-inf
    L1  q = -2,  z ~ b xi (b > 0),  U(1) = -2,  f(1) finite,  U + 2 ~ 4/s
    L2  q = -2,  z -> a,             U(1) = -2,  f(1) infinite
For kappa = 0 (no genuine swirl) only G and a recessive no-swirl branch "R0" occur.

A finite window can never certify a logarithmic stratum (mu = 0 exactly); the
classifier returns an *estimated* stratum with a confidence in [0, 1] and a note.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Optional

import numpy as np

from .solver import Solution, terminal_safe, xi_from_s

# thresholds of the velocity-label table (task brief section 1)
MU_II5 = 2.0 / 3.0
MU_III = 0.5
MU_TOL = 1e-3          # |mu - 1| below this counts as mu = 1


@dataclass
class PoleDiagnostics:
    pole: str                       # "north" | "south"
    xi_eval: float                  # where the reading was taken (|xi|)
    s_eval: float                   # s = -ln(1 -+ x) at the reading point
    Q_end: float                    # raw reading of Q at the reading point
    q: float                        # Q(+-1), extrapolated
    q_err: float
    mu: float                       # sqrt(1 + q/2) (0 if q < -2)
    h_end: float                    # mirrored Riccati variable at the reading point
    h_lim: float                    # extrapolated limit of h_m
    U_end: float                    # U(x) at the reading point
    U_lim: float                    # predicted pole value of U from the stratum
    f_end: float
    f_lim: float                    # extrapolated f(+-1) (inf for R / L2)
    f_err: float
    stratum: str                    # G | R | L1 | L2 | R0 | unsafe | blowup | ?
    confidence: float
    note: str = ""
    li_yan: str = ""                # Type 1 / 2 / 3
    bp: str = ""                    # Bradshaw-Palmer: I, II, II.5, III, beyond III
    exponents: dict = field(default_factory=dict)          # fitted
    predicted_exponents: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        if self.stratum in ("blowup", "unsafe"):
            return self.stratum
        return f"{self.stratum} (μ={self.mu:.4g}, conf {self.confidence:.2f})"


# ------------------------------------------------------------------ helpers
def _aitken(a0: float, a1: float, a2: float) -> tuple[float, float]:
    """Aitken Delta^2 limit of three equally spaced samples, with an error estimate.

    Falls back to the last value when the differences are not geometric.
    """
    d1, d2 = a1 - a0, a2 - a1
    if not all(map(math.isfinite, (a0, a1, a2))):
        return a2, math.inf
    if abs(d2) < 1e-15 * (1 + abs(a2)):
        return a2, abs(d2) + 1e-15 * (1 + abs(a2))
    r = d2 / d1 if d1 != 0 else math.inf
    if 0.0 < r < 0.95:
        lim = a2 + d2 * r / (1.0 - r)
        return lim, abs(lim - a2) * 0.5 + abs(d2) * r * r
    return a2, abs(d2) * 10.0           # slow (logarithmic) or oscillating convergence


def _mirror_profiles(sol: Solution, xi: np.ndarray, side: int) -> dict:
    """Profiles at side*xi, mirrored to the north-pole picture for side = -1."""
    pr = sol.profiles(side * np.asarray(xi, float))
    if side < 0:
        pr = dict(pr)
        pr["h"] = -pr["h"]
        pr["U"] = -pr["U"]
        pr["u_theta"] = -pr["u_theta"]
        pr["u_r"] = pr["u_r"]          # U_x is even under sigma_2
    return pr


def fit_exponent(d: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ln|v| against ln d, and its residual spread."""
    m = np.isfinite(v) & (np.abs(v) > 0) & np.isfinite(d) & (d > 0)
    if m.sum() < 4:
        return math.nan, math.nan
    X, Y = np.log(d[m]), np.log(np.abs(v[m]))
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    return float(coef[0]), float(np.std(resid))


def predicted_exponents(stratum: str, mu: float, f_lim: float, U_lim: float) -> dict:
    """Velocity rates of the task-brief table (hypotheses to be tested), as powers of d.

    Returns a dict component -> (exponent, description).  'log' rates are given as
    the leading power with a comment.
    """
    out: dict = {}
    if stratum == "G" or stratum == "R0":
        out["u_theta"] = (-1.0, "(2μ-2)/d" if stratum == "G" else "(-2-2μ)/d")
        out["u_phi"] = (-1.0, "f(1)/d")
        if stratum == "G":
            if mu > 1 + MU_TOL:
                out["u_r"] = (0.0, "O(1)")
            elif abs(mu - 1) <= MU_TOL:
                out["u_r"] = (0.0, "~ κ ln(1/d)")
            else:
                out["u_r"] = (2 * mu - 2, "d^(2μ-2)")
        else:
            out["u_r"] = (0.0, "O(1)")
    elif stratum == "R":
        out["u_phi"] = (-1.0 - 2 * mu, "d^(-1-2μ)")
        out["u_theta"] = (-1.0, "(-2-2μ)/d")
    elif stratum == "L1":
        out["u_r"] = (-2.0, "-2/(d² ln²(1/d))")
        out["u_theta"] = (-1.0, "-2/d")
        out["u_phi"] = (-1.0, "f(1)/d")
    elif stratum == "L2":
        out["u_phi"] = (-1.0, "ln(1/d)/d")
        out["u_theta"] = (-1.0, "-2/d")
    return out


def type_labels(stratum: str, mu: float, f_lim: float, fitted_ur: float = math.nan
                ) -> tuple[str, str, str]:
    """(Li-Yan type, Bradshaw-Palmer type, comment) from the brief's rate table."""
    if stratum in ("R", "L1", "L2"):
        return "3", "beyond III", "pole stratum " + stratum
    if stratum not in ("G", "R0"):
        return "?", "?", ""
    fzero = (not math.isfinite(f_lim)) or abs(f_lim) < 1e-8
    if stratum == "R0":
        return "3", "III / II.5", "no-swirl recessive branch, u_r = O(1)"
    if mu > 1 + MU_TOL:
        return "3", "II.5 (and III)", "u_r = O(1), u_θ,u_φ ~ c/d"
    if abs(mu - 1) <= MU_TOL:
        if fzero and abs(f_lim) < 1e-8:
            return "1 or 2", "I or II", "μ = 1, f(1) = 0: II iff κ ∝ Q_x(1) ≠ 0 (log u_r)"
        return "3", "II.5 (and III)", "μ = 1, u_φ ~ f(1)/d"
    if mu > MU_II5:
        return "3", "II.5 (and III)", "u_r ~ d^(2μ-2), 2μ-2 > -2/3"
    if mu >= MU_III:
        return "3", "III", "u_r ~ d^(2μ-2) with -1 ≤ 2μ-2 ≤ -2/3"
    return "3", "beyond III", "u_r ~ d^(2μ-2), 2μ-2 < -1"


# ------------------------------------------------------------------ main entry
def diagnose_pole(sol: Solution, pole: str = "north", s_eval: Optional[float] = None,
                  spacing: float = 0.75, fit_window: float = 6.0) -> PoleDiagnostics:
    """Pole limits and stratum at x = +1 ('north') or x = -1 ('south').

    ``s_eval`` reads the solution at s = -ln(1 -+ x) (as the book does, e.g. s = 14)
    instead of at the end of the integration window.
    """
    side = +1 if pole == "north" else -1
    blow = sol.blowup_north if side > 0 else sol.blowup_south
    xi_end = sol.xi_hi if side > 0 else -sol.xi_lo
    if s_eval is not None:
        xi_end = min(xi_end, xi_from_s(s_eval))
    s_end = float(np.logaddexp(0, 2 * xi_end) - math.log(2))
    nan = math.nan
    if blow and s_eval is None:
        xb = sol.xi_blowup_north if side > 0 else sol.xi_blowup_south
        return PoleDiagnostics(pole, xi_end, s_end, nan, nan, nan, nan, nan, nan, nan, nan,
                               nan, nan, nan, "blowup", 1.0,
                               note=f"z -> 0 at x = {math.tanh(xb):.6f} (xi = {xb:.4f})")

    xs = xi_end - spacing * np.arange(3)[::-1]
    pr = _mirror_profiles(sol, xs, side)
    q, q_err = _aitken(*pr["Q"])
    h_lim, h_err = _aitken(*pr["h"])
    f_lim, f_err = _aitken(*pr["f"])
    h_end = float(pr["h"][-1])
    Q_end = float(pr["Q"][-1])
    U_end = float(pr["U"][-1])
    f_end = float(pr["f"][-1])
    D = 1.0 + 0.5 * q
    mu = math.sqrt(max(D, 0.0))
    kappa0 = sol.params.kappa == 0.0

    safe = terminal_safe(h_end, Q_end, +1, tol=1e-7)
    stratum, conf, note = "?", 0.0, ""
    if not safe:
        stratum, conf = "unsafe", 1.0
        note = "terminal test fails: D < 0 or h ≤ -√D (the solution dies beyond the window)"
    else:
        # how fast is h moving?  logarithmic approach has h ~ 1/xi
        xi_eff = xi_end
        log_like = (mu < 0.5 / max(xi_eff, 1.0)) or (mu < 0.05 and h_end > 3 * mu)
        if kappa0:
            if h_end >= 0:
                stratum, conf = "G", 1.0
            else:
                stratum, conf = "R0", 1.0
                note = "no-swirl recessive branch (κ = 0)"
        elif log_like:
            xh = xi_eff * h_end
            if xh > 0.4:
                stratum = "L1"
                conf = min(1.0, max(0.0, 1.0 - mu / max(h_end, 1e-12))) * min(1.0, xh)
                note = (f"q+2 = {q + 2:.2e}; ξ·h = {xh:.3f} (→1 for L1); "
                        f"U+2 = {U_end + 2:.4f} vs 4/s = {4 / s_end:.4f}")
            else:
                stratum = "L2"
                conf = min(1.0, 1.0 - xh / 0.4)
                note = f"q+2 = {q + 2:.2e}; ξ·h = {xh:.3f} (→0 for L2); f grows like ξ"
        elif h_lim > 0:
            stratum = "G"
            dev = abs(h_lim - mu) / max(mu, 1e-12)
            conf = max(0.0, min(1.0, 1.0 - dev * 10))
            note = f"h → {h_lim:.6g}, μ = {mu:.6g}"
        else:
            stratum = "R"
            dev = abs(h_lim + mu) / max(mu, 1e-12)
            conf = max(0.0, min(1.0, 1.0 - dev * 10))
            if mu >= 0.5:
                conf = 0.0
                note = "inconsistent: recessive with μ ≥ 1/2 (Theorem 4.9 forbids it)"
            else:
                note = f"h → {h_lim:.6g} ≈ -μ = {-mu:.6g}"

    U_lim = {"G": -2 + 2 * mu, "R": -2 - 2 * mu, "R0": -2 - 2 * mu,
             "L1": -2.0, "L2": -2.0}.get(stratum, nan)
    if stratum in ("R", "L2"):
        f_lim = math.copysign(math.inf, sol.params.kappa * side)   # f monotone, sign of kappa
        f_err = 0.0

    # --------------------------------------------------- fitted exponents
    exps: dict = {}
    if stratum not in ("unsafe", "blowup", "?"):
        lo = max(1.0, xi_end - fit_window)
        xi_fit = np.linspace(lo, xi_end, 40)
        pf = _mirror_profiles(sol, xi_fit, side)
        d = pf["sin"]
        U_xi = pf["U_x"] * d * d
        ok_r = np.abs(U_xi) > 1e-9                # below this u_r is rounding noise
        ur = np.where(ok_r, pf["u_r"], np.nan)
        for name, v in (("u_r", ur), ("u_theta", pf["u_theta"]), ("u_phi", pf["u_phi"])):
            slope, spread = fit_exponent(d, v)
            exps[name] = {"slope": slope, "spread": spread,
                          "window_d": [float(d[-1]), float(d[0])]}
    pred = predicted_exponents(stratum, mu, f_lim, U_lim)
    ly, bp, comment = type_labels(stratum, mu, f_lim)
    if comment:
        note = (note + "; " if note else "") + comment
    return PoleDiagnostics(pole, float(xi_end), s_end, Q_end, float(q), float(q_err), mu, h_end,
                           float(h_lim), U_end, float(U_lim), f_end, float(f_lim),
                           float(f_err), stratum, float(conf), note, ly, bp, exps,
                           {k: v[0] for k, v in pred.items()})


# ------------------------------------------------------------------ summary
@dataclass
class Summary:
    is_global: bool
    blowup_x: Optional[float]
    cls: str                       # "Z" | "N" | "face" | "-"
    north: PoleDiagnostics
    south: PoleDiagnostics

    def status_line(self) -> str:
        if not self.is_global:
            where = f" (blow-up at x = {self.blowup_x:+.6f})" if self.blowup_x is not None else ""
            return f"NOT global{where} | N: {self.north.short()} | S: {self.south.short()}"
        return (f"global | class {self.cls} | N: {self.north.short()} "
                f"[LY {self.north.li_yan}, BP {self.north.bp}] | S: {self.south.short()} "
                f"[LY {self.south.li_yan}, BP {self.south.bp}]")


def summarize(sol: Solution) -> Summary:
    n = diagnose_pole(sol, "north")
    s = diagnose_pole(sol, "south")
    bx = None
    if sol.blowup_north:
        bx = sol.x_blowup_north
    elif sol.blowup_south:
        bx = sol.x_blowup_south
    glob = (not sol.blowup) and n.stratum not in ("unsafe", "blowup") and \
        s.stratum not in ("unsafe", "blowup")
    if not glob:
        cls = "-"
    elif sol.params.kappa == 0:
        cls = "face"
    else:
        fN = n.f_lim if math.isfinite(n.f_lim) else math.copysign(1e300, n.f_lim)
        fS = s.f_lim if math.isfinite(s.f_lim) else math.copysign(1e300, s.f_lim)
        # f is monotone: it has a zero iff the pole values have opposite signs
        cls = "Z" if fN * fS < 0 else "N"
    return Summary(glob, bx, cls, n, s)
