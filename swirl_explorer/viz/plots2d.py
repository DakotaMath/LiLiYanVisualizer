"""2D profile plots (pure matplotlib; the Qt panel only hosts the Figure).

One axes per selected quantity, stacked vertically and sharing the horizontal axis.
Horizontal variable: x = cos θ, θ (degrees), ξ = artanh x, or s = −ln(1 ∓ x).
Modes:
  * ``linear``  : the quantity itself
  * ``log``     : |quantity| on a log scale
  * ``loglog``  : near one pole, against d = sin θ (log, reversed so the pole is on the
                  right). Velocities are shown as |u|; U, f, Q, h, W as the distance to
                  their pole limit |q − q(pole)|. The predicted rates are overlaid
                  (dashed), anchored at the smallest d.
Predicted pole limits (dashed, colour of the curve) are drawn in the linear mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..core.solver import Solution
from ..core.poles import Summary, PoleDiagnostics
from . import style

QUANTITIES: dict[str, str] = {
    "U": r"$U = u_\theta\sin\theta$",
    "f": r"$f = u_\varphi\sin\theta$",
    "Q": r"$Q$",
    "z": r"$z$  ($z(0)=1$)",
    "W": r"$W = 1 + Q/2$",
    "h": r"$h = z'/z$",
    "u_r": r"$u_r$",
    "u_theta": r"$u_\theta$",
    "u_phi": r"$u_\varphi$",
}
XVARS: dict[str, str] = {
    "s_north": r"$s = -\ln(1-x)$   (north pole $\rightarrow$)",
    "s_south": r"$s = -\ln(1+x)$   (south pole $\rightarrow$)",
    "x": r"$x = \cos\theta$",
    "theta": r"$\theta$ (degrees)",
    "xi": r"$\xi = \mathrm{artanh}\,x$",
}
MODES = ("linear", "log", "loglog")

CURRENT = "#2a78d6"
PRED = "#d62f7a"   # predictions: magenta, drawn on top of the curves
PIN_COLORS = ["#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VELOCITIES = ("u_r", "u_theta", "u_phi")


@dataclass
class Curve:
    """A solution sampled for plotting (current or pinned)."""
    label: str
    sol: Solution
    summary: Summary
    color: str = CURRENT
    pinned: bool = False
    _cache: dict = field(default_factory=dict)

    def profiles(self, n: int = 1600) -> dict:
        if "pr" not in self._cache:
            margin = 0.02 if self.sol.blowup else 0.0
            xi = self.sol.grid(n, margin=margin)
            pr = self.sol.profiles(xi)
            pr["theta_deg"] = np.degrees(pr["theta"])
            self._cache["pr"] = pr
        return self._cache["pr"]


def _geometry(xi: np.ndarray) -> dict:
    """The horizontal-axis variables at given xi (no solution needed)."""
    return {"x": np.tanh(xi), "theta_deg": np.degrees(2 * np.arctan(np.exp(-xi))), "xi": xi,
            "s_north": np.logaddexp(0, 2 * xi) - math.log(2), "s_south": np.logaddexp(0, -2 * xi) - math.log(2)}


def _xvalues(pr: dict, xvar: str) -> np.ndarray:
    return {"x": pr["x"], "theta": pr["theta_deg"], "xi": pr["xi"],
            "s_north": pr["s_north"], "s_south": pr["s_south"]}[xvar]


def _pole_limit(diag: PoleDiagnostics, q: str, side: int) -> Optional[float]:
    """Predicted pole value of quantity q at the pole of ``diag`` (actual, not mirrored)."""
    if diag.stratum in ("blowup", "unsafe", "?"):
        return None
    if q == "U":
        v = diag.U_lim
        return None if not math.isfinite(v) else (v if side > 0 else -v)
    if q == "Q":
        return diag.q
    if q == "W":
        return 1 + 0.5 * diag.q
    if q == "h":
        if not math.isfinite(diag.h_lim):
            return None
        return diag.h_lim if side > 0 else -diag.h_lim
    if q == "f":
        return diag.f_lim if math.isfinite(diag.f_lim) else None
    return None


def _robust_ylim(ys: list[np.ndarray]) -> Optional[tuple[float, float]]:
    v = np.concatenate([y[np.isfinite(y)] for y in ys if y.size]) if ys else np.array([])
    if v.size < 5:
        return None
    lo, hi = np.percentile(v, [1, 99])
    full_lo, full_hi = v.min(), v.max()
    span = hi - lo
    if span <= 0:
        return None
    # only clip if the extremes are far outside the bulk (blow-up, pole singularities)
    if full_hi - full_lo < 4 * span:
        lo, hi = full_lo, full_hi
        span = hi - lo
    return lo - 0.08 * span, hi + 0.08 * span


def draw_profiles(fig: Figure, curves: Sequence[Curve], quantities: Sequence[str],
                  xvar: str = "s_north", mode: str = "linear", pole: str = "north",
                  overlay: bool = True) -> list:
    """Draw stacked profile plots onto ``fig`` in the current colour theme. Returns the axes."""
    import matplotlib
    global INK, INK2, GRID
    p = style.c()
    INK, INK2, GRID = p["plot_ink"], p["plot_ink2"], p["plot_grid"]
    fig.set_facecolor(p["plot_bg"])
    with matplotlib.rc_context(style.mpl_rc()):
        return _draw_profiles(fig, curves, quantities, xvar, mode, pole, overlay)


def _draw_profiles(fig, curves, quantities, xvar, mode, pole, overlay) -> list:
    fig.clear()
    quantities = [q for q in quantities if q in QUANTITIES] or ["U"]
    n = len(quantities)
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]
    side = +1 if pole == "north" else -1
    for ax, q in zip(axes, quantities):
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ys_for_lim = []
        for c in curves:
            pr = c.profiles()
            lw = 2.0 if not c.pinned else 1.4
            if mode == "loglog":
                m = (pr["xi"] * side) > 3.7          # near the pole: d < 0.05
                d = pr["sin"][m]
                y = pr[q][m]
                if q in VELOCITIES:
                    yy = np.abs(y)
                    if q == "u_r":
                        ok = np.abs(pr["U_x"][m] * d * d) > 1e-9
                        yy = np.where(ok, yy, np.nan)
                    ylab = rf"$|{QUANTITIES[q].strip('$')}|$"
                else:
                    diag = c.summary.north if side > 0 else c.summary.south
                    lim = _pole_limit(diag, q, side)
                    if lim is None:
                        ax.text(0.5, 0.5, f"no finite pole limit of {q}", transform=ax.transAxes,
                                ha="center", color=INK2, fontsize=9)
                        continue
                    yy = np.abs(y - lim)
                    yy = np.where(yy > 1e-13 * (1 + abs(lim)), yy, np.nan)
                    ylab = rf"$|{q} - {q}(\mathrm{{pole}})|$"
                    if not c.pinned:
                        ax.text(0.99, 0.96, "no rate prediction for |U − U(pole)| (not in the rate table)",
                                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
                ax.loglog(d, yy, color=c.color, lw=lw, label=c.label)
                ax.set_ylabel(ylab)
                if overlay and not c.pinned and q in VELOCITIES:
                    diag = c.summary.north if side > 0 else c.summary.south
                    pe = diag.predicted_exponents.get(q)
                    fin = np.isfinite(yy) & (yy > 0)
                    if pe is not None and fin.any():
                        i = np.flatnonzero(fin)[-1]
                        ref = yy[i] * (d / d[i]) ** pe
                        ax.loglog(d, ref, color=PRED, lw=1.5, ls=(0, (4, 3)), zorder=6)
                        ex = diag.exponents.get(q, {})
                        fit, spr = ex.get("slope", math.nan), ex.get("spread", math.nan)
                        ax.text(0.99, 0.04, f"predicted exponent {pe:.3f}   ·   fitted {fit:.3f}"
                                f" (residual spread {spr:.1e})".replace("-", "\u2212"),
                                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            else:
                xv = _xvalues(pr, xvar)
                y = pr[q]
                if mode == "log":
                    y = np.abs(y)
                    ax.set_yscale("log")
                ax.plot(xv, y, color=c.color, lw=lw, label=c.label)
                yl = y
                if c.sol.blowup and not c.pinned:     # keep the blow-up spike out of the y-range
                    near = np.zeros_like(xv, dtype=bool)
                    for xb in (c.sol.xi_blowup_north, c.sol.xi_blowup_south):
                        if math.isfinite(xb):
                            near |= np.abs(pr["xi"] - xb) < 1.0
                    yl = np.where(near, np.nan, y)
                ys_for_lim.append(yl if mode != "log" else np.log10(np.where(y > 0, y, np.nan)))
                ax.set_ylabel(QUANTITIES[q])
                if not c.pinned:
                    for xb in (c.sol.xi_blowup_north, c.sol.xi_blowup_south):
                        if math.isfinite(xb):
                            xbv = _xvalues(_geometry(np.array([xb])), xvar)[0]
                            ax.axvline(xbv, color=INK, lw=1.3, ls="-.")
                            if ax is axes[0]:
                                what = "U → −∞" if xb > 0 else "U → +∞"
                                ax.annotate(f"blow-up: {what} at x = {math.tanh(xb):+.6f} (ξ = {xb:.3f})"
                                            .replace("-", "\u2212"),
                                            xy=(xbv, 1.0),
                                            xycoords=("data", "axes fraction"), fontsize=8.5, color=INK,
                                            ha="right", va="top", rotation=0,
                                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
                if overlay and mode == "linear":
                    north_right = xvar in ("x", "xi", "s_north")
                    for s, diag, nm in ((+1, c.summary.north, "N"), (-1, c.summary.south, "S")):
                        lim = _pole_limit(diag, q, s)
                        if lim is not None:
                            right = (s > 0) == north_right
                            x0, x1 = (0.72, 1.0) if right else (0.0, 0.28)
                            ax.plot([x0, x1], [lim, lim], transform=ax.get_yaxis_transform(),
                                    color=PRED if not c.pinned else c.color, lw=1.5 if not c.pinned else 1.0,
                                    ls=(0, (4, 3)), zorder=6)
                            if c.pinned:
                                continue
                            ax.annotate(f"{nm}: {lim:.4g}".replace("-", "\u2212"),
                                        xy=(1.0, lim), xycoords=ax.get_yaxis_transform(),
                                        xytext=(4, 0), textcoords="offset points", annotation_clip=False,
                                        fontsize=8, color=PRED, ha="left", va="center")
                    if q == "U" and xvar in ("s_north", "s_south") and not c.pinned:
                        diag = c.summary.north if xvar == "s_north" else c.summary.south
                        if diag.stratum == "L1":
                            sv = xv[xv > 2.5]
                            sgn = 1 if xvar == "s_north" else -1
                            ax.plot(sv, sgn * (-2 + 4 / sv), color=c.color, lw=1.0, ls=":")
        if mode == "linear":
            lim = _robust_ylim(ys_for_lim)
            if lim:
                ax.set_ylim(*lim)
    ax_last = axes[-1]
    if mode == "loglog":
        ax_last.set_xlabel(r"$d = \sin\theta$, distance to the axis — axis reversed, the "
                           + ("north" if side > 0 else "south") + r" pole is on the right $\rightarrow$")
        for ax in axes:
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
                break
    else:
        ax_last.set_xlabel(XVARS[xvar])
        north_right = xvar in ("x", "xi", "s_north")
        lt, rt = ("← S pole (x = −1)", "N pole (x = +1) →") if north_right else \
                 ("← N pole (x = +1)", "S pole (x = −1) →")
        ax_last.annotate(lt, xy=(0, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
                         ha="left", va="top", fontsize=8.5, color=INK2)
        ax_last.annotate(rt, xy=(1, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
                         ha="right", va="top", fontsize=8.5, color=INK2)
    ordered = [c for c in curves if not c.pinned] + [c for c in curves if c.pinned]
    handles = [Line2D([], [], color=c.color, lw=2.0 if not c.pinned else 1.4, label=c.label) for c in ordered]
    if overlay:
        if mode == "linear":
            handles.append(Line2D([], [], color=PRED, lw=1.5, ls=(0, (4, 3)), label="predicted pole limit"))
        if any(c.pinned for c in curves) and mode == "linear":
            pc = next(c.color for c in curves if c.pinned)
            handles.append(Line2D([], [], color=pc, lw=1.0, ls=(0, (4, 3)), label="pin's pole limits"))
        if any(c.sol.blowup and not c.pinned for c in curves) and mode != "loglog":
            handles.append(Line2D([], [], color=INK, lw=1.3, ls="-.", label="blow-up point"))
        elif mode == "loglog":
            handles.append(Line2D([], [], color=PRED, lw=1.5, ls=(0, (4, 3)),
                                  label="predicted rate (matched at the smallest d)"))
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=8.5,
                   frameon=False, borderaxespad=0.0)
    fig.align_ylabels(axes)
    eng = fig.get_layout_engine()
    if eng is not None and mode == "linear" and overlay:
        try:
            eng.set(rect=(0, 0, 0.9, 1))          # room for the pole values in the right margin
        except Exception:
            pass
    return list(axes)


def export_csv(path: str, curves: Sequence[Curve], quantities: Sequence[str], xvar: str) -> None:
    """Write the plotted curves as CSV (one block of columns per curve)."""
    cols, header = [], []
    for c in curves:
        pr = c.profiles()
        cols.append(_xvalues(pr, xvar))
        header.append(f"{c.label}:{xvar}")
        for q in quantities:
            cols.append(pr[q])
            header.append(f"{c.label}:{q}")
    n = max(len(a) for a in cols)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            w.writerow([f"{a[i]:.12g}" if i < len(a) else "" for a in cols])
