"""Milestone-1 diagnostic figures (core solver only, no GUI).

fig_pole_types.png : U against s for the three runs of Numerical observation 4.28 with the
                     predicted pole limits (top), and |u_r| against d near the axis with the
                     predicted rate of each stratum (bottom).
fig_lens_map.png   : Sigma-slice (gamma, kappa) at c = 0, A = 0 (book Fig. 1.3): global region
                     coloured by the north-pole regularity class, lens edges by bisection.
Run:  python scripts/make_m1_figures.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import solve
from swirl_explorer.core.poles import diagnose_pole, fit_exponent
from swirl_explorer.core.edges import gamma_minus
from swirl_explorer.core.sigma_map import SigmaMapSpec, compute_sigma_map

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VIOLET = "#6250d6"
HALO = [pe.withStroke(linewidth=3, foreground=SURF)]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "legend.frameon": False, "mathtext.fontset": "dejavusans",
})


def _runs():
    gm0 = gamma_minus(Params(kappa=1.0))
    gm9 = gamma_minus(Params(c2=-0.9, kappa=0.3))
    return [
        ("G", "G (generic): inside the lens", Params(gamma=gm0 + 0.1, kappa=1.0), S1),
        ("L1", "L1 (logarithmic): on the lens edge", Params(gamma=gm0 + 1e-11, kappa=1.0), S2),
        ("R", r"R (recessive): lens edge for $c_2=-0.9$", Params(c2=-0.9, gamma=gm9 + 1e-11, kappa=0.3), S3),
    ]


def fig_pole_types() -> str:
    runs = _runs()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.5, 9.6))
    # ---------------------------------------------------------------- panel A
    for strat, name, p, col in runs:
        sol = solve(p)
        xi = np.linspace(0, 7.35, 600)
        pr = sol.profiles(xi)
        d = diagnose_pole(sol, "north", s_eval=14)
        ax.plot(pr["s_north"], pr["U"], color=col, lw=2.2, zorder=3)
        ax.axhline(d.U_lim, color=col, lw=1.1, ls=(0, (5, 3)), zorder=2)
        if strat == "L1":
            s = pr["s_north"][pr["s_north"] > 2.6]
            ax.plot(s, -2 + 4 / s, color=col, lw=1.2, ls=":")
        y = pr["U"][-1]
        ax.text(14.35, y, f"{name}\nU(1) = {d.U_lim:.3g}", fontsize=9, va="center", color=INK)
        ax.plot([14.05, 14.28], [y, y], color=col, lw=2.2, clip_on=False)
    ax.set_xlim(0, 14)
    ax.set_ylim(-2.6, 0.1)
    ax.set_xlabel(r"$s = -\ln(1-x)$      toward the north pole $\rightarrow$      ($s = 14$: $d \approx 10^{-3}$)")
    ax.set_ylabel(r"$U = u_\theta \sin\theta$")
    ax.set_title("A. U tends to the limit predicted by the pole type", loc="left", fontsize=11, color=INK)
    style = [Line2D([], [], color=INK2, lw=2.2, label="computed"),
             Line2D([], [], color=INK2, lw=1.1, ls=(0, (5, 3)), label="predicted limit U(1)"),
             Line2D([], [], color=INK2, lw=1.2, ls=":", label=r"L1 law $U \approx -2 + 4/s$")]
    ax.legend(handles=style, loc="upper right", fontsize=8.5)
    # ---------------------------------------------------------------- panel B
    xi2 = np.linspace(np.log(2 / 0.05), 14.5, 500)        # d from 0.05 to 1e-6 (edge runs drift later)
    for strat, name, p, col in runs:
        sol = solve(p)
        dfull = diagnose_pole(sol, "north")
        p2 = sol.profiles(xi2)
        good = np.abs(p2["U_x"] * p2["sin"] ** 2) > 1e-9      # drop rounding noise
        d = p2["sin"][good]
        ur = np.abs(p2["u_r"][good])
        fit, _ = fit_exponent(d[d < 1e-3], ur[d < 1e-3])
        if strat == "G":
            ref = d ** (2 * dfull.mu - 2)
            lab = (rf"G: fitted $|u_r| \propto d^{{{fit:.2f}}}$" + "\n"
                   + rf"predicted $d^{{2\mu-2}} = d^{{{2 * dfull.mu - 2:.2f}}}$")
        elif strat == "L1":
            ref = 1 / (d ** 2 * np.log(1 / d) ** 2)
            lab = (rf"L1: fitted $|u_r| \propto d^{{{fit:.2f}}}$" + "\n"
                   + r"predicted $1/(d^2 \ln^2(1/d))$")
        else:
            ref = None
            lab = rf"R: fitted $|u_r| \propto d^{{{fit:.2f}}}$" + "\n(no rate predicted)"
        if ref is not None:
            bx.loglog(d, ref * ur[-1] / ref[-1], color=col, lw=1.1, ls=(0, (5, 3)), zorder=2)
        bx.loglog(d, ur, color=col, lw=2.2, zorder=3)
        yl = ur[-1] * (0.06 if strat == "R" else (8.0 if strat == "G" else 1.0))
        bx.text(8.5e-7, yl, lab, fontsize=9, va="center", ha="left", color=INK)
        bx.plot([9.6e-7, 8.8e-7], [ur[-1], yl], color=col, lw=0.8, clip_on=False)
    bx.set_xlim(0.05, 1e-6)                                  # pole on the right, as in panel A
    bx.set_ylim(2, 3e10)
    bx.set_xlabel(r"$d = \sin\theta$ = distance to the axis      toward the axis $\rightarrow$")
    bx.set_ylabel(r"$|u_r|$")
    bx.set_title("B. The radial velocity blows up at a rate set by the type (log–log)", loc="left",
                 fontsize=11, color=INK)
    style2 = [Line2D([], [], color=INK2, lw=2.2, label="computed"),
              Line2D([], [], color=INK2, lw=1.1, ls=(0, (5, 3)), label="predicted rate, matched at d = 10⁻⁶")]
    bx.legend(handles=style2, loc="upper left", fontsize=8.5)
    fig.suptitle("Three behaviours at the north pole (the axis ray x = cos θ = 1):\n"
                 "each pole type has its own limit of U and its own blow-up rate of $u_r$",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.008,
             "Notation: x = cos θ, d = sin θ, U = u_θ sin θ, u = r⁻¹(u_r e_r + u_θ e_θ + u_φ e_φ), μ = √(1 + Q(1)/2). "
             "Data at the equator: c = (c₁, c₂, c₃)\nare the Taylor constants of the potential Q; "
             "G and L1: c = 0, κ = f_x(0) = 1; R: c = (0, −0.9, 0), κ = 0.3; A = f(0) = 0. "
             "Values match Numerical observation 4.28 of the monograph.", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.subplots_adjust(hspace=0.32, right=0.72, top=0.89)
    path = os.path.join(OUT, "fig_pole_types.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_lens_map() -> str:
    spec = SigmaMapSpec(Params(), "gamma", "kappa", (-2.5, 2.5), (0.0, 4.6), nx=251, ny=185)
    res = compute_sigma_map(spec, workers=os.cpu_count() or 4)
    xs, ys = spec.axes()
    G = res.is_global
    mu = np.where(G, res.mu_n, np.nan)
    ks = np.concatenate([np.linspace(0.005, 4.0, 50), np.linspace(4.0, 4.1607, 12)])
    gm = np.array([gamma_minus(Params(kappa=k), tol=1e-8) for k in ks])
    gp = -gm                                   # mirror symmetry at x0 = 0, c = 0
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.grid(False)
    ax.pcolormesh(xs, ys, np.where(G, np.nan, 1.0), cmap=ListedColormap(["#e6e5e1"]),
                  shading="nearest", rasterized=True)
    bounds = [0, 0.5, 2 / 3, 1.0]
    cols = ["#163f73", "#2a78d6", "#a9cbf2"]             # one hue: darker = more singular (smaller μ₊)
    cmap = ListedColormap(cols)
    ax.pcolormesh(xs, ys, mu, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), shading="nearest",
                  rasterized=True)
    cs = ax.contour(xs, ys, mu, levels=[0.5, 2 / 3], colors=["white"], linewidths=1.0)

    def on_contour(level_idx: int, kappa: float):
        seg = max(cs.allsegs[level_idx], key=len)
        j = int(np.argmin(np.abs(seg[:, 1] - kappa)))
        return float(seg[j, 0]), float(seg[j, 1])

    labels = ax.clabel(cs, fmt={0.5: "μ₊ = ½", 2 / 3: "μ₊ = ⅔"}, fontsize=9.5, inline=True,
                       manual=[on_contour(0, 2.2), on_contour(1, 2.2)])
    for t in labels:
        t.set_color(INK)
        t.set_bbox(dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))
    ax.plot(np.concatenate([gm, gp[::-1]]), np.concatenate([ks, ks[::-1]]), color=INK, lw=1.6)
    ax.plot([0], [4.1608149], marker="o", ms=7, color=INK)
    ax.text(-0.12, 4.32, r"apex $\kappa_{\max} \approx 4.16$", fontsize=10, color=INK, ha="right",
            path_effects=HALO)
    ax.plot([-2, 2], [0, 0], color=VIOLET, lw=5, solid_capstyle="butt", clip_on=False, zorder=5)
    ax.text(0, 0.1, r"Landau window $-2 \leq \gamma \leq 2$ (no swirl, $\kappa \to 0$)", ha="center",
            fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
    ann = dict(fontsize=9.3, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID),
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate("lower edge γ₋(κ):\nnorth pole is L1", xy=(-1.62, 1.3), xytext=(-2.45, 2.2), **ann)
    ax.annotate("upper edge γ₊(κ): decided\nby the south pole", xy=(1.66, 1.2), xytext=(1.55, 2.05), **ann)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("γ = U(0)   (meridional circulation at the equator)")
    ax.set_ylabel("κ = f_x(0)   (swirl rate at the equator)")
    ax.set_title("Global solutions form a lens (slice c = 0, A = 0), shaded by the north-pole regularity class",
                 loc="left", fontsize=11.5, color=INK)
    handles = [Patch(color=cols[2], label=r"Type II.5: $2/3 < \mu_+ < 1$"),
               Patch(color=cols[1], label=r"Type III: $1/2 \leq \mu_+ \leq 2/3$"),
               Patch(color=cols[0], label=r"beyond III: $\mu_+ < 1/2$"),
               Patch(color="#e6e5e1", label="not global (z → 0 at some angle)")]
    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True, facecolor=SURF, edgecolor=GRID,
              title="north pole: all lens points are type G;\nclass set by " + r"$\mu_+ = \sqrt{1+Q(1)/2}$",
              title_fontsize=8.8)
    fig.text(0.01, 0.01, f"Data at the equator x = 0. Darker = more singular at the north pole. "
             f"Grid {spec.nx}×{spec.ny}; edges by bisection. No point of this slice has μ₊ ≥ 1.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(OUT, "fig_lens_map.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(fig_pole_types())
    print(fig_lens_map())
