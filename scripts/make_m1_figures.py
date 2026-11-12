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
        ("R", "R (recessive): edge of another lens", Params(c2=-0.9, gamma=gm9 + 1e-11, kappa=0.3), S3),
    ]


def fig_pole_types() -> str:
    runs = _runs()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.5, 9.8))
    # ---------------------------------------------------------------- panel A
    for strat, name, p, col in runs:
        sol = solve(p)
        xi = np.linspace(0, 7.35, 600)
        pr = sol.profiles(xi)
        d = diagnose_pole(sol, "north", s_eval=14)
        ax.plot(pr["s_north"], pr["U"], color=col, lw=2.2, zorder=3)
        ax.axhline(d.U_lim, color=col, lw=1.1, ls=(0, (5, 3)), zorder=2)
        if strat == "L1":
            s = pr["s_north"][pr["s_north"] > 5.0]
            ax.plot(s, -2 + 4 / s, color=col, lw=1.3, ls=":")
            ax.annotate("L1 converges only logarithmically:\nU + 2 ≈ 4/s is still 0.32 at s = 14",
                        xy=(12.5, -1.645), xytext=(9.0, -1.08), fontsize=8.8, color=INK,
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID),
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
        y = pr["U"][-1]
        extra = "\n(reached only like 4/s)" if strat == "L1" else ""
        ax.text(14.35, y, f"{name}\nU(1) = {d.U_lim:.3g}{extra}", fontsize=9, va="center", color=INK)
        ax.plot([14.05, 14.28], [y, y], color=col, lw=2.2, clip_on=False)
    ax.set_xlim(0, 14)
    ax.set_ylim(-2.6, -0.5)
    ax.set_xlabel(r"$s = -\ln(1-x)$   (toward the north pole $\rightarrow$)")
    ax.set_ylabel(r"$U = u_\theta \sin\theta$")
    ax.set_title("A. U tends to the limit predicted by the pole type", loc="left", fontsize=11, color=INK)
    style = [Line2D([], [], color=INK2, lw=2.2, label="computed"),
             Line2D([], [], color=INK2, lw=1.1, ls=(0, (5, 3)), label="predicted limit U(1)"),
             Line2D([], [], color=INK2, lw=1.3, ls=":", label=r"L1 law $U \approx -2 + 4/s$")]
    ax.legend(handles=style, loc="lower right", fontsize=8.5, frameon=True, facecolor=SURF, edgecolor=GRID)
    # ---------------------------------------------------------------- panel B: local exponent
    xi2 = np.linspace(np.log(2 / 0.03), 13.8, 700)        # d from 0.05 to 1e-6 (edge runs drift later)
    d_end = 1.0 / np.cosh(13.8)                              # = sin(theta) at the last sample
    label_y = {"G": None, "L1": None, "R": None}
    for strat, name, p, col in runs:
        sol = solve(p)
        dfull = diagnose_pole(sol, "north")
        p2 = sol.profiles(xi2)
        good = np.abs(p2["U_x"] * p2["sin"] ** 2) > 1e-9      # drop rounding noise
        d = p2["sin"][good]
        ur = np.abs(p2["u_r"][good])
        slope = np.gradient(np.log(ur), np.log(d))            # local exponent of |u_r| ~ d^slope
        if strat == "G":
            pred = np.full_like(d, 2 * dfull.mu - 2)
            how = r"$2\mu-2$"
        elif strat == "L1":
            pred = -2 + 2 / np.log(1 / d)                      # local exponent of 1/(d^2 ln^2(1/d))
            how = r"from $1/(d^2\ln^2(1/d))$"
        else:
            pred = np.full_like(d, -4 * dfull.mu)
            how = r"$-4\mu$ (derived here)"
        bx.semilogx(d, pred, color=col, lw=1.2, ls=(0, (5, 3)), zorder=2)
        bx.semilogx(d, slope, color=col, lw=2.2, zorder=3)
        plateau = (d > 3e-6) & (d < 3e-5)
        y = float(np.median(slope[plateau]))
        label_y[strat] = (y, f"{name.split(':')[0]}: exponent {y:.2f}\npredicted {np.median(pred[plateau]):.2f}, {how}", col)
    # right-end labels, nudged apart vertically if they collide
    items = sorted(label_y.values(), key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.22)
    for (y0, txt, col), yt in zip(items, ys):
        bx.text(d_end * 0.62, yt, txt, fontsize=9, va="center", ha="left", color=INK)
        bx.plot([d_end, d_end * 0.7], [y0, yt], color=col, lw=2.2, clip_on=False)
    bx.set_xlim(0.03, d_end)                                 # pole on the right, as in panel A
    bx.set_ylim(-2.05, -0.5)
    bx.set_xlabel(r"$d = \sin\theta$, distance to the axis   (toward the axis $\rightarrow$)")
    bx.set_ylabel(r"local exponent $\;d\ln|u_r|\,/\,d\ln d$")
    bx.set_title(r"B. The blow-up exponent of $u_r$ tends to the rate predicted by the pole type",
                 loc="left", fontsize=11, color=INK)
    style2 = [Line2D([], [], color=INK2, lw=2.2, label=r"computed  ($|u_r| \sim d^{\,\mathrm{exponent}}$)"),
              Line2D([], [], color=INK2, lw=1.2, ls=(0, (5, 3)), label="predicted exponent")]
    bx.legend(handles=style2, loc="upper left", fontsize=8.5, frameon=True, facecolor=SURF, edgecolor=GRID)
    fig.suptitle("Three pole types at the north pole: each has its own limit of $U$ and blow-up rate of $u_r$",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.008,
             r"$x=\cos\theta$, $d=\sin\theta$, $\mu=\sqrt{1+Q(1)/2}$.  Data at the equator: "
             r"G, L1: $c=0$, $\kappa=1$;  R: $c=(0,-0.9,0)$, $\kappa=0.3$;  $A=0$ "
             "(Numerical observation 4.28 of the monograph).", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.subplots_adjust(hspace=0.3, right=0.72, top=0.92)
    path = os.path.join(OUT, "fig_pole_types.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _nearest_fill(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(np.isnan(a), return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _shade_lens(ax, xs, ys, mu, gm, ks, cols, smooth=1.0, kmin=0.0):
    """Shade classes, draw smooth class boundaries clipped to the bisected lens."""
    from scipy.ndimage import gaussian_filter
    from matplotlib.patches import Polygon
    bounds = [0, 0.5, 2 / 3, 1.0]
    cmap = ListedColormap(cols)
    ax.pcolormesh(xs, ys, np.where(np.isnan(mu), 1.0, np.nan), cmap=ListedColormap(["#e6e5e1"]),
                  shading="nearest", rasterized=True)
    ax.pcolormesh(xs, ys, mu, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), shading="nearest",
                  rasterized=True)
    sm = gaussian_filter(_nearest_fill(mu), smooth)
    sm[ys < kmin, :] = np.nan                      # coarse grid cannot resolve the thin corner
    cs = ax.contour(xs, ys, sm, levels=[0.5, 2 / 3], colors=["white"], linewidths=2.4)
    poly = Polygon(np.column_stack([np.concatenate([gm, -gm[::-1]]), np.concatenate([ks, ks[::-1]])]),
                   closed=True, transform=ax.transData, fill=False, lw=0)
    ax.add_patch(poly)
    cs.set_clip_path(poly)
    ax.plot(np.concatenate([gm, -gm[::-1]]), np.concatenate([ks, ks[::-1]]), color=INK, lw=2.0)
    return cs


def _inline_label(ax, cs, level_idx, kappa, text):
    seg = max(cs.allsegs[level_idx], key=len)
    j = int(np.argmin(np.abs(seg[:, 1] - kappa)))
    a, b = seg[max(j - 3, 0)], seg[min(j + 3, len(seg) - 1)]
    pa, pb = ax.transData.transform(a), ax.transData.transform(b)
    ang = np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    ax.text(seg[j, 0], seg[j, 1], text, rotation=ang, rotation_mode="anchor", ha="center", va="center",
            fontsize=10, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"), zorder=6)


def fig_lens_map() -> str:
    spec = SigmaMapSpec(Params(), "gamma", "kappa", (-2.2, 2.2), (0.0, 4.4), nx=241, ny=200)
    res = compute_sigma_map(spec, workers=os.cpu_count() or 4)
    xs, ys = spec.axes()
    ks = np.concatenate([np.linspace(0.002, 0.2, 15), np.linspace(0.2, 4.0, 45), np.linspace(4.0, 4.1607, 12)])
    gm = np.array([gamma_minus(Params(kappa=k), tol=1e-9) for k in ks])

    def masked_mu(res, xs, ys):
        Xg, Yg = np.meshgrid(xs, ys)
        lo = np.interp(Yg, ks, gm, right=np.inf)
        inside = (Yg <= ks[-1]) & (Xg >= lo) & (Xg <= -lo)
        return np.where(res.is_global & inside, res.mu_n, np.nan)

    mu = masked_mu(res, xs, ys)
    cols = ["#163f73", "#2a78d6", "#a9cbf2"]             # one hue: darker = more singular (smaller μ₊)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.grid(False)
    cs = _shade_lens(ax, xs, ys, mu, gm, ks, cols, kmin=0.45)
    fig.canvas.draw()
    _inline_label(ax, cs, 0, 3.0, r"$\mu_+ = 1/2$")
    _inline_label(ax, cs, 1, 2.45, r"$\mu_+ = 2/3$")
    ax.plot([0], [4.1608149], marker="o", ms=7, color=INK)
    ax.text(-0.1, 4.2, r"apex $\kappa_{\max} \approx 4.16$", fontsize=10, color=INK, ha="right",
            path_effects=HALO)
    ax.plot([-2, 2], [0, 0], color=INK2, lw=5, solid_capstyle="butt", clip_on=False, zorder=5)
    ax.text(0, 0.2, r"Landau window $-2 \leq \gamma \leq 2$ (no swirl, $\kappa \to 0$)", ha="center",
            fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
    ann = dict(fontsize=9.3, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID),
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate("lower edge $\\gamma_-(\\kappa)$:\nnorth pole is L1", xy=(float(np.interp(2.6, ks, gm)), 2.6), xytext=(-2.15, 3.3), **ann)
    ax.annotate("upper edge $\\gamma_+(\\kappa)$:\ndecided by the south pole", xy=(float(-np.interp(0.9, ks, gm)), 0.9),
                xytext=(0.75, 1.45), **ann)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0, 4.4)
    ax.set_xlabel(r"$\gamma = U(0)$   (meridional circulation at the equator)")
    ax.set_ylabel(r"$\kappa = f_x(0)$   (swirl rate at the equator)")
    ax.set_title(r"Global solutions form a lens (slice $c = 0$, $A = 0$), shaded by how singular the north pole is",
                 loc="left", fontsize=11.5, color=INK)
    handles = [Patch(color=cols[2], label=r"Type II.5:  $2/3 < \mu_+ < 1$"),
               Patch(color=cols[1], label=r"Type III:  $1/2 \leq \mu_+ \leq 2/3$"),
               Patch(color=cols[0], label=r"beyond III:  $\mu_+ < 1/2$  ($|u_r| \gg 1/d$)"),
               Patch(color="#e6e5e1", label="not global")]
    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True, facecolor=SURF, edgecolor=GRID,
              title="North-pole class by the exponent " + r"$\mu_+ = \sqrt{1+Q(1)/2}$" + "\n(darker = more singular; none has "
              + r"$\mu_+ \geq 1$)", title_fontsize=9)
    # ---- inset: the thin lower-left part of the lens, recomputed on a fine grid
    spec2 = SigmaMapSpec(Params(), "gamma", "kappa", (-2.02, -1.55), (0.0, 1.2), nx=160, ny=160)
    res2 = compute_sigma_map(spec2, workers=os.cpu_count() or 4)
    xs2, ys2 = spec2.axes()
    _shade_lens(ax, xs2, ys2, masked_mu(res2, xs2, ys2), gm, ks, cols, smooth=1.5)
    ins = ax.inset_axes([0.83, 0.53, 0.155, 0.21])
    ins.grid(False)
    _shade_lens(ins, xs2, ys2, masked_mu(res2, xs2, ys2), gm, ks, cols, smooth=1.5)
    ins.set_xlim(-2.02, -1.55)
    ins.set_ylim(0, 1.2)
    ins.tick_params(labelsize=8)
    ins.set_title("zoom of the dashed box", fontsize=8.5, color=INK, loc="left", pad=3)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-2.02, 0.0), 0.47, 1.2, fill=False, ec=INK, lw=1.0, ls="--", zorder=7))
    ax.text(-1.53, 1.22, "zoomed at right", fontsize=8.5, color=INK, va="bottom", path_effects=HALO)
    fig.text(0.01, 0.01, f"Data at the equator x = 0. Grid {spec.nx}×{spec.ny} (inset {spec2.nx}×{spec2.ny}); "
             "lens edges by bisection.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(OUT, "fig_lens_map.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path



if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(fig_pole_types())
    print(fig_lens_map())
