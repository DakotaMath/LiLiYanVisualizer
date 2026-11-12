"""The 2D slice figure (no Qt): u·n colour, LIC texture, projected streamlines, axis and cut-outs.

Shared by the desktop slice pane and the browser version.
"""
from __future__ import annotations

import math

import numpy as np

from . import style


def _axis_name(v) -> str:
    """'x', '−z', ... for (anti)parallel coordinate directions, else the vector."""
    for k, name in enumerate("xyz"):
        if abs(abs(v[k]) - 1.0) < 1e-6:
            return name if v[k] > 0 else "−" + name
    return _fmt_vec(v)


def _plane_name(spec) -> str:
    n = spec.normal
    off = f"{spec.offset:+.2f}".replace("-", "−")
    if abs(n[2]) > 1 - 1e-6:
        return f"Horizontal slice z = {spec.offset * np.sign(n[2]):+.2f}".replace("-", "−")
    if abs(n[2]) < 1e-6 and abs(spec.offset) < 1e-9:
        return "Meridional slice through the axis"
    return f"Slice at offset {off}"


def _fmt_vec(v) -> str:
    return "(" + ", ".join(f"{c:+.2f}".replace("-", "−") for c in v) + ")"



def draw_slice(fig, data, lines, domain, show_colour: bool = True):
    """Draw a sampled slice (SliceData) with its projected streamlines into `fig`; returns the axes."""
    import matplotlib
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    p = style.c()
    fig.clear()
    fig.set_facecolor(p["plot_bg"])
    with matplotlib.rc_context(style.mpl_rc()):
        ax = fig.add_subplot()
        L = data.spec.half_size
        ext = (-L, L, -L, L)
        AXIS_COL = "#e09a2d"
        m = data.clim
        norm = matplotlib.colors.Normalize(-m, m)
        cmap = matplotlib.colormaps["RdBu_r"]
        if show_colour:
            rgba = cmap(norm(np.nan_to_num(data.show)))
            if data.lic is not None:       # LIC only modulates brightness: the colour stays readable
                g = np.nan_to_num(data.lic, nan=0.5)
                lo, hi = np.percentile(g, [2, 98])          # contrast-stretch the texture
                g = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)
                f = 0.55 + 0.6 * g
                rgba[..., :3] = np.clip(rgba[..., :3] * f[..., None], 0, 1)
            rgba[..., 3] = np.isfinite(data.show)
            ax.imshow(rgba, origin="lower", extent=ext, interpolation="bilinear")
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            tk = [-m, -m / 2, 0, m / 2, m]
            if data.symlog:                                 # ticks at decades of the real value
                dec = [10.0 ** k for k in range(-6, 12) if data.lin_thresh <= 10.0 ** k <= data.vmax]
                pos = [float(np.log10(1.0 + v / data.lin_thresh)) for v in dec[-3:]]
                if pos:
                    tk = [-t for t in pos[::-1]] + [0.0] + pos
            cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09,
                                   extend="neither" if data.symlog else "both", aspect=35, ticks=tk)
            cb.ax.set_xticklabels([f"{v:.3g}".replace("-", "−") for v in
                                   (np.round(data.to_value(tk), 6) if data.symlog else tk)])
            if data.is_zero:
                tk = [-m, 0, m]
                cb.set_ticks(tk)
                cb.ax.set_xticklabels(["", "0", ""])
                note = f"({data.show_label} = 0 on this plane, up to round-off: no flow through it)"
            elif data.symlog:
                note = f"(signed-log scale, linear below {data.lin_thresh:.2g}; max |{data.show_label}| = {data.vmax:.3g})"
            else:
                note = f"(values beyond ±{m:.3g} saturate; max |{data.show_label}| = {data.vmax:.3g})"
            cb.set_label(f"{data.show_label}:  red = toward you,  blue = away" + "\n" + note, fontsize=8.5)
            sat = np.abs(np.nan_to_num(data.show)) > m
            if sat.any():
                ax.contour(data.s, data.t, sat.astype(float), levels=[0.5], colors=[p["plot_ink"]],
                           linewidths=0.8, linestyles=":")
        elif data.lic is not None:
            ax.imshow(np.ma.masked_invalid(data.lic), origin="lower", extent=ext, cmap="gray",
                      interpolation="bilinear")
        # geometric cut-outs as smooth hatched shapes
        from matplotlib.patches import Circle, Rectangle
        R = domain.R
        d = data.spec.offset
        if abs(d) < domain.r_min:
            rc = math.sqrt(domain.r_min ** 2 - d * d)
            ax.add_patch(Circle((0, 0), rc, facecolor=p["plot_bg"], edgecolor=p["plot_ink2"], hatch="////",
                                lw=0.8, zorder=4))
        line_col = "#151515"                         # dark on the light colour map, in both themes
        shown = lines if data.lic is None else sorted(lines, key=len, reverse=True)[:8]   # 8 longest
        from matplotlib import patheffects as _pe
        halo = [_pe.Stroke(linewidth=2.6, foreground="#f4f4f4", alpha=0.55), _pe.Normal()]
        for poly in shown:
            ax.plot(poly[:, 0], poly[:, 1], color=line_col, lw=0.9 if data.lic is None else 1.0, zorder=5,
                    path_effects=halo)      # light halo keeps lines visible on saturated colour
            if len(poly) > 6:                       # one arrowhead per line, mid-way
                k = len(poly) // 2
                ax.annotate("", xy=poly[k + 1], xytext=poly[k], zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=line_col, lw=0.9, mutation_scale=10))
        if abs(d) < R:
            rr = math.sqrt(R * R - d * d)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(rr * np.cos(th), rr * np.sin(th), color=p["plot_ink2"], lw=0.9, ls="--")
        flow_lab = "in-plane flow (u projected)" if data.lic is None else "in-plane flow: texture + a few arrows"
        handles = [Line2D([], [], color=p["plot_ink"], lw=0.9, marker=">", ms=4, label=flow_lab),
                   Line2D([], [], color=p["plot_ink2"], lw=0.9, ls="--", label=f"edge of the ball |X| = {R:g}")]
        if abs(d) < domain.r_min:
            handles.append(Patch(facecolor="none", edgecolor=p["plot_ink2"], hatch="////",
                                 label="cut out around the origin"))
        if show_colour and (np.abs(np.nan_to_num(data.show)) > m).any():
            handles.append(Line2D([], [], color=p["plot_ink"], lw=1.0, ls=":",
                                  label=f"|{data.show_label}| beyond the colour range"))
        for kind, a, b in data.axis_hits:
            if kind == "point" and abs(a) <= L and abs(b) <= L:
                ax.plot([a], [b], marker="o", ms=9, mfc="white", mec=AXIS_COL, mew=2.2, zorder=6)
                handles.append(Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=AXIS_COL, mew=2,
                                      label="symmetry axis crosses here"))
            elif kind == "line":
                w = 2 * domain.axis_a
                ax.add_patch(Rectangle((-w / 2, -L), w, 2 * L, facecolor=p["plot_bg"], edgecolor="none",
                                       zorder=4))
                ax.plot([-a * L, a * L], [-b * L, b * L], color=AXIS_COL, lw=1.6, ls="-.", zorder=6)
                handles.append(Line2D([], [], color=AXIS_COL, lw=1.6, ls="-.", label="symmetry axis (z)"))
        e1, e2, n = data.spec.basis()
        ax.set_xlabel(_axis_name(e1) + "  →", fontsize=11)
        ax.set_ylabel(_axis_name(e2) + "  →", fontsize=11)
        where = ""
        if abs(d) > domain.r_min:
            where = f"\norigin {abs(d):.2f} " + ("below" if d > 0 else "above") + " the plane"
        ax.set_title(f"{_plane_name(data.spec)}\nn = {_axis_name(n)} points toward you{where}", fontsize=10,
                     loc="left")
        ticks = np.arange(-L, L + 1e-9, 0.5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=1, fontsize=8.5,
                  frameon=False)
    return ax
