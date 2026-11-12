"""Drawing a Σ-map (core.sigma_map.SigmaMapResult) with matplotlib (no Qt).

Shared by the desktop Σ-map tab and the browser version.  ``draw_sigma`` draws one of three
views into a Figure and returns the map axes; ``readout`` describes the grid point nearest to
a position.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.params import PARAM_NAMES, PARAM_UI
from ..core.sigma_map import BLOWUP, UNSAFE, G, R, L1, L2, CODE_NAMES
from . import style
from .scene3d_colors import STRATUM_COLORS

MODES = {"global": "global / not global",
         "strata": "pole strata (north | south)",
         "mu": "pole exponents μ₊ | μ₋"}
MU_LEVELS = (0.5, 2 / 3, 1.0)
MU_LABELS = {0.5: "½", 2 / 3: "⅔", 1.0: "1"}
MU_STYLES = (":", "--", "-.")
MU_COLORS = ("#ffffff", "#7fd3ff", "#ffb000")


def MU_NORM():
    """Colour scale for μ that gives 0..1 (where the classes change) 80 % of the colours."""
    import matplotlib.colors as mc
    f = lambda x: np.where(np.asarray(x) <= 1.0, 0.8 * np.asarray(x), 0.8 + 0.4 * (np.asarray(x) - 1.0))
    g = lambda y: np.where(np.asarray(y) <= 0.8, np.asarray(y) / 0.8, 1.0 + (np.asarray(y) - 0.8) / 0.4)
    return mc.FuncNorm((f, g), vmin=0.0, vmax=1.5)   # ½ white, ⅔ sky blue, 1 amber         # ½ dotted, ⅔ dashed, 1 dash-dot (the edge of Σ is solid)
MARK = "#ff4fd8"
CLICK_HELP = ("<br><br>Hover to read a point (crosshair); <b>click</b> to jump there, <b>drag</b> to move. "
          "With 'lock to global solutions' on, non-global points are refused. "
          "<b>Shift-click</b> adds a keyframe to the path (Path dock).")
EXPLAIN = {
    "global": "<b>Green</b>: global, i.e. the solution exists on the whole sphere and both poles pass the "
              "book's terminal test.<br><b>Olive</b>: not global, the solution blows up at some x before "
              "reaching a pole.<br><b>Grey</b>: not global, it reaches a pole that fails the terminal test "
              "(usually a thin fringe along the edge, where the pole types R, L1, L2 live).",
    "strata": "Each panel shows the type of one pole. <b>G</b> (generic, green as in the global view) fills "
              "the interior of Σ; R, L1, L2 occur only on edges of the global region, so on a grid they "
              "appear as thin slivers if at all.<br><b>Hatched</b>: this pole is fine but the other pole "
              "fails, so the point is not global. Olive / grey: this pole itself blows up / fails.",
    "mu": "<b>μ</b> = √(1 + Q/2) at a generic pole sets the radial velocity near the axis, "
          "u<sub>r</sub> ~ d<sup>2μ−2</sup>: μ &lt; ½ beyond class III, ½–⅔ class III, ⅔–1 class II.5, "
          "μ &gt; 1 u<sub>r</sub> bounded. Contours: ½ white dotted, ⅔ blue dashed, 1 amber dash-dot. "
          "Cross-hatched: not global (switch to the global view to see why).",
}                     # current point (magenta, as the predictions in the profile plots)


PATH_FAIL = {"blowup": "#c9ad72", "unsafe": "#d9dde3"}   # the path where it is not global (timeline colours)


class _Labelled:
    """A legend handle made of overlaid lines, with a label (matplotlib draws tuples with HandlerTuple)."""

    def __init__(self, parts, label):
        self.parts, self._label = parts, label

    def get_label(self):
        return self._label


def _legend_map():
    from matplotlib.legend_handler import HandlerTuple, HandlerBase

    class _H(HandlerBase):
        def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
            return HandlerTuple(ndivide=1).create_artists(legend, orig.parts, xd, yd, w, h, fs, trans)
    return {_Labelled: _H()}


def _name(n: str) -> str:
    return PARAM_UI[n][0]


def _fail(kind: str) -> str:
    """Colours of the two ways to be non-global; distinct from each other and from the background."""
    dark = style.MODE == "dark"
    return {"blow": "#6b5d45" if dark else "#b3a68c", "unsafe": "#b9bdc7" if dark else "#8e9097"}[kind]


GLOBAL_COL = "#3fa877"
NEUTRAL = "#141518"                   # not global in the μ view: off the colour ramp (near black)                   # not global, in the continuous (μ) view


def _stratum_col(name: str) -> str:
    """Pole-type colours on the Σ-map: G (the interior of Σ) in the same green as 'global'."""
    return GLOBAL_COL if name == "G" else STRATUM_COLORS[name]
KAPPA_MAX_BOOK = 4.1608               # lens apex at c = 0, A = 0, x₀ = 0 (monograph, Example 5.71)


def _is_book_lens(spec) -> bool:
    b = spec.base
    return ({spec.x_name, spec.y_name} == {"gamma", "kappa"} and spec.x_name == "gamma"
            and b.c1 == b.c2 == b.c3 == b.A == 0.0 and b.xb == 0.0)




def fill(ax, xs, ys, mask, color=None, hatch=None, z=1):
    """Smooth-edged fill of a boolean mask (marching squares, not grid steps)."""
    if not mask.any():
        return
    kw = dict(hatches=[hatch]) if hatch else {}
    ax.contourf(xs, ys, mask.astype(float), levels=[0.5, 1.5], colors=[color] if color else "none",
                antialiased=True, zorder=z, **kw)

def fill_failures(ax, res, xs, ys, code=None):
    """Not global: brown = blow-up before a pole, grey = a pole fails the terminal test.
    With ``code`` (one pole) only that pole's own failure is shown."""
    if code is None:
        blow = res.done & ((res.code_n == BLOWUP) | (res.code_s == BLOWUP))
        unsafe = res.done & ~res.is_global & ~blow
    else:
        blow = res.done & (code == BLOWUP)
        unsafe = res.done & (code == UNSAFE)
    fill(ax, xs, ys, blow, _fail("blow"))
    fill(ax, xs, ys, unsafe, _fail("unsafe"))

def mu_lines(ax, xs, ys, m, color, halo, labels=True, extra_ls=None):
    from matplotlib import patheffects as pe
    if np.isfinite(m).sum() <= 4:
        return
    ymid = 0.5 * (ys[0] + ys[-1])
    for lev, ls, lw, lc in zip(MU_LEVELS, MU_STYLES, (1.6, 1.6, 2.0), MU_COLORS):
        if color != "#ffffff":           # overlay of the second pole: one colour for all levels
            lc = color
        cs = ax.contour(xs, ys, m, levels=[lev], colors=[lc], linewidths=lw,
                        linestyles=[extra_ls or ls], zorder=4)
        cs.set_path_effects([pe.Stroke(linewidth=lw + 1.8, foreground=halo, alpha=0.6), pe.Normal()])
        segs = sorted((sg for sg in cs.allsegs[0] if len(sg) > 3), key=len, reverse=True)[:2] \
            if cs.allsegs else []
        if labels and segs:
            # one label per piece, at 55 % of its height span: away from the tips and from the
            # places where several contours meet the edge
            spots = []
            for sg in segs:
                y0, y1 = sg[:, 1].min(), sg[:, 1].max()
                fr = {0.5: 0.3, 2 / 3: 0.6, 1.0: 0.85}[lev]      # stagger the labels of nearby levels
                target = y0 + fr * (y1 - y0) if abs(y1 - ymid) >= abs(y0 - ymid) else y1 - fr * (y1 - y0)
                spots.append(tuple(sg[np.argmin(abs(sg[:, 1] - target))]))
            for t in ax.clabel(cs, fmt={lev: MU_LABELS[lev]}, fontsize=10, inline=True, inline_spacing=4,
                               manual=spots):
                t.set_rotation(0)
                t.set_color(lc)
                t.set_path_effects([pe.Stroke(linewidth=2.5, foreground=halo), pe.Normal()])

def draw_sigma(fig, res, mode: str = "global", edge: bool = True, mu: bool = False,
               current=None, path_overlay=None) -> list:
    """Draw a Σ-map result into `fig`; returns the map axes (for click-to-data mapping)."""
    import matplotlib
    import matplotlib.colors as mc
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    p = style.c()
    halo = "#101114" if style.MODE == "dark" else "#ffffff"
    fig.clear()
    fig.set_facecolor(p["plot_bg"])
    rc = {**style.mpl_rc(), "hatch.color": "#d9dde3", "hatch.linewidth": 0.8}
    with matplotlib.rc_context(rc):
        if res is None:
            ax = fig.add_subplot()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No map yet: choose two parameters and press 'Compute map'.",
                    ha="center", va="center", color=p["muted"], transform=ax.transAxes)
            return
        spec = res.spec
        xs, ys = spec.axes()
        two = mode in ("strata", "mu")
        axs = fig.subplots(1, 2, sharex=True, sharey=True) if two else [fig.add_subplot()]
        # the same shape in every view (the lens must not stretch between views)
        box = float(np.clip((ys[-1] - ys[0]) / (xs[-1] - xs[0]), 0.6, 1.15)) if not two else None
        for ax in (axs if not two else []):
            ax.set_box_aspect(box)
        if two:
            fig.get_layout_engine().set(wspace=0.08)
        handles = []
        for ax in axs:                          # not-yet-computed cells show the plain background
            ax.set_facecolor(p["plot_bg"])
        fail_handles = [Patch(color=_fail("unsafe"), label="not global: a pole fails the terminal test"),
                        Patch(color=_fail("blow"), label="not global: blows up before a pole")]
        if mode == "global":
            fill_failures(axs[0], res, xs, ys)
            fill(axs[0], xs, ys, res.is_global, GLOBAL_COL)
            unsafe = res.done & ~res.is_global & (res.code_n != BLOWUP) & (res.code_s != BLOWUP)
            fill(axs[0], xs, ys, unsafe, _fail("unsafe"), z=6)      # on top of the edge line
            handles += [Patch(color=GLOBAL_COL, label="global (in Σ)")] + fail_handles
        elif mode == "strata":
            for ax, code, title in ((axs[0], res.code_n, "north pole x = +1"),
                                    (axs[1], res.code_s, "south pole x = −1")):
                fill_failures(ax, res, xs, ys, code)
                for k, name in ((G, "G"), (R, "R"), (L1, "L1"), (L2, "L2")):
                    fill(ax, xs, ys, res.done & (code == k), _stratum_col(name))
                other = res.done & ~res.is_global & (code >= G)
                fill(ax, xs, ys, other, (0, 0, 0, 0.45), z=2)          # dimmed ...
                fill(ax, xs, ys, other, None, "////", z=2)             # ... with pale stripes
                ax.set_title(title, fontsize=10)
            present = set(np.unique(res.code_n[res.done])) | set(np.unique(res.code_s[res.done]))
            absent = []
            for k, name, lab in ((G, "G", "G: generic"), (R, "R", "R: recessive"),
                                 (L1, "L1", "L1: logarithmic"), (L2, "L2", "L2: logarithmic, on the wall")):
                if k in present:
                    handles.append(Patch(color=_stratum_col(name), label=lab))
                else:
                    absent.append(name)
            if absent:
                handles.append(Line2D([], [], ls="", label=", ".join(absent)
                                      + ": edge only, not resolved on this grid"))
            import matplotlib.colors as mc2
            dim = 0.55 * np.array(mc2.to_rgb(_stratum_col("G")))
            handles.append(Patch(facecolor=dim, hatch="////", edgecolor="#d9dde3",
                                 label="hatched: this pole is fine, the other pole fails"))
            codes = np.concatenate([res.code_n[res.done], res.code_s[res.done]])
            if (codes == UNSAFE).any():
                handles.append(Patch(color=_fail("unsafe"), label="this pole fails the terminal test"))
            if (codes == BLOWUP).any():
                handles.append(Patch(color=_fail("blow"), label="blows up before reaching this pole"))
        else:
            import matplotlib
            SEQUENTIAL = matplotlib.colormaps["cividis"]      # no green: green means 'global' elsewhere
            vmax = 1.5
            lv = np.linspace(0.0, vmax, 151)
            for ax, mu, title in ((axs[0], res.mu_n, "μ₊ at the north pole"), (axs[1], res.mu_s, "μ₋ at the south pole")):
                fill(ax, xs, ys, res.done & ~res.is_global, NEUTRAL)
                fill(ax, xs, ys, res.done & ~res.is_global, None, "x", z=1)     # sparse cross-hatch
                m = np.where(res.is_global, mu, np.nan)
                if np.isfinite(m).sum() > 4:
                    cf = ax.contourf(xs, ys, np.clip(m, 0, vmax), levels=lv, cmap=SEQUENTIAL, norm=MU_NORM(),
                                     zorder=1)
                mu_lines(ax, xs, ys, m, "#ffffff", "#101114")
                ax.set_title(title, fontsize=10)
            sm = matplotlib.cm.ScalarMappable(norm=MU_NORM(), cmap=SEQUENTIAL)
            cb = fig.colorbar(sm, ax=list(axs), orientation="horizontal", fraction=0.05, pad=0.03,
                                   aspect=30, shrink=0.8, extend="max", ticks=[0, 0.5, 2 / 3, 1.0, 1.5])
            cb.ax.set_xticklabels(["0", "½", "⅔", "1", "1.5"])
            gm_all = np.concatenate([res.mu_n[res.is_global], res.mu_s[res.is_global]])
            top = float(np.nanmax(gm_all)) if gm_all.size else vmax
            top = min(vmax, max(0.75, top * 1.03))
            cb.ax.set_xlim(0, top)                     # only the range present on this map
            for x0, x1, lab in ((0, 0.5, r"beyond III: $u_r$ worse than $1/d$"), (0.5, 2 / 3, "III"),
                                (2 / 3, 1.0, "II.5"), (1.0, 1.5, r"$u_r$ bounded")):
                if x0 >= top:
                    continue
                x1 = min(x1, top)
                cb.ax.text(0.5 * (x0 + x1), 1.25, lab, transform=cb.ax.get_xaxis_transform(), ha="center",
                           va="bottom", fontsize=8.5, color=p["plot_ink"])
                if x0 > 0:
                    cb.ax.axvline(x0, color="#101114", lw=1.2)
            cb.set_label(r"pole exponent $\mu = \sqrt{1 + Q/2}$ at a generic (G) pole: "
                         r"$u_r \sim d^{\,2\mu-2}$ near the axis",
                         fontsize=9)
            gmu = np.concatenate([res.mu_n[res.is_global], res.mu_s[res.is_global]])
            for lev, ls, lc in zip(MU_LEVELS, MU_STYLES, MU_COLORS):
                crosses = gmu.size and np.nanmin(gmu) < lev < np.nanmax(gmu)
                if crosses:
                    handles.append(Line2D([], [], color=lc, lw=1.6, ls=ls, label=f"μ = {MU_LABELS[lev]}"))
            handles.append(Patch(facecolor=NEUTRAL, hatch="x", edgecolor="#d9dde3",
                                 label="not global (see the global view for why)"))
        for ax in axs:
            if edge and res.done.sum() > 4:
                gm = np.where(res.done, res.is_global.astype(float), np.nan)
                ax.contour(xs, ys, gm, levels=[0.5], colors=[p["plot_ink"]], linewidths=1.4, zorder=5)
            if mu and mode != "mu":
                mu_lines(ax, xs, ys, np.where(res.is_global, res.mu_n, np.nan), "#ffffff", "#101114")
                mu_lines(ax, xs, ys, np.where(res.is_global, res.mu_s, np.nan), "#f2c14e", "#101114",
                               labels=False)
            px, py = 0.015 * (xs[-1] - xs[0]), 0.015 * (ys[-1] - ys[0])    # keep edge marks off the frame
            ax.set_xlim(xs[0] - px, xs[-1] + px)
            ax.set_ylim(ys[0] - py, ys[-1] + py)
            ax.set_xlabel(_name(spec.x_name), fontsize=11)
            ax.axhline(0, color=p["plot_ink2"], lw=0.5, alpha=0.5, zorder=3)
            ax.axvline(0, color=p["plot_ink2"], lw=0.5, alpha=0.5, zorder=3)
        axs[0].set_ylabel(_name(spec.y_name), fontsize=11)
        if _is_book_lens(spec) and mode == "global":   # the book's landmarks (Figure 1.3), to check by eye
            for ax in axs:
                for yk in (KAPPA_MAX_BOOK, -KAPPA_MAX_BOOK):
                    ax.plot([0], [yk], marker="_", ms=16, mew=2.2, color="#f2c14e", zorder=6)
                ax.plot([-2, 2], [0, 0], ls="", marker="|", ms=16, mew=2.2, color="#f2c14e", zorder=6)
                box = dict(boxstyle="round,pad=0.2", fc="#101114", ec="none", alpha=0.75)
                for yk, va, dy in ((KAPPA_MAX_BOOK, "bottom", 6), (-KAPPA_MAX_BOOK, "top", -6)):
                    ax.annotate(f"κ = {yk:+.4f}".replace("-", "−"), (0, yk), xytext=(-12, dy),
                                textcoords="offset points", fontsize=8.5, color="#f2c14e", va=va, ha="right",
                                bbox=box)
                for xk, ha, dx in ((2.0, "left", 8), (-2.0, "right", -8)):
                    ax.annotate(f"γ = {xk:+g}".replace("-", "−"), (xk, 0), xytext=(dx, -12),
                                textcoords="offset points", fontsize=8.5, color="#f2c14e", ha=ha, va="top",
                                bbox=box)
            handles.append(Line2D([], [], ls="", marker="|", ms=10, mew=2, color="#f2c14e",
                                  label="book's values (Fig. 1.3): apex κ_max, Landau window"))
        if mu and mode != "mu":
            handles += [Line2D([], [], color="#ffffff", lw=1.4, label="μ₊ contours (north): ½ dotted, ⅔ dashed, 1 dash-dot"),
                        Line2D([], [], color="#f2c14e", lw=1.4, label="μ₋ contours (south), same styles")]
        if edge:
            handles.append(Line2D([], [], color=p["plot_ink"], lw=1.4, label="edge of the global region"))
        draw_path(axs, spec, handles, path_overlay)
        # the current solution
        cur = current
        if cur is not None:
            cx, cy = getattr(cur, spec.x_name), getattr(cur, spec.y_name)
            same = all(abs(getattr(cur, n) - getattr(spec.base, n)) < 1e-12
                       for n in PARAM_NAMES if n not in (spec.x_name, spec.y_name)) \
                and abs(cur.xb - spec.base.xb) < 1e-12
            for ax in axs:
                ring = bool(path_overlay and path_overlay.get("keys"))
                if ring and same:          # a ring, so keyframe badges under it stay readable
                    ax.plot([cx], [cy], marker="o", ms=17, mfc="none", mec=MARK, mew=2.6, zorder=12,
                            clip_on=False)
                else:
                    ax.plot([cx], [cy], marker="o", ms=11, mfc=MARK if same else "none",
                            mec="white" if same else MARK, mew=2.0, zorder=12, clip_on=False)
            if path_overlay and path_overlay.get("keys"):
                handles.append(Line2D([], [], ls="", marker="o", ms=12, mfc="none", mec=MARK, mew=2.2,
                                      label="current solution (on the path: the preview position)"))
            else:
                handles.append(Line2D([], [], ls="", marker="o", ms=9, mfc=MARK, mec="white", mew=1.5,
                                      label="current solution"))
            if not same:
                handles.append(Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=MARK, mew=1.8,
                                      label="current solution (other fixed values: recompute)"))
        others = [n for n in PARAM_NAMES if n not in (spec.x_name, spec.y_name)]
        fixed = ",  ".join(f"{_name(n)} = {getattr(spec.base, n):.4g}" for n in others).replace("-", "−")
        fig.suptitle(f"Σ-slice in ({_name(spec.x_name)}, {_name(spec.y_name)})   at   {fixed},  "
                          f"x₀ = {spec.base.xb:g}", fontsize=10, color=p["plot_ink"])
        fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="outside lower center",
                        ncol=2, fontsize=8.5, frameon=False, handler_map=_legend_map())
    return list(axs)


def draw_path(axs, spec, handles, path_overlay):
    """The path dock's path projected on the map axes: samples, red where not global, numbered keys."""
    ov = path_overlay
    if not ov or not ov.get("keys"):
        return
    from matplotlib.lines import Line2D
    xn, yn = spec.x_name, spec.y_name
    xs = np.array([getattr(q, xn) for q in ov["samples"]])
    ys = np.array([getattr(q, yn) for q in ov["samples"]])
    kind = ov.get("kind")
    kind = np.asarray(kind) if kind is not None and len(kind) == len(xs) else np.array(["ok"] * len(xs))
    bad = kind != "ok"
    kx = [getattr(q, xn) for q in ov["keys"]]
    ky = [getattr(q, yn) for q in ov["keys"]]
    halo = "#101114" if style.MODE == "dark" else "#ffffff"
    for ax in axs:
        ax.plot(xs, ys, color=halo, lw=4.2, zorder=7, solid_capstyle="round")
        ax.plot(xs, ys, color="#ffffff" if style.MODE == "dark" else "#222222", lw=2.0, zorder=8)
        for k, ls in (("blowup", "-"), ("unsafe", "-")):
            m = kind == k
            if m.any():   # include the neighbours so short runs still show as segments
                mm = m | np.r_[m[1:], False] | np.r_[False, m[:-1]]
                yb = np.where(mm, ys, np.nan)
                ax.plot(xs, yb, color="#000000", lw=7.5, zorder=8.4, solid_capstyle="butt")
                ax.plot(xs, yb, color=PATH_FAIL[k], lw=4.5, ls=ls, zorder=8.5, solid_capstyle="butt")
        for q in ov.get("events", []):            # the timeline's event triangles, on the map
            ax.plot([getattr(q, xn)], [getattr(q, yn)], marker="v", ms=11, mfc="#f2c14e", mec=halo,
                    mew=1.0, zorder=10.5, ls="")
        ax.plot(kx, ky, ls="", marker="o", ms=13, mfc="#f2c14e", mec=halo, mew=1.2, zorder=11,
                clip_on=False)
        for i, (a, b) in enumerate(zip(kx, ky)):
            ax.annotate(str(i + 1), (a, b), ha="center", va="center", fontsize=7.5, color="#111111",
                        fontweight="bold", zorder=11.5, annotation_clip=False)
    handles.append(Line2D([], [], color="#ffffff" if style.MODE == "dark" else "#222222", lw=2,
                          marker="o", ms=8, mfc="#f2c14e", mec=halo, label="path, global part (numbered keyframes)"))
    labels_extra = []
    from matplotlib.legend_handler import HandlerTuple   # noqa: F401  (tuples drawn as overlays)
    if (kind == "blowup").any():
        handles.append((Line2D([], [], color="#000000", lw=7.5),
                        Line2D([], [], color=PATH_FAIL["blowup"], lw=4.5)))
        labels_extra.append("path: not global, blows up (as on the timeline)")
    if (kind == "unsafe").any():
        handles.append((Line2D([], [], color="#000000", lw=7.5),
                        Line2D([], [], color=PATH_FAIL["unsafe"], lw=4.5)))
        labels_extra.append("path: not global, a pole fails the terminal test")
    # tuple handles carry no label of their own: wrap them so the legend can name them
    for i, h in enumerate(handles):
        if isinstance(h, tuple):
            handles[i] = _Labelled(h, labels_extra.pop(0))
    if ov.get("events"):
        handles.append(Line2D([], [], ls="", marker="v", ms=9, mfc="#f2c14e", mec=halo,
                              label="path event (as on the timeline)"))
    other = [n for n in ov.get("varying", []) if n not in (xn, yn)]
    if other:
        handles.append(Line2D([], [], ls="", label="path also varies " + ", ".join(_name(n) for n in other)
                              + " (not shown on this map)"))



def readout(res, x, y) -> str:
    """One line about the grid point nearest (x, y)."""
    if res is None or x is None:
        return ""
    xs, ys = res.spec.axes()
    i, j = int(np.argmin(abs(xs - x))), int(np.argmin(abs(ys - y)))
    sx, sy = _name(res.spec.x_name), _name(res.spec.y_name)
    head = f"{sx} = {xs[i]:.4g},  {sy} = {ys[j]:.4g}:  ".replace("-", "−")
    if not res.done[j, i]:
        return head + "not computed yet"

    def pole(code, mu):
        if code == BLOWUP:
            return "blows up before it"
        if code == UNSAFE:
            return "fails the terminal test"
        return f"{CODE_NAMES[code]}, μ = {mu:.3f}"
    state = "GLOBAL" if res.is_global[j, i] else "not global"
    extra = ""
    if math.isfinite(res.blow_x[j, i]):
        extra = f" (blow-up at x ≈ {res.blow_x[j, i]:.3f})".replace("-", "−")
    return (head + f"{state}{extra}.   North: {pole(res.code_n[j, i], res.mu_n[j, i])}.   "
            f"South: {pole(res.code_s[j, i], res.mu_s[j, i])}.")

