"""The numbers behind the 3D view, without PyVista: colours, scalar values, axis labels, cone
positions, swirl arcs.  ``scene3d`` turns them into meshes for VTK; the browser version turns
them into WebGL geometry.  Both therefore show the same values and labels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.fields import FieldInterpolator
from ..core.poles import Summary
from .scene3d_colors import STRATUM_COLORS
from .slices import ROUNDOFF
from .streamlines import Domain

DIVERGING = "coolwarm"     # grey midpoint, visible on the white background


def _truncated(name: str, lo: float = 0.22, hi: float = 1.0):
    import matplotlib
    from matplotlib.colors import ListedColormap
    base = matplotlib.colormaps[name]
    return ListedColormap(base(np.linspace(lo, hi, 256)), name=f"{name}_trunc")


SEQUENTIAL = _truncated("viridis")      # the darkest purples vanish on the dark background


def scalar_values(fi: FieldInterpolator, X: np.ndarray, which: str) -> np.ndarray:
    key = {"speed": "speed", "u_phi": "u_phi", "helicity": "helicity", "pressure": "pressure"}[which]
    v = fi.scalar(X, key)
    if which in ("u_phi", "helicity") and len(v):
        # floating-point noise (e.g. u_φ ~ 1e-32 without swirl) is zero, not a colour pattern
        sp = fi.scalar(X, "speed")
        sp = sp[np.isfinite(sp)]
        ref = float(sp.max()) if sp.size else 0.0
        floor = ROUNDOFF * (ref if which == "u_phi" else ref * ref)
        v = np.where(np.abs(v) <= floor, 0.0, v)
    return v


def colour_spec(values: np.ndarray, which: str) -> dict:
    """Colour map, limits and log flag for a scalar."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return dict(cmap=SEQUENTIAL, clim=(0, 1), log_scale=False)
    if which == "speed":
        lo, hi = np.percentile(v[v > 0], [2, 98]) if (v > 0).any() else (1e-3, 1)
        lo = max(float(lo), 1e-6)                 # round-off speeds (rest state) must not invert the limits
        return dict(cmap=SEQUENTIAL, clim=(lo, max(float(hi), lo * 10)), log_scale=True)
    m = np.percentile(np.abs(v), 50)        # robust: the axis/origin spikes would wash out the map
    m = m if m > 0 else 1.0
    return dict(cmap=DIVERGING, clim=(-m, m), log_scale=False)


def outline_lines(domain: Domain) -> list[np.ndarray]:
    """The equator and the meridian in the plane y = 0 (the cut plane): enough to read the ball;
    for the box region, its twelve edges."""
    if domain.shape == "box":
        R = domain.R
        c = np.array([[x, y, z] for x in (-R, R) for y in (-R, R) for z in (-R, R)])
        return [c[[i, j]] for i in range(8) for j in range(i + 1, 8) if np.sum(c[i] != c[j]) == 1]
    t = np.linspace(0, 2 * np.pi, 145)
    eq = np.column_stack([domain.R * np.cos(t), domain.R * np.sin(t), np.zeros_like(t)])
    mer = np.column_stack([domain.R * np.cos(t), np.zeros_like(t), domain.R * np.sin(t)])
    return [eq, mer]


def axis_info(summary: Summary, domain: Domain, fi: Optional[FieldInterpolator] = None) -> list[dict]:
    """Per pole: the rod (z0, z1), its colour (pole stratum), the label, and the flow direction
    along the axis there (+1 up / −1 down / 0 unknown), from u_z just outside the axis cut-out."""
    out = []
    zmax = domain.R * 1.08
    for sign, diag, name in ((+1, summary.north, "N"), (-1, summary.south, "S")):
        col = STRATUM_COLORS.get(diag.stratum, "#8a8986")
        up = 0
        if diag.stratum == "blowup":
            lab = f"{name}: not reached (blow-up first)"
        elif diag.stratum == "unsafe":
            lab = f"{name}: fails the terminal test"
        else:
            lab = f"{name}: {diag.stratum}, μ={diag.mu:.3g}"
            if fi is not None:
                probe = np.array([[2.0 * domain.axis_a + 0.01, 0.0, sign * 0.8]])
                uz = fi.velocity(probe)[0, 2]
                if np.isfinite(uz) and uz != 0:
                    up = 1 if uz > 0 else -1
                    lab += ", " + ("outflow" if up * sign > 0 else "inflow")
        out.append(dict(sign=sign, z0=sign * domain.r_min, z1=sign * zmax, color=col, label=lab, flow=up,
                        radius=max(domain.axis_a * 0.6, 0.01)))
    return out


def arrow_points(lines: list[np.ndarray], spacing: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Positions and unit directions for cones every ``spacing`` (arclength) along each line."""
    P, D = [], []
    for L in lines:
        if len(L) < 3:
            continue
        seg = np.linalg.norm(np.diff(L, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        for t in np.arange(spacing / 2, s[-1], spacing):
            i = int(np.searchsorted(s, t))
            i = min(max(i, 1), len(L) - 1)
            d = L[i] - L[i - 1]
            n = np.linalg.norm(d)
            if n > 0:
                P.append(L[i])
                D.append(d / n)
    return (np.array(P).reshape(-1, 3), np.array(D).reshape(-1, 3))


def glyph_vectors(fi: FieldInterpolator, domain: Domain, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Arrow positions and vectors on a coarse n³ grid: unit direction of u times log(1 + |u|/median)."""
    ax = np.linspace(-domain.R, domain.R, n)
    X = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
    X = X[domain.inside(X)]
    if len(X) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    U = fi.velocity(X)
    ok = np.isfinite(U).all(axis=1)
    X, U = X[ok], U[ok]
    sp = np.linalg.norm(U, axis=1)
    med = np.median(sp) if len(sp) else 1.0
    length = np.log1p(sp / med)
    return X, U / sp[:, None] * length[:, None]


def iso_filled(v: np.ndarray, level: float) -> np.ndarray:
    """Grid values with the undefined points (outside the region) set to a value on the far side of
    the level, so the isosurface does not close along the cut-outs."""
    fill = np.nanmin(v) - 1.0 if level > np.nanmedian(v) else np.nanmax(v) + 1.0
    return np.where(np.isfinite(v), v, fill)


def iso_good_points(P: np.ndarray, domain: Domain, spacing: float) -> np.ndarray:
    """Isosurface points away from the excluded ball/cylinder and the outer boundary."""
    r = np.linalg.norm(P, axis=1)
    rho = np.hypot(P[:, 0], P[:, 1])
    outer = r if domain.shape == "ball" else np.max(np.abs(P), axis=1)
    return (r > domain.r_min + 1.5 * spacing) & (rho > domain.axis_a + 1.5 * spacing) &         (outer < domain.R - 1.0 * spacing)


def swirl_arc_points(fi: FieldInterpolator) -> list[tuple[np.ndarray, str]]:
    """Arcs around the axis at z = ±0.55 in the sense of rotation (sign of u_φ); the arrow is at the end."""
    out = []
    for z in (0.55, -0.55):
        rho = 0.32
        probe = np.array([[rho, 0.0, z]])
        uphi = fi.velocity(probe)[0, 1]          # at φ = 0, e_φ = e_y
        if not np.isfinite(uphi) or abs(uphi) < 1e-12:
            continue
        sgn = 1.0 if uphi > 0 else -1.0
        ph = np.linspace(-0.3 * np.pi, 0.85 * np.pi, 60) * sgn + np.pi    # visible side (x < 0)
        arc = np.column_stack([rho * np.cos(ph), rho * np.sin(ph), np.full_like(ph, z)])
        out.append((arc, "counter-clockwise" if sgn > 0 else "clockwise"))
    return out


def legend_items(data, s, cs, overlays, show_handle=False, slice_shown=False) -> list[str]:
    """The "What you see" lines (HTML), shared by the desktop 3D view and the browser version.
    `data` needs n_lines, surface, iso, iso_level, vortex, glyphs, selfsim, swirl_arcs."""
    import matplotlib
    from . import style
    from .scene3d_colors import COLOR_BY, ISO_FIELDS
    P = style.c()
    sw = "<span style='color:{c};font-size:15px'>■</span>&nbsp;"
    cm = matplotlib.colormaps[cs["cmap"]] if isinstance(cs["cmap"], str) else cs["cmap"]
    ramp = "".join(f"<span style='color:{matplotlib.colors.to_hex(cm(t))};font-size:15px'>■</span>"
                   for t in (0.05, 0.35, 0.65, 0.95)) + "&nbsp;"
    items = []
    if s.show_streamlines:
        if overlays:
            items.append(sw.format(c=P["outline"]) + f"streamlines of u ({data.n_lines}), dimmed "
                         "while vortex lines / rescalings are shown")
        else:
            items.append(ramp + f"streamlines of u ({data.n_lines}), coloured by "
                         f"{COLOR_BY[s.color_by]} (colour bar); small cones point downstream")
    if s.show_surface and data.surface is not None:
        items.append(ramp + f"stream surface (translucent), coloured by {COLOR_BY[s.color_by]}")
    if s.show_iso and data.iso is not None:
        items.append(sw.format(c="#9fb3c8") + f"isosurface {ISO_FIELDS[s.iso_field]} = {data.iso_level:.3g}")
    if s.show_vortex and data.vortex is not None:
        items.append(sw.format(c="#2e9d3a") + "vortex lines (curves along ω)")
    if s.show_glyphs and data.glyphs is not None:
        items.append(sw.format(c=P["view_text"]) + "arrows: direction of u; length grows with log |u|")
    if s.show_selfsim and data.selfsim:
        items.append(sw.format(c=P["view_text"]) + "thick: one streamline X(s); thin: its rescalings λX "
                     "(labelled), also streamlines since u is (−1)-homogeneous")
    if show_handle:
        items.append(sw.format(c="#d81b60") + ("magenta handle: drag it; its distance from the axis is the "
                                                 "ring radius" if s.seed_mode in ("rings", "ring") else
                                                 "magenta ball: drag to move the seeds, handles resize"))
    items.append(sw.format(c=STRATUM_COLORS["G"]) + "axis coloured by pole type (blue G, red R, amber L1, "
                 "purple L2); big cone = direction of the flow along the axis · " + sw.format(c="#9a9ca3")
                 + f"cut-outs: ball r &lt; {s.r_min:g} around 0, cylinder ρ &lt; {s.axis_a:g} around the axis"
                 + (" · near half y &lt; 0 cut away" if s.clip else ""))
    if data.swirl_arcs:
        items.append(sw.format(c="#e65100") + "orange arcs: sense of the swirl u_φ near each pole")
    items.append(sw.format(c=P["origin"]) + "grey ball: origin cut out (u ~ 1/|X| is singular there)")
    if slice_shown:
        items.append("<span style='color:#b2182b;font-size:15px'>■</span><span style='color:#2166ac;"
                     "font-size:15px'>■</span>&nbsp;slice plane: red = flow through it toward +n, blue = "
                     "against n; dark lines = in-plane streamlines")
    items.append(sw.format(c=P["outline"]) + "thin ellipses: equator and cut-plane meridian of |X| = 1")
    return items
