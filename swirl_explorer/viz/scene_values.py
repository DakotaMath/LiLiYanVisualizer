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
        return dict(cmap=SEQUENTIAL, clim=(max(lo, 1e-6), max(hi, lo * 10)), log_scale=True)
    m = np.percentile(np.abs(v), 50)        # robust: the axis/origin spikes would wash out the map
    m = m if m > 0 else 1.0
    return dict(cmap=DIVERGING, clim=(-m, m), log_scale=False)


def outline_lines(domain: Domain) -> list[np.ndarray]:
    """The equator and the meridian in the plane y = 0 (the cut plane): enough to read the ball."""
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
        if diag.stratum in ("blowup", "unsafe"):
            lab = f"{name}: not reached"
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
