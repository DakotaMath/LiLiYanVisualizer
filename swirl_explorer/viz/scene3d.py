"""PyVista mesh builders for the 3D view (no Qt). Every builder returns plain pyvista data.

Scalars that can colour streamlines / isosurfaces:
    speed     |u|                       (log colour scale)
    u_phi     azimuthal velocity        (diverging, symmetric)
    helicity  u · ω                     (diverging, symmetric)
    pressure  p                         (diverging, symmetric about 0)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pyvista as pv

from ..core.fields import FieldInterpolator
from ..core.poles import Summary
from .streamlines import Domain, TracedLines, stream_surface

STRATUM_COLORS = {"G": "#2a78d6", "R": "#e34948", "L1": "#eda100", "L2": "#8f3fd1",
                  "R0": "#e87ba4", "unsafe": "#8a8986", "blowup": "#8a8986", "?": "#8a8986"}
COLOR_BY = {"speed": "|u| (log scale)", "u_phi": "u_φ (swirl velocity)", "helicity": "helicity u·ω",
            "pressure": "pressure p"}
VTK_TITLES = {"speed": "speed (log scale)", "u_phi": "swirl velocity",
              "helicity": "helicity", "pressure": "pressure"}
DIVERGING = "coolwarm"     # grey midpoint, visible on the white background
def _truncated(name: str, lo: float = 0.22, hi: float = 1.0):
    import matplotlib
    from matplotlib.colors import ListedColormap
    base = matplotlib.colormaps[name]
    return ListedColormap(base(np.linspace(lo, hi, 256)), name=f"{name}_trunc")


SEQUENTIAL = _truncated("viridis")      # the darkest purples vanish on the dark background


def scalar_values(fi: FieldInterpolator, X: np.ndarray, which: str) -> np.ndarray:
    key = {"speed": "speed", "u_phi": "u_phi", "helicity": "helicity", "pressure": "pressure"}[which]
    return fi.scalar(X, key)


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


# ----------------------------------------------------------------- context geometry
def domain_outline(domain: Domain) -> pv.PolyData:
    if domain.shape == "box":
        return pv.Box(bounds=(-domain.R, domain.R) * 3).extract_feature_edges()
    # the equator and the meridian in the plane y = 0 (the cut plane): enough to read the ball
    t = np.linspace(0, 2 * np.pi, 145)
    eq = np.column_stack([domain.R * np.cos(t), domain.R * np.sin(t), np.zeros_like(t)])
    mer = np.column_stack([domain.R * np.cos(t), np.zeros_like(t), domain.R * np.sin(t)])
    return _polylines([eq, mer])


def swirl_arcs(fi: FieldInterpolator, domain: Domain) -> list[tuple[pv.PolyData, str]]:
    """Curved arrows around the axis at z = ±0.55 showing the sense of rotation (sign of u_φ)."""
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
        tube = pv.lines_from_points(arc).tube(radius=0.012, n_sides=10)
        tip, prev = arc[-1], arc[-4]
        d = tip - prev
        cone = pv.Cone(center=tip + 0.04 * d / np.linalg.norm(d), direction=d, height=0.11, radius=0.045,
                       resolution=20)
        out.append((tube.merge(cone), "counter-clockwise" if sgn > 0 else "clockwise"))
    return out


def _polylines(lines: list[np.ndarray]) -> pv.PolyData:
    pts = np.vstack(lines)
    cells = []
    off = 0
    for L in lines:
        cells.append(np.concatenate([[len(L)], np.arange(off, off + len(L))]))
        off += len(L)
    return pv.PolyData(pts, lines=np.concatenate(cells))     # lines only, no vertex cells


def axis_meshes(summary: Summary, domain: Domain,
                fi: Optional[FieldInterpolator] = None) -> list[tuple[pv.PolyData, str, str]]:
    """Two rods along the axis coloured by the pole stratum, each with a large cone showing the
    direction of the flow along the axis there (sign of u_z just outside the axis cut-out)."""
    out = []
    zmax = domain.R * 1.08
    for sign, diag, name in ((+1, summary.north, "N"), (-1, summary.south, "S")):
        z0, z1 = sign * domain.r_min, sign * zmax
        mesh = pv.Line((0, 0, z0), (0, 0, z1)).tube(radius=max(domain.axis_a * 0.6, 0.01))
        col = STRATUM_COLORS.get(diag.stratum, "#8a8986")
        if diag.stratum in ("blowup", "unsafe"):
            lab = f"{name}: not reached"
        else:
            lab = f"{name}: {diag.stratum}, μ={diag.mu:.3g}"
            if fi is not None:
                probe = np.array([[2.0 * domain.axis_a + 0.01, 0.0, sign * 0.8]])
                uz = fi.velocity(probe)[0, 2]
                if np.isfinite(uz) and uz != 0:
                    up = 1.0 if uz > 0 else -1.0
                    cone = pv.Cone(center=(0, 0, sign * 0.5), direction=(0, 0, up), height=0.26, radius=0.08,
                                   resolution=24)
                    mesh = mesh.merge(cone)
                    lab += ", " + ("outflow" if up * sign > 0 else "inflow")
        out.append((mesh, col, lab))
    return out


# ----------------------------------------------------------------- lines
def lines_polydata(traced: TracedLines, fi: FieldInterpolator, color_by: str) -> Optional[pv.PolyData]:
    if not traced.lines:
        return None
    pd = _polylines(traced.lines)
    pd["value"] = scalar_values(fi, pd.points, color_by)
    return pd


def arrowheads(traced: TracedLines, fi: FieldInterpolator, color_by: str,
               spacing: float = 0.6) -> Optional[pv.PolyData]:
    """Cones every ``spacing`` (arclength) along each line, pointing downstream."""
    P, D = [], []
    for L in traced.lines:
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
    if not P:
        return None
    pd = pv.PolyData(np.array(P))
    pd["dir"] = np.array(D)
    pd["value"] = scalar_values(fi, pd.points, color_by)
    cones = pd.glyph(orient="dir", scale=False, factor=0.12, geom=pv.Cone(radius=0.28, height=1.0, resolution=12))
    return cones


def surface_polydata(traced: TracedLines, fi: FieldInterpolator, color_by: str) -> Optional[pv.PolyData]:
    P, quads = stream_surface(traced.lines)
    if len(quads) == 0:
        return None
    faces = np.hstack([np.full((len(quads), 1), 4), quads]).ravel()
    pd = pv.PolyData(P, faces)
    pd["value"] = scalar_values(fi, P, color_by)
    return pd


# ----------------------------------------------------------------- volume sampling
def sample_grid(fi: FieldInterpolator, domain: Domain, n: int = 64) -> pv.ImageData:
    """Sample speed, swirl f/sin θ, u_φ, helicity and pressure on a cube; NaN outside the domain."""
    ax = np.linspace(-domain.R, domain.R, n)
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(ax[1] - ax[0],) * 3, origin=(ax[0],) * 3)
    P = grid.points
    ok = domain.inside(P)
    for key, name in (("speed", "speed"), ("swirl", "swirl"), ("helicity", "helicity"),
                      ("pressure", "pressure")):
        v = np.full(len(P), np.nan)
        v[ok] = fi.scalar(P[ok], key)
        grid[name] = v
    grid["valid"] = ok.astype(float)
    return grid


def isosurface(grid: pv.ImageData, field: str, level: float) -> Optional[pv.PolyData]:
    """Isosurface of ``field`` = level, cut away where the field is undefined."""
    v = np.array(grid[field])
    fill = np.nanmin(v) - 1.0 if level > np.nanmedian(v) else np.nanmax(v) + 1.0
    g = grid.copy()
    g[field] = np.where(np.isfinite(v), v, fill)
    g.set_active_scalars(field)
    try:
        surf = g.contour([level], scalars=field)
    except Exception:
        return None
    return surf if surf.n_points else None


def clean_isosurface(surf: pv.PolyData, domain: Domain, spacing: float) -> Optional[pv.PolyData]:
    """Drop the parts of an isosurface that hug the excluded ball/cylinder or the outer boundary."""
    if surf is None or surf.n_points == 0:
        return None
    P = surf.points
    r = np.linalg.norm(P, axis=1)
    rho = np.hypot(P[:, 0], P[:, 1])
    outer = r if domain.shape == "ball" else np.max(np.abs(P), axis=1)
    good = (r > domain.r_min + 1.5 * spacing) & (rho > domain.axis_a + 1.5 * spacing) & \
           (outer < domain.R - 1.0 * spacing)
    surf["good"] = good.astype(float)
    out = surf.threshold(0.5, scalars="good", preference="point", all_scalars=True)
    out = out.extract_surface(algorithm="dataset_surface") if out.n_points else None
    return out if out is not None and out.n_points else None


def glyphs(fi: FieldInterpolator, domain: Domain, n: int = 6, scale: float = 0.22) -> Optional[pv.PolyData]:
    """Arrow glyphs on a coarse grid; arrow length ∝ log(1 + |u|/median)."""
    ax = np.linspace(-domain.R, domain.R, n)
    X = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
    X = X[domain.inside(X)]
    if len(X) == 0:
        return None
    U = fi.velocity(X)
    ok = np.isfinite(U).all(axis=1)
    X, U = X[ok], U[ok]
    sp = np.linalg.norm(U, axis=1)
    med = np.median(sp) if len(sp) else 1.0
    length = np.log1p(sp / med)
    dirs = U / sp[:, None]
    pd = pv.PolyData(X)
    pd["vec"] = dirs * length[:, None]
    pd["speed"] = sp
    return pd.glyph(orient="vec", scale="vec", factor=scale, geom=pv.Arrow(tip_length=0.35, shaft_radius=0.04))
