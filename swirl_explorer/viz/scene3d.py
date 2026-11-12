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
from .slices import ROUNDOFF

from .scene3d_colors import STRATUM_COLORS, COLOR_BY, VTK_TITLES  # noqa: F401  (re-exported)
from .scene_values import (DIVERGING, SEQUENTIAL, _truncated, scalar_values, colour_spec,  # noqa: F401
                           outline_lines, axis_info, arrow_points, swirl_arc_points)


# ----------------------------------------------------------------- context geometry
def domain_outline(domain: Domain) -> pv.PolyData:
    if domain.shape == "box":
        return pv.Box(bounds=(-domain.R, domain.R) * 3).extract_feature_edges()
    return _polylines(outline_lines(domain))


def swirl_arcs(fi: FieldInterpolator, domain: Domain) -> list[tuple[pv.PolyData, str]]:
    """Curved arrows around the axis at z = ±0.55 showing the sense of rotation (sign of u_φ)."""
    out = []
    for arc, sense in swirl_arc_points(fi):
        tube = pv.lines_from_points(arc).tube(radius=0.012, n_sides=10)
        tip, prev = arc[-1], arc[-4]
        d = tip - prev
        cone = pv.Cone(center=tip + 0.04 * d / np.linalg.norm(d), direction=d, height=0.11, radius=0.045,
                       resolution=20)
        out.append((tube.merge(cone), sense))
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
    for ai in axis_info(summary, domain, fi):
        mesh = pv.Line((0, 0, ai["z0"]), (0, 0, ai["z1"])).tube(radius=ai["radius"])
        if ai["flow"]:
            cone = pv.Cone(center=(0, 0, ai["sign"] * 0.5), direction=(0, 0, ai["flow"]), height=0.26,
                           radius=0.08, resolution=24)
            mesh = mesh.merge(cone)
        out.append((mesh, ai["color"], ai["label"]))
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
    P, D = arrow_points(traced.lines, spacing)
    if not len(P):
        return None
    pd = pv.PolyData(P)
    pd["dir"] = D
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
