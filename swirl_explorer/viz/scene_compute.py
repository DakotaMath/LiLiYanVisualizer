"""Compute everything the 3D view shows for one solution + settings (runs in a worker thread)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time
from typing import Optional

import numpy as np

from ..core.fields import FieldInterpolator
from ..core.poles import Summary
from ..core.solver import Solution
from .streamlines import (Domain, trace, seeds_sphere, seeds_line, seeds_disc, seeds_ring, seeds_rings,
                          rescaled_copies,
                          TracedLines)


@dataclass
class SceneSettings:
    seed_mode: str = "rings"               # "rings" | "ring" | "sphere" | "line" | "disc"
    n_seeds: int = 40
    sphere_center: tuple = (0.35, 0.25, 0.45)
    sphere_radius: float = 0.18
    line_p0: tuple = (-0.7, 0.35, 0.35)
    line_p1: tuple = (0.7, 0.35, 0.35)
    color_by: str = "speed"
    show_streamlines: bool = True
    show_surface: bool = False
    show_vortex: bool = False
    show_glyphs: bool = False
    show_iso: bool = False
    iso_field: str = "speed"               # "speed" | "swirl" | "helicity"
    iso_quantile: float = 0.6              # level as a quantile of the sampled values
    show_selfsim: bool = False
    domain_shape: str = "ball"
    r_min: float = 0.12
    axis_a: float = 0.025
    opacity: float = 0.45
    clip: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSettings":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("sphere_center", "line_p0", "line_p1"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)


@dataclass
class SceneData:
    domain: Domain
    seeds: np.ndarray
    streamlines: Optional[object] = None
    arrows: Optional[object] = None
    surface: Optional[object] = None
    vortex: Optional[object] = None
    glyphs: Optional[object] = None
    iso: Optional[object] = None
    iso_level: float = float("nan")
    selfsim: list = field(default_factory=list)
    axis: list = field(default_factory=list)
    outline: Optional[object] = None
    origin_ball: Optional[object] = None
    swirl_arcs: list = field(default_factory=list)
    color_spec: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    n_lines: int = 0
    stop_reasons: dict = field(default_factory=dict)


def make_seeds(s: SceneSettings) -> np.ndarray:
    if s.seed_mode == "line":
        return seeds_line(s.line_p0, s.line_p1, s.n_seeds)
    if s.seed_mode == "rings":
        rho = max(float(np.hypot(*s.sphere_center[:2])), 0.08)
        near = seeds_rings(2.5 * s.axis_a + 0.01, 2, heights=(0.6, -0.6))   # a few lines for the axial flow
        return np.vstack([seeds_rings(rho, max(s.n_seeds // 5, 1)), near])
    if s.seed_mode == "ring":
        return seeds_ring(s.sphere_center, s.n_seeds)
    if s.seed_mode == "disc":
        return seeds_disc(s.sphere_center, (0, 0, 1), s.sphere_radius, s.n_seeds)
    return seeds_sphere(s.sphere_center, s.sphere_radius, s.n_seeds)


@dataclass
class NumericScene:
    """Everything that can be computed without VTK (safe to build in a worker thread)."""
    settings: "SceneSettings"
    domain: Domain
    fi: FieldInterpolator
    seeds: np.ndarray
    traced: Optional[TracedLines] = None
    surface_traced: Optional[TracedLines] = None
    vortex_traced: Optional[TracedLines] = None
    selfsim: list = field(default_factory=list)
    grid_axis: Optional[np.ndarray] = None
    grid_values: Optional[np.ndarray] = None
    iso_level: float = float("nan")
    timings: dict = field(default_factory=dict)


def iso_grid(fi: FieldInterpolator, dom: Domain, field: str, n: int = 56) -> tuple[np.ndarray, np.ndarray]:
    """The isosurface field sampled on an n³ grid ('ij' order), NaN outside the region."""
    ax = np.linspace(-dom.R, dom.R, n)
    P = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
    ok = dom.inside(P)
    vals = np.full(len(P), np.nan)
    vals[ok] = fi.scalar(P[ok], field)
    return ax, vals


def iso_level(vals: np.ndarray, field: str, quantile: float) -> float:
    """The isosurface level as a quantile of the sampled values (of log |u| for the speed)."""
    v = vals[np.isfinite(vals)]
    if not v.size:
        return float("nan")
    if field == "speed":
        return float(np.exp(np.quantile(np.log(v[v > 0]), quantile)))
    return float(np.quantile(v, quantile))


def compute_numeric(sol: Solution, s: SceneSettings, fi: FieldInterpolator | None = None) -> NumericScene:
    """Worker-thread part: interpolation tables, seeds, line tracing, grid sampling (NumPy only).
    ``fi`` may be passed in to reuse the interpolation tables of the same solution."""
    t0 = time.perf_counter()
    dom = Domain(R=1.0, r_min=s.r_min, axis_a=s.axis_a, shape=s.domain_shape)
    fi = fi if fi is not None else FieldInterpolator(sol, dxi=0.002, fast=True)
    seeds = make_seeds(s)
    num = NumericScene(settings=s, domain=dom, fi=fi, seeds=seeds)
    tm = num.timings
    if s.show_streamlines or s.show_surface:
        t = time.perf_counter()
        num.traced = trace(fi.velocity, seeds, dom)
        if s.show_surface:
            line_seeds = seeds_line(s.line_p0, s.line_p1, 40) if s.seed_mode == "line" else \
                seeds_line(np.asarray(s.sphere_center) - [s.sphere_radius, 0, 0],
                           np.asarray(s.sphere_center) + [s.sphere_radius, 0, 0], 40)
            num.surface_traced = trace(fi.velocity, line_seeds, dom, min_points=3)
        tm["streamlines"] = time.perf_counter() - t
    if s.show_vortex:
        t = time.perf_counter()
        num.vortex_traced = trace(fi.vorticity, seeds, dom)
        tm["vortex"] = time.perf_counter() - t
    if s.show_iso:
        t = time.perf_counter()
        num.grid_axis, num.grid_values = iso_grid(fi, dom, s.iso_field)
        num.iso_level = iso_level(num.grid_values, s.iso_field, s.iso_quantile)
        tm["iso"] = time.perf_counter() - t
    if s.show_selfsim:
        one = trace(fi.velocity, np.asarray(s.sphere_center, float)[None, :],
                    Domain(R=0.85, r_min=s.r_min, axis_a=s.axis_a, shape="ball"))
        if one.lines:
            base = one.lines[0]
            copies = [base] + rescaled_copies(base)
            num.selfsim = [L for L in (L[np.linalg.norm(L, axis=1) <= dom.R] for L in copies) if len(L) > 1]
    tm["numeric"] = time.perf_counter() - t0
    return num


def build_scene(num: NumericScene, summary: Summary) -> SceneData:
    """Main-thread part: all VTK / PyVista objects (VTK is not thread-safe)."""
    from . import scene3d          # PyVista: only needed here, not for the numbers (browser build)
    import pyvista as pv
    t0 = time.perf_counter()
    s, dom, fi = num.settings, num.domain, num.fi
    data = SceneData(domain=dom, seeds=num.seeds, timings=dict(num.timings))
    data.outline = scene3d.domain_outline(dom)
    data.origin_ball = pv.Sphere(radius=dom.r_min, theta_resolution=16, phi_resolution=12)
    data.axis = scene3d.axis_meshes(summary, dom, fi)
    if num.traced is not None:
        tl = num.traced
        data.n_lines = len(tl.lines)
        reasons: dict = {}
        for r in tl.reasons:
            for part in r.split(" / "):
                reasons[part] = reasons.get(part, 0) + 1
        data.stop_reasons = reasons
        if s.show_streamlines:
            data.streamlines = scene3d.lines_polydata(tl, fi, s.color_by)
            data.arrows = scene3d.arrowheads(tl, fi, s.color_by)
    if num.surface_traced is not None:
        data.surface = scene3d.surface_polydata(num.surface_traced, fi, s.color_by)
    if num.vortex_traced is not None:
        data.vortex = scene3d.lines_polydata(num.vortex_traced, fi, s.color_by)
    if s.show_glyphs:
        data.glyphs = scene3d.glyphs(fi, dom)
    if num.grid_values is not None and np.isfinite(num.iso_level):
        ax = num.grid_axis
        n = len(ax)
        grid = pv.ImageData(dimensions=(n, n, n), spacing=(ax[1] - ax[0],) * 3, origin=(ax[0],) * 3)
        # meshgrid 'ij' order -> VTK x-fastest order
        grid[s.iso_field] = num.grid_values.reshape(n, n, n).ravel(order="F")
        data.iso_level = num.iso_level
        surf = scene3d.isosurface(grid, s.iso_field, num.iso_level)
        data.iso = scene3d.clean_isosurface(surf, dom, grid.spacing[0])
        if data.iso is not None:
            data.iso["value"] = scene3d.scalar_values(fi, data.iso.points, s.color_by)
    data.selfsim = num.selfsim
    vals = [m["value"] for m in (data.streamlines, data.surface, data.vortex, data.iso) if m is not None]
    data.color_spec = scene3d.colour_spec(np.concatenate(vals) if vals else np.array([]), s.color_by)
    if s.clip:
        for name in ("streamlines", "arrows", "surface", "vortex", "glyphs", "iso"):
            m = getattr(data, name)
            if m is not None and m.n_points:
                setattr(data, name, m.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False))
    data.timings["build"] = time.perf_counter() - t0
    data.timings["total"] = data.timings.get("numeric", 0) + data.timings["build"]
    return data


def compute_scene(sol: Solution, summary: Summary, s: SceneSettings) -> SceneData:
    """Both stages in the calling thread (tests, scripts)."""
    return build_scene(compute_numeric(sol, s), summary)
