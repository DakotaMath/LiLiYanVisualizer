"""Drawing a computed 3D scene into any PyVista plotter (no Qt).

Shared by the desktop 3D view (pyvistaqt) and the browser version (trame), so both show
exactly the same picture.  ``draw_scene`` replaces the scene actors (names ``sx_*``) and
returns the legend lines as small HTML snippets; ``draw_slice_plane`` does the same for the
slice plane (actors ``sl_*``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from . import style
from .scene3d import COLOR_BY, STRATUM_COLORS, VTK_TITLES, _polylines

ISO_FIELDS = {"speed": "|u|", "swirl": "swirl f / sin θ", "helicity": "helicity u·ω"}
SLICE_BARS = ("slice u.n", "slice rho*u.n")
SLICE_ACTORS = ("sl_plane", "sl_lines", "sl_normal", "sl_outline")


@dataclass
class SceneDrawn:
    legend: list[str] = field(default_factory=list)   # HTML snippets, one per line
    coloured: bool = False                             # is anything colour-mapped (does "colour by" matter)?


def clear_scene(pl) -> None:
    for n in list(pl.actors):
        if n.startswith("sx_"):
            pl.remove_actor(n, render=False)
    for title in list(pl.scalar_bars.keys()):           # the slice keeps its own bar
        if title not in SLICE_BARS:
            pl.remove_scalar_bar(title)


def draw_scene(pl, data, s, show_handle: bool = False, slice_shown: bool = False) -> SceneDrawn:
    """Draw `data` (a SceneData from build_scene) with settings `s` (SceneSettings) into `pl`."""
    clear_scene(pl)
    P = style.c()
    pl.add_mesh(data.outline, color=P["outline"], line_width=1, name="sx_outline", pickable=False)
    for (mesh, col, lab), nm in zip(data.axis, ("sx_axisN", "sx_axisS")):
        pl.add_mesh(mesh, color=col, name=nm, smooth_shading=True)
    for (mesh, col, lab), z, nm in zip(data.axis, (1.1, -1.1), ("sx_labN", "sx_labS")):
        pl.add_point_labels(np.array([[0, 0, z]]), [lab], name=nm, text_color="white", font_size=13,
                            shape_color=col, shape_opacity=0.95, show_points=False, always_visible=False)
    if s.clip:
        disc = pv.Disc(center=(0, 0, 0), normal=(0, 1, 0), inner=0.0, outer=data.domain.R, c_res=72)
        pl.add_mesh(disc, color=P["cut_plane"], opacity=0.07, name="sx_cutplane")
    cs = data.color_spec
    hi = max(abs(cs["clim"][0]), abs(cs["clim"][1]))
    fmt = "%.0f" if hi >= 100 else "%.3g"
    coloured = bool((s.show_streamlines and not ((s.show_selfsim and data.selfsim)
                                                 or (s.show_vortex and data.vortex is not None)))
                    or (s.show_surface and data.surface is not None))
    if coloured:
        pl.add_text(VTK_TITLES[s.color_by], position=(0.04, 0.08), viewport=True, font_size=8,
                    color=P["view_text"], name="sx_bartitle")
    bar = dict(title="", color=P["view_text"], vertical=False, position_x=0.04,
               position_y=0.02, height=0.045, width=0.42, title_font_size=1, label_font_size=10,
               n_labels=5, fmt=fmt)
    first = True

    def add_coloured(mesh, name, **kw):
        nonlocal first
        if mesh is None or mesh.n_points == 0:
            return
        pl.add_mesh(mesh, scalars="value", cmap=cs["cmap"], clim=cs["clim"], log_scale=cs["log_scale"],
                    name=name, show_scalar_bar=first, scalar_bar_args=bar, nan_opacity=0.0, **kw)
        first = False

    overlays = bool((s.show_selfsim and data.selfsim) or (s.show_vortex and data.vortex is not None))
    if data.streamlines is not None:
        if overlays:
            pl.add_mesh(data.streamlines, color=P["outline"], line_width=1, opacity=0.5, name="sx_streamlines")
        else:
            add_coloured(data.streamlines.tube(radius=0.007, n_sides=10), "sx_streamlines", smooth_shading=True)
            if data.arrows is not None:
                add_coloured(data.arrows, "sx_arrows", smooth_shading=True)
    if data.origin_ball is not None:
        pl.add_mesh(data.origin_ball, color="#8a8c92", ambient=0.3, name="sx_origin")
        pl.add_point_labels(np.array([[-0.75, 0.0, -0.3]]), ["origin (cut out)"],
                            name="sx_origin_lab", font_size=11, text_color=P["view_text"], shape_opacity=0.0,
                            show_points=False, always_visible=False)
    for i, (arc, sense) in enumerate(data.swirl_arcs):
        pl.add_mesh(arc, color="#e65100", name=f"sx_arc{i}", smooth_shading=True)
    if s.seed_mode in ("ring", "rings") and show_handle:
        c = np.asarray(s.sphere_center, float)
        guide = pv.Line((0.0, 0.0, c[2]), tuple(c))
        pl.add_mesh(guide, color="#d81b60", line_width=2, name="sx_guide")

    add_coloured(data.surface, "sx_surface", opacity=s.opacity, smooth_shading=True)
    if data.iso is not None and data.iso.n_points:      # one neutral colour: does not compete with the lines
        pl.add_mesh(data.iso, color="#9fb3c8", opacity=s.opacity, smooth_shading=True, name="sx_iso")
    if data.vortex is not None and data.vortex.n_points:
        pl.add_mesh(data.vortex.tube(radius=0.006, n_sides=8), color="#2e9d3a", name="sx_vortex",
                    smooth_shading=True)
    if data.glyphs is not None:
        pl.add_mesh(data.glyphs, color=P["view_text"], name="sx_glyphs", smooth_shading=True)
    lam = (1.0, 0.6, 0.8, 1.25)
    for i, L in enumerate(data.selfsim[:4]):
        tube = pv.lines_from_points(L).tube(radius=(0.016, 0.006, 0.008, 0.011)[i], n_sides=10)
        pl.add_mesh(tube, color=P["view_text"], name=f"sx_selfsim{i}", smooth_shading=True)
    if data.selfsim:
        fr = (0.85, 0.15, 0.4, 0.65)
        ends = np.array([L[int(fr[i] * (len(L) - 1))] for i, L in enumerate(data.selfsim[:4])])
        pl.add_point_labels(ends, ["X(s)"] + [f"{l}·X" for l in lam[1:len(ends)]], name="sx_selfsim_lab",
                            font_size=12, text_color=P["view_text"], shape_opacity=0.6, shape_color=P["view_bg"],
                            show_points=False, always_visible=False)
    return SceneDrawn(_legend(data, s, cs, overlays, show_handle, slice_shown), coloured)


def _legend(data, s, cs, overlays, show_handle, slice_shown) -> list[str]:
    import matplotlib
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
                 + (" · half y > 0 cut away" if s.clip else ""))
    if data.swirl_arcs:
        items.append(sw.format(c="#e65100") + "orange arcs: sense of the swirl u_φ near each pole")
    items.append(sw.format(c=P["origin"]) + "grey ball: origin cut out (u ~ 1/|X| is singular there)")
    if slice_shown:
        items.append("<span style='color:#b2182b;font-size:15px'>■</span><span style='color:#2166ac;"
                     "font-size:15px'>■</span>&nbsp;slice plane: red = flow through it toward +n, blue = "
                     "against n; dark lines = in-plane streamlines")
    items.append(sw.format(c=P["outline"]) + "thin ellipses: equator and cut-plane meridian of |X| = 1")
    return items


def clear_slice_plane(pl) -> None:
    for n in SLICE_ACTORS:
        pl.remove_actor(n, render=False)
    for t in SLICE_BARS:
        if t in pl.scalar_bars:
            pl.remove_scalar_bar(t)


def draw_slice_plane(pl, data, lines) -> None:
    """The slice plane in 3D: coloured by the slice's display values, with its in-plane streamlines."""
    clear_slice_plane(pl)
    X = data.X
    grid = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2])
    if data.show is None:
        data.set_display(False)
    grid["un"] = data.show.ravel(order="F")
    m = data.clim
    pl.add_mesh(grid, scalars="un", cmap="RdBu_r", clim=(-m, m), nan_opacity=0.0, opacity=0.95,
                show_scalar_bar=False, name="sl_plane", smooth_shading=False, show_edges=False,
                lighting=False,                       # unlit: the same colours as the 2D slice
                scalar_bar_args=dict(title="slice u.n" if data.show_label == "u·n" else "slice rho*u.n",
                                     n_labels=3, vertical=False, position_x=0.53, position_y=0.02, width=0.42,
                                     height=0.045, color=style.c()["view_text"], fmt="%.2g",
                                     title_font_size=11, label_font_size=10))
    d = data.spec.offset
    if abs(d) < 1.0:                               # the region is the unit ball
        rr = math.sqrt(1.0 - d * d)
        th = np.linspace(0, 2 * np.pi, 145)
        ring = data.spec.to_world(rr * np.cos(th), rr * np.sin(th))
        pl.add_mesh(pv.lines_from_points(ring), color="#f2c14e", line_width=2, name="sl_outline")
    lift = 0.006 * data.spec.normal                # draw the lines just above the plane (no z-fighting)
    polys = [data.spec.to_world(L[:, 0], L[:, 1]) + lift for L in lines[::2] if len(L) > 1]
    if polys:
        pl.add_mesh(_polylines(polys), color="#111111", line_width=1.0, name="sl_lines")
    c, n = data.spec.center, data.spec.normal
    pl.add_mesh(pv.Arrow(start=c, direction=n, scale=0.35, tip_radius=0.08, shaft_radius=0.025),
                color="#f2c14e", name="sl_normal")


OBLIQUE_CAMERA = [(2.7, -2.9, 2.4), (0, 0, 0), (0, 0, 1)]
POLE_CAMERA = [(0, 0, 6), (0, 0, 0), (0, 1, 0)]
SCENE_BOUNDS = (-1.1, 1.1, -1.1, 1.1, -1.3, 1.3)


def oblique_camera(pl, zoom: float = 1.85) -> None:
    pl.camera_position = OBLIQUE_CAMERA
    pl.reset_camera(bounds=SCENE_BOUNDS)
    pl.camera.zoom(zoom)


def pole_camera(pl) -> None:
    pl.camera_position = POLE_CAMERA
    pl.reset_camera(bounds=SCENE_BOUNDS)
