"""The browser bridge: everything the web page asks of Python, with JSON-friendly inputs and outputs.

Runs inside Pyodide (Python compiled to WebAssembly) in a Web Worker; also runs in ordinary
CPython, which is how the tests exercise it.  Needs only numpy, scipy and matplotlib: no Qt, no
VTK.  Plots are drawn by the same matplotlib code as the desktop app (``viz.plots2d``,
``viz.sigma_plot``, ``viz.slice_plot``) and returned as PNG data URLs; the 3D scene is returned
as plain arrays (base64 float32 / uint8) that the page turns into WebGL geometry.
"""
from __future__ import annotations

import base64
import io
import math
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from .. import __version__  # noqa: E402
from ..core.params import Params, PARAM_NAMES, PARAM_UI, DEFAULT_RANGES  # noqa: E402
from ..core.solver import solve as _solve, is_global_fast  # noqa: E402
from ..core.poles import summarize  # noqa: E402
from ..core.edges import gamma_minus  # noqa: E402
from ..core.sigma_map import SigmaMapSpec, SigmaMapResult, quick_pole  # noqa: E402
from ..core.paths import ParamPath, scan_path, failure_kind, _why  # noqa: E402
from ..viz import style  # noqa: E402
from ..viz.plots2d import QUANTITIES, XVARS, Curve, draw_profiles, PIN_COLORS  # noqa: E402
from ..viz.scene3d_colors import COLOR_BY, VTK_TITLES, STRATUM_COLORS  # noqa: E402
from ..viz.scene_values import (scalar_values, colour_spec, outline_lines, axis_info, arrow_points,  # noqa: E402
                                swirl_arc_points)
from ..viz.status_text import head_html, pole_line  # noqa: E402

_S: dict = {"p": None, "sol": None, "sm": None, "pins": [], "sigma": None}


# ----------------------------------------------------------------------------- helpers
def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _png(fig: Figure, dpi: float) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _figure(w: int, h: int, ratio: float = 1.0) -> tuple[Figure, float]:
    """A figure of w × h CSS pixels, drawn at the device pixel ratio for sharp text."""
    dpi = 100.0 * max(1.0, min(ratio, 3.0))
    fig = Figure(figsize=(max(w, 50) / 100.0, max(h, 50) / 100.0), layout="constrained")
    fig.set_facecolor(style.c()["plot_bg"])
    return fig, dpi


def _params(d: dict) -> Params:
    return Params(*(float(d.get(n, 0.0)) for n in PARAM_NAMES), xb=float(d.get("xb", 0.0)))


def set_theme(mode: str) -> None:
    style.set_mode("light" if mode == "light" else "dark")


# ----------------------------------------------------------------------------- static info
def info() -> dict:
    """Names, labels, ranges and option lists for building the page."""
    from ..viz.sigma_plot import EXPLAIN, CLICK_HELP
    from ..viz.slices import PRESETS
    return {
        "version": __version__,
        "sigma_explain": EXPLAIN, "sigma_click": CLICK_HELP.replace("(Path dock)", "(Path &amp; movie panel)"),
        "slice_presets": [{"label": k, "az": v[0], "el": v[1]} for k, v in PRESETS.items()],
        "start": Params(gamma=-1.6806045, kappa=1.0).to_dict(),
        "params": [{"name": n, "label": PARAM_UI[n][0], "tip": PARAM_UI[n][1], "range": DEFAULT_RANGES[n]}
                   for n in PARAM_NAMES],
        "quantities": [{"key": k, "label": lab} for k, lab in (
            ("U", "U"), ("f", "f"), ("Q", "Q"), ("z", "z"), ("W", "W = 1+Q/2"), ("h", "h = z′/z"),
            ("u_r", "u_r"), ("u_theta", "u_θ"), ("u_phi", "u_φ"))],
        "xvars": [{"key": "xi", "label": "ξ = artanh x"}, {"key": "x", "label": "x = cos θ"},
                  {"key": "theta", "label": "θ (degrees)"}, {"key": "s_north", "label": "s = −ln(1−x) (north)"},
                  {"key": "s_south", "label": "s = −ln(1+x) (south)"}],
        "modes": [{"key": "linear", "label": "linear"}, {"key": "log", "label": "log |·|"},
                  {"key": "loglog", "label": "log–log near a pole"}],
        "color_by": [{"key": k, "label": v} for k, v in COLOR_BY.items()],
        "presets": [{"key": k, "label": lab, "tip": tip} for k, lab, tip in (
            ("rest", "Rest state", "c = 0, γ = A = κ = 0: U ≡ 0, no flow"),
            ("landau", "Landau jet…", "no swirl: U = 2(1−x²)/(λ_L + x), global iff |λ_L| ≥ 1"),
            ("default", "Generic pole (type G)", "inside the lens: c = 0, κ = 1, γ = γ₋ + 0.1 (monograph, Numerical observation 4.28)"),
            ("lens_edge", "Lens edge (type L1)", "on the lower lens edge: c = 0, κ = 1, γ = γ₋ (north pole logarithmic, Obs. 4.28)"),
            ("recessive", "Recessive edge (type R)", "lens edge for c = (0, −0.9, 0), κ = 0.3: north pole recessive, μ ≈ 0.22 (Obs. 4.28)"))],
    }


def preset(key: str, landau: float = 1.5) -> dict:
    """The desktop presets, with the same values."""
    if key == "rest":
        p = Params()
    elif key == "landau":
        p = Params(gamma=2.0 / landau) if landau else Params()
    elif key == "lens_edge":
        p = Params(gamma=gamma_minus(Params(kappa=1.0)) + 1e-11, kappa=1.0)
    elif key == "recessive":
        q = Params(c2=-0.9, kappa=0.3)
        p = q.with_(gamma=gamma_minus(q) + 1e-11)
    else:                                   # generic pole (type G): the start-up solution
        p = Params(gamma=gamma_minus(Params(kappa=1.0)) + 0.1, kappa=1.0)
    return p.to_dict()


# ----------------------------------------------------------------------------- the global lock
def lock(pdict: dict, name: str, old: float, new: float) -> dict:
    """The desktop's 'lock to global solutions': an edit that leaves the global set stops at the edge."""
    base = _params(pdict)
    at = lambda v: base.with_(**{name: v})
    if is_global_fast(at(new)) or not is_global_fast(at(old)):
        return {"value": new, "note": ""}
    g, b = old, new
    tol = 1e-9 * max(1.0, abs(old), abs(new))
    while abs(b - g) > tol:
        mid = 0.5 * (g + b)
        if is_global_fast(at(mid)):
            g = mid
        else:
            b = mid
    side = "maximum" if new > old else "minimum"
    return {"value": g, "note": f"Locked: {PARAM_UI[name][0]} held at its global {side} {g:.7g} "
                                "(untick 'lock to global solutions' to go past it)"}


def is_global(pdict: dict) -> bool:
    return bool(is_global_fast(_params(pdict)))


# ----------------------------------------------------------------------------- solve + status
def solve(pdict: dict) -> dict:
    t = time.perf_counter()
    p = _params(pdict)
    sol = _solve(p)
    sm = summarize(sol)
    _S.update(p=p, sol=sol, sm=sm)
    north, ntip = pole_line("North x = +1", sm.north, +1)
    south, stip = pole_line("South x = −1", sm.south, -1)
    return {"params": p.to_dict(), "global": bool(sm.is_global), "head": head_html(sm),
            "north": north, "south": south, "north_tip": ntip, "south_tip": stip,
            "status_line": sm.status_line(), "ms": 1e3 * (time.perf_counter() - t)}


# ----------------------------------------------------------------------------- profile plots
def pin() -> int:
    if _S["sol"] is not None and len(_S["pins"]) < len(PIN_COLORS):
        p = _S["p"]
        names = {"c1": "c₁", "c2": "c₂", "c3": "c₃", "gamma": "γ", "A": "A", "kappa": "κ"}
        parts = [f"{names[k]}={v:.4g}" for k, v in zip(names, p.as_tuple()) if v != 0.0]
        lab = (f"pin {len(_S['pins']) + 1}: " + (", ".join(parts) or "rest state")).replace("-", "−")
        _S["pins"].append(Curve(lab, _S["sol"], _S["sm"], PIN_COLORS[len(_S["pins"])], pinned=True))
    return len(_S["pins"])


def clear_pins() -> int:
    _S["pins"].clear()
    return 0


def profiles(opts: dict) -> dict:
    if _S["sol"] is None:
        return {"png": ""}
    from ..viz.plots2d import CURRENT
    fig, dpi = _figure(opts.get("w", 600), opts.get("h", 700), opts.get("ratio", 1.0))
    curves = list(_S["pins"]) + [Curve("current", _S["sol"], _S["sm"], CURRENT)]
    q = [k for k in opts.get("quantities", ["U", "f", "Q"]) if k in QUANTITIES] or ["U"]
    xvar = opts.get("xvar", "xi")
    draw_profiles(fig, curves, q, xvar=xvar if xvar in XVARS else "xi", mode=opts.get("mode", "linear"),
                  pole=opts.get("pole", "north"), overlay=bool(opts.get("overlay", True)))
    return {"png": _png(fig, dpi)}


# ----------------------------------------------------------------------------- 3D scene
def _colours(values: np.ndarray, cs: dict) -> np.ndarray:
    import matplotlib.colors as mc
    cmap = matplotlib.colormaps[cs["cmap"]] if isinstance(cs["cmap"], str) else cs["cmap"]
    lo, hi = cs["clim"]
    v = np.nan_to_num(values, nan=lo)
    norm = mc.LogNorm(lo, hi, clip=True) if cs["log_scale"] else mc.Normalize(lo, hi, clip=True)
    if cs["log_scale"]:
        v = np.clip(v, lo, None)
    return (np.asarray(cmap(norm(v)))[:, :3] * 255).astype(np.uint8)


def _lines_payload(lines: list[np.ndarray], fi=None, color_by: str | None = None, cs: dict | None = None,
                   step: int = 1) -> dict:
    lines = [L[::step] if len(L) > 3 * step else L for L in lines if len(L) > 1]
    if not lines:
        return {"n": 0}
    P = np.vstack(lines)
    out = {"n": len(lines), "points": _b64(P, np.float32), "offsets": np.cumsum([0] + [len(L) for L in lines]).tolist()}
    if fi is not None and color_by and cs is not None:
        out["colors"] = _b64(_colours(scalar_values(fi, P, color_by), cs), np.uint8)
    return out


def scene(settings: dict) -> dict:
    """Streamlines and context of the current solution, as arrays for WebGL."""
    if _S["sol"] is None:
        return {}
    from ..viz.scene_compute import SceneSettings, compute_numeric
    from ..viz.streamlines import stream_surface
    t0 = time.perf_counter()
    s = SceneSettings.from_dict(settings)
    num = compute_numeric(_S["sol"], s)
    fi, dom = num.fi, num.domain
    traced = num.traced.lines if num.traced is not None else []
    surf = None
    if s.show_surface and num.surface_traced is not None and num.surface_traced.lines:
        SP, quads = stream_surface(num.surface_traced.lines)
        if len(quads):
            surf = (SP, quads)
    # one colour scale over everything that is colour-mapped (as the desktop: lines, surface, vortex)
    vals = [scalar_values(fi, np.vstack(traced), s.color_by)] if traced else []
    if surf is not None:
        vals.append(scalar_values(fi, surf[0], s.color_by))
    if s.show_vortex and num.vortex_traced is not None and num.vortex_traced.lines:
        vals.append(scalar_values(fi, np.vstack(num.vortex_traced.lines), s.color_by))
    cs = colour_spec(np.concatenate(vals) if vals else np.array([]), s.color_by)
    overlays = bool((s.show_selfsim and num.selfsim) or (s.show_vortex and num.vortex_traced is not None))
    out: dict = {"color_by": s.color_by, "overlays": overlays, "clip": bool(s.clip)}
    if s.show_streamlines:
        out["streamlines"] = _lines_payload(traced, fi, s.color_by, cs)
        if not overlays:
            P, D = arrow_points(traced)
            if len(P):
                out["arrows"] = {"n": len(P), "points": _b64(P, np.float32), "dirs": _b64(D, np.float32),
                                 "colors": _b64(_colours(scalar_values(fi, P, s.color_by), cs), np.uint8)}
    if surf is not None:
        SP, quads = surf
        if True:
            tri = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
            out["surface"] = {"points": _b64(SP, np.float32), "triangles": _b64(tri, np.uint32),
                              "colors": _b64(_colours(scalar_values(fi, SP, s.color_by), cs), np.uint8),
                              "opacity": s.opacity}
    if s.show_vortex and num.vortex_traced is not None:
        out["vortex"] = _lines_payload(num.vortex_traced.lines)
    if s.show_selfsim and num.selfsim:
        out["selfsim"] = _lines_payload(num.selfsim)
    out["axis"] = axis_info(_S["sm"], dom, fi)
    out["outline"] = _lines_payload(outline_lines(dom))
    out["origin_r"] = dom.r_min
    out["arcs"] = [{"points": _b64(a, np.float32), "n": len(a), "sense": sense} for a, sense in swirl_arc_points(fi)]
    # the colour bar
    cmap = matplotlib.colormaps[cs["cmap"]] if isinstance(cs["cmap"], str) else cs["cmap"]
    out["colorbar"] = {"title": VTK_TITLES[s.color_by], "clim": [float(c) for c in cs["clim"]],
                       "log": bool(cs["log_scale"]),
                       "stops": [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, 9)]}
    out["n_lines"] = len(traced)
    out["stops"] = {k: v for k, v in sorted(getattr(num.traced, "stop_reasons", {}).items())} if num.traced else {}
    out["legend"] = _legend(out, s, cs, overlays)
    out["ms"] = 1e3 * (time.perf_counter() - t0)
    return out


def _legend(out: dict, s, cs, overlays) -> list[dict]:
    """The 'what you see' list: one entry per element, each with its colour swatch(es)."""
    ramp = out["colorbar"]["stops"][1::2]
    items = []
    if s.show_streamlines:
        if overlays:
            items.append({"colors": [style.c()["outline"]], "text": f"streamlines of u ({out['n_lines']}), dimmed "
                          "while vortex lines / rescalings are shown"})
        else:
            items.append({"colors": ramp, "text": f"streamlines of u ({out['n_lines']}), coloured by "
                          f"{COLOR_BY[s.color_by]} (colour bar); small cones point downstream"})
    if "surface" in out:
        items.append({"colors": ramp, "text": f"stream surface (translucent), coloured by {COLOR_BY[s.color_by]}"})
    if "vortex" in out:
        items.append({"colors": ["#2e9d3a"], "text": "vortex lines (curves along ω)"})
    if "selfsim" in out:
        items.append({"colors": [style.c()["view_text"]], "text": "thick: one streamline X(s); thin: its rescalings "
                      "λX = 0.6X, 0.8X, 1.25X, also streamlines since u is (−1)-homogeneous"})
    items.append({"colors": [STRATUM_COLORS["G"], STRATUM_COLORS["R"], STRATUM_COLORS["L1"]],
                  "text": "axis coloured by pole type (blue G, red R, amber L1, purple L2); the cone shows the "
                          "direction of the flow along the axis"})
    if out["arcs"]:
        items.append({"colors": ["#e65100"], "text": "orange arcs: sense of the swirl u_φ near each pole"})
    items.append({"colors": [style.c()["origin"]], "text": f"grey ball: origin cut out (u ~ 1/|X| is singular there); "
                  f"cut-outs: ball r < {s.r_min:g} around 0, cylinder ρ < {s.axis_a:g} around the axis"
                  + (" · near half y < 0 cut away" if s.clip else "")})
    items.append({"colors": [style.c()["outline"]], "text": "thin circles: equator and meridian of |X| = 1"})
    return items


# ----------------------------------------------------------------------------- slice
def slice_view(plane: dict, opts: dict) -> dict:
    """The 2D slice (PNG) and its colour image for texturing the plane in 3D."""
    if _S["sol"] is None:
        return {}
    from ..core.fields import FieldInterpolator
    from ..viz.slices import PlaneSpec, sample_plane, lic, projected_streamlines
    from ..viz.streamlines import Domain
    from ..viz.slice_plot import draw_slice
    t0 = time.perf_counter()
    spec = PlaneSpec.from_dict({**plane, "n": int(opts.get("n", 121))})
    dom = Domain(R=1.0, r_min=float(opts.get("r_min", 0.12)), axis_a=float(opts.get("axis_a", 0.025)))
    fi = FieldInterpolator(_S["sol"], dxi=0.002, fast=True)
    data = sample_plane(fi, spec, dom)
    if opts.get("lic"):
        data.lic = lic(data.us, data.ut)
    lines = projected_streamlines(data, density=float(opts.get("density", 1.4)) * (0.5 if opts.get("lic") else 1.0)) \
        if opts.get("lines", True) else []
    data.set_display(bool(opts.get("rho")), bool(opts.get("log")))
    fig, dpi = _figure(opts.get("w", 520), opts.get("h", 640), opts.get("ratio", 1.0))
    draw_slice(fig, data, lines, dom, show_colour=True)
    # the same colours as a small texture for the plane in 3D
    import matplotlib.colors as mc
    m = data.clim
    rgba = matplotlib.colormaps["RdBu_r"](mc.Normalize(-m, m)(np.nan_to_num(data.show)))
    rgba[..., 3] = np.isfinite(data.show) * 0.95
    tex = (rgba * 255).astype(np.uint8)
    e1, e2, n = spec.basis()
    return {"png": _png(fig, dpi), "texture": _b64(tex, np.uint8), "tex_n": int(spec.n),
            "center": spec.center.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "normal": n.tolist(),
            "half": float(spec.half_size),
            "lines": _lines_payload([np.column_stack([L[:, 0], L[:, 1], np.zeros(len(L))]) for L in lines[::2] if len(L) > 1])
            if lines else {"n": 0},
            "ms": 1e3 * (time.perf_counter() - t0)}


# ----------------------------------------------------------------------------- Σ-map
def _spec(d: dict) -> SigmaMapSpec:
    return SigmaMapSpec(_params(d["base"]), d["x_name"], d["y_name"], tuple(d["x_range"]), tuple(d["y_range"]),
                        nx=int(d["nx"]), ny=int(d["ny"]))


def sigma_rows(spec_d: dict, rows: list[int]) -> list[dict]:
    from .rows import sigma_rows as _rows
    return _rows(spec_d, rows)


def sigma_start(spec_d: dict) -> None:
    spec = _spec(spec_d)
    shape = (spec.ny, spec.nx)
    _S["sigma"] = SigmaMapResult(spec, np.full(shape, -1, int), np.full(shape, -1, int),
                                 np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan),
                                 np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, bool))


def sigma_add(rows: list[dict]) -> dict:
    res = _S["sigma"]
    for r in rows:
        j = r["j"]
        nan = lambda a: np.array([np.nan if v is None else v for v in a], float)
        res.code_n[j], res.code_s[j] = r["cn"], r["cs"]
        res.mu_n[j], res.mu_s[j], res.f_n[j], res.f_s[j], res.blow_x[j] = (nan(r[k]) for k in ("mn", "ms", "fn", "fs", "bx"))
        res.done[j] = True
    return {"done": int(res.done.sum()), "global": int(res.is_global.sum())}


def sigma_draw(opts: dict) -> dict:
    """The Σ-map picture plus what the page needs to map clicks to data coordinates."""
    from ..viz.sigma_plot import draw_sigma
    res = _S["sigma"]
    fig, dpi = _figure(opts.get("w", 700), opts.get("h", 700), opts.get("ratio", 1.0))
    cur = _params(opts["current"]) if opts.get("current") else None
    ov = None
    if opts.get("path"):
        ov = _path_overlay(opts["path"])
    axs = draw_sigma(fig, res, opts.get("mode", "global"), edge=bool(opts.get("edge", True)),
                     mu=bool(opts.get("mu", False)), current=cur, path_overlay=ov)
    png = _png(fig, dpi)
    W, H = fig.bbox.width, fig.bbox.height
    boxes = []
    for ax in axs:
        bb = ax.get_window_extent()
        boxes.append({"x0": bb.x0 / W, "x1": bb.x1 / W, "y0": 1 - bb.y1 / H, "y1": 1 - bb.y0 / H,
                      "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim())})
    stats = None
    if res is not None:
        stats = {"done": int(res.done.sum()), "global": int(res.is_global.sum()), "nx": res.spec.nx,
                 "ny": res.spec.ny}
    return {"png": png, "axes": boxes, "stats": stats}


def sigma_point(x: float, y: float) -> dict:
    from ..viz.sigma_plot import readout
    res = _S["sigma"]
    if res is None:
        return {}
    return {"text": readout(res, x, y), "params": res.spec.params_at(x, y).to_dict()}


# ----------------------------------------------------------------------------- paths
def _path(d: dict) -> ParamPath:
    return ParamPath.from_dict(d)


def path_scan(path_d: dict, n: int) -> dict:
    path = _path(path_d)
    if len(path) < 2:
        return {"frames": [], "events": []}
    frames, events = scan_path(path, int(n))
    return {"frames": [{"t": f.t, "global": f.is_global, "kind": failure_kind(f), "north": f.north,
                        "south": f.south, "mu_n": None if not math.isfinite(f.mu_n) else f.mu_n,
                        "mu_s": None if not math.isfinite(f.mu_s) else f.mu_s,
                        "why": "" if f.is_global else _why(f), "params": f.params.to_dict()} for f in frames],
            "events": [{"index": e.index, "t": e.t, "kind": e.kind, "detail": e.detail} for e in events]}


def path_at(path_d: dict, t: float) -> dict:
    return _path(path_d).at(float(t))[0].to_dict()


def _path_overlay(d: dict) -> dict | None:
    path = _path(d["path"])
    if not len(path):
        return None
    n = int(d.get("n", 2))
    if len(path) < 2:
        keys = [k.params for k in path.keyframes]
        return {"keys": keys, "samples": keys, "kind": None, "cur": None, "varying": []}
    samples = path.at(np.linspace(*path.duration, n))
    kind = d.get("kind")
    return {"keys": [k.params for k in path.keyframes], "samples": samples,
            "kind": kind if kind and len(kind) == n else None, "cur": None,
            "events": [path.at(t)[0] for t in d.get("event_t", [])], "varying": path.varying()}


# ----------------------------------------------------------------------------- sessions
def session_json(state: dict) -> str:
    """A session file in the desktop app's format, from the page's state."""
    from ..io.session import Session
    import json
    path = _path(state["path"]) if state.get("path") else None
    sec = {k: v for k, v in state.items() if k not in ("params", "path")}
    return json.dumps(Session(_params(state["params"]), sec, path).to_dict(), indent=1, ensure_ascii=False)


def session_load(text: str) -> dict:
    from ..io.session import Session
    import json
    s = Session.from_dict(json.loads(text))
    return {"params": s.params.to_dict(), "sections": s.sections, "path": s.path.to_dict() if s.path else None}


# ----------------------------------------------------------------------------- dispatcher
_PUBLIC = ("info", "preset", "set_theme", "lock", "is_global", "solve", "pin", "clear_pins", "profiles", "scene",
           "slice_view", "sigma_rows", "sigma_start", "sigma_add", "sigma_draw", "sigma_point", "path_scan",
           "path_at", "session_json", "session_load")


def _plain(o):
    """json.dumps fallback for numpy scalars and arrays."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def call(name: str, args_json: str) -> str:
    """Entry point for the page's worker: call(name, JSON list of arguments) -> JSON result."""
    import json
    if name not in _PUBLIC:
        raise ValueError(f"unknown call {name!r}")
    out = globals()[name](*json.loads(args_json))
    return json.dumps(out, default=_plain, allow_nan=False) if out is not None else "null"
