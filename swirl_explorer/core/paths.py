"""Paths in parameter space: keyframes, interpolation, and event detection.

A path is a list of keyframes (time t_k, Params_k).  Frames are sampled by linear
or cubic-spline interpolation in R^6.  ``scan_path`` shoots every frame (fast path)
and reports the frames where the solution leaves the global set or a pole stratum
changes, so that movies never silently render garbage.  Each event is located more
precisely between its two frames by bisection.

Keyframes may also carry *extras*: the camera, the slice plane and the seeds.  An
extra present on at least one keyframe is keyframed (interpolated between the
keyframes that carry it, held constant outside them); an extra on no keyframe is
held fixed by the UI.  The helpers below interpolate plain dicts, so this module
stays free of GUI imports.

The architecture leaves room for the later milestone (effective coordinates and
pseudo-arclength continuation, Ch. 8-9): a path may carry ``coords = "cauchy"``
(the six data at the base point, implemented here) or ``"effective"`` (reserved).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline

from .params import Params, PARAM_NAMES
from .sigma_map import quick_pole, CODE_NAMES, G

EXTRA_KEYS = ("camera", "plane", "seeds")


# ----------------------------------------------------------------------------- extras
def _lerp(a, b, s):
    return (1.0 - s) * np.asarray(a, float) + s * np.asarray(b, float)


def _slerp(u, v, s):
    """Spherical interpolation of two directions (not necessarily unit length)."""
    u = np.asarray(u, float) / (np.linalg.norm(u) or 1.0)
    v = np.asarray(v, float) / (np.linalg.norm(v) or 1.0)
    d = float(np.clip(u @ v, -1.0, 1.0))
    om = math.acos(d)
    if om < 1e-9:
        return u
    if abs(om - math.pi) < 1e-6:          # opposite: turn through any perpendicular
        perp = np.cross(u, [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u, [1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)
        return math.cos(math.pi * s) * u + math.sin(math.pi * s) * perp
    return (math.sin((1 - s) * om) * u + math.sin(s * om) * v) / math.sin(om)


def interp_camera(c0: dict, c1: dict, s: float) -> dict:
    """Camera {position, focal_point, view_up, parallel_scale}: the viewing direction turns on a
    great circle about the focal point (no swing through the scene), the distance and zoom are linear."""
    f0, f1 = np.asarray(c0["focal_point"], float), np.asarray(c1["focal_point"], float)
    d0 = np.asarray(c0["position"], float) - f0
    d1 = np.asarray(c1["position"], float) - f1
    f = _lerp(f0, f1, s)
    dist = (1 - s) * np.linalg.norm(d0) + s * np.linalg.norm(d1)
    d = _slerp(d0, d1, s) * dist
    up = _slerp(c0["view_up"], c1["view_up"], s)
    dn = d / (np.linalg.norm(d) or 1.0)
    up = up - (up @ dn) * dn                       # keep "up" perpendicular to the view direction
    up /= np.linalg.norm(up) or 1.0
    out = {"position": tuple(map(float, f + d)), "focal_point": tuple(map(float, f)),
           "view_up": tuple(map(float, up))}
    if "parallel_scale" in c0 and "parallel_scale" in c1:
        out["parallel_scale"] = float((1 - s) * c0["parallel_scale"] + s * c1["parallel_scale"])
    return out


def interp_plane(p0: dict, p1: dict, s: float) -> dict:
    """Slice plane {azimuth, elevation, offset, ...}: angles the short way round, offset linear."""
    out = dict(p1 if s >= 0.5 else p0)
    da = ((p1["azimuth"] - p0["azimuth"] + 180.0) % 360.0) - 180.0
    az = p0["azimuth"] + s * da
    out["azimuth"] = float(((az + 180.0) % 360.0) - 180.0)
    out["elevation"] = float((1 - s) * p0["elevation"] + s * p1["elevation"])
    out["offset"] = float((1 - s) * p0["offset"] + s * p1["offset"])
    return out


def interp_seeds(a: dict, b: dict, s: float) -> dict:
    """Seeds: positions and radius linear; the seed shape and count jump at the midpoint."""
    out = dict(b if s >= 0.5 else a)
    for k in ("sphere_center", "line_p0", "line_p1"):
        if k in a and k in b:
            out[k] = tuple(map(float, _lerp(a[k], b[k], s)))
    if "sphere_radius" in a and "sphere_radius" in b:
        out["sphere_radius"] = float((1 - s) * a["sphere_radius"] + s * b["sphere_radius"])
    return out


_INTERP = {"camera": interp_camera, "plane": interp_plane, "seeds": interp_seeds}


# ----------------------------------------------------------------------------- path
@dataclass
class Keyframe:
    t: float
    params: Params
    extras: dict = field(default_factory=dict)   # camera, plane, seeds (keyframed by the UI)

    def to_dict(self) -> dict:
        return {"t": self.t, "params": self.params.to_dict(), "extras": self.extras}

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        return cls(float(d["t"]), Params.from_dict(d["params"]), dict(d.get("extras", {})))


@dataclass
class ParamPath:
    keyframes: list[Keyframe] = field(default_factory=list)
    interpolation: str = "linear"          # "linear" | "spline"
    coords: str = "cauchy"                 # "cauchy" | "effective" (later milestone)

    def add(self, t: float, p: Params, **extras) -> None:
        self.keyframes.append(Keyframe(float(t), p, extras))
        self.sort()

    def sort(self) -> None:
        self.keyframes.sort(key=lambda k: k.t)

    def __len__(self) -> int:
        return len(self.keyframes)

    @property
    def duration(self) -> tuple[float, float]:
        return self.keyframes[0].t, self.keyframes[-1].t

    def _matrix(self) -> tuple[np.ndarray, np.ndarray]:
        ts = np.array([k.t for k in self.keyframes])
        M = np.array([k.params.as_tuple() for k in self.keyframes])
        return ts, M

    def at(self, t) -> list[Params]:
        """Interpolated parameters at time(s) t (clamped to the keyframe range)."""
        if not self.keyframes:
            raise ValueError("empty path")
        ts, M = self._matrix()
        t = np.atleast_1d(np.asarray(t, float))
        xb = self.keyframes[0].params.xb
        if len(ts) == 1:
            return [self.keyframes[0].params for _ in t]
        tc = np.clip(t, ts[0], ts[-1])
        if self.interpolation == "spline" and len(ts) >= 3 and np.all(np.diff(ts) > 0):
            V = CubicSpline(ts, M, axis=0, bc_type="natural")(tc)
        else:
            V = np.column_stack([np.interp(tc, ts, M[:, i]) for i in range(6)])
        return [Params.from_tuple(v, xb) for v in V]

    def varying(self) -> list[str]:
        """Names of the data that change along the path."""
        if len(self.keyframes) < 2:
            return []
        _, M = self._matrix()
        return [n for i, n in enumerate(PARAM_NAMES) if np.ptp(M[:, i]) > 0]

    def keyed(self, key: str) -> bool:
        return any(key in k.extras for k in self.keyframes)

    def extra_at(self, t: float, key: str) -> Optional[dict]:
        """The extra `key` at time t, interpolated between the keyframes that carry it (None if none do)."""
        ks = [k for k in self.keyframes if key in k.extras]
        if not ks:
            return None
        if t <= ks[0].t or len(ks) == 1:
            return dict(ks[0].extras[key])
        if t >= ks[-1].t:
            return dict(ks[-1].extras[key])
        for a, b in zip(ks, ks[1:]):
            if a.t <= t <= b.t:
                s = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
                return _INTERP[key](a.extras[key], b.extras[key], s)
        return dict(ks[-1].extras[key])          # pragma: no cover

    def sample(self, n_frames: int) -> tuple[np.ndarray, list[Params]]:
        t0, t1 = self.duration
        ts = np.linspace(t0, t1, n_frames)
        return ts, self.at(ts)

    def to_dict(self) -> dict:
        return {"interpolation": self.interpolation, "coords": self.coords,
                "keyframes": [k.to_dict() for k in self.keyframes]}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamPath":
        p = cls([Keyframe.from_dict(k) for k in d.get("keyframes", [])],
                d.get("interpolation", "linear"), d.get("coords", "cauchy"))
        p.sort()
        return p


# ----------------------------------------------------------------------------- scan
@dataclass
class FrameStatus:
    index: int
    t: float
    params: Params
    is_global: bool
    north: str
    south: str
    mu_n: float
    mu_s: float
    blow_x: float = math.nan          # where z -> 0 (x = cos θ) when a pole blows up

    @property
    def state(self) -> tuple:
        return (self.is_global, self.north, self.south)


def failure_kind(fs: "FrameStatus") -> str:
    """'ok', 'blowup' (z = 0 before a pole) or 'unsafe' (a pole fails the terminal test)."""
    if fs.is_global:
        return "ok"
    return "blowup" if "blow-up" in (fs.north, fs.south) else "unsafe"


@dataclass
class PathEvent:
    index: int             # first frame after the event
    t: float               # event time, refined by bisection between frames index-1 and index
    kind: str              # "leaves_global" | "enters_global" | "stratum_change" | "failure_change"
    detail: str


def frame_status(p: Params, i: int = 0, t: float = 0.0, xi_max: float = 14.0, rtol: float = 1e-9) -> FrameStatus:
    cn, mn, _, bn = quick_pole(p, +1, xi_max, rtol)
    cs, ms, _, bs = quick_pole(p, -1, xi_max, rtol)
    bx = bn if math.isfinite(bn) else bs
    return FrameStatus(i, float(t), p, cn >= G and cs >= G, CODE_NAMES[cn], CODE_NAMES[cs], mn, ms, bx)


def _why(fs: FrameStatus) -> str:
    """Why a frame is not global, in the same words as the status panel."""
    bits = []
    if "blow-up" in (fs.north, fs.south):
        if math.isfinite(fs.blow_x):
            side = "north" if fs.blow_x > 0 else "south"
            if abs(fs.blow_x) > 0.9999:
                bits.append(f"blows up just before the {side} pole")
            else:
                near = f" (just before the {side} pole)" if abs(fs.blow_x) > 0.99 else ""
                bits.append(f"blows up at x ≈ {fs.blow_x:.4f}{near}".replace("-", "−"))
        else:
            bits.append("blows up before a pole")
    for name, side in ((fs.north, "north"), (fs.south, "south")):
        if name == "unsafe":
            bits.append(f"the {side} pole fails the terminal test")
    return "; ".join(bits) or f"north {fs.north}, south {fs.south}"


def scan_path(path: ParamPath, n_frames: int, xi_max: float = 14.0, rtol: float = 1e-9,
              progress: Optional[Callable[[float], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None, refine: int = 14
              ) -> tuple[list[FrameStatus], list[PathEvent]]:
    """Status of every frame and the events between consecutive frames.

    ``refine`` bisection steps place each event between its two frames (0: frame time)."""
    ts, ps = path.sample(n_frames)
    frames: list[FrameStatus] = []
    events: list[PathEvent] = []
    for i, (t, p) in enumerate(zip(ts, ps)):
        if cancelled and cancelled():
            break
        fs = frame_status(p, i, t, xi_max, rtol)
        if frames and frames[-1].state != fs.state:
            prev = frames[-1]
            te = _bisect(path, prev, fs, refine, xi_max, rtol)
            if prev.is_global and not fs.is_global:
                events.append(PathEvent(i, te, "leaves_global", _why(fs)))
            elif not prev.is_global and fs.is_global:
                events.append(PathEvent(i, te, "enters_global", f"N {fs.north}, S {fs.south}"))
            elif fs.is_global:
                changed = [f"{side} {a}->{b}" for side, a, b in (("north", prev.north, fs.north),
                                                                  ("south", prev.south, fs.south)) if a != b]
                events.append(PathEvent(i, te, "stratum_change", ", ".join(changed)))
            else:
                events.append(PathEvent(i, te, "failure_change", _why(fs)))
        frames.append(fs)
        if progress:
            progress((i + 1) / n_frames)
    return frames, events


def _bisect(path: ParamPath, a: FrameStatus, b: FrameStatus, steps: int, xi_max: float, rtol: float) -> float:
    """Time where the state of frame `a` changes, between a.t and b.t."""
    lo, hi = a.t, b.t
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        s = frame_status(path.at(mid)[0], 0, mid, xi_max, rtol)
        if s.state == a.state:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
