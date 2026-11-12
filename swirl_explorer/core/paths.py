"""Paths in parameter space: keyframes, interpolation, and event detection.

A path is a list of keyframes (time t_k, Params_k).  Frames are sampled by linear
or cubic-spline interpolation in R^6.  ``scan_path`` shoots every frame (fast path)
and reports the frames where the solution leaves the global set or a pole stratum
changes, so that movies never silently render garbage.

The architecture leaves room for the later milestone (effective coordinates and
pseudo-arclength continuation, Ch. 8-9): a path may carry ``coords = "cauchy"``
(the six data at the base point, implemented here) or ``"effective"`` (reserved).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline

from .params import Params, PARAM_NAMES
from .sigma_map import quick_pole, CODE_NAMES, G


@dataclass
class Keyframe:
    t: float
    params: Params
    extras: dict = field(default_factory=dict)   # camera, seeds, plane ... (keyframed by the UI)

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
        self.keyframes.sort(key=lambda k: k.t)

    @property
    def duration(self) -> tuple[float, float]:
        return self.keyframes[0].t, self.keyframes[-1].t

    def _matrix(self) -> tuple[np.ndarray, np.ndarray]:
        ts = np.array([k.t for k in self.keyframes])
        M = np.array([k.params.as_tuple() for k in self.keyframes])
        return ts, M

    def at(self, t) -> list[Params]:
        """Interpolated parameters at time(s) t."""
        if not self.keyframes:
            raise ValueError("empty path")
        ts, M = self._matrix()
        t = np.atleast_1d(np.asarray(t, float))
        xb = self.keyframes[0].params.xb
        if len(ts) == 1:
            return [self.keyframes[0].params for _ in t]
        if self.interpolation == "spline" and len(ts) >= 3:
            V = CubicSpline(ts, M, axis=0, bc_type="natural")(np.clip(t, ts[0], ts[-1]))
        else:
            V = np.column_stack([np.interp(t, ts, M[:, i]) for i in range(6)])
        return [Params.from_tuple(v, xb) for v in V]

    def sample(self, n_frames: int) -> tuple[np.ndarray, list[Params]]:
        t0, t1 = self.duration
        ts = np.linspace(t0, t1, n_frames)
        return ts, self.at(ts)

    def to_dict(self) -> dict:
        return {"interpolation": self.interpolation, "coords": self.coords,
                "keyframes": [k.to_dict() for k in self.keyframes]}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamPath":
        return cls([Keyframe.from_dict(k) for k in d.get("keyframes", [])],
                   d.get("interpolation", "linear"), d.get("coords", "cauchy"))


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


@dataclass
class PathEvent:
    index: int
    t: float
    kind: str              # "leaves_global" | "enters_global" | "stratum_change"
    detail: str


def scan_path(path: ParamPath, n_frames: int, xi_max: float = 14.0, rtol: float = 1e-9,
              progress: Optional[Callable[[float], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None
              ) -> tuple[list[FrameStatus], list[PathEvent]]:
    ts, ps = path.sample(n_frames)
    frames: list[FrameStatus] = []
    events: list[PathEvent] = []
    for i, (t, p) in enumerate(zip(ts, ps)):
        if cancelled and cancelled():
            break
        cn, mn, _, _ = quick_pole(p, +1, xi_max, rtol)
        cs, ms, _, _ = quick_pole(p, -1, xi_max, rtol)
        fs = FrameStatus(i, float(t), p, cn >= G and cs >= G, CODE_NAMES[cn], CODE_NAMES[cs], mn, ms)
        if frames:
            prev = frames[-1]
            if prev.is_global and not fs.is_global:
                events.append(PathEvent(i, fs.t, "leaves_global",
                                        f"N: {fs.north}, S: {fs.south}"))
            elif not prev.is_global and fs.is_global:
                events.append(PathEvent(i, fs.t, "enters_global", ""))
            elif fs.is_global and (prev.north, prev.south) != (fs.north, fs.south):
                events.append(PathEvent(i, fs.t, "stratum_change",
                                        f"N {prev.north}->{fs.north}, S {prev.south}->{fs.south}"))
        frames.append(fs)
        if progress:
            progress((i + 1) / n_frames)
    return frames, events
