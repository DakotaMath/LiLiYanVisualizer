"""Slice planes through the 3D field (no Qt imports).

A plane is given by its unit normal n = (cos el cos az, cos el sin az, sin el) and its signed offset
d from the origin (points X with X·n = d). In-plane coordinates (s, t) use the basis

    e2 = the projection of the axis direction z onto the plane (so "up" is up), or y if n ∥ z,
    e1 = e2 × n,             so that e1 × e2 = n,

so the 2D panel shows the plane with e1 to the right, e2 up and n pointing at the viewer. On the
plane we sample

    un   = u·n               normal velocity (diverging colour map: red = toward the viewer, blue = away)
    us, ut = u·e1, u·e2      tangential components (projected streamlines, LIC)

and NaN where the field is excluded (origin ball, axis cylinder, outside the region).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Optional

import numpy as np

from ..core.fields import FieldInterpolator
from .streamlines import Domain

ROUNDOFF = 1e-10                # relative size below which u·n counts as zero (floating-point noise)

PRESETS = {                     # name -> (azimuth°, elevation°)
    "meridional x–z": (-90.0, 0.0),
    "meridional y–z": (0.0, 0.0),
    "horizontal x–y": (0.0, 90.0),
}


@dataclass
class PlaneSpec:
    azimuth: float = -90.0       # degrees (default: meridional x–z plane, +x to the right)
    elevation: float = 0.0       # degrees
    offset: float = 0.0          # signed distance from the origin
    half_size: float = 1.0       # the plane square is [-L, L]^2 in (s, t)
    n: int = 181                 # samples per side

    @property
    def normal(self) -> np.ndarray:
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.normal
        z = np.array([0.0, 0.0, 1.0])
        p = z - (z @ n) * n
        if np.linalg.norm(p) < 1e-6:
            p = np.array([0.0, 1.0, 0.0])
        e2 = p / np.linalg.norm(p)
        e1 = np.cross(e2, n)            # e1 × e2 = n: the viewer looks at the plane from the +n side
        return e1, e2, n

    @property
    def center(self) -> np.ndarray:
        return self.offset * self.normal

    def to_world(self, s, t) -> np.ndarray:
        e1, e2, _ = self.basis()
        s, t = np.asarray(s, float), np.asarray(t, float)
        return self.center + s[..., None] * e1 + t[..., None] * e2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlaneSpec":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    @classmethod
    def from_normal_origin(cls, normal, origin, **kw) -> "PlaneSpec":
        """Spec for the plane through ``origin`` with normal ``normal`` (as given by a 3D widget)."""
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        el = math.degrees(math.asin(max(-1.0, min(1.0, n[2]))))
        az = math.degrees(math.atan2(n[1], n[0]))
        return cls(azimuth=az, elevation=el, offset=float(np.asarray(origin, float) @ n), **kw)


@dataclass
class SliceData:
    spec: PlaneSpec
    s: np.ndarray                 # (n,) in-plane coordinate along e1
    t: np.ndarray                 # (n,) along e2
    X: np.ndarray                 # (n, n, 3) world points, index [j(t), i(s)]
    un: np.ndarray                # (n, n)
    us: np.ndarray
    ut: np.ndarray
    speed_t: np.ndarray           # |tangential part|
    axis_hits: list = field(default_factory=list)       # (s, t) where the symmetry axis crosses the plane
    lic: Optional[np.ndarray] = None
    # what is shown in colour (set by the panel, shared with the 3D view)
    show: Optional[np.ndarray] = None
    show_label: str = "u·n"
    clim: float = float("nan")          # symmetric colour limit
    vmax: float = float("nan")          # true max |shown value|

    symlog: bool = False
    lin_thresh: float = 1.0
    is_zero: bool = False

    def set_display(self, scale_by_rho: bool, symlog: bool = False, percentile: float = 85.0) -> None:
        """Choose the coloured field (u·n or ρ·u·n, optionally signed-log) and one symmetric colour limit
        shared by the 2D and 3D views."""
        # round-off: |u·n| below ROUNDOFF·max|u| on the plane is zero (e.g. ~1e-32 on a meridional plane
        # of a swirl-free flow) -- otherwise the colour range stretches over pure noise
        speed = np.sqrt(self.un ** 2 + self.us ** 2 + self.ut ** 2)
        speed = speed[np.isfinite(speed)]
        floor = ROUNDOFF * (float(speed.max()) if speed.size else 0.0)
        un = np.where(np.abs(self.un) <= floor, 0.0, self.un)
        if scale_by_rho:
            rho = np.hypot(self.X[..., 0], self.X[..., 1])
            raw, self.show_label = un * rho, "ρ·(u·n)"
        else:
            raw, self.show_label = un, "u·n"
        v = np.abs(raw[np.isfinite(raw)])
        self.vmax = float(v.max()) if v.size else 0.0
        self.is_zero = self.vmax == 0.0                 # no flow through the plane at all
        self.symlog = bool(symlog) and not self.is_zero
        if self.is_zero:
            self.show = np.where(np.isfinite(raw), 0.0, np.nan)
            self.clim, self.lin_thresh = 1.0, 1.0
            return
        if symlog:
            nz = v[v > 0]
            lt = float(np.percentile(nz, 50)) if nz.size else 1.0
            self.lin_thresh = lt if lt > 0 else 1.0
            self.show = np.sign(raw) * np.log10(1.0 + np.abs(raw) / self.lin_thresh)
            self.clim = float(np.log10(1.0 + self.vmax / self.lin_thresh)) or 1.0      # nothing saturates
        else:
            self.show = raw
            m = float(np.percentile(v, percentile)) if v.size else 1.0
            self.clim = m if m > 0 else 1.0

    def to_value(self, t):
        """Inverse of the displayed transform (for colour-bar tick labels)."""
        t = np.asarray(t, float)
        return np.sign(t) * self.lin_thresh * (10.0 ** np.abs(t) - 1.0) if self.symlog else t

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.un)


def sample_plane(fi: FieldInterpolator, spec: PlaneSpec, domain: Domain) -> SliceData:
    L = spec.half_size
    s = np.linspace(-L, L, spec.n)
    t = np.linspace(-L, L, spec.n)
    S, T = np.meshgrid(s, t)                       # [j, i] = (t_j, s_i)
    X = spec.to_world(S, T)
    P = X.reshape(-1, 3)
    ok = domain.inside(P)
    U = np.full_like(P, np.nan)
    if ok.any():
        U[ok] = fi.velocity(P[ok])
    e1, e2, n = spec.basis()
    un = (U @ n).reshape(S.shape)
    us = (U @ e1).reshape(S.shape)
    ut = (U @ e2).reshape(S.shape)
    hits = _axis_hits(spec)
    return SliceData(spec, s, t, X, un, us, ut, np.hypot(us, ut), hits)


def _axis_hits(spec: PlaneSpec) -> list:
    """Where the line X = (0, 0, z) meets the plane (or the whole line if it lies in it)."""
    e1, e2, n = spec.basis()
    d = spec.offset
    if abs(n[2]) > 1e-9:
        z = d / n[2]
        X = np.array([0.0, 0.0, z]) - spec.center
        return [("point", float(X @ e1), float(X @ e2))]
    if abs(d) < 1e-9:                               # the axis lies in the plane
        return [("line", float(np.array([0, 0, 1.0]) @ e1), float(np.array([0, 0, 1.0]) @ e2))]
    return []


# ----------------------------------------------------------------- LIC
def lic(us: np.ndarray, ut: np.ndarray, length: int = 18, seed: int = 0) -> np.ndarray:
    """Line-integral convolution of the in-plane direction field (white noise, box kernel).

    Vectorised over pixels: each pixel is advected ``length`` steps forward and backward along the
    normalised field with bilinear sampling, and the noise is averaged along the path.
    """
    ny, nx = us.shape
    rng = np.random.default_rng(seed)
    noise = rng.random((ny, nx))
    mag = np.hypot(us, ut)
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = np.where(mag > 0, us / mag, 0.0)
        vy = np.where(mag > 0, ut / mag, 0.0)
    vx = np.nan_to_num(vx)
    vy = np.nan_to_num(vy)
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    acc = noise.copy()
    cnt = np.ones_like(noise)
    for sign in (1.0, -1.0):
        x, y = ii.copy(), jj.copy()
        alive = np.isfinite(us)
        for _ in range(length):
            i0 = np.clip(np.round(x).astype(int), 0, nx - 1)
            j0 = np.clip(np.round(y).astype(int), 0, ny - 1)
            x = x + sign * vx[j0, i0] * 0.8
            y = y + sign * vy[j0, i0] * 0.8
            inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
            alive &= inside
            i1 = np.clip(np.round(x).astype(int), 0, nx - 1)
            j1 = np.clip(np.round(y).astype(int), 0, ny - 1)
            alive &= np.isfinite(us[j1, i1])
            acc += np.where(alive, noise[j1, i1], 0.0)
            cnt += alive
    out = acc / cnt
    out[~np.isfinite(us)] = np.nan
    return out


# ----------------------------------------------------------------- projected streamlines
def projected_streamlines(data: SliceData, density: float = 1.4) -> list[np.ndarray]:
    """Evenly spaced streamlines of the tangential field, as (k, 2) arrays in (s, t).

    Uses matplotlib's streamplot (evenly spaced seeding, Jobard–Lefer style) on an off-screen axes.
    """
    from matplotlib.figure import Figure
    fig = Figure()
    ax = fig.add_subplot()
    us = np.ma.masked_invalid(data.us)
    ut = np.ma.masked_invalid(data.ut)
    if us.count() < 10:
        return []
    sp = ax.streamplot(data.s, data.t, us, ut, density=density, broken_streamlines=True,
                       arrowstyle="-")
    lines = []
    for seg in sp.lines.get_segments():
        if len(seg):
            lines.append(np.asarray(seg))
    return _join_segments(lines)


def _join_segments(segs: list[np.ndarray], tol: float = 1e-9) -> list[np.ndarray]:
    """Join streamplot output into polylines. Older matplotlib returns 2-point segments that must be
    chained; newer versions already return one polyline per streamline."""
    if any(len(sgm) > 2 for sgm in segs):
        return [np.asarray(sgm) for sgm in segs if len(sgm) > 1]
    out: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    for sgm in segs:
        if cur and np.linalg.norm(cur[-1] - sgm[0]) < tol:
            cur.append(sgm[1])
        else:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = [sgm[0], sgm[1]]
    if len(cur) > 1:
        out.append(np.array(cur))
    return out
