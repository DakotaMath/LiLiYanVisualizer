"""Σ-map rows for the browser's worker pool: numpy + scipy only (no matplotlib), so pool workers load fast."""
from __future__ import annotations

import json
import math

from ..core.params import Params, PARAM_NAMES
from ..core.sigma_map import SigmaMapSpec, quick_pole


def _spec(d: dict) -> SigmaMapSpec:
    b = d["base"]
    base = Params(*(float(b.get(n, 0.0)) for n in PARAM_NAMES), xb=float(b.get("xb", 0.0)))
    return SigmaMapSpec(base, d["x_name"], d["y_name"], tuple(d["x_range"]), tuple(d["y_range"]),
                        nx=int(d["nx"]), ny=int(d["ny"]))


def sigma_rows(spec_d: dict, rows: list[int]) -> list[dict]:
    """Shoot both poles at every point of the given rows (same calls as core.sigma_map)."""
    spec = _spec(spec_d)
    xs, ys = spec.axes()
    out = []
    for j in rows:
        r = {"j": int(j), "cn": [], "cs": [], "mn": [], "ms": [], "fn": [], "fs": [], "bx": []}
        for xv in xs:
            p = spec.params_at(float(xv), float(ys[j]))
            cn, mn, fn, bn = quick_pole(p, +1, spec.xi_max, spec.rtol)
            cs, ms, fs, bs = quick_pole(p, -1, spec.xi_max, spec.rtol)
            r["cn"].append(int(cn))
            r["cs"].append(int(cs))
            for k, v in (("mn", mn), ("ms", ms), ("fn", fn), ("fs", fs), ("bx", bn if math.isfinite(bn) else bs)):
                r[k].append(float(v) if math.isfinite(v) else None)
        out.append(r)
    return out


def call(name: str, args_json: str) -> str:
    if name != "sigma_rows":
        raise ValueError(name)
    return json.dumps(sigma_rows(*json.loads(args_json)))
