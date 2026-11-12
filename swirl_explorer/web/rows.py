"""Σ-map rows for the browser's worker pool: numpy + scipy only (no matplotlib), so pool workers load fast."""
from __future__ import annotations

import json
import math

from ..core.params import Params, PARAM_NAMES
from ..core.sigma_map import SigmaMapSpec, rows_poles


def _spec(d: dict) -> SigmaMapSpec:
    b = d["base"]
    base = Params(*(float(b.get(n, 0.0)) for n in PARAM_NAMES), xb=float(b.get("xb", 0.0)))
    return SigmaMapSpec(base, d["x_name"], d["y_name"], tuple(d["x_range"]), tuple(d["y_range"]),
                        nx=int(d["nx"]), ny=int(d["ny"]))


def sigma_rows(spec_d: dict, rows: list[int]) -> list[dict]:
    """Both poles at every point of the given rows, integrated together (core.sigma_map.rows_poles)."""
    spec = _spec(spec_d)
    rows, n, s = rows_poles(spec, rows)
    fin = lambda a: [float(v) if math.isfinite(v) else None for v in a]
    out = []
    for k, j in enumerate(rows):
        bx = [bn if math.isfinite(bn) else bs for bn, bs in zip(n[3][k], s[3][k])]
        out.append({"j": int(j), "cn": [int(c) for c in n[0][k]], "cs": [int(c) for c in s[0][k]],
                    "mn": fin(n[1][k]), "ms": fin(s[1][k]), "fn": fin(n[2][k]), "fs": fin(s[2][k]), "bx": fin(bx)})
    return out


def call(name: str, args_json: str) -> str:
    if name != "sigma_rows":
        raise ValueError(name)
    return json.dumps(sigma_rows(*json.loads(args_json)))
