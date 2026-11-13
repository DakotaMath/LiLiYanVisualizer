"""The browser's data picker: the lock for (γ, κ) (site/sigma-kernel.js) against the solver, and the
Li–Li–Yan set J and its projection (site/picker.js).  Runs the JavaScript with Node; skipped where
Node is not installed."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import is_global_fast

ROOT = Path(__file__).resolve().parents[1]
KERNEL = (ROOT / "site" / "sigma-kernel.js").as_uri()
PICKER = (ROOT / "site" / "picker.js").as_uri()
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")


def _js(body: str):
    src = (f'import {{ isGlobal, gkClamp }} from "{KERNEL}";\n'
           f'import {{ cbar3, sStar, inJ, inJlock, projectC, C_MIN, C3_GAP }} from "{PICKER}";\n'
           f"process.stdout.write(JSON.stringify(({body})()));")
    out = subprocess.run([NODE, "--input-type=module", "-e", src], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


POINTS = [dict(gamma=-1.68, kappa=1.0), dict(gamma=0.0, kappa=4.0), dict(gamma=0.0, kappa=4.3),
          dict(gamma=2.5, kappa=0.2), dict(c3=-3.0, gamma=-0.6, kappa=0.25), dict(c1=0.5, c2=-0.4, A=1.2, gamma=0.3, kappa=-1.0),
          dict(c2=-0.9, kappa=0.3, gamma=-1.2), dict(A=2.0, gamma=0.1, kappa=1.5, xb=0.3)]


def test_is_global_matches_the_solver():
    full = [{**dict(c1=0.0, c2=0.0, c3=0.0, gamma=0.0, A=0.0, kappa=0.0, xb=0.0), **p} for p in POINTS]
    js = _js(f"() => {json.dumps(full)}.map(isGlobal)")
    py = [is_global_fast(Params.from_dict(p)) for p in full]
    assert js == py
    assert any(py) and not all(py)


@pytest.mark.parametrize("target", [(3.0, 3.0), (-2.5, 0.0), (0.0, 5.0)])
def test_gk_clamp_stops_on_the_global_side_of_the_edge(target):
    base = dict(c1=0.0, c2=0.0, c3=0.0, A=0.0, xb=0.0)
    r = _js(f"() => gkClamp({{ p: {json.dumps(base)}, target: {list(target)}, seeds: [[0, 0]] }})")
    g, k = r["gk"]
    assert r["moved"]
    at = lambda gg, kk: Params(gamma=gg, kappa=kk)
    assert is_global_fast(at(g, k))
    # a little further toward the target is not global: (γ, κ) is on the edge
    d = math.hypot(target[0] - g, target[1] - k)
    step = 1e-6 / d
    assert not is_global_fast(at(g + step * (target[0] - g), k + step * (target[1] - k)))


def test_j_and_its_projection():
    r = _js("""() => {
      const out = { cbar00: cbar3(0, 0), sstar: [sStar(-4), sStar(0), sStar(1)], cases: [] };
      for (const [c1, c2, c3] of [[0, 0, -6], [-1.5, 2, 0], [-3, -3, -2], [5, -2, -10], [0.3, 0.2, -1], [-1, -1, 1]]) {
        const [p1, p2] = projectC(c1, c2, c3);
        // brute force: the nearest point of the locked set on a fine grid around the answer
        let best = Infinity;
        for (let i = -400; i <= 400; i++) for (let j = -400; j <= 400; j++) {
          const a = p1 + i * 0.002, b = p2 + j * 0.002;
          if (inJlock(a, b, c3)) best = Math.min(best, Math.hypot(a - c1, b - c2));
        }
        out.cases.push({ inside: inJlock(p1, p2, c3), again: projectC(p1, p2, c3), p: [p1, p2], d: Math.hypot(p1 - c1, p2 - c2), best,
                         gap: c3 - cbar3(p1, p2) });
      }
      return out; }""")
    assert r["cbar00"] == pytest.approx(-4.0)                      # c̄₃(0, 0) = −½·2·4 (monograph Thm 3.44)
    assert r["sstar"] == pytest.approx([2.0, 0.0, 0.0])
    for c in r["cases"]:
        assert c["inside"]
        assert c["again"] == pytest.approx(c["p"], abs=1e-12)       # a projected point stays put
        assert c["d"] <= c["best"] + 1e-9                           # and it is the nearest one
        assert c["gap"] >= 0.05 - 1e-9 and min(c["p"]) >= -0.99 - 1e-12
