"""The browser's Σ-map kernel (site/sigma-kernel.js) against the Python one, point by point.

Runs the JavaScript with Node; skipped where Node is not installed.  Also checks that the
generated part of sigma-kernel.js is what scripts/gen_sigma_kernel_js.py produces.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from swirl_explorer.web.rows import sigma_rows

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "site" / "sigma-kernel.js"
NODE = shutil.which("node")

SPECS = [
    dict(base=dict(), x_name="gamma", y_name="kappa", x_range=[-2.6, 2.6], y_range=[-4.6, 4.6]),
    dict(base=dict(c1=0.3, c2=-0.2, gamma=0.5, A=0.4, kappa=0.3), x_name="c1", y_name="kappa",
         x_range=[-2, 2], y_range=[-2, 2]),
    dict(base=dict(c3=0.5, A=0.2, kappa=0.5), x_name="c2", y_name="gamma", x_range=[-3, 3], y_range=[-4, 4]),
    dict(base=dict(gamma=0.5, xb=0.3), x_name="gamma", y_name="A", x_range=[-3, 3], y_range=[-3, 3]),
]


def _js_rows(spec, rows):
    src = (f'import {{ sigmaRows }} from "{KERNEL.as_uri()}";\n'
           f"process.stdout.write(JSON.stringify(sigmaRows({json.dumps(spec)}, {json.dumps(rows)})));")
    out = subprocess.run([NODE, "--input-type=module", "-e", src], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
@pytest.mark.parametrize("k", range(len(SPECS)))
def test_js_kernel_equals_python(k):
    spec = dict(SPECS[k], nx=31, ny=31)
    rows = list(range(31))
    js, py = _js_rows(spec, rows), sigma_rows(spec, rows)
    for a, b in zip(js, py):
        assert a["j"] == b["j"]
        assert a["cn"] == b["cn"] and a["cs"] == b["cs"]
        for key in ("mn", "ms", "fn", "fs", "bx"):
            for x, y in zip(a[key], b[key]):
                assert (x is None) == (y is None), key
                if x is not None:
                    assert x == pytest.approx(y, rel=1e-8, abs=1e-9), key


def test_generated_code_is_current():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "gen_sigma_kernel_js.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    src = KERNEL.read_text(encoding="utf-8").replace("\r\n", "\n")     # git may check out CRLF
    a, b = src.index(gen.BEGIN), src.index(gen.END) + len(gen.END)
    assert src[a:b] == gen.generate(), "run python scripts/gen_sigma_kernel_js.py"
