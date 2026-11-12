"""Assemble the static browser version into _site/ (what GitHub Pages serves).

    python scripts/build_site.py            # build
    python scripts/build_site.py --serve    # build, then serve on http://localhost:8000

_site/ = site/* + the app icon + swirl.zip (the Python package the page runs in Pyodide).
Only what the browser needs goes into the zip: core/, io/, viz/ (its PyVista modules are never
imported there), web/ and the package's __init__.  Qt code (ui/) stays out.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
SKIP_DIRS = {"ui", "__pycache__"}
SKIP_FILES = {"selftest.py", "__main__.py"}


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "site", OUT)
    shutil.copy(ROOT / "packaging" / "swirl.png", OUT / "swirl.png")
    (OUT / ".nojekyll").write_text("")            # serve files as they are
    pkg = ROOT / "swirl_explorer"
    n = 0
    with zipfile.ZipFile(OUT / "swirl.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*.py")):
            rel = f.relative_to(ROOT)
            if SKIP_DIRS & set(rel.parts) or (f.parent == pkg and f.name in SKIP_FILES):
                continue
            z.write(f, rel.as_posix())
            n += 1
    # stamp a content hash on every asset URL, so browsers never mix old and new files
    import hashlib
    h = hashlib.sha1()
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    v = h.hexdigest()[:10]
    for name, pairs in {
        "index.html": [('href="style.css"', f'href="style.css?v={v}"'), ('src="app.js"', f'src="app.js?v={v}"')],
        "app.js": [('from "./view3d.js"', f'from "./view3d.js?v={v}"'), ('from "./path.js"', f'from "./path.js?v={v}"'),
                   ("py-worker.js?role=${role}", f"py-worker.js?v={v}&role=${{role}}"),
                   ('new Worker("sigma-worker.js"', f'new Worker("sigma-worker.js?v={v}"')],
        "sigma-worker.js": [('from "./sigma-kernel.js"', f'from "./sigma-kernel.js?v={v}"')],
        "py-worker.js": [('fetch("swirl.zip"', f'fetch("swirl.zip?v={v}"')],
    }.items():
        f = OUT / name
        t = f.read_text(encoding="utf-8")
        for x, y in pairs:
            assert x in t, (name, x)
            t = t.replace(x, y)
        f.write_text(t, encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"built {OUT} ({n} Python files, {size / 1024:.0f} kB)")
    return OUT


if __name__ == "__main__":
    out = build()
    if "--serve" in sys.argv:
        import functools
        import http.server
        port = 8000
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        print(f"serving on http://localhost:{port}")
        http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
