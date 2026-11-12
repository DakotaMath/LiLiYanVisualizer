# The browser version (GitHub Pages)

Swirl Explorer also runs in a web browser, with nothing to install and no server. The page loads
**Pyodide** (CPython compiled to WebAssembly, with numpy, scipy and matplotlib). Inside the visitor's
browser it then runs the **same Python code** as the desktop app: the solver, the pole diagnostics,
the plots and the Σ-map. GitHub Pages only serves the static files, which is free.

## Turning it on (one time, about 1 minute)

1. In the GitHub repository, open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Merge the pull request that adds the web version, or run the **Web version (GitHub Pages)**
   workflow by hand (**Actions → Web version (GitHub Pages) → Run workflow**).

The site then appears at `https://<owner>.github.io/<repository>/`, for this repository
<https://dakotamath.github.io/LiLiYanVisualizer/>. The address is also on the workflow run's page.

After that, every push to `main` that changes `site/`, `swirl_explorer/` or the build script
republishes it. Pull requests are only built and tested, not published.

## How it fits together

```
site/                     the page: index.html, style.css, app.js (UI), view3d.js (WebGL 3D view),
                          path.js (path panel), py-worker.js (Pyodide in a Web Worker)
swirl_explorer/web/       the Python side of the page: api.py (every call the page makes, JSON in/out),
                          rows.py (Σ-map rows for the worker pool)
swirl_explorer/core, viz, io   shared with the desktop app, unchanged
scripts/build_site.py     assembles _site/: the page + swirl.zip (the package, without the Qt code)
.github/workflows/pages.yml    tests, builds and publishes
```

- **Plots:** the profiles, the Σ-map and the slice are drawn by the desktop's own matplotlib code
  (`viz/plots2d.py`, `viz/sigma_plot.py`, `viz/slice_plot.py`) and sent to the page as images.
- **3D view:** the numbers come from the desktop's code (`viz/scene_compute.py`, `viz/scene_values.py`);
  the page draws them with three.js in place of VTK.
- **Σ-map:** rows are shared between several background workers (up to 6, one per spare CPU core).

## Running it locally

```bash
python scripts/build_site.py --serve
```

Then open <http://localhost:8000>. Pages are served over HTTP because browsers refuse to start
workers from `file://` addresses.

## Differences from the desktop app

- **Speed:** the first visit downloads about 40 MB (Pyodide and the scientific packages); the
  browser caches them. The page is ready after about 6–10 s. Computing is about 4–5 times slower
  than the desktop app: a solve takes 30–130 ms, a 101 × 101 Σ-map about 10 s on a 6-core machine.
- **Seeds and the slice plane:** there are no drag handles; type the positions instead.
- **Isosurfaces:** the first time *isosurface* is ticked, the page downloads scikit-image (its
  marching cubes stands in for VTK's contour filter) and then draws the same surface.
- **Movies:** *Record video* captures the 3D view along the path, with the same caption strip, as
  WebM, or MP4 where the browser supports it. It replaces the desktop's composited MP4 export.
- **Sessions:** files are compatible both ways (*Save session* / *Open session*).
