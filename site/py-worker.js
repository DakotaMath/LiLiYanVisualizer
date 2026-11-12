// Python in a Web Worker (Pyodide). Two roles:
//   main  numpy + scipy + matplotlib, the full bridge (swirl_explorer.web.api)
//   pool  numpy + scipy only, Σ-map rows (swirl_explorer.web.rows); started on the first Σ-map
// Messages in:  {id, fn, args}          out: {id, ok, result | error}   (result is JSON text)
// (a module worker: importScripts() of the classic pyodide.js is blocked in some browsers)
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

const role = new URL(self.location.href).searchParams.get("role") || "main";
let call = null, py = null, skimage = false;

const ready = (async () => {
  const t0 = performance.now();
  py = await loadPyodide({ indexURL: PYODIDE });
  self.postMessage({ progress: "Python loaded", ms: performance.now() - t0 });
  await py.loadPackage(role === "pool" ? ["numpy", "scipy"] : ["numpy", "scipy", "matplotlib"]);
  self.postMessage({ progress: "numpy, scipy" + (role === "pool" ? "" : ", matplotlib") + " loaded",
                     ms: performance.now() - t0 });
  const zip = await (await fetch("swirl.zip", { cache: "no-cache" })).arrayBuffer();
  py.unpackArchive(zip, "zip");
  const mod = py.pyimport(role === "pool" ? "swirl_explorer.web.rows" : "swirl_explorer.web.api");
  call = mod.call;
  self.postMessage({ ready: true, ms: performance.now() - t0 });
})().catch((e) => self.postMessage({ fatal: String(e) }));

self.onmessage = async (ev) => {
  const { id, fn, args } = ev.data;
  try {
    await ready;
    // scikit-image (marching cubes) is downloaded only when an isosurface is first asked for
    if (fn === "scene" && args && args[0] && args[0].show_iso && !skimage) {
      self.postMessage({ progress: "loading scikit-image for the isosurface" });
      await py.loadPackage(["scikit-image"]);
      skimage = true;
    }
    const result = call(fn, JSON.stringify(args || []));
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message ? e.message : e) });
  }
};
