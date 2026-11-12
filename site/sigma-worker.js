// Σ-map pool worker: plain JavaScript (sigma-kernel.js), no Python.  Speaks the PyWorker protocol:
// in {id, fn: "sigma_rows", args: [spec, rows]}, out {id, ok, result | error} with result JSON text.
import { sigmaRows } from "./sigma-kernel.js";

self.onmessage = (ev) => {
  const { id, fn, args } = ev.data;
  try {
    if (fn !== "sigma_rows") throw new Error("unknown call " + fn);
    self.postMessage({ id, ok: true, result: JSON.stringify(sigmaRows(...args)) });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message ? e.message : e) });
  }
};
self.postMessage({ ready: true, ms: 0 });
