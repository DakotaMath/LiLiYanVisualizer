// Σ-map pool worker: plain JavaScript (sigma-kernel.js), no Python.  Speaks the PyWorker protocol:
// in {id, fn, args}, out {id, ok, result | error} with result JSON text.
//   sigma_rows(spec, rows)   rows of a Σ-map (the pool)
//   gk_clamp(req)            the data picker's lock for (γ, κ): the target if it is global, else
//                            the edge of the global set between a global seed and the target
//   refine_edges(req)        where the map's global/not-global change sits on grid edges (bisection)
import { sigmaRows, gkClamp, refineEdges } from "./sigma-kernel.js";

const FNS = { sigma_rows: sigmaRows, gk_clamp: gkClamp, refine_edges: refineEdges };

self.onmessage = (ev) => {
  const { id, fn, args } = ev.data;
  try {
    if (!FNS[fn]) throw new Error("unknown call " + fn);
    self.postMessage({ id, ok: true, result: JSON.stringify(FNS[fn](...args)) });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message ? e.message : e) });
  }
};
self.postMessage({ ready: true, ms: 0 });
