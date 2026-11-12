// Swirl Explorer in the browser: the page. The numerics and the plots are the desktop app's Python
// (swirl_explorer.web.api in a Pyodide worker); this file is the UI around them.
import { View3D, OBLIQUE, POLE } from "./view3d.js";
import { PathDock } from "./path.js";

const $ = (id) => document.getElementById(id);
const NAMES = ["c1", "c2", "c3", "gamma", "A", "kappa"];
const SHORT = { c1: "c₁", c2: "c₂", c3: "c₃", gamma: "γ", A: "A", kappa: "κ" };
const fmt = (v, n = 7) => Number(v).toPrecision(n).replace(/(\.\d*?)0+(e|$)/, "$1$2").replace(/\.(e|$)/, "$1").replace("-", "−");
const ratio = () => Math.min(window.devicePixelRatio || 1, 2);

// ============================================================================ Python workers
class PyWorker {
  constructor(role, onProgress) {
    this.w = new Worker(`py-worker.js?role=${role}`, { type: "module" });
    this.n = 0; this.pending = new Map();
    this.ready = new Promise((res, rej) => { this._ready = res; this._fail = rej; });
    this.w.onmessage = (ev) => {
      const m = ev.data;
      if (m.progress && onProgress) onProgress(m.progress);
      if (m.ready) this._ready();
      if (m.fatal) { this._fail(new Error(m.fatal)); if (onProgress) onProgress("error: " + m.fatal); }
      if (m.id !== undefined) {
        const p = this.pending.get(m.id); this.pending.delete(m.id);
        if (!p) return;
        if (m.ok) p.res(JSON.parse(m.result)); else p.rej(new Error(m.error));
      }
    };
  }
  call(fn, ...args) {
    const id = this.n++;
    return new Promise((res, rej) => { this.pending.set(id, { res, rej }); this.w.postMessage({ id, fn, args }); });
  }
  terminate() { this.w.terminate(); }
}

// ============================================================================ state
const S = {
  info: null, params: null, lock: true, ranges: {}, theme: "dark",
  scene: { seed_mode: "rings", n_seeds: 40, sphere_center: [0.35, 0.25, 0.45], sphere_radius: 0.18,
           line_p0: [-0.7, 0.35, 0.35], line_p1: [0.7, 0.35, 0.35], color_by: "speed",
           show_streamlines: true, show_surface: false, show_vortex: false, show_glyphs: false, show_iso: false,
           iso_field: "speed", iso_quantile: 0.6, show_selfsim: false, domain_shape: "ball", r_min: 0.12,
           axis_a: 0.025, opacity: 0.45, clip: true },
  showSlice3d: false, sliceShown: false,
  plane: { azimuth: -90, elevation: 0, offset: 0, half_size: 1.0 },
  sliceOpts: { un: true, rho: false, log: false, lines: true, lic: false, density: 1.4, follow: false },
  plots: { xvar: "xi", mode: "linear", pole: "north", quantities: ["U", "f", "Q"], overlay: true },
  sigma: { x_name: "gamma", y_name: "kappa", x_range: [-2.6, 2.6], y_range: [-4.6, 4.6], n: 101, mode: "global",
           edge: true, mu: false, computed: null, running: false, axes: [] },
  tab: "view3d", rtab: "ctl3d", solved: null,
};

let py = null, view = null, path = null, syncPlots = () => {}, syncSigma = () => {};
const status = (t) => { $("sb-info").textContent = t; };

// ============================================================================ scheduling
// One Python worker does everything in order. Requests of the same kind replace each other while
// waiting (latest wins), so dragging a slider never piles up work.
const jobs = new Map();          // kind -> {fn, args, resolve}
let running = false;
const ORDER = ["lock", "jump", "frame", "solve", "scene", "slice", "profiles", "sigma_draw", "sigma_point", "scan"];
function schedule(kind, fn, args) {
  return new Promise((resolve, reject) => {
    const old = jobs.get(kind);
    if (old) old.resolve(null);            // superseded
    jobs.set(kind, { fn, args, resolve, reject });
    pump();
  });
}
async function pump() {
  if (running) return;
  running = true;
  try {
    while (jobs.size) {
      const rank = (k) => { const i = ORDER.indexOf(k.split(":")[0]); return i < 0 ? ORDER.length : i; };
      const kind = [...jobs.keys()].sort((a, b) => rank(a) - rank(b))[0];
      const j = jobs.get(kind); jobs.delete(kind);
      try { j.resolve(await py.call(j.fn, ...j.args)); } catch (e) { console.error(kind, e); status(`${kind}: ${e.message}`); j.resolve(null); }
    }
  } finally { running = false; }
}

// ============================================================================ parameters
function buildParams() {
  const grid = $("param-grid");
  grid.innerHTML = "";
  for (const p of S.info.params) {
    S.ranges[p.name] = [...p.range];
    const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = p.label; lab.title = p.tip;
    const sl = document.createElement("input"); sl.type = "range"; sl.min = 0; sl.max = 2000; sl.step = 1; sl.id = `p-${p.name}-s`; sl.title = p.tip;
    const num = document.createElement("input"); num.type = "number"; num.step = "any"; num.id = `p-${p.name}`; num.title = p.tip;
    const rng = document.createElement("span"); rng.className = "rng"; rng.id = `p-${p.name}-r`;
    grid.append(lab, sl, num, rng);
    sl.addEventListener("input", () => {
      const [a, b] = S.ranges[p.name];
      edit(p.name, a + (b - a) * (sl.value / 2000));
    });
    num.addEventListener("change", () => { const v = parseFloat(num.value); if (isFinite(v)) edit(p.name, v); });
  }
  const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = "base point x₀";
  const xb = document.createElement("input"); xb.type = "number"; xb.id = "p-xb"; xb.step = "0.001"; xb.min = -0.99; xb.max = 0.99;
  xb.title = "base point x_b = cos θ_b of the Cauchy data (default: equator)";
  xb.addEventListener("change", () => setParams({ ...S.params, xb: Math.max(-0.99, Math.min(0.99, parseFloat(xb.value) || 0)) }));
  grid.append(lab, document.createElement("span"), xb, document.createElement("span"));
  renderRanges();
}

function renderRanges() {
  const edit = $("show-ranges").checked;
  for (const n of NAMES) {
    const r = $(`p-${n}-r`), [a, b] = S.ranges[n];
    if (edit) {
      r.innerHTML = `<input type="number" step="any" value="${a}" data-i="0"> <input type="number" step="any" value="${b}" data-i="1">`;
      r.querySelectorAll("input").forEach((inp) => inp.addEventListener("change", () => {
        const v = parseFloat(inp.value); if (!isFinite(v)) return;
        S.ranges[n][+inp.dataset.i] = v;
        if (S.ranges[n][0] >= S.ranges[n][1]) S.ranges[n][1] = S.ranges[n][0] + 1;
        syncWidgets();
      }));
    } else {
      r.textContent = `${+a.toPrecision(6)} … ${+b.toPrecision(6)}`.replace(/-/g, "−");
      r.title = "slider range (tick 'edit slider ranges' to change it)";
    }
  }
}

function syncWidgets() {
  let widened = false;
  for (const n of NAMES) {
    const v = S.params[n];
    if (v < S.ranges[n][0]) { S.ranges[n][0] = v; widened = true; }
    if (v > S.ranges[n][1]) { S.ranges[n][1] = v; widened = true; }
  }
  if (widened) renderRanges();
  for (const n of NAMES) {
    const v = S.params[n], [a, b] = S.ranges[n];
    $(`p-${n}`).value = Number(v.toPrecision(7));
    $(`p-${n}-s`).value = Math.round(2000 * Math.min(1, Math.max(0, (v - a) / (b - a))));
  }
  $("p-xb").value = S.params.xb;
}

// a user edit of one datum: through the global lock, then solve
let lockGen = 0;
async function edit(name, value) {
  const old = S.params[name];
  $("lock-note").hidden = true;
  if (S.lock && value !== old) {
    const g = ++lockGen;
    const r = await schedule(`lock:${name}`, "lock", [S.params, name, old, value]);
    if (r === null || g !== lockGen) return;            // superseded by a later edit
    if (r.note) { $("lock-note").textContent = r.note; $("lock-note").hidden = false; }
    value = r.value;
  }
  setParams({ ...S.params, [name]: value });
}

function setParams(p) {             // presets, the Σ-map and the path set values directly (no lock)
  S.params = { ...S.params, ...p };
  syncWidgets();
  requestSolve();
}

// ============================================================================ solve and the views
let solveGen = 0;
async function requestSolve() {
  const g = ++solveGen;
  $("st-head").innerHTML = `<span class="muted">computing…</span>`;
  const r = await schedule("solve", "solve", [S.params]);
  if (!r || g !== solveGen) return;
  S.solved = r;
  $("st-head").innerHTML = r.head;
  $("st-north").innerHTML = r.north; $("st-north").title = r.north_tip;
  $("st-south").innerHTML = r.south; $("st-south").title = r.south_tip;
  $("sb-params").textContent = `(c₁, c₂, c₃, γ, A, κ) = (${NAMES.map((n) => fmt(r.params[n])).join(", ")}) at x₀ = ${r.params.xb}   ·   solve ${r.ms.toFixed(0)} ms`;
  refreshViews(g);
  if (path) path.setCurrent(r.params);
}

function refreshViews(g = solveGen) {
  if (!S.showSlice3d) view.setSlice(null);
  if (S.tab === "view3d") { refreshScene(g); if (S.sliceShown || S.showSlice3d) refreshSlice(g); }
  if (S.rtab === "profiles") refreshProfiles(g);
  if (S.tab === "sigma") refreshSigma();
}

async function refreshProfiles(g = solveGen) {
  const img = $("profiles-img");
  const w = Math.max(280, $("right").clientWidth - 20);
  const n = S.plots.quantities.length;
  const r = await schedule("profiles", "profiles", [{ ...S.plots, w, h: Math.max(320, Math.min(160 * n + 120, 1100)), ratio: ratio() }]);
  if (r && r.png && g === solveGen) img.src = r.png;
}

let sceneGen = 0;
async function refreshScene(g = solveGen) {
  const k = ++sceneGen;
  $("busy3d").hidden = false;
  const r = await schedule("scene", "scene", [{ ...S.scene, _slice_shown: S.showSlice3d }]);
  if (k === sceneGen) $("busy3d").hidden = true;
  if (!r || g !== solveGen || !r.n_lines && r.n_lines !== 0) return;
  view.setScene(r);
  renderLegend(r);
  const why = Object.entries(r.stops || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${v}× ${k}`).join(", ");
  $("info3d").innerHTML = (why ? `streamline ends (both directions): ${why}<br>` : "") + `computed in ${r.ms.toFixed(0)} ms`;
}

function renderLegend(r) {
  $("legend3d").innerHTML = (r.legend || []).map((t) => `<div>${t}</div>`).join("");
  $("s-color_by").disabled = !r.coloured;
  $("s-color_by").title = r.coloured ? "" : "nothing on screen is colour-mapped right now (streamlines are dimmed while overlays are shown)";
}

let sliceGen = 0;
async function refreshSlice(g = solveGen) {
  if (!S.sliceShown && !S.showSlice3d) return;
  const k = ++sliceGen;
  if (S.sliceShown) $("busy-slice").hidden = false;
  const wrap = $("slice-wrap");
  const opts = { ...S.sliceOpts, r_min: S.scene.r_min, axis_a: S.scene.axis_a, shape: S.scene.domain_shape, n: S.plane.n || 181,
                 w: Math.max(300, wrap.clientWidth || 500), h: Math.max(360, wrap.clientHeight || 600), ratio: ratio() };
  const r = await schedule("slice", "slice_view", [S.plane, opts]);
  if (k === sliceGen) $("busy-slice").hidden = true;
  if (!r || g !== solveGen || !r.png) return;
  S.lastSlice = r;
  if (S.sliceShown) $("slice-img").src = r.png;
  view.setSlice(S.showSlice3d ? r : null);
  updateNormalText(r);
  if (S.sliceOpts.follow) view.viewAlong(r.center, r.normal, r.e2);
}

function updateNormalText(r) {
  const v = (a) => "(" + a.map((x) => (Math.abs(x) < 5e-4 ? "0" : x.toFixed(2).replace("-", "−"))).join(", ") + ")";
  $("sl-normal").innerHTML = `normal n = ${v(r.normal)} points toward you<br>right = ${v(r.e1)}, up = ${v(r.e2)}`;
}

// ============================================================================ Σ-map
let pool = [], poolSize = Math.max(1, Math.min((navigator.hardwareConcurrency || 4) - 1, 6));
let sigmaRun = 0;

function sigmaSpec(base) {
  const s = S.sigma;
  return { base: { ...(base || S.params) }, x_name: s.x_name, y_name: s.y_name, x_range: [...s.x_range], y_range: [...s.y_range], nx: s.n, ny: s.n };
}

function startPool() {                 // also on hovering the Σ-map tab: a head start for the first map
  if (!pool.length) pool = Array.from({ length: poolSize }, () => new PyWorker("pool"));
}

async function computeSigma(base) {
  const s = S.sigma;
  if (s.x_name === s.y_name) { $("sigma-note").textContent = "Pick two different parameters for the two axes."; return; }
  if (s.x_range[0] >= s.x_range[1] || s.y_range[0] >= s.y_range[1]) { $("sigma-note").textContent = "Each range needs 'from' < 'to'."; return; }
  const run = ++sigmaRun, spec = sigmaSpec(base), t0 = performance.now();
  s.running = true; s.spec = spec;
  $("sg-compute").disabled = true; $("sg-cancel").disabled = false; $("sg-bar").style.width = "0%";
  if (!pool.length) $("sigma-note").textContent = `starting ${poolSize} background workers (first map only)…`;
  startPool();
  await py.call("sigma_start", spec);
  const order = [...Array(spec.ny).keys()].sort((a, b) => Math.abs(a - (spec.ny - 1) / 2) - Math.abs(b - (spec.ny - 1) / 2));
  let next = 0, done = 0, lastDraw = 0;
  const chunk = Math.max(1, Math.round(spec.ny / (poolSize * 6)));
  $("sigma-note").textContent = "computing…";
  const adds = [];
  const worker = async (w) => {
    await w.ready;
    while (next < order.length && run === sigmaRun) {
      const rows = order.slice(next, next + chunk); next += chunk;
      const res = await w.call("sigma_rows", spec, rows);
      if (run !== sigmaRun) return;
      adds.push(py.call("sigma_add", res));          // the pool does not wait for the main worker
      done += rows.length;
      $("sg-bar").style.width = `${(100 * done) / spec.ny}%`;
      $("sigma-note").textContent = `computing… ${(done * spec.nx).toLocaleString()} of ${(spec.nx * spec.ny).toLocaleString()} points`;
      if (performance.now() - lastDraw > 1500) { lastDraw = performance.now(); refreshSigma(); }
    }
  };
  try {
    await Promise.all(pool.map(worker));
    await Promise.all(adds);
  } catch (e) {
    console.error(e);
    pool.forEach((w) => w.terminate()); pool = [];      // start fresh workers next time
    if (run === sigmaRun) {
      s.running = false; sigmaRun++;
      $("sg-compute").disabled = false; $("sg-cancel").disabled = true;
      $("sigma-note").textContent = `Σ-map error: ${e.message} (the rows computed so far are shown)`;
      refreshSigma();
    }
    return;
  }
  if (run !== sigmaRun) return;
  s.running = false; s.computed = spec;
  $("sg-compute").disabled = false; $("sg-cancel").disabled = true;
  const r = await refreshSigma();
  const secs = (performance.now() - t0) / 1000;
  if (r && r.stats) {
    const st = r.stats, pct = (100 * st.global) / Math.max(1, st.done);
    $("sigma-note").textContent = `done in ${secs.toFixed(1)} s:  ${st.global.toLocaleString()} of ${st.done.toLocaleString()} grid points are global (${pct.toFixed(1)}%), on a ${st.nx} × ${st.ny} grid`;
  }
}

function cancelSigma() {
  sigmaRun++;
  S.sigma.running = false;
  $("sg-compute").disabled = false; $("sg-cancel").disabled = true;
  $("sigma-note").textContent = "cancelled (the rows computed so far are shown)";
  refreshSigma();
}

async function refreshSigma() {
  const s = S.sigma;
  if (!s.spec) return null;
  const wrap = $("sigma-wrap");
  const w = Math.max(360, wrap.clientWidth - 10), h = Math.max(360, wrap.clientHeight - 10);
  const r = await schedule("sigma_draw", "sigma_draw", [{ mode: s.mode, edge: s.edge, mu: s.mu, current: S.params,
    w, h, ratio: ratio(), path: path ? path.overlay() : null }]);
  if (!r) return null;
  $("sigma-img").src = r.png;
  s.axes = r.axes;
  updateSigmaExplain();
  return r;
}

function updateSigmaExplain() {
  $("sg-explain").innerHTML = S.info.sigma_explain[S.sigma.mode] + S.info.sigma_click;
  $("sg-mu-wrap").hidden = S.sigma.mode === "mu";
  const p = S.params, others = NAMES.filter((n) => n !== S.sigma.x_name && n !== S.sigma.y_name);
  const nb = (t) => t.replace(/ /g, "\u00a0");          // keep "c₁ = 0" together when the line wraps
  $("sg-fixed").textContent = `Fixed at the current solution: ${others.map((n) => nb(`${label(n)} = ${fmt(p[n], 4)}`)).join(", ")}, ${nb(`x₀ = ${p.xb}`)}`;
}
const label = (n) => S.info.params.find((q) => q.name === n).label;

// pixel on the image -> data coordinates of the axes under it
function sigmaData(ev) {
  const img = $("sigma-img"), r = img.getBoundingClientRect();
  const fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
  for (const a of S.sigma.axes) {
    if (fx >= a.x0 && fx <= a.x1 && fy >= a.y0 && fy <= a.y1) {
      const x = a.xlim[0] + ((fx - a.x0) / (a.x1 - a.x0)) * (a.xlim[1] - a.xlim[0]);
      const y = a.ylim[1] - ((fy - a.y0) / (a.y1 - a.y0)) * (a.ylim[1] - a.ylim[0]);
      return { x, y, fx, fy };
    }
  }
  return null;
}

function drawCross(d) {
  const ch = $("sigma-cross-h"), cv = $("sigma-cross-v");
  if (!d || !S.sigma.axes.length) { ch.hidden = cv.hidden = true; return; }
  const img = $("sigma-img"), r = img.getBoundingClientRect(), wr = $("sigma-wrap").getBoundingClientRect();
  const a0 = S.sigma.axes[0], aN = S.sigma.axes[S.sigma.axes.length - 1];
  const ox = r.left - wr.left, oy = r.top - wr.top;
  ch.hidden = cv.hidden = false;
  Object.assign(ch.style, { left: `${ox + a0.x0 * r.width}px`, width: `${(aN.x1 - a0.x0) * r.width}px`, top: `${oy + d.fy * r.height}px` });
  Object.assign(cv.style, { top: `${oy + a0.y0 * r.height}px`, height: `${(a0.y1 - a0.y0) * r.height}px`, left: `${ox + d.fx * r.width}px` });
}

let dragging = false, lastJump = 0, hoverT = 0;
function sigmaParams(d) {                // the clicked point, from the map's own spec
  const sp = S.sigma.spec;
  return { ...sp.base, [sp.x_name]: d.x, [sp.y_name]: d.y };
}

async function sigmaJump(d) {
  const sp = S.sigma.spec, q = sigmaParams(d);
  if (S.lock && !(await schedule("jump", "is_global", [q]))) {
    $("sigma-note").textContent = `Locked: (${label(sp.x_name)}, ${label(sp.y_name)}) = (${+d.x.toPrecision(4)}, ${+d.y.toPrecision(4)}) is not global. Untick 'lock to global solutions' to go there.`;
    return;
  }
  setParams(q);
}

function bindSigma() {
  const img = $("sigma-img");
  img.addEventListener("mousemove", async (ev) => {
    const d = sigmaData(ev);
    drawCross(d);
    if (dragging && d && performance.now() - lastJump > 40) { lastJump = performance.now(); sigmaJump(d); return; }
    if (!d) { $("sigma-hover").textContent = "Hover the map to read a point; click to jump there, drag to move; shift-click adds a path keyframe."; return; }
    const t = ++hoverT;
    const r = await schedule("sigma_point", "sigma_point", [d.x, d.y]);
    if (r && t === hoverT) $("sigma-hover").textContent = r.text;
  });
  img.addEventListener("mouseleave", () => { drawCross(null); dragging = false; });
  img.addEventListener("mousedown", (ev) => {
    const d = sigmaData(ev);
    if (!d || ev.button !== 0) return;
    if (ev.shiftKey) {                            // a keyframe for the path panel, no jump
      showPath(true);
      path.addKeyframe(sigmaParams(d));
      status(`keyframe ${path.path.keyframes.length} added from the Σ-map`);
      return;
    }
    dragging = true;
    sigmaJump(d);
  });
  window.addEventListener("mouseup", () => { dragging = false; });
}

// ============================================================================ tabs, panels
function setTab(t) {
  S.tab = t;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  $("tab-view3d").hidden = t !== "view3d";
  $("tab-sigma").hidden = t !== "sigma";
  $("btn-slice").hidden = t === "sigma";
  if (t === "sigma") {
    setRTab("ctlsigma");
    if (!S.sigma.spec && !S.sigma.running) computeSigma();   // the first visit computes a map right away
    else refreshSigma();
  } else {
    setRTab("ctl3d");
    view.resize();
    refreshViews();
  }
}

function setRTab(t) {
  S.rtab = t;
  document.querySelectorAll(".rtab").forEach((b) => b.classList.toggle("active", b.dataset.rtab === t));
  for (const p of ["profiles", "ctl3d", "ctlslice", "ctlsigma"]) $(`rtab-${p}`).hidden = p !== t;
  if (t === "profiles" && S.solved) refreshProfiles();
}

// Show 2D slice: the slice pane and its controls, the plane in 3D on and the other layers off;
// Hide puts the layers back (as the desktop).
let layersBefore = null;
function showSlice(on) {
  S.sliceShown = on;
  $("slice-wrap").hidden = !on;
  $("rtab-slice").hidden = !on;
  $("btn-slice").textContent = on ? "Hide 2D slice" : "Show 2D slice";
  $("btn-slice").classList.toggle("on", on);
  const keys = ["show_streamlines", "show_surface", "show_vortex", "show_selfsim", "clip"];
  if (on) {
    layersBefore = Object.fromEntries(keys.map((k) => [k, S.scene[k]]));
    for (const k of keys) S.scene[k] = false;
  } else if (layersBefore) {
    Object.assign(S.scene, layersBefore); layersBefore = null;
  }
  S.showSlice3d = on;
  if (!on) view.setSlice(null);
  syncSceneWidgets();
  if (on) setRTab("ctlslice"); else if (S.rtab === "ctlslice") setRTab("ctl3d");
  view.resize();
  refreshScene();
  refreshSlice();
}

function showPath(on) {
  $("bottom").hidden = !on;
  $("notation").open = !on;
  $("btn-path").classList.toggle("on", on);
  view.resize();
}

// ============================================================================ 3D controls
function syncSceneWidgets() {
  const s = S.scene;
  $("s-seed_mode").value = s.seed_mode; $("s-n_seeds").value = s.n_seeds;
  [$("s-cx").value, $("s-cy").value, $("s-cz").value] = s.sphere_center;
  $("s-sphere_radius").value = s.sphere_radius; $("s-color_by").value = s.color_by; $("s-shape").value = s.domain_shape;
  document.querySelectorAll("#layers input").forEach((c) => { c.checked = !!s[c.dataset.k]; });
  $("s-r_min").value = s.r_min; $("s-axis_a").value = s.axis_a; $("s-opacity").value = Math.round(s.opacity * 100);
  $("s-show_slice").checked = S.showSlice3d;
}

function bindScene() {
  const num = (id, k, lo, hi) => $(id).addEventListener("change", () => {
    let v = parseFloat($(id).value); if (!isFinite(v)) return;
    S.scene[k] = Math.max(lo, Math.min(hi, v)); refreshScene(); if (k !== "n_seeds") refreshSlice();
  });
  $("s-seed_mode").addEventListener("change", () => { S.scene.seed_mode = $("s-seed_mode").value; refreshScene(); });
  $("s-shape").addEventListener("change", () => { S.scene.domain_shape = $("s-shape").value; refreshScene(); refreshSlice(); });
  num("s-n_seeds", "n_seeds", 1, 600);
  num("s-sphere_radius", "sphere_radius", 0.02, 0.9);
  num("s-r_min", "r_min", 0.01, 0.8);
  num("s-axis_a", "axis_a", 0.002, 0.3);
  ["s-cx", "s-cy", "s-cz"].forEach((id, i) => $(id).addEventListener("change", () => {
    const v = parseFloat($(id).value); if (!isFinite(v)) return;
    S.scene.sphere_center = S.scene.sphere_center.map((c, j) => (j === i ? v : c)); refreshScene();
  }));
  $("s-opacity").addEventListener("change", () => { S.scene.opacity = $("s-opacity").value / 100; refreshScene(); });
  $("s-color_by").addEventListener("change", () => { S.scene.color_by = $("s-color_by").value; refreshScene(); });
  document.querySelectorAll("#layers input").forEach((c) => c.addEventListener("change", () => { S.scene[c.dataset.k] = c.checked; refreshScene(); }));
  $("s-show_slice").addEventListener("change", () => { S.showSlice3d = $("s-show_slice").checked; if (S.showSlice3d) refreshSlice(); else view.setSlice(null); refreshScene(); });
  $("cam-oblique").addEventListener("click", () => view.setCamera(OBLIQUE, 1.85));
  $("cam-pole").addEventListener("click", () => view.setCamera(POLE, 1.0));
}

// ============================================================================ slice controls
function setPlane(p) {
  S.plane = { ...S.plane, ...p };
  $("sl-az").value = $("sl-az-n").value = Math.round(S.plane.azimuth * 100) / 100;
  $("sl-el").value = $("sl-el-n").value = Math.round(S.plane.elevation * 100) / 100;
  $("sl-off").value = $("sl-off-n").value = Math.round(S.plane.offset * 1000) / 1000;
  refreshSlice();
}

function bindSlice() {
  const pres = $("sl-presets");
  for (const p of S.info.slice_presets) {
    const b = document.createElement("button"); b.textContent = p.label;
    b.addEventListener("click", () => setPlane({ azimuth: p.az, elevation: p.el, offset: 0 }));
    pres.appendChild(b);
  }
  for (const [k, ids] of Object.entries({ azimuth: ["sl-az", "sl-az-n"], elevation: ["sl-el", "sl-el-n"], offset: ["sl-off", "sl-off-n"] })) {
    $(ids[0]).addEventListener("input", () => setPlane({ [k]: parseFloat($(ids[0]).value) }));
    $(ids[1]).addEventListener("change", () => { const v = parseFloat($(ids[1]).value); if (isFinite(v)) setPlane({ [k]: v }); });
  }
  for (const [id, k] of [["sl-un", "un"], ["sl-rho", "rho"], ["sl-log", "log"], ["sl-lines", "lines"], ["sl-lic", "lic"], ["sl-follow", "follow"]]) {
    $(id).addEventListener("change", () => { S.sliceOpts[k] = $(id).checked; refreshSlice(); });
  }
  $("sl-density").addEventListener("change", () => { S.sliceOpts.density = parseFloat($("sl-density").value) || 1.4; refreshSlice(); });
  $("sl-view-n").addEventListener("click", () => { const r = S.lastSlice; if (r) view.viewAlong(r.center, r.normal, r.e2); });
  $("sl-export").addEventListener("click", () => S.lastSlice && download(S.lastSlice.png, "slice.png"));
  setPlane({});
}

// ============================================================================ Σ-map controls
function bindSigmaControls() {
  for (const id of ["sg-x", "sg-y"]) {
    for (const p of S.info.params) { const o = document.createElement("option"); o.value = p.name; o.textContent = p.label; $(id).appendChild(o); }
  }
  const sync = () => {
    const s = S.sigma;
    $("sg-x").value = s.x_name; $("sg-y").value = s.y_name;
    $("sg-x0").value = s.x_range[0]; $("sg-x1").value = s.x_range[1];
    $("sg-y0").value = s.y_range[0]; $("sg-y1").value = s.y_range[1];
    $("sg-n").value = s.n; $("sg-mode").value = s.mode; $("sg-edge").checked = s.edge; $("sg-mu").checked = s.mu;
  };
  syncSigma = sync;
  sync();
  const rng = (n) => S.info.params.find((q) => q.name === n).range;
  $("sg-x").addEventListener("change", () => { S.sigma.x_name = $("sg-x").value; S.sigma.x_range = [...rng(S.sigma.x_name)]; sync(); updateSigmaExplain(); });
  $("sg-y").addEventListener("change", () => { S.sigma.y_name = $("sg-y").value; S.sigma.y_range = [...rng(S.sigma.y_name)]; sync(); updateSigmaExplain(); });
  for (const [id, k, i] of [["sg-x0", "x_range", 0], ["sg-x1", "x_range", 1], ["sg-y0", "y_range", 0], ["sg-y1", "y_range", 1]]) {
    $(id).addEventListener("change", () => { const v = parseFloat($(id).value); if (isFinite(v)) S.sigma[k][i] = v; });
  }
  $("sg-n").addEventListener("change", () => { S.sigma.n = Math.max(11, Math.min(401, parseInt($("sg-n").value) || 101)); sync(); });
  $("sg-mode").addEventListener("change", () => { S.sigma.mode = $("sg-mode").value; refreshSigma(); });
  $("sg-edge").addEventListener("change", () => { S.sigma.edge = $("sg-edge").checked; refreshSigma(); });
  $("sg-mu").addEventListener("change", () => { S.sigma.mu = $("sg-mu").checked; refreshSigma(); });
  $("sg-compute").addEventListener("click", () => computeSigma());
  $("sg-cancel").addEventListener("click", cancelSigma);
  $("sg-lens").addEventListener("click", () => {
    const base = { c1: 0, c2: 0, c3: 0, gamma: 0, A: 0, kappa: 1, xb: 0 };
    Object.assign(S.sigma, { x_name: "gamma", y_name: "kappa", x_range: [-2.6, 2.6], y_range: [-4.6, 4.6], mode: "global" });
    sync();
    setParams(base);
    computeSigma(base);
  });
  $("sg-export").addEventListener("click", () => download($("sigma-img").src, "sigma_map.png"));
}

// ============================================================================ profiles controls
function bindProfiles() {
  for (const x of S.info.xvars) { const o = document.createElement("option"); o.value = x.key; o.textContent = x.label; $("pl-xvar").appendChild(o); }
  for (const m of S.info.modes) { const o = document.createElement("option"); o.value = m.key; o.textContent = m.label; $("pl-mode").appendChild(o); }
  const qs = $("pl-quantities");
  for (const q of S.info.quantities) {
    const l = document.createElement("label"); l.className = "chk";
    l.innerHTML = `<input type="checkbox" value="${q.key}" ${S.plots.quantities.includes(q.key) ? "checked" : ""}> ${q.label}`;
    qs.appendChild(l);
  }
  const sync = () => {
    $("pl-xvar").value = S.plots.xvar; $("pl-mode").value = S.plots.mode; $("pl-pole").value = S.plots.pole;
    $("pl-overlay").checked = S.plots.overlay;
    qs.querySelectorAll("input").forEach((c) => { c.checked = S.plots.quantities.includes(c.value); });
    const ll = S.plots.mode === "loglog";
    $("pl-pole").disabled = !ll; $("pl-xvar").disabled = ll;
  };
  syncPlots = sync;
  sync();
  const ch = () => {
    S.plots.xvar = $("pl-xvar").value; S.plots.mode = $("pl-mode").value; S.plots.pole = $("pl-pole").value;
    S.plots.overlay = $("pl-overlay").checked;
    S.plots.quantities = [...qs.querySelectorAll("input")].filter((c) => c.checked).map((c) => c.value);
    sync(); refreshProfiles();
  };
  for (const id of ["pl-xvar", "pl-mode", "pl-pole", "pl-overlay"]) $(id).addEventListener("change", ch);
  qs.addEventListener("change", ch);
  $("pl-pin").addEventListener("click", async () => { const n = await py.call("pin"); $("pl-pins").textContent = n ? `${n} pinned` : ""; refreshProfiles(); });
  $("pl-clear").addEventListener("click", async () => { await py.call("clear_pins"); $("pl-pins").textContent = ""; refreshProfiles(); });
  $("pl-export").addEventListener("change", async () => {
    const kind = $("pl-export").value; $("pl-export").value = "";
    if (kind === "png") { download($("profiles-img").src, "profiles.png"); return; }
    const w = Math.max(280, $("right").clientWidth - 20), n = S.plots.quantities.length;
    const r = await py.call("profiles", { ...S.plots, w, h: Math.max(320, Math.min(160 * n + 120, 1100)), format: kind });
    const text = kind === "svg" ? r.svg : r.csv;
    download(URL.createObjectURL(new Blob([text], { type: kind === "svg" ? "image/svg+xml" : "text/csv" })), `profiles.${kind}`);
  });
}

// ============================================================================ sessions (the desktop's format)
function collectSession() {
  return {
    params: S.params, path: path.path.keyframes.length ? path.path : null,
    lock: S.lock, scene: { ...S.scene, show_slice: S.showSlice3d, show_handle: false },
    camera: view.getCamera(), plane: { ...S.plane, n: S.plane.n || 181 },
    slice: { chk_un: S.sliceOpts.un, chk_lines: S.sliceOpts.lines, chk_lic: S.sliceOpts.lic, chk_rho: S.sliceOpts.rho,
             chk_log: S.sliceOpts.log, follow: S.sliceOpts.follow, density: S.sliceOpts.density },
    plots: { xvar: S.plots.xvar, mode: S.plots.mode, pole: S.plots.pole, quantities: S.plots.quantities, overlay: S.plots.overlay },
    sigma: { x_name: S.sigma.x_name, y_name: S.sigma.y_name, x_range: S.sigma.x_range, y_range: S.sigma.y_range,
             grid: S.sigma.n, mode: S.sigma.mode, edge: S.sigma.edge, mu: S.sigma.mu },
    movie: path.state(),
    ui: { theme: S.theme, tab: S.tab === "sigma" ? 1 : 0, show_slice: S.sliceShown, show_path: !$("bottom").hidden },
  };
}

async function applySession(text) {
  let s;
  try { s = await py.call("session_load", text); } catch (e) { alert("Could not open this file:\n" + e.message); return; }
  const sec = s.sections || {};
  if (S.sigma.running) cancelSigma();
  if (sec.ui && sec.ui.theme) setTheme(sec.ui.theme);
  if ("lock" in sec) { S.lock = !!sec.lock; $("lock").checked = S.lock; }
  const ui = sec.ui || {};
  if (!!ui.show_slice !== S.sliceShown) showSlice(!!ui.show_slice);     // first: it switches 3D layers
  if (sec.scene) { const sc = { ...sec.scene }; delete sc.show_handle; S.showSlice3d = !!sc.show_slice; delete sc.show_slice; Object.assign(S.scene, sc); }
  if (sec.plane) S.plane = { azimuth: sec.plane.azimuth, elevation: sec.plane.elevation, offset: sec.plane.offset, half_size: sec.plane.half_size || 1, n: sec.plane.n || 181 };
  if (sec.slice) Object.assign(S.sliceOpts, { un: sec.slice.chk_un !== false, lines: sec.slice.chk_lines, lic: sec.slice.chk_lic,
    rho: sec.slice.chk_rho, log: sec.slice.chk_log, density: sec.slice.density, follow: !!sec.slice.follow });
  if (sec.plots) { Object.assign(S.plots, sec.plots); syncPlots(); }
  if (sec.sigma) {
    Object.assign(S.sigma, { x_name: sec.sigma.x_name, y_name: sec.sigma.y_name, x_range: sec.sigma.x_range,
      y_range: sec.sigma.y_range, n: Math.min(sec.sigma.grid || 101, 401), mode: sec.sigma.mode, edge: sec.sigma.edge, mu: sec.sigma.mu });
    S.sigma.spec = null; syncSigma();
  }
  path.setPath(s.path, sec.movie || {});
  syncSceneWidgets();
  for (const [id, k] of [["sl-un", "un"], ["sl-rho", "rho"], ["sl-log", "log"], ["sl-lines", "lines"], ["sl-lic", "lic"], ["sl-follow", "follow"]]) $(id).checked = !!S.sliceOpts[k];
  $("sl-density").value = S.sliceOpts.density;
  view.setSlice(null);
  showPath("show_path" in ui ? !!ui.show_path : !!s.path);
  if (sec.camera && sec.camera.position) view.setCamera(sec.camera);
  setPlane({});
  setParams(s.params);
  setTab(ui.tab === 1 ? "sigma" : "view3d");
  status("session loaded");
}

function download(url, name) {
  const a = document.createElement("a"); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
}

// ============================================================================ theme
function setTheme(t) {
  S.theme = t;
  document.documentElement.dataset.theme = t;
  $("btn-theme").textContent = t === "dark" ? "Light theme" : "Dark theme";
  if (!py) return;
  py.call("set_theme", t).then(() => { view.setTheme(t); requestSolve(); if (path) { path.drawTimeline(); path.updateEvents(); } });
}

// ============================================================================ start
window.__errors = [];
window.addEventListener("error", (e) => { if (!/ResizeObserver loop/.test(e.message)) window.__errors.push(String(e.message)); });
window.addEventListener("unhandledrejection", (e) => window.__errors.push(String(e.reason)));

async function main() {
  const steps = ["Python loaded", "numpy, scipy, matplotlib loaded"];
  const t0 = performance.now();
  py = new PyWorker("main", (m) => {
    const k = steps.indexOf(m);
    $("load-msg").textContent = m;
    $("load-bar").style.width = `${k < 0 ? 90 : 30 + 40 * k}%`;
  });
  view = new View3D($("view3d"), $("labels3d"), $("colorbar"));
  try { await py.ready; } catch (e) { $("load-msg").textContent = "Could not start Python: " + e.message; return; }
  S.info = await py.call("info");
  $("load-bar").style.width = "100%";
  for (const c of S.info.color_by) { const o = document.createElement("option"); o.value = c.key; o.textContent = c.label; $("s-color_by").appendChild(o); }
  buildParams();
  const pres = $("presets");
  for (const p of S.info.presets) {
    const b = document.createElement("button"); b.textContent = p.label; b.title = p.tip;
    b.addEventListener("click", async () => {
      let L = 1.5;
      if (p.key === "landau") {
        const dlg = $("landau-dlg"); dlg.returnValue = ""; dlg.showModal();
        const v = await new Promise((res) => dlg.addEventListener("close", () => res(dlg.returnValue), { once: true }));
        if (v !== "ok") return;
        L = parseFloat($("landau-L").value);
        if (!isFinite(L) || L === 0 || Math.abs(L) > 100) return;
      }
      setParams(await py.call("preset", p.key, L));
    });
    pres.appendChild(b);
  }
  S.idle = () => !running && jobs.size === 0;
  window.__idle = S.idle;                            // for scripts/web_screenshots.py
  S.show3d = () => { if (S.tab !== "view3d") setTab("view3d"); };
  path = new PathDock({ py, $, getParams: () => S.params, setParams, view, S,
    onChange: () => { if (S.tab === "sigma") refreshSigma(); }, schedule,
    captureExtras: (keys) => {
      const e = {};
      if (keys.includes("camera")) e.camera = view.getCamera();
      if (keys.includes("plane")) e.plane = { ...S.plane };
      if (keys.includes("seeds")) e.seeds = { seed_mode: S.scene.seed_mode, n_seeds: S.scene.n_seeds, sphere_center: S.scene.sphere_center,
        sphere_radius: S.scene.sphere_radius, line_p0: S.scene.line_p0, line_p1: S.scene.line_p1 };
      return e;
    },
    applyExtras: (e) => {
      if (e.seeds) { Object.assign(S.scene, e.seeds); syncSceneWidgets(); }
      if (e.plane) setPlane(e.plane);
      if (e.camera) view.setCamera(e.camera);
    } });
  bindScene(); bindSlice(); bindSigmaControls(); bindProfiles(); bindSigma();
  syncSceneWidgets();
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  document.querySelector('.tab[data-tab="sigma"]').addEventListener("mouseenter", startPool, { once: true });
  document.querySelectorAll(".rtab").forEach((b) => b.addEventListener("click", () => setRTab(b.dataset.rtab)));
  $("btn-slice").addEventListener("click", () => showSlice(!S.sliceShown));
  $("btn-path").addEventListener("click", () => showPath($("bottom").hidden));
  $("lock").addEventListener("change", () => { S.lock = $("lock").checked; });
  $("show-ranges").addEventListener("change", renderRanges);
  $("btn-theme").addEventListener("click", () => setTheme(S.theme === "dark" ? "light" : "dark"));
  $("btn-help").addEventListener("click", () => $("help").showModal());
  $("btn-save").addEventListener("click", async () => {
    const text = await py.call("session_json", collectSession());
    download(URL.createObjectURL(new Blob([text], { type: "application/json" })), "session.swirl.json");
  });
  $("btn-open").addEventListener("click", () => $("file-open").click());
  $("file-open").addEventListener("change", async () => {
    const f = $("file-open").files[0]; if (!f) return;
    applySession(await f.text()); $("file-open").value = "";
  });
  window.addEventListener("keydown", (ev) => {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    const k = ev.key.toLowerCase();
    if (k === "o") { ev.preventDefault(); $("btn-open").click(); }
    if (k === "s") { ev.preventDefault(); $("btn-save").click(); }
  });
  window.addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(() => refreshViews(), 300); });
  S.params = S.info.start;
  syncWidgets();
  setRTab(S.rtab);                                  // 3D controls raised, as the desktop
  $("loading").classList.add("done");
  view.resize();
  await requestSolve();
  status(`ready in ${((performance.now() - t0) / 1000).toFixed(1)} s · Swirl Explorer ${S.info.version} running in your browser (Pyodide)`);
}
main();
