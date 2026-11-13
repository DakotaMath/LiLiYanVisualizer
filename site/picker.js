// The data picker in the left panel. Four controls set the six data at the base point:
//   the (c₁, c₂) plane, with the c₃ tape beside it    the slice {c₃ fixed} of the Li–Li–Yan set J
//   the A tape                                         drag left/right as far as you like; no limit
//   the (γ, κ) map                                     the global region for the current c, A, x₀
// Both planes: click to pick a point, drag the marker to move it, drag elsewhere to pan, the wheel
// zooms, double-click fits the view again. With 'lock to global solutions' on, c stays in J and
// (γ, κ) stays in the global region: an edit that leaves it stops at the edge, and when c, A or x₀
// change, a (γ, κ) that is no longer global moves to the nearest edge point.

const GREEN = "#3fa877", MAGENTA = "#ff4fd8";
const GK_DEFAULT = { x0: -3, x1: 3, y0: -5, y1: 5 };      // γ × κ: the parameters' default ranges
const C_DEFAULT = { x0: -1.6, x1: 3, y0: -1.6, y1: 3 };
const GRID = 81;
const TAPE_PPU = 40;                                      // c₃ tape: pixels per unit                                          // the (γ, κ) map's grid, per axis

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, n = 5) => (Math.abs(v) < 1e-12 ? "0" : Number(v).toPrecision(n).replace(/(\.\d*?)0+(e|$)/, "$1$2").replace(/\.(e|$)/, "$1")).replace("-", "−");
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ============================================================================ J (Theorem 3.44)
// J = {c₁ ≥ −1, c₂ ≥ −1, c₃ ≥ c̄₃(c₁, c₂)}, c̄₃ = −½ S (S + 2), S = √(1+c₁) + √(1+c₂).
// For fixed c₃ the slice is {c₁, c₂ ≥ −1, S ≥ S*(c₃)}, S* = √(1 − 2c₃) − 1 (0 for c₃ ≥ 0): the
// quadrant minus a corner that grows as c₃ decreases. S is concave, so the slice is convex.
export const cbar3 = (c1, c2) => { const S = Math.sqrt(Math.max(0, 1 + c1)) + Math.sqrt(Math.max(0, 1 + c2)); return -0.5 * S * (S + 2); };
export const sStar = (c3) => (c3 >= 0 ? 0 : Math.sqrt(1 - 2 * c3) - 1);
export const inJ = (c1, c2, c3) => c1 >= -1 && c2 >= -1 && Math.sqrt(1 + c1) + Math.sqrt(1 + c2) >= sStar(c3) * (1 - 1e-12);

// The lock keeps c a little inside J: on the edge of J itself the global (γ, κ) set at A = 0 is
// only a segment of the line κ = 0 (a point on the curved edge), with nothing around it to pick.
export const C_MIN = -0.99, C3_GAP = 0.05;
export const inJlock = (c1, c2, c3) => c1 >= C_MIN && c2 >= C_MIN && Math.sqrt(1 + c1) + Math.sqrt(1 + c2) >= sStar(c3 - C3_GAP) * (1 - 1e-12);

/** The point of {c₁, c₂ ≥ C_MIN, c₃ − c̄₃(c₁, c₂) ≥ C3_GAP} nearest to (c₁, c₂) (itself when it is
 *  inside): the convex set {u₁, u₂ ≥ e, √u₁ + √u₂ ≥ s} in u = 1 + c. */
export function projectC(c1, c2, c3) {
  if (inJlock(c1, c2, c3)) return [c1, c2];
  const e = 1 + C_MIN, s = Math.max(sStar(c3 - C3_GAP), 2 * Math.sqrt(e)), u1 = 1 + c1, u2 = 1 + c2;
  const d = (a, b) => (a - u1) ** 2 + (b - u2) ** 2;
  // the boundary: two rays u_i = e and the curve √u = s (t, 1 − t), √e/s ≤ t ≤ 1 − √e/s
  const r0 = (s - Math.sqrt(e)) ** 2;
  const cands = [[e, Math.max(r0, u2)], [Math.max(r0, u1), e]];
  const t0 = Math.sqrt(e) / s, t1 = 1 - t0;
  if (t1 > t0) {
    const f = (t) => [(s * t) ** 2, (s * (1 - t)) ** 2], g = (t) => d(...f(t)), n = 400;
    let bt = t0;
    for (let i = 1; i <= n; i++) { const t = t0 + ((t1 - t0) * i) / n; if (g(t) < g(bt)) bt = t; }
    const h = (t1 - t0) / n;
    let lo = Math.max(t0, bt - h), hi = Math.min(t1, bt + h);
    for (let k = 0; k < 60; k++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (g(m1) < g(m2)) hi = m2; else lo = m1; }
    cands.push(f(0.5 * (lo + hi)));
  }
  let [a, b] = cands.reduce((p, q) => (d(...q) < d(...p) ? q : p));
  const S = Math.sqrt(a) + Math.sqrt(b);
  if (S < s) { const k = (s / S) ** 2 * (1 + 1e-12); a *= k; b *= k; }       // rounding: stay inside
  return [Math.max(a, e) - 1, Math.max(b, e) - 1];
}

// ============================================================================ ticks
function ticks(a, b, n) {
  const span = b - a;
  if (!(span > 0)) return [];
  const raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
const tickText = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(6))).replace("-", "−");

// ============================================================================ a 2D plane on a canvas
// Pan (drag), zoom (wheel), click, a draggable marker, hover. The view is clamped: its span to
// [minSpan, maxSpan] and its centre to `limits`.
class Plane {
  constructor(canvas, o) {
    this.cv = canvas; this.o = o; this.v = { ...o.view };
    this.pad = { l: 34, r: 6, t: 18, b: 22 };
    this._raf = 0;
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointercancel", () => { this.drag = null; });
    canvas.addEventListener("pointerleave", () => { if (!this.drag) { this.hover = null; this.render(); } });
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("dblclick", (e) => { e.preventDefault(); this.o.onFit && this.o.onFit(); });
    new ResizeObserver(() => this.render()).observe(canvas);
  }
  // geometry (CSS pixels)
  get W() { return this.cv.clientWidth; }
  get H() { return this.cv.clientHeight; }
  get box() { const p = this.pad; return { x: p.l, y: p.t, w: Math.max(10, this.W - p.l - p.r), h: Math.max(10, this.H - p.t - p.b) }; }
  X(x) { const b = this.box, v = this.v; return b.x + ((x - v.x0) / (v.x1 - v.x0)) * b.w; }
  Y(y) { const b = this.box, v = this.v; return b.y + b.h - ((y - v.y0) / (v.y1 - v.y0)) * b.h; }
  data(px, py) { const b = this.box, v = this.v; return [v.x0 + ((px - b.x) / b.w) * (v.x1 - v.x0), v.y0 + ((b.y + b.h - py) / b.h) * (v.y1 - v.y0)]; }
  local(e) { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

  setView(v) {
    const o = this.o, b = this.box;
    let { x0, x1, y0, y1 } = v;
    let sx = clamp(x1 - x0, o.minSpan, o.maxSpan), sy = clamp(y1 - y0, o.minSpan, o.maxSpan);
    if (o.equal) {                                   // one unit is as long on both axes
      const k = b.h / b.w;
      if (sy / sx > k) sx = sy / k; else sy = sx * k;
      if (sx > o.maxSpan) { sx = o.maxSpan; sy = sx * k; }
    }
    const cx = clamp(0.5 * (x0 + x1), o.limits.x[0], o.limits.x[1]), cy = clamp(0.5 * (y0 + y1), o.limits.y[0], o.limits.y[1]);
    this.v = { x0: cx - sx / 2, x1: cx + sx / 2, y0: cy - sy / 2, y1: cy + sy / 2 };
    this.render();
  }

  down(e) {
    if (e.button !== 0) return;
    const [px, py] = this.local(e), m = this.o.marker && this.o.marker();
    const onMarker = m && Math.hypot(this.X(m[0]) - px, this.Y(m[1]) - py) < 11;
    this.drag = { mode: onMarker ? "marker" : "press", px, py, v: { ...this.v } };
    this.cv.setPointerCapture(e.pointerId);
    this.cv.style.cursor = onMarker ? "grabbing" : "";
  }
  move(e) {
    const [px, py] = this.local(e), d = this.drag;
    if (!d) {
      const m = this.o.marker && this.o.marker();
      this.cv.style.cursor = m && Math.hypot(this.X(m[0]) - px, this.Y(m[1]) - py) < 11 ? "grab" : "crosshair";
      this.hover = this.data(px, py); this.render();
      return;
    }
    if (d.mode === "press" && Math.hypot(px - d.px, py - d.py) > 4) { d.mode = "pan"; this.cv.style.cursor = "move"; }
    if (d.mode === "pan") {
      const b = this.box, sx = (d.v.x1 - d.v.x0) / b.w, sy = (d.v.y1 - d.v.y0) / b.h;
      const dx = (px - d.px) * sx, dy = (py - d.py) * sy;
      this.setView({ x0: d.v.x0 - dx, x1: d.v.x1 - dx, y0: d.v.y0 + dy, y1: d.v.y1 + dy });
      this.o.onView && this.o.onView(false);
    } else if (d.mode === "marker") {
      this.o.onPick(...this.data(px, py), "drag");
    }
    this.hover = this.data(px, py); this.render();
  }
  up(e) {
    const d = this.drag; this.drag = null;
    if (!d) return;
    const [px, py] = this.local(e);
    this.cv.style.cursor = "crosshair";
    if (d.mode === "press" || d.mode === "marker") this.o.onPick(...this.data(px, py), "end");
    if (d.mode === "pan") this.o.onView && this.o.onView(true);
  }
  wheel(e) {
    e.preventDefault();
    const [px, py] = this.local(e), [x, y] = this.data(px, py);
    const k = Math.exp(clamp(e.deltaY, -300, 300) * 0.0018), v = this.v;
    this.setView({ x0: x + (v.x0 - x) * k, x1: x + (v.x1 - x) * k, y0: y + (v.y0 - y) * k, y1: y + (v.y1 - y) * k });
    this.o.onView && this.o.onView(true, true);
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.paint(); });
  }
  paint() {
    const cv = this.cv, r = Math.min(window.devicePixelRatio || 1, 2), W = this.W, H = this.H;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * r) || cv.height !== Math.round(H * r)) { cv.width = Math.round(W * r); cv.height = Math.round(H * r); }
    const ctx = cv.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const b = this.box;
    ctx.save();
    ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    this.o.draw(ctx, this);
    // grid lines
    ctx.strokeStyle = css("--text"); ctx.globalAlpha = 0.09; ctx.lineWidth = 1;
    const tx = ticks(this.v.x0, this.v.x1, Math.max(5, b.w / 45)), ty = ticks(this.v.y0, this.v.y1, Math.max(4, b.h / 34));
    ctx.beginPath();
    for (const t of tx) { const x = Math.round(this.X(t)) + 0.5; ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); }
    for (const t of ty) { const y = Math.round(this.Y(t)) + 0.5; ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y); }
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (this.o.overlay) this.o.overlay(ctx, this);
    if (this.hover && !this.drag) {                  // a faint crosshair at the cursor
      const hx = Math.round(this.X(this.hover[0])) + 0.5, hy = Math.round(this.Y(this.hover[1])) + 0.5;
      ctx.strokeStyle = css("--text"); ctx.globalAlpha = 0.28; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, b.y); ctx.lineTo(hx, b.y + b.h); ctx.moveTo(b.x, hy); ctx.lineTo(b.x + b.w, hy); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    const m = this.o.marker && this.o.marker();
    if (m) {
      const x = this.X(m[0]), y = this.Y(m[1]);
      ctx.lineWidth = 2.5; ctx.strokeStyle = css("--window");
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = MAGENTA; ctx.lineWidth = 1.5; ctx.strokeStyle = document.documentElement.dataset.theme === "light" ? "#5a1050" : "#fff";
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
    if (this.o.readoutEl) this.o.readoutEl.textContent = this.hover && this.o.readout ? this.o.readout(this.hover) : "";
    // frame, tick labels, axis names
    ctx.strokeStyle = css("--border"); ctx.lineWidth = 1;
    ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
    ctx.fillStyle = css("--muted"); ctx.font = "10px 'Segoe UI', system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of tx) { const x = this.X(t); if (x > b.x + 8 && x < b.x + b.w - 26) ctx.fillText(tickText(t), x, b.y + b.h + 3); }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of ty) { const y = this.Y(t); if (y > b.y + 5 && y < b.y + b.h - 8) ctx.fillText(tickText(t), b.x - 4, y); }
    ctx.fillStyle = css("--text"); ctx.font = "italic 13px 'Segoe UI', system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText(this.o.xlabel, b.x + b.w - 2, b.y + b.h + 2);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(this.o.ylabel, b.x - 4, b.y / 2);
  }
}

// ============================================================================ drag-anywhere controls
// Press and drag: the value follows the mouse's motion (not its position), with the pointer locked,
// so one drag can go as far as it likes. Shift: ten times finer.
function scrubber(el, { onDelta, onEnd, axis }) {
  let active = false;
  const move = (e) => {
    if (!active) return;
    const m = axis === "x" ? e.movementX : -e.movementY;
    if (m) onDelta(m * (e.shiftKey ? 0.1 : 1));
  };
  const up = () => {
    if (!active) return;
    active = false;
    if (document.pointerLockElement === el) document.exitPointerLock();
    document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up);
    el.classList.remove("scrubbing");
    onEnd();
  };
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    active = true;
    el.classList.add("scrubbing");
    try { const p = el.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch (err) { /* no pointer lock */ }
    document.addEventListener("pointermove", move); document.addEventListener("pointerup", up);
  });
}

function hidpi(cv) {
  const r = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * r) || cv.height !== Math.round(H * r)) { cv.width = Math.round(W * r); cv.height = Math.round(H * r); }
  const ctx = cv.getContext("2d"); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H);
  return [ctx, W, H];
}

// ============================================================================ the picker
export class DataPicker {
  /** o: {$, getParams, commit(p), isLocked(), getPool() -> workers, clampWorker} */
  constructor(o) {
    this.o = o; const $ = o.$;
    this.p = { ...o.getParams() };
    this.gen = 0;
    this.grid = null; this.gkAuto = null; this.gkUser = false; this.gkBusy = false;
    this.cUser = false;

    // ---- the (c₁, c₂) plane
    this.cPlane = new Plane($("pk-c"), {
      view: C_DEFAULT, equal: true, minSpan: 0.02, maxSpan: 400, limits: { x: [-1.5, 300], y: [-1.5, 300] },
      xlabel: "c₁", ylabel: "c₂", readoutEl: $("pk-cur-c"),
      draw: (ctx, P) => this.drawC(ctx, P), overlay: (ctx, P) => this.overlayC(ctx, P),
      marker: () => [this.p.c1, this.p.c2],
      onPick: (x, y, phase) => this.editData({ c1: x, c2: y }, phase),
      readout: (d) => `c₁ ${fmt(d[0], 4)}, c₂ ${fmt(d[1], 4)} · ${inJ(d[0], d[1], this.p.c3) ? "in J" : "outside J"}`,
      onView: () => { this.cUser = true; },
      onFit: () => { this.cUser = false; this.fitC(true); },
    });
    // ---- the c₃ tape
    this.tape = $("pk-c3");
    scrubber(this.tape, { axis: "y", onDelta: (m) => this.editData({ c3: this.p.c3 + m * this.c3Step() }, "drag"),
                          onEnd: () => this.editData({}, "end") });
    this.tape.addEventListener("wheel", (e) => {
      e.preventDefault();
      const st = e.shiftKey ? 0.01 : 0.1;
      this.editData({ c3: +(this.p.c3 - Math.sign(e.deltaY) * st).toFixed(10) }, "end");
    }, { passive: false });
    new ResizeObserver(() => this.drawTape()).observe(this.tape);
    // ---- the A bar
    this.abar = $("pk-A");
    scrubber(this.abar, { axis: "x", onDelta: (m) => this.editData({ A: this.p.A + m / TAPE_PPU }, "drag"),
                          onEnd: () => this.editData({}, "end") });
    this.abar.addEventListener("dblclick", () => this.editData({ A: 0 }, "end"));
    this.abar.addEventListener("wheel", (e) => {
      e.preventDefault();
      const st = e.shiftKey ? 0.01 : 0.1;
      this.editData({ A: +(this.p.A - Math.sign(e.deltaY) * st).toFixed(10) }, "end");
    }, { passive: false });
    new ResizeObserver(() => this.drawA()).observe(this.abar);
    // ---- the (γ, κ) map
    this.gkPlane = new Plane($("pk-gk"), {
      view: GK_DEFAULT, equal: false, minSpan: 1e-3, maxSpan: 200, limits: { x: [-60, 60], y: [-120, 120] },
      xlabel: "γ", ylabel: "κ", readoutEl: $("pk-cur-gk"),
      draw: (ctx, P) => this.drawGK(ctx, P), overlay: (ctx, P) => this.overlayGK(ctx, P),
      marker: () => [this.p.gamma, this.p.kappa],
      onPick: (x, y, phase) => this.editGK(x, y, phase),
      readout: (d) => this.readGK(d),
      onView: (done) => { this.gkUser = true; if (done) this.gkViewChanged(); },
      onFit: () => { this.gkUser = false; this.refreshMap(true); },
    });
    // ---- typed values
    for (const n of ["c1", "c2", "c3", "A", "gamma", "kappa", "xb"]) {
      $(`p-${n}`).addEventListener("change", () => {
        const v = parseFloat($(`p-${n}`).value);
        if (!isFinite(v)) { this.syncInputs(); return; }
        if (n === "gamma" || n === "kappa") this.editGK(n === "gamma" ? v : this.p.gamma, n === "kappa" ? v : this.p.kappa, "end");
        else if (n === "xb") this.editData({ xb: clamp(v, -0.99, 0.99) }, "end");
        else this.editData({ [n]: v }, "end");
      });
    }
    this.$ = $;
    this.syncInputs(); this.fitC(false); this.render();
  }

  // ---------------------------------------------------------------- outside changes (no lock)
  /** Presets, the Σ-map, the path and sessions set the data directly: show them, nothing clamped. */
  sync(p) {
    const same = ["c1", "c2", "c3", "gamma", "A", "kappa", "xb"].every((k) => p[k] === this.p[k]);
    this.p = { ...p };
    if (!same && !this.committing) { this.gen++; this.note(""); }     // cancels a pending edit
    this.syncInputs(); this.fitC(false); this.render();
    this.refreshMap(false);
  }

  commit(p) { this.committing = true; try { this.o.commit({ ...p }); } finally { this.committing = false; } }

  /** A short message in a toast over the plane it concerns ("c" or "gk"); it fades out. */
  note(t, where = "gk") {
    for (const w of ["c", "gk"]) {
      const n = this.$(`pk-toast-${w}`);
      if (w !== where && t) continue;
      clearTimeout(n._t);
      n.textContent = t; n.classList.toggle("on", !!t);
      if (t) n._t = setTimeout(() => n.classList.remove("on"), 5000);
    }
  }

  syncInputs() {
    for (const n of ["c1", "c2", "c3", "A", "gamma", "kappa"]) {
      const el = this.$(`p-${n}`);
      if (document.activeElement !== el) el.value = Number(this.p[n].toPrecision(4));
    }
    this.$("p-xb").value = this.p.xb;
  }

  render() { this.cPlane.render(); this.gkPlane.render(); this.drawTape(); this.drawA(); }

  // ---------------------------------------------------------------- edits of c, A, x₀
  // c is moved into J at once (exact); then the (γ, κ) map is redrawn for the new data and a
  // (γ, κ) that has left the global region moves to its edge. Only then is the solution solved.
  async editData(ch, phase) {
    let p = { ...this.p, ...ch };
    const lock = this.o.isLocked();
    if (lock && !inJlock(p.c1, p.c2, p.c3)) {
      [p.c1, p.c2] = projectC(p.c1, p.c2, p.c3);
      this.note("(c₁, c₂) held at the edge of J", "c");
    }
    const changed = ["c1", "c2", "c3", "A", "xb"].some((k) => p[k] !== this.p[k]);
    this.p = p;
    this.syncInputs(); this.fitC(false); this.render();
    if (!changed && phase !== "end") return;
    if (!lock) { this.note(""); this.commit(p); this.refreshMap(false); return; }
    const g = ++this.gen;
    try {
      const grid = await this.refreshMap(false);
      if (g !== this.gen || !grid) return;
      const r = await this.clampGK(p, p.gamma, p.kappa, grid);
      if (g !== this.gen || !r) return;
      if (!r.gk) this.note("no global (γ, κ) here: left as it was");
      else if (r.moved) { [p.gamma, p.kappa] = r.gk; this.note("(γ, κ) moved to the edge of the global set"); }
      this.p = { ...this.p, gamma: p.gamma, kappa: p.kappa };
      this.syncInputs(); this.render();
      this.commit(this.p);
    } catch (e) { console.error(e); }
  }

  // ---------------------------------------------------------------- edits of (γ, κ)
  async editGK(gm, kp, phase) {
    const lock = this.o.isLocked();
    if (!lock) {
      this.p = { ...this.p, gamma: gm, kappa: kp }; this.note("");
      this.syncInputs(); this.render(); this.commit(this.p); return;
    }
    const g = ++this.gen;
    try {
      const grid = this.grid && this.grid.key === this.dataKey(this.p) ? this.grid : await this.refreshMap(false);
      if (g !== this.gen) return;
      const r = await this.clampGK(this.p, gm, kp, grid, phase === "drag");
      if (g !== this.gen || !r) return;
      if (!r.gk) { this.note("no global (γ, κ) here"); return; }
      if (r.moved) this.note("held at the edge of the global set");
      this.p = { ...this.p, gamma: r.gk[0], kappa: r.gk[1] };
      this.syncInputs(); this.render();
      this.commit(this.p);
    } catch (e) { console.error(e); }
  }

  // Lock for (γ, κ), on the clamp worker: one request at a time, the newest waiting one wins.
  clampGK(p, gm, kp, grid, quick = false) {
    const seeds = [];
    if (grid && grid.n) {
      const sx = grid.v.x1 - grid.v.x0, sy = grid.v.y1 - grid.v.y0, N = grid.N;
      const cand = [];
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        if (!grid.glob[j * N + i]) continue;
        const x = grid.v.x0 + (sx * i) / (N - 1), y = grid.v.y0 + (sy * j) / (N - 1);
        cand.push([((x - gm) / sx) ** 2 + ((y - kp) / sy) ** 2, x, y]);
      }
      cand.sort((a, b) => a[0] - b[0]);
      for (const c of cand.slice(0, 8)) seeds.push([c[1], c[2]]);
    }
    if (grid && grid.k0) seeds.push([clamp(gm, grid.k0[0], grid.k0[1]), 0]);     // the face κ = 0 (always global)
    if (seeds.length === 0) seeds.push([this.p.gamma, this.p.kappa]);
    const req = { p: { c1: p.c1, c2: p.c2, c3: p.c3, A: p.A, xb: p.xb, gamma: gm, kappa: kp }, target: [gm, kp], seeds, quick };
    return new Promise((res) => {
      if (this.clampWait) this.clampWait.res(null);          // superseded
      this.clampWait = { req, res };
      this.pumpClamp();
    });
  }
  async pumpClamp() {
    if (this.clampBusy) return;
    this.clampBusy = true;
    try {
      while (this.clampWait) {
        const { req, res } = this.clampWait; this.clampWait = null;
        try { res(await this.o.clampWorker.call("gk_clamp", req)); } catch (e) { console.error(e); res(null); }
      }
    } finally { this.clampBusy = false; }
  }

  // ---------------------------------------------------------------- the (γ, κ) map
  dataKey(p) { return JSON.stringify([p.c1, p.c2, p.c3, p.A, p.xb]); }

  /** The map for the current data: in the user's view, else fitted to the whole global region.
   *  Resolves to the grid (null if superseded). force: compute even if the data are the same. */
  refreshMap(force) {
    const key = this.dataKey(this.p);
    if (!force && this.grid && this.grid.key === key && !this.gkStale) return Promise.resolve(this.grid);
    if (!force && this.mapJob && this.mapJob.key === key) return this.mapJob.promise;
    const job = { key, run: (this.mapRun || 0) + 1 };
    this.mapRun = job.run;
    job.promise = (this.gkUser ? this.computeGrid(this.p, this.gkPlane.v, job.run) : this.fitGK(this.p, job.run)).then((grid) => {
      if (job.run !== this.mapRun) return this.mapJob ? this.mapJob.promise : this.grid;   // superseded: the newer map
      if (grid && grid.n) return this.refineGrid(grid, job.run).then(() => grid);
      return grid;
    }).then((grid) => {
      if (job.run !== this.mapRun) return this.mapJob ? this.mapJob.promise : this.grid;
      this.mapJob = null; this.gkStale = false;
      if (grid) { this.grid = grid; this.gkPlane.render(); }
      this.$("pk-busy").hidden = true;
      return grid;
    });
    this.mapJob = job;
    this.$("pk-busy").hidden = false;
    return job.promise;
  }

  gkViewChanged() {                       // after a pan or zoom: recompute over the new view, soon
    clearTimeout(this._gkT);
    this._gkT = setTimeout(() => { this.gkStale = true; this.refreshMap(true); }, 220);
  }

  async computeGrid(p, v, run, N = GRID) {
    const pool = this.o.getPool();
    const spec = { base: { ...p }, x_name: "gamma", y_name: "kappa", x_range: [v.x0, v.x1], y_range: [v.y0, v.y1], nx: N, ny: N };
    const rows = [...Array(N).keys()], per = Math.ceil(N / pool.length);
    const parts = await Promise.all(pool.map(async (w, k) => {
      const mine = rows.slice(k * per, (k + 1) * per);
      if (!mine.length || run !== this.mapRun) return [];
      await w.ready;
      return w.call("sigma_rows", spec, mine);
    }));
    if (run !== this.mapRun) return null;
    const glob = new Uint8Array(N * N);
    let n = 0;
    for (const part of parts) for (const r of part) for (let i = 0; i < N; i++) {
      if (r.cn[i] >= 2 && r.cs[i] >= 2) { glob[r.j * N + i] = 1; n++; }
    }
    return { key: this.dataKey(p), v: { ...v }, N, glob, n };
  }

  /** Where the edge crosses each grid edge between a global and a non-global point (bisection on the
   *  pool): grid.frac[key] = fraction from the lower-index end; key "h"/"v" + index of that end. */
  async refineGrid(grid, run) {
    const N = grid.N, v = grid.v, G0 = grid.glob, edges = [], keys = [];
    const X = (i) => v.x0 + ((v.x1 - v.x0) * i) / (N - 1), Y = (j) => v.y0 + ((v.y1 - v.y0) * j) / (N - 1);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const a = G0[j * N + i];
      for (const [di, dj, h] of [[1, 0, "h"], [0, 1, "v"]]) {
        if (i + di >= N || j + dj >= N) continue;
        const b = G0[(j + dj) * N + i + di];
        if (a === b) continue;
        keys.push([h + (j * N + i), a ? 0 : 1]);         // measured from the global end
        edges.push(a ? [X(i), Y(j), X(i + di), Y(j + dj)] : [X(i + di), Y(j + dj), X(i), Y(j)]);
      }
    }
    if (!edges.length) return;
    const pool = this.o.getPool(), per = Math.ceil(edges.length / pool.length);
    const base = { ...this.p };
    const parts = await Promise.all(pool.map(async (w, k) => {
      const mine = edges.slice(k * per, (k + 1) * per);
      if (!mine.length || run !== this.mapRun) return [];
      return w.call("refine_edges", { p: base, edges: mine, steps: 6 });
    }));
    if (run !== this.mapRun) return;
    const fr = parts.flat(), frac = {};
    keys.forEach(([key, flip], n) => { frac[key] = flip ? 1 - fr[n] : fr[n]; });
    grid.frac = frac;
  }

  /** The global γ on the line κ = 0 (fine scan over the pool): [γ_min, γ_max], or null. */
  async scanK0(p, run) {
    const pool = this.o.getPool(), lo = -30, hi = 30, n = 300, w = (hi - lo) / pool.length;
    const parts = await Promise.all(pool.map(async (wk, k) => {
      await wk.ready;
      if (run !== this.mapRun) return [];
      const spec = { base: { ...p, kappa: 0 }, x_name: "gamma", y_name: "kappa", x_range: [lo + k * w, lo + (k + 1) * w],
                     y_range: [0, 0], nx: n, ny: 1 };
      const r = await wk.call("sigma_rows", spec, [0]);
      return r[0].cn.map((c, i) => (c >= 2 && r[0].cs[i] >= 2 ? spec.x_range[0] + (w * i) / (n - 1) : null)).filter((x) => x !== null);
    }));
    const g = parts.flat();
    return g.length ? [Math.min(...g), Math.max(...g)] : null;
  }

  extent(grid) {
    if (!grid.n) return null;
    const N = grid.N, v = grid.v;
    let i0 = N, i1 = -1, j0 = N, j1 = -1;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) if (grid.glob[j * N + i]) {
      i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j);
    }
    const X = (i) => v.x0 + ((v.x1 - v.x0) * i) / (N - 1), Y = (j) => v.y0 + ((v.y1 - v.y0) * j) / (N - 1);
    return { x: [X(i0), X(i1)], y: [Y(j0), Y(j1)], left: i0 === 0, right: i1 === N - 1, bottom: j0 === 0, top: j1 === N - 1,
             cx: (v.x1 - v.x0) / (N - 1), cy: (v.y1 - v.y0) / (N - 1) };
  }

  // Zoomed out to the whole global region: widen while it touches an edge, zoom in while it fills
  // less than 60% of the view. Starts from the last fitted view, so small changes take one pass.
  async fitGK(p, run) {
    let v = { ...(this.gkAuto || GK_DEFAULT) }, grid = null, k0;
    for (let round = 0; round < 8; round++) {
      grid = await this.computeGrid(p, v, run);
      if (!grid) return null;
      const e = this.extent(grid);
      const sx = v.x1 - v.x0, sy = v.y1 - v.y0;
      if (!e) {
        // a region too thin for the grid: it always contains the face κ = 0, U₋(x₀) ≤ γ ≤ U₊(x₀)
        if (k0 === undefined) {
          k0 = await this.scanK0(p, run);
          if (run !== this.mapRun) return null;
          if (k0) {
            const w = k0[1] - k0[0], px = Math.max(0.4 * w, 0.05), py = Math.max(3 * w, 0.3);
            v = { x0: k0[0] - px, x1: k0[1] + px, y0: -py, y1: py };
            continue;
          }
        }
        break;
      }
      if ((e.left || e.right || e.bottom || e.top) && sx < 100) {
        v = { x0: v.x0 - (e.left ? sx / 2 : 0), x1: v.x1 + (e.right ? sx / 2 : 0),
              y0: v.y0 - (e.bottom ? sy / 2 : 0), y1: v.y1 + (e.top ? sy / 2 : 0) };
        continue;
      }
      const px = Math.max(0.1 * (e.x[1] - e.x[0]), 2 * e.cx, 0.01), py = Math.max(0.1 * (e.y[1] - e.y[0]), 2 * e.cy, 0.01);
      const t = { x0: e.x[0] - px, x1: e.x[1] + px, y0: e.y[0] - py, y1: e.y[1] + py };
      const clear = e.x[0] - v.x0 >= 1.5 * e.cx && v.x1 - e.x[1] >= 1.5 * e.cx && e.y[0] - v.y0 >= 1.5 * e.cy && v.y1 - e.y[1] >= 1.5 * e.cy;
      if (clear && t.x1 - t.x0 >= 0.85 * sx && t.y1 - t.y0 >= 0.85 * sy) break;    // fits well enough
      v = t;
    }
    if (run !== this.mapRun) return null;
    if (grid && !grid.n && k0) grid.k0 = k0;
    if (grid && grid.n || k0) this.gkAuto = { ...v };
    this.gkPlane.setView(v);
    return grid;
  }

  drawGK(ctx, P) {
    const dark = document.documentElement.dataset.theme !== "light";
    ctx.fillStyle = css("--plot-bg");
    ctx.fillRect(0, 0, P.W, P.H);
    const g = this.grid;
    if (!g) return;
    if (g.k0) {                                   // only the segment of κ = 0 is global
      ctx.strokeStyle = GREEN; ctx.lineWidth = 4; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(P.X(g.k0[0]), P.Y(0)); ctx.lineTo(P.X(g.k0[1]), P.Y(0)); ctx.stroke();
      ctx.lineCap = "butt";
    }
    const N = g.N;
    if (!this._off || this._off.width !== N) { this._off = document.createElement("canvas"); this._off.width = this._off.height = N; }
    if (this._offFor !== g) {
      const o = this._off.getContext("2d"), img = o.createImageData(N, N);
      const [r, gg, b] = [0x3f, 0xa8, 0x77];
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const k = 4 * ((N - 1 - j) * N + i), on = g.glob[j * N + i];
        img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b; img.data[k + 3] = on ? 235 : 0;
      }
      o.putImageData(img, 0, 0);
      this._offFor = g;
    }
    const cx = (g.v.x1 - g.v.x0) / (N - 1), cy = (g.v.y1 - g.v.y0) / (N - 1);
    const at = (i, j) => [P.X(g.v.x0 + i * cx), P.Y(g.v.y0 + j * cy)];
    // where the edge cuts the grid edge from (i, j) to the right ("h") or up ("v"): refined, else halfway
    const cut = (i, j, h) => {
      const f = g.frac ? g.frac[h + (j * N + i)] : undefined, s = f === undefined ? 0.5 : f;
      return h === "h" ? [i + s, j] : [i, j + s];
    };
    // the inside of each grid cell (marching squares, edges cut at the midpoints), filled as one path
    ctx.beginPath();
    for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
      const q = [g.glob[j * N + i], g.glob[j * N + i + 1], g.glob[(j + 1) * N + i + 1], g.glob[(j + 1) * N + i]];
      const s = q[0] + q[1] + q[2] + q[3];
      if (!s) continue;
      const cor = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], pts = [];
      const mids = [cut(i, j, "h"), cut(i + 1, j, "v"), cut(i, j + 1, "h"), cut(i, j, "v")];   // s, e, n, w
      for (let k = 0; k < 4; k++) {
        const n = (k + 1) % 4;
        if (q[k]) pts.push(cor[k]);
        if (q[k] !== q[n]) pts.push(mids[k]);
      }
      const p0 = at(...pts[0]); ctx.moveTo(p0[0], p0[1]);
      for (const p of pts.slice(1)) { const pp = at(...p); ctx.lineTo(pp[0], pp[1]); }
      ctx.closePath();
    }
    ctx.fillStyle = dark ? "rgba(63,168,119,0.92)" : "rgba(63,168,119,0.88)";
    ctx.fill("nonzero");
    // the edge: marching squares on the 0/1 field
    ctx.strokeStyle = dark ? "#cfe9dc" : "#1d5e40"; ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
      const a = g.glob[j * N + i], b = g.glob[j * N + i + 1], c = g.glob[(j + 1) * N + i + 1], d = g.glob[(j + 1) * N + i];
      const k = a | (b << 1) | (c << 2) | (d << 3);
      if (k === 0 || k === 15) continue;
      const e = { s: at(...cut(i, j, "h")), e: at(...cut(i + 1, j, "v")), n: at(...cut(i, j + 1, "h")), w: at(...cut(i, j, "v")) };
      const seg = { 1: ["w", "s"], 2: ["s", "e"], 3: ["w", "e"], 4: ["e", "n"], 6: ["s", "n"], 7: ["w", "n"], 8: ["n", "w"],
                    9: ["n", "s"], 11: ["n", "e"], 12: ["e", "w"], 13: ["e", "s"], 14: ["s", "w"] }[k];
      const segs = k === 5 ? [["w", "n"], ["s", "e"]] : k === 10 ? [["w", "s"], ["n", "e"]] : [seg];
      for (const [u, w] of segs) { ctx.moveTo(...e[u]); ctx.lineTo(...e[w]); }
    }
    ctx.stroke();
  }
  overlayGK(ctx, P) {
    const g = this.grid;
    if (g && !g.n && !g.k0 && !this.mapJob) {
      ctx.fillStyle = css("--muted"); ctx.font = "12px 'Segoe UI', system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const b = P.box;
      ctx.fillText("no global (γ, κ) for these c, A, x₀", b.x + b.w / 2, b.y + b.h / 2);
    }
  }
  readGK(d) {
    const g = this.grid;
    let s = "";
    if (g && g.n && g.key === this.dataKey(this.p)) {
      const N = g.N, i = Math.round(((d[0] - g.v.x0) / (g.v.x1 - g.v.x0)) * (N - 1)), j = Math.round(((d[1] - g.v.y0) / (g.v.y1 - g.v.y0)) * (N - 1));
      if (i >= 0 && i < N && j >= 0 && j < N) s = g.glob[j * N + i] ? " · global" : " · not global";
    }
    return `γ ${fmt(d[0], 4)}, κ ${fmt(d[1], 4)}${s}`;
  }

  // ---------------------------------------------------------------- the (c₁, c₂) plane
  c3Step() { return 1 / TAPE_PPU; }           // per pixel of drag: the tape moves with the mouse

  fitC(force) {                               // keep the slice's corner and the marker in view
    if (this.cUser && !force) {
      const v = this.cPlane.v, x = this.p.c1, y = this.p.c2;
      if (x >= v.x0 && x <= v.x1 && y >= v.y0 && y <= v.y1) return;
      const sx = v.x1 - v.x0, sy = v.y1 - v.y0;
      this.cPlane.setView({ x0: x - sx / 2, x1: x + sx / 2, y0: y - sy / 2, y1: y + sy / 2 });
      return;
    }
    const s2 = sStar(this.p.c3) ** 2, m = Math.max(this.p.c1, this.p.c2);
    let hi = Math.max(C_DEFAULT.x1, 1.35 * s2 + 0.5);
    if (m > hi - 0.05 * (hi + 1)) hi = m + 0.2 * (hi + 1);     // the marker stays in view
    const lo = -1 - 0.18 * (hi + 1);
    this.cPlane.setView({ x0: lo, x1: hi, y0: lo, y1: hi });
  }

  drawC(ctx, P) {
    const dark = document.documentElement.dataset.theme !== "light";
    // outside J: hatched
    ctx.fillStyle = dark ? "#2a2326" : "#efe4e2"; ctx.fillRect(0, 0, P.W, P.H);
    ctx.save();
    ctx.strokeStyle = dark ? "rgba(255,107,107,0.36)" : "rgba(192,57,43,0.42)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = -P.H; k < P.W; k += 7) { ctx.moveTo(k, P.H); ctx.lineTo(k + P.H, 0); }
    ctx.stroke(); ctx.restore();
    // the slice of J
    const s = sStar(this.p.c3), s2 = s * s, v = P.v, R = Math.max(v.x1, v.y1) + 10 * (v.x1 - v.x0);
    ctx.beginPath();
    ctx.moveTo(P.X(-1), P.Y(R)); ctx.lineTo(P.X(-1), P.Y(s2 - 1));
    for (let i = 1; i <= 120; i++) { const t = i / 120; ctx.lineTo(P.X(s2 * t * t - 1), P.Y(s2 * (1 - t) * (1 - t) - 1)); }
    ctx.lineTo(P.X(R), P.Y(-1)); ctx.lineTo(P.X(R), P.Y(R)); ctx.closePath();
    ctx.fillStyle = css("--plot-bg"); ctx.fill();
  }
  overlayC(ctx, P) {
    const s = sStar(this.p.c3), s2 = s * s, v = P.v, R = Math.max(v.x1, v.y1) + 10 * (v.x1 - v.x0);
    ctx.lineWidth = 2; ctx.strokeStyle = css("--bad");
    ctx.beginPath();
    ctx.moveTo(P.X(-1), P.Y(R)); ctx.lineTo(P.X(-1), P.Y(s2 - 1));
    for (let i = 1; i <= 120; i++) { const t = i / 120; ctx.lineTo(P.X(s2 * t * t - 1), P.Y(s2 * (1 - t) * (1 - t) - 1)); }
    ctx.lineTo(P.X(R), P.Y(-1));
    ctx.stroke();
    const b = P.box;
    ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillStyle = css("--muted");
    if (P.X(-1) > b.x + 70 || P.Y(-1) < b.y + b.h - 18) ctx.fillText("outside J", b.x + 4, b.y + b.h - 3);
  }
  drawTape() {
    const cv = this.tape, [ctx, W, H] = hidpi(cv);
    if (!W) return;
    const dark = document.documentElement.dataset.theme !== "light";
    const ppu = TAPE_PPU, c3 = this.p.c3, Y = (v) => H / 2 - (v - c3) * ppu;
    ctx.fillStyle = css("--base"); ctx.fillRect(0, 0, W, H);
    // below c̄₃(c₁, c₂) the marked c is outside J: lowering c₃ there pushes (c₁, c₂) along the edge
    const cb = cbar3(this.p.c1, this.p.c2), yb = Y(cb);
    if (yb < H) {
      const y0 = Math.max(0, yb);
      ctx.fillStyle = dark ? "#2a2326" : "#efe4e2"; ctx.fillRect(0, y0, W, H - y0);
      ctx.save(); ctx.beginPath(); ctx.rect(0, y0, W, H - y0); ctx.clip();
      ctx.strokeStyle = dark ? "rgba(255,107,107,0.36)" : "rgba(192,57,43,0.42)"; ctx.lineWidth = 1; ctx.beginPath();
      for (let k = -W; k < H + W; k += 7) { ctx.moveTo(0, k + W); ctx.lineTo(W, k); }
      ctx.stroke(); ctx.restore();
    }
    ctx.strokeStyle = css("--muted"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
    ctx.font = "10px 'Segoe UI', system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const lo = c3 - H / 2 / ppu, hi = c3 + H / 2 / ppu;
    ctx.beginPath();
    for (let v = Math.ceil(lo * 4) / 4; v <= hi; v += 0.25) {
      const y = Math.round(Y(v)) + 0.5, major = Math.abs(v - Math.round(v)) < 1e-9;
      ctx.moveTo(W - (major ? 9 : 5), y); ctx.lineTo(W, y);
      if (major && Math.abs(y - H / 2) > 12 && Math.abs(y - yb) > 10) ctx.fillText(String(Math.round(v)).replace("-", "−"), W - 11, y);
    }
    ctx.stroke();
    // the current value: a fixed pointer with its value, in the middle
    ctx.fillStyle = MAGENTA; ctx.fillRect(0, H / 2 - 9, W, 18);
    ctx.fillStyle = "#2b0a28"; ctx.font = "600 11px 'Segoe UI', system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(fmt(c3, 4), W / 2, H / 2 + 0.5);
    ctx.font = "10px 'Segoe UI', system-ui, sans-serif"; ctx.fillStyle = css("--bad"); ctx.textAlign = "left";
    if (yb > -2 && yb < H + 2) {                  // c̄₃ on top: visible even under the pointer
      const under = Math.abs(yb - H / 2) <= 10;      // under the thumb: only stubs beside it
      ctx.strokeStyle = css("--bad"); ctx.lineWidth = 2.5; ctx.setLineDash([4, 2]);
      ctx.beginPath();
      if (under) { ctx.moveTo(0, H / 2 + 11); ctx.lineTo(W, H / 2 + 11); } else { ctx.moveTo(0, yb); ctx.lineTo(W, yb); }
      ctx.stroke(); ctx.setLineDash([]);
      const y = under ? H / 2 + 11 : yb, below = under || y < 14;
      ctx.textBaseline = below ? "top" : "bottom";
      ctx.fillText("c̄₃", 3, below ? y + 2 : y - 2);
    } else if (yb >= H + 2) {                     // below the visible part: an arrow at the bottom
      ctx.textBaseline = "bottom"; ctx.fillText(`c̄₃ ${fmt(cb, 3)} ▾`, 3, H - 2);
    }
    this.$("pk-cbar").textContent = fmt(cb, 4);
    ctx.strokeStyle = css("--border"); ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  }

  drawA() {                                // a tape under a fixed needle, like c₃: no limit
    const cv = this.abar, [ctx, W, H] = hidpi(cv);
    if (!W) return;
    const A = this.p.A, X = (v) => W / 2 + (v - A) * TAPE_PPU;
    ctx.fillStyle = css("--base"); ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = css("--muted"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
    ctx.font = "10px 'Segoe UI', system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.beginPath();
    const lo = A - W / 2 / TAPE_PPU, hi = A + W / 2 / TAPE_PPU;
    for (let v = Math.ceil(lo * 4) / 4; v <= hi; v += 0.25) {
      const x = Math.round(X(v)) + 0.5, major = Math.abs(v - Math.round(v)) < 1e-9;
      ctx.moveTo(x, H - (major ? 9 : 5)); ctx.lineTo(x, H);
      if (major && x > 8 && x < W - 8 && Math.abs(x - W / 2) > 26) ctx.fillText(String(Math.round(v)).replace("-", "−"), x, 15);
    }
    ctx.stroke();
    // the needle with the value
    ctx.fillStyle = MAGENTA; ctx.fillRect(W / 2 - 22, 1, 44, 13);
    ctx.fillRect(W / 2 - 1, 1, 2, H - 2);
    ctx.fillStyle = "#2b0a28"; ctx.font = "600 11px 'Segoe UI', system-ui, sans-serif"; ctx.textBaseline = "middle";
    ctx.fillText(fmt(A, 4), W / 2, 8);
    ctx.strokeStyle = css("--border"); ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  }
}
