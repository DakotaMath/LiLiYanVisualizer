// The path panel: keyframes, the scanned timeline, playback and video recording. Mirrors the desktop's
// path dock (swirl_explorer/ui/panels/path_panel.py); the scan and the events are core.paths in Python.
const NAMES = ["c1", "c2", "c3", "gamma", "A", "kappa"];
const SHORT = { c1: "c₁", c2: "c₂", c3: "c₃", gamma: "γ", A: "A", kappa: "κ" };
const COL = { ok: "#3fa877", other: "#2f8f9d", blowup: "#8a7650", unsafe: "#b9bdc7", bad: "#d64045", event: "#f2c14e" };
const EXTRA = ["camera", "plane", "seeds"];
const m = (x) => Number(x).toPrecision(6).replace(/(\.\d*?)0+(e|$)/, "$1$2").replace(/\.(e|$)/, "$1").replace("-", "−");

// ---------------------------------------------------------------- extras interpolation (core/paths.py)
const lerp = (a, b, s) => a.map((x, i) => (1 - s) * x + s * b[i]);
const norm = (v) => Math.hypot(...v) || 1;
function slerp(u, v, s) {
  u = u.map((x) => x / norm(u)); v = v.map((x) => x / norm(v));
  const d = Math.max(-1, Math.min(1, u[0] * v[0] + u[1] * v[1] + u[2] * v[2])), om = Math.acos(d);
  if (om < 1e-9) return u;
  if (Math.abs(om - Math.PI) < 1e-6) {
    let p = [u[1], -u[0], 0]; if (norm(p) < 1e-6) p = [0, u[2], -u[1]];
    p = p.map((x) => x / norm(p));
    return u.map((x, i) => Math.cos(Math.PI * s) * x + Math.sin(Math.PI * s) * p[i]);
  }
  return u.map((x, i) => (Math.sin((1 - s) * om) * x + Math.sin(s * om) * v[i]) / Math.sin(om));
}
const INTERP = {
  camera(c0, c1, s) {
    const f0 = c0.focal_point, f1 = c1.focal_point;
    const d0 = c0.position.map((x, i) => x - f0[i]), d1 = c1.position.map((x, i) => x - f1[i]);
    const f = lerp(f0, f1, s), dist = (1 - s) * norm(d0) + s * norm(d1);
    const d = slerp(d0, d1, s).map((x) => x * dist);
    let up = slerp(c0.view_up, c1.view_up, s);
    const dn = d.map((x) => x / norm(d)), k = up[0] * dn[0] + up[1] * dn[1] + up[2] * dn[2];
    up = up.map((x, i) => x - k * dn[i]); up = up.map((x) => x / norm(up));
    const out = { position: f.map((x, i) => x + d[i]), focal_point: f, view_up: up };
    if (c0.parallel_scale && c1.parallel_scale) out.parallel_scale = (1 - s) * c0.parallel_scale + s * c1.parallel_scale;
    return out;
  },
  plane(p0, p1, s) {
    const out = { ...(s >= 0.5 ? p1 : p0) };
    const da = ((p1.azimuth - p0.azimuth + 540) % 360) - 180;
    out.azimuth = ((p0.azimuth + s * da + 540) % 360) - 180;
    out.elevation = (1 - s) * p0.elevation + s * p1.elevation;
    out.offset = (1 - s) * p0.offset + s * p1.offset;
    return out;
  },
  seeds(a, b, s) {
    const out = { ...(s >= 0.5 ? b : a) };
    for (const k of ["sphere_center", "line_p0", "line_p1"]) if (a[k] && b[k]) out[k] = lerp(a[k], b[k], s);
    if ("sphere_radius" in a && "sphere_radius" in b) out.sphere_radius = (1 - s) * a.sphere_radius + s * b.sphere_radius;
    return out;
  },
};

export class PathDock {
  constructor(o) {
    Object.assign(this, o);
    this.path = { interpolation: "linear", coords: "cauchy", keyframes: [] };
    this.frames = []; this.events = []; this.index = 0; this.sel = -1; this.scanGen = 0;
    const $ = this.$;
    this.table = $("pa-table"); this.canvas = $("pa-timeline");
    this.table.querySelector("thead").innerHTML = `<tr><th title="the path parameter at this keyframe (not physical time: the solutions are stationary)">path t</th>${NAMES.map((n) => `<th id="pa-h-${n}">${SHORT[n]}</th>`).join("")}<th title="the view this keyframe restores (camera, slice plane, seeds), chosen by the boxes above">restores view</th></tr>`;
    $("pa-add").onclick = () => this.addKeyframe(this.getParams());
    $("pa-set").onclick = () => {
      if (this.sel < 0) return;
      const k = this.path.keyframes[this.sel];
      k.params = { ...this.getParams() }; k.extras = this.captureExtras(this.keys());
      this.changed(k.t);
    };
    $("pa-del").onclick = () => { if (this.sel >= 0) { this.path.keyframes.splice(this.sel, 1); this.sel = -1; this.changed(); } };
    $("pa-clear").onclick = () => { this.path.keyframes = []; this.sel = -1; this.changed(); };
    $("pa-interp").onchange = () => { this.path.interpolation = $("pa-interp").value; this.changed(); };
    $("pa-frames").onchange = () => this.changed();
    $("pa-fps").onchange = () => this.updateLength();
    $("pa-first").onclick = () => this.goto(0);
    $("pa-last").onclick = () => this.goto(this.n() - 1);
    $("pa-play").onclick = () => this.play(!this.playing);
    $("pa-video").onclick = () => this.record();
    this.canvas.addEventListener("mousedown", (ev) => { this.scrubbing = true; this.goto(this.frameAt(ev)); });
    this.canvas.addEventListener("mousemove", (ev) => { if (this.scrubbing) this.goto(this.frameAt(ev)); this.tip(ev); });
    window.addEventListener("mouseup", () => { this.scrubbing = false; });
    new ResizeObserver(() => this.drawTimeline()).observe(this.canvas);
    $("pa-events").addEventListener("click", (ev) => { const a = ev.target.closest("a[data-i]"); if (a) this.goto(+a.dataset.i); });
    this.refresh();
  }

  n() { return Math.max(2, Math.min(2000, parseInt(this.$("pa-frames").value) || 48)); }
  fps() { return Math.max(1, Math.min(60, parseInt(this.$("pa-fps").value) || 24)); }
  keys() { return EXTRA.filter((k) => this.$(`pa-k-${k}`).checked); }
  duration() { const k = this.path.keyframes; return [k[0].t, k[k.length - 1].t]; }
  tOf(i) { const [a, b] = this.duration(); return a + ((b - a) * i) / Math.max(1, this.n() - 1); }

  addKeyframe(p, extras = null) {
    const k = this.path.keyframes;
    const t = k.length ? k[k.length - 1].t + 1 : 0;
    const xb = k.length ? k[0].params.xb : p.xb;
    k.push({ t, params: { ...p, xb }, extras: extras || this.captureExtras(this.keys()) });
    this.changed(t);
  }

  setPath(d, movie = {}) {
    this.path = d ? { interpolation: d.interpolation || "linear", coords: d.coords || "cauchy", keyframes: d.keyframes || [] }
                  : { interpolation: "linear", coords: "cauchy", keyframes: [] };
    this.$("pa-interp").value = this.path.interpolation;
    for (const k of EXTRA) this.$(`pa-k-${k}`).checked = this.path.keyframes.some((f) => f.extras && k in f.extras);
    if (movie.frames) this.$("pa-frames").value = movie.frames;
    if (movie.fps) this.$("pa-fps").value = movie.fps;
    this.$("pa-loop").checked = !!movie.loop;
    this.sel = -1;
    this.changed();
  }

  state() { return { frames: this.n(), fps: this.fps(), loop: this.$("pa-loop").checked }; }

  changed(select = null) {
    this.path.keyframes.sort((a, b) => a.t - b.t);
    if (select !== null) this.sel = this.path.keyframes.findIndex((k) => k.t === select);
    this.frames = []; this.events = [];
    this.index = Math.min(this.index, this.n() - 1);
    this.refresh();
    this.scan();
    this.onChange();
  }

  varying() {
    const k = this.path.keyframes;
    return k.length < 2 ? [] : NAMES.filter((n) => Math.max(...k.map((f) => f.params[n])) > Math.min(...k.map((f) => f.params[n])));
  }

  refresh() {
    const $ = this.$, ks = this.path.keyframes, vary = this.varying();
    for (const n of NAMES) $(`pa-h-${n}`).textContent = SHORT[n] + (vary.includes(n) ? " (varies)" : "");
    const tb = this.table.querySelector("tbody");
    tb.innerHTML = ks.map((k, r) => `<tr data-r="${r}" class="${r === this.sel ? "sel" : ""}"><td contenteditable data-c="t">${+k.t.toPrecision(10)}</td>` +
      NAMES.map((n) => `<td contenteditable data-c="${n}" class="${vary.includes(n) ? "" : "const"}" ${vary.includes(n) ? 'style="font-weight:600"' : ""}>${+Number(k.params[n]).toPrecision(10)}</td>`).join("") +
      `<td class="const">${EXTRA.filter((e) => k.extras && e in k.extras).join(", ") || "—"}</td></tr>`).join("");
    tb.querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => {
      this.sel = +tr.dataset.r; this.refreshSel();
      if (ks.length >= 2) { const [a, b] = this.duration(); this.goto(Math.round(((ks[this.sel].t - a) / ((b - a) || 1)) * (this.n() - 1))); }
    }));
    tb.querySelectorAll("td[contenteditable]").forEach((td) => {
      td.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); td.blur(); } });
      td.addEventListener("blur", () => {
        const v = parseFloat(td.textContent.replace("−", "-")), r = +td.parentElement.dataset.r, c = td.dataset.c;
        if (!isFinite(v)) { this.refresh(); return; }
        const k = ks[r];
        if (c === "t") { if (v !== k.t) { k.t = v; this.changed(v); } }
        else if (v !== k.params[c]) { k.params = { ...k.params, [c]: v }; this.changed(k.t); }
      });
    });
    const n = ks.length, ok = n >= 2;
    for (const id of ["pa-set", "pa-del", "pa-clear"]) $(id).disabled = n === 0;
    for (const id of ["pa-play", "pa-first", "pa-last", "pa-video"]) $(id).disabled = !ok;
    if (!ok && this.playing) this.play(false);
    if (!ok) {
      $("pa-events").innerHTML = `<span class="muted">A path needs two keyframes. Add the current solution with <b>+ Add current</b>, or <b>shift-click</b> points on the Σ-map. Type values in the table to make them exact. The timeline then shows where along the path the solution is global (green), global with a special pole (teal) or not global (grey / olive).</span>`;
      $("pa-frame").textContent = "";
    }
    this.updateLength();
    this.drawTimeline();
  }

  refreshSel() { this.table.querySelectorAll("tbody tr").forEach((tr) => tr.classList.toggle("sel", +tr.dataset.r === this.sel)); }
  updateLength() { this.$("pa-len").textContent = `= ${(this.n() / this.fps()).toFixed(1)} s`; }

  async scan() {
    const g = ++this.scanGen;
    if (this.path.keyframes.length < 2) { this.drawTimeline(); return; }
    this.$("pa-frame").innerHTML = `<span class="muted">checking every frame…</span>`;
    this.$("pa-events").innerHTML = "";
    const r = await this.schedule("scan", "path_scan", [this.path, this.n()]);
    if (!r || g !== this.scanGen) return;
    this.frames = r.frames; this.events = r.events;
    this.drawTimeline(); this.updateEvents(); this.updateFrame();
    this.onChange();
  }

  kind(f) { return !f.global ? f.kind : (f.north === "G" && f.south === "G" ? "ok" : "other"); }

  updateEvents() {
    const $ = this.$, fr = this.frames, n = fr.length;
    if (!n) return;
    const bad = fr.filter((f) => !f.global).length, kinds = new Set(fr.map((f) => this.kind(f)));
    const sw = (c, t) => `<span style="color:${c}">■</span> ${t} &nbsp; `;
    let head = sw(COL.ok, "global");
    if (kinds.has("other")) head += sw(COL.other, "global, an R / L1 / L2 pole");
    if (kinds.has("unsafe")) head += sw(COL.unsafe, "not global: a pole fails the terminal test");
    if (kinds.has("blowup")) head += sw(COL.blowup, "not global: blows up before a pole");
    if (this.events.length) head += `<span style="color:${COL.event}">▼</span> event &nbsp; `;
    head += `<span style="color:${COL.event}">●</span> keyframe (numbers as on the Σ-map)<br>`;
    const summary = bad === 0 ? `<b style="color:${COL.ok}">All ${n} frames are global.</b>`
      : `<b style="color:${COL.bad}">${bad} of ${n} frames are not global.</b> A recorded video holds the last global frame there; the preview shows them as they are.`;
    const names = { leaves_global: "leaves the global set", enters_global: "comes back into the global set",
                    stratum_change: "a pole changes type", failure_change: "still not global, now" };
    const lines = this.events.slice(0, 12).map((e) => {
      const detail = e.detail.replace(/->/g, "→"), sep = e.kind === "failure_change" ? " " : " — ";
      const extra = e.kind === "leaves_global" && e.index > 0
        ? ` <span class="muted">(last global frame: ${e.index}, t = ${Number(this.tOf(e.index - 1)).toPrecision(4)})</span>` : "";
      return `<a data-i="${e.index}">t = ${Number(e.t).toPrecision(4)}</a> (frame ${e.index + 1}): ${names[e.kind]}${detail ? sep + detail : ""}${extra}`;
    });
    if (this.events.length > 12) lines.push(`… and ${this.events.length - 12} more`);
    $("pa-events").innerHTML = head + summary + (lines.length ? "<br>" + lines.join("<br>") : "");
  }

  updateFrame() {
    const $ = this.$;
    if (this.path.keyframes.length < 2) return;
    const i = this.index;
    let txt = `frame ${i + 1} / ${this.n()}, path t = ${m(this.tOf(i))}`;
    const f = this.frames[i];
    if (f) {
      txt += f.global
        ? ` &nbsp; <b style="color:${COL.ok}">global</b> · north ${f.north}, μ₊ = ${f.mu_n?.toFixed(3)} · south ${f.south}, μ₋ = ${f.mu_s?.toFixed(3)}`
        : ` &nbsp; <b style="color:${COL.bad}">NOT GLOBAL</b>: ${f.why}`;
    }
    $("pa-frame").innerHTML = txt;
  }

  frameAt(ev) {
    const r = this.canvas.getBoundingClientRect(), f = (ev.clientX - r.left - 8) / Math.max(1, r.width - 16);
    return Math.round(Math.max(0, Math.min(1, f)) * (this.n() - 1));
  }

  tip(ev) {
    if (this.path.keyframes.length < 2) return;
    const r = this.canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    const [t0, t1] = this.duration(), span = (t1 - t0) || 1, X = (f) => 8 + f * (r.width - 16);
    const names = { leaves_global: "leaves the global set", enters_global: "back in the global set",
                    stratum_change: "a pole changes type", failure_change: "not global, now" };
    for (const e of this.events) {
      if (Math.abs(X((e.t - t0) / span) - x) < 6) { this.canvas.title = `event at t = ${m(e.t)}: ${names[e.kind] || e.kind} ${e.detail.replace(/->/g, "→")}`; return; }
    }
    for (const [j, k] of this.path.keyframes.entries()) {
      if (Math.abs(X((k.t - t0) / span) - x) < 8 && y > 38) { this.canvas.title = `keyframe ${j + 1} at t = ${m(k.t)}`; return; }
    }
    const i = this.frameAt(ev), f = this.frames[i];
    this.canvas.title = f ? `frame ${i + 1}, t = ${m(f.t)}: ${f.global ? "global" : "NOT global"}; north ${f.north}, south ${f.south}` : "";
  }

  drawTimeline() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    if (!W) return;
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const panel = css.getPropertyValue("--panel"), border = css.getPropertyValue("--border");
    const text = css.getPropertyValue("--text"), faint = css.getPropertyValue("--faint");
    g.fillStyle = panel; g.fillRect(0, 0, W, H);
    const top = 18, h = 20, X = (f) => 8 + f * (W - 16), n = this.n();
    if (this.path.keyframes.length < 2) {
      g.fillStyle = faint; g.font = "12px system-ui"; g.textAlign = "center";
      g.fillText("add two keyframes to make a path", W / 2, H / 2); return;
    }
    const w = (W - 16) / n;
    if (this.frames.length) {
      this.frames.forEach((f, i) => { g.fillStyle = COL[this.kind(f)]; g.fillRect(8 + i * w, top, w + 0.6, h); });
    } else { g.fillStyle = border; g.fillRect(8, top, W - 16, h); }
    const [t0, t1] = this.duration(), span = (t1 - t0) || 1;
    // ticks at whole / half units of the path parameter
    const step = span <= 4 ? 0.5 : span <= 12 ? 1 : Math.pow(10, Math.floor(Math.log10(span / 4)));
    g.strokeStyle = faint; g.lineWidth = 1;
    for (let tk = Math.ceil(t0 / step) * step; tk <= t1 + 1e-9; tk += step) {
      const x = X((tk - t0) / span); g.beginPath(); g.moveTo(x, top + h); g.lineTo(x, top + h + 4); g.stroke();
    }
    for (const e of this.events) {
      const x = X((e.t - t0) / span);
      g.strokeStyle = COL.event; g.lineWidth = 2; g.beginPath(); g.moveTo(x, top - 2); g.lineTo(x, top + h + 2); g.stroke();
      g.fillStyle = COL.event; g.beginPath(); g.moveTo(x - 5, 4); g.lineTo(x + 5, 4); g.lineTo(x, 12); g.fill();
    }
    g.font = "bold 10px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
    this.path.keyframes.forEach((k, j) => {
      const x = X((k.t - t0) / span);
      g.fillStyle = COL.event; g.beginPath(); g.arc(x, top + h + 12, 7, 0, 2 * Math.PI); g.fill();
      g.fillStyle = "#111"; g.fillText(String(j + 1), x, top + h + 12.5);
    });
    const x = X(this.index / Math.max(1, n - 1));
    g.strokeStyle = text; g.lineWidth = 2; g.beginPath(); g.moveTo(x, top - 6); g.lineTo(x, top + h + 6); g.stroke();
    g.fillStyle = text; g.beginPath(); g.moveTo(x - 5, top - 9); g.lineTo(x + 5, top - 9); g.lineTo(x, top - 3); g.fill();
    g.fillStyle = faint; g.font = "11px system-ui"; g.textBaseline = "alphabetic";
    g.textAlign = "left"; g.fillText(`path t = ${m(t0)}`, 8, top + h + 34);
    g.textAlign = "right"; g.fillText(`path t = ${m(t1)}`, W - 8, top + h + 34);
  }

  extraAt(t, key) {
    const ks = this.path.keyframes.filter((k) => k.extras && key in k.extras);
    if (!ks.length) return null;
    if (t <= ks[0].t || ks.length === 1) return ks[0].extras[key];
    if (t >= ks[ks.length - 1].t) return ks[ks.length - 1].extras[key];
    for (let i = 0; i < ks.length - 1; i++) {
      const a = ks[i], b = ks[i + 1];
      if (a.t <= t && t <= b.t) return INTERP[key](a.extras[key], b.extras[key], b.t === a.t ? 0 : (t - a.t) / (b.t - a.t));
    }
    return null;
  }

  async frameParams(i) {
    if (this.frames[i] && this.frames[i].params) return this.frames[i].params;
    return await this.schedule("frame", "path_at", [this.path, this.tOf(i)]);
  }

  async goto(i) {
    if (this.path.keyframes.length < 2) return;
    this.index = Math.max(0, Math.min(this.n() - 1, i));
    this.drawTimeline(); this.updateFrame();
    const t = this.tOf(this.index), extras = {};
    for (const k of EXTRA) { const e = this.extraAt(t, k); if (e) extras[k] = e; }
    const p = await this.frameParams(this.index);
    if (!p) return;                                   // superseded by a later frame
    this.applyExtras(extras);
    this.setParams(p);
  }

  setCurrent() { if (this.S.tab === "sigma") this.onChange(); }

  play(on) {
    this.playing = on;
    this.$("pa-play").textContent = on ? "⏸ Pause" : "▶ Play";
    clearInterval(this.timer);
    if (!on) return;
    if (this.index >= this.n() - 1) this.goto(0);
    this.timer = setInterval(() => {
      if (this.index >= this.n() - 1) { if (this.$("pa-loop").checked) this.goto(0); else this.play(false); return; }
      this.goto(this.index + 1);
    }, 1000 / this.fps());
  }

  overlay() {
    const ks = this.path.keyframes;
    if (!ks.length) return null;
    const n = this.n();
    const kind = this.frames.length === n ? this.frames.map((f) => (f.global ? "ok" : f.kind)) : null;
    return { path: this.path, n, kind, event_t: kind ? this.events.map((e) => e.t) : [], index: this.index };
  }

  // ------------------------------------------------------------ video: the 3D view along the path
  async record() {
    if (this.path.keyframes.length < 2 || this.recording) return;
    const { view, S } = this;
    const n = this.n(), fps = this.fps();
    const src = view.renderer.domElement;
    const W = src.width - (src.width % 2), cap = Math.max(48, Math.round(0.095 * src.height)), H = src.height + cap - ((src.height + cap) % 2);
    const out = document.createElement("canvas"); out.width = W; out.height = H;
    const g = out.getContext("2d");
    const types = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"];
    const mime = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    if (!mime) { alert("This browser cannot record video (MediaRecorder is missing)."); return; }
    const stream = out.captureStream(0), track = stream.getVideoTracks()[0];
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
    const chunks = []; rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    this.recording = true; this.play(false);
    this.S.show3d();                                   // the 3D view must be live while recording
    const btn = this.$("pa-video"); btn.textContent = "Recording… (click to stop)";
    let stop = false; btn.disabled = false; btn.onclick = () => { stop = true; };
    rec.start();
    rec.pause();
    const frameMs = 1000 / fps;
    let last = null, held = 0;
    for (let i = 0; i < n && !stop; i++) {
      await this.goto(i);
      await this.waitIdle();
      const f = this.frames[i];
      const ok = !f || f.global;
      if (ok || !last) {
        view.renderNow();
        g.fillStyle = "#101114"; g.fillRect(0, 0, W, H);
        g.drawImage(src, 0, 0);
        if (ok) last = g.getImageData(0, 0, W, src.height);
      } else {
        g.putImageData(last, 0, 0); held++;
        g.fillStyle = "rgba(0,0,0,0.5)"; g.fillRect(0, 0, W, src.height);
      }
      this.caption(g, W, H, cap, i, f, !ok ? (last ? "HELD · last global frame" : "NOT GLOBAL") : "");
      rec.resume();                                   // the frame is on the timeline for exactly 1/fps
      track.requestFrame();
      await new Promise((r) => setTimeout(r, frameMs));
      rec.pause();
    }
    if (rec.state === "paused") rec.resume();
    const stopped = new Promise((r) => (rec.onstop = r));
    rec.stop();
    await stopped;
    const ext = mime.startsWith("video/mp4") ? "mp4" : "webm";
    const url = URL.createObjectURL(new Blob(chunks, { type: mime.split(";")[0] }));
    const a = document.createElement("a"); a.href = url; a.download = `swirl_path.${ext}`; a.click();
    this.recording = false; btn.textContent = "Record video…"; btn.onclick = () => this.record();
    this.$("sb-info").textContent = `recorded ${n} frames (${(n / fps).toFixed(1)} s) as ${ext.toUpperCase()}` + (held ? ` — ${held} not-global frames held at the last global frame` : "");
  }

  waitIdle() {
    return new Promise((res) => {
      const t0 = performance.now();
      const tick = () => (this.S.idle() || performance.now() - t0 > 20000 ? res() : setTimeout(tick, 20));
      setTimeout(tick, 20);
    });
  }

  caption(g, W, H, ch, i, f, banner) {
    const y0 = H - ch, n = this.n(), sh = Math.max(8, Math.round(ch * 0.17));
    g.fillStyle = "#18191d"; g.fillRect(0, y0, W, ch);
    const x0 = sh * 1.5, ws = W - 3 * sh;
    this.frames.forEach((fr, k) => { g.fillStyle = COL[this.kind(fr)]; g.fillRect(x0 + (k * ws) / n, y0 + 2, ws / n + 0.6, sh - 2); });
    const [t0, t1] = this.duration(), span = (t1 - t0) || 1;
    g.font = `bold ${Math.round(sh * 0.85)}px system-ui`; g.textAlign = "center"; g.textBaseline = "middle";
    this.path.keyframes.forEach((k, j) => {
      const x = x0 + ((k.t - t0) / span) * ws;
      g.fillStyle = COL.event; g.beginPath(); g.arc(x, y0 + sh / 2, sh * 0.6, 0, 2 * Math.PI); g.fill();
      g.fillStyle = "#111"; g.fillText(String(j + 1), x, y0 + sh / 2 + 0.5);
    });
    const x = x0 + ((i + 0.5) * ws) / n;
    g.fillStyle = "#fff"; g.beginPath(); g.moveTo(x - sh * 0.7, y0 - sh); g.lineTo(x + sh * 0.7, y0 - sh); g.lineTo(x, y0); g.fill();
    const body = ch - sh, yb = y0 + sh, pad = Math.round(body * 0.25);
    const p = (f && f.params) || this.getParams(), vary = this.varying();
    const moving = NAMES.filter((n) => vary.includes(n)).map((n) => `${SHORT[n]} = ${Number(p[n]).toPrecision(4)}`).join(",  ");
    g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillStyle = "#e2e4e9"; g.font = `bold ${Math.round(body * 0.3)}px system-ui`;
    g.fillText(`path t = ${Number(this.tOf(i)).toPrecision(4)}    frame ${i + 1} / ${n}`, pad, yb + body * 0.42);
    g.fillStyle = "#969aa3"; g.font = `${Math.round(body * 0.25)}px system-ui`;
    g.fillText((moving || "").replace(/-/g, "−"), pad, yb + body * 0.82);
    const ok = !f || f.global;
    const st = !f ? "" : ok ? `global · N ${f.north} μ₊=${f.mu_n?.toFixed(3)} · S ${f.south} μ₋=${f.mu_s?.toFixed(3)}`
      : `NOT GLOBAL · ${f.kind === "blowup" ? "blows up" : "a pole fails the terminal test"}`;
    if (st) {
      g.font = `bold ${Math.round(body * 0.3)}px system-ui`;
      const tw = g.measureText(st).width, bx1 = W - pad, bx0 = bx1 - tw - 2 * pad;
      g.fillStyle = ok ? "#3fa877" : "#d64045";
      g.beginPath(); g.roundRect(bx0, yb + body * 0.18, bx1 - bx0, body * 0.64, body * 0.15); g.fill();
      g.fillStyle = "#fff"; g.textBaseline = "middle"; g.fillText(st, bx0 + pad, yb + body * 0.5);
    }
    if (banner) {
      g.font = `bold ${Math.round(H * 0.022)}px system-ui`;
      const tw = g.measureText(banner).width, mm = Math.round(H * 0.012);
      g.fillStyle = "#d64045"; g.beginPath(); g.roundRect(W - tw - 3 * mm, mm, tw + 2 * mm, H * 0.035, mm * 0.6); g.fill();
      g.fillStyle = "#fff"; g.textBaseline = "middle"; g.textAlign = "left"; g.fillText(banner, W - tw - 2 * mm, mm + H * 0.0175);
    }
  }
}
