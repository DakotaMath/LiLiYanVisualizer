// The 3D view in WebGL (three.js). Draws the arrays from swirl_explorer.web.api.scene() the way the
// desktop draws its PyVista meshes (viz/scene_draw.py): same colours, sizes, labels and camera.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function b64(s, Type) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}

const PAL = {
  dark: { text: "#e8e6e3", outline: "#6f737b", cut: "#8fa3bf", origin: "#8a8c92", bg: "#16181c", bgTop: "#23262c" },
  light: { text: "#1d1c1a", outline: "#9a9ca3", cut: "#8fa3bf", origin: "#8a8c92", bg: "#ffffff", bgTop: "#ffffff" },
};
const CLIP = [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)];   // keeps y ≥ 0, as the desktop's clip
export const OBLIQUE = { position: [2.7, -2.9, 2.4], focal_point: [0, 0, 0], view_up: [0, 0, 1] };
export const POLE = { position: [0, 0, 6], focal_point: [0, 0, 0], view_up: [0, 1, 0] };

// ---------------------------------------------------------------------------- geometry builders
function tubes(lines, radius, sides = 10) {
  // lines: [{pts: Float32Array (k*3), cols: Uint8Array (k*3) | null}]; parallel-transport frames
  let nv = 0, ni = 0;
  for (const L of lines) { const k = L.pts.length / 3; if (k < 2) continue; nv += k * sides; ni += (k - 1) * sides * 6; }
  const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
  const idx = new Uint32Array(ni);
  let v = 0, f = 0;
  const T = new THREE.Vector3(), N = new THREE.Vector3(), B = new THREE.Vector3(), P = new THREE.Vector3();
  const prevT = new THREE.Vector3(), tmp = new THREE.Vector3();
  for (const L of lines) {
    const p = L.pts, k = p.length / 3;
    if (k < 2) continue;
    const base = v;
    for (let i = 0; i < k; i++) {
      const a = Math.max(i - 1, 0), b = Math.min(i + 1, k - 1);
      T.set(p[3 * b] - p[3 * a], p[3 * b + 1] - p[3 * a + 1], p[3 * b + 2] - p[3 * a + 2]);
      if (T.lengthSq() < 1e-20) T.copy(prevT.lengthSq() ? prevT : tmp.set(0, 0, 1));
      T.normalize();
      if (i === 0) {
        tmp.set(Math.abs(T.x) < 0.9 ? 1 : 0, Math.abs(T.x) < 0.9 ? 0 : 1, 0);
        N.crossVectors(T, tmp).normalize();
      } else {             // parallel transport of the normal
        N.sub(tmp.copy(T).multiplyScalar(N.dot(T)));
        if (N.lengthSq() < 1e-12) N.crossVectors(T, tmp.set(0, 0, 1));
        N.normalize();
      }
      B.crossVectors(T, N);
      prevT.copy(T);
      P.set(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      const r = L.cols ? L.cols[3 * i] / 255 : 1, g = L.cols ? L.cols[3 * i + 1] / 255 : 1, bl = L.cols ? L.cols[3 * i + 2] / 255 : 1;
      for (let s = 0; s < sides; s++) {
        const th = (2 * Math.PI * s) / sides, c = Math.cos(th), sn = Math.sin(th);
        const nx = c * N.x + sn * B.x, ny = c * N.y + sn * B.y, nz = c * N.z + sn * B.z;
        pos[3 * v] = P.x + radius * nx; pos[3 * v + 1] = P.y + radius * ny; pos[3 * v + 2] = P.z + radius * nz;
        nor[3 * v] = nx; nor[3 * v + 1] = ny; nor[3 * v + 2] = nz;
        col[3 * v] = r; col[3 * v + 1] = g; col[3 * v + 2] = bl;
        v++;
      }
    }
    for (let i = 0; i < k - 1; i++) {
      for (let s = 0; s < sides; s++) {
        const a = base + i * sides + s, b = base + i * sides + ((s + 1) % sides);
        const c = a + sides, d = b + sides;
        idx[f++] = a; idx[f++] = c; idx[f++] = b; idx[f++] = b; idx[f++] = c; idx[f++] = d;
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

function splitLines(payload, withColors = true) {
  if (!payload || !payload.n) return [];
  const pts = b64(payload.points, Float32Array);
  const cols = withColors && payload.colors ? b64(payload.colors, Uint8Array) : null;
  const off = payload.offsets, out = [];
  for (let i = 0; i < off.length - 1; i++) {
    out.push({ pts: pts.subarray(3 * off[i], 3 * off[i + 1]), cols: cols ? cols.subarray(3 * off[i], 3 * off[i + 1]) : null });
  }
  return out;
}

function polyline(pts, color, opacity = 1) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pts instanceof Float32Array ? pts : new Float32Array(pts), 3));
  return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
}

function coneAlong(from, dir, height, radius, color, segs = 24) {
  const g = new THREE.ConeGeometry(radius, height, segs);
  const m = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ color, shininess: 30 }));
  const d = new THREE.Vector3(...dir).normalize();
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
  m.position.set(...from);
  return m;
}

// ---------------------------------------------------------------------------- the view
export class View3D {
  constructor(canvas, labelsDiv, colorbarDiv) {
    this.canvas = canvas; this.labelsDiv = labelsDiv; this.colorbarDiv = colorbarDiv;
    this.theme = "dark";
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.localClippingEnabled = true;
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -100, 100);
    this.camera.up.set(0, 0, 1);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    this.light = new THREE.DirectionalLight(0xffffff, 1.6);   // a head light, as VTK's
    this.scene.add(this.light);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.zoomSpeed = 1.2;
    this.controls.addEventListener("change", () => this.render());
    this.groups = { scene: new THREE.Group(), slice: new THREE.Group() };
    this.scene.add(this.groups.scene, this.groups.slice);
    this.labels = [];
    this.halfHeight = 1.25;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.setCamera(OBLIQUE, 1.85);
  }

  resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this._frustum();
    this.render();
  }

  _frustum() {
    const el = this.canvas.parentElement;
    const a = (el.clientWidth || 1) / (el.clientHeight || 1);
    const H = this.halfHeight;
    Object.assign(this.camera, { left: -H * a, right: H * a, top: H, bottom: -H });
    this.camera.updateProjectionMatrix();
  }

  // camera in the desktop's terms: position / focal point / up / parallel scale (half height)
  setCamera(c, zoom = null) {
    const f = new THREE.Vector3(...(c.focal_point || [0, 0, 0]));
    const p = new THREE.Vector3(...c.position);
    // the desktop resets to the scene bounds and zooms: half height ≈ 1.5 / zoom for the ball
    this.camera.position.copy(p.clone().sub(f).normalize().multiplyScalar(8).add(f));
    this.camera.up.set(...c.view_up);
    this.controls.target.copy(f);
    this.halfHeight = c.parallel_scale ? c.parallel_scale : 2.3 / (zoom || 1);
    this.camera.zoom = 1;
    this._frustum();
    this.controls.update();
    this.render();
  }

  getCamera() {
    const t = this.controls.target;
    return { position: this.camera.position.toArray(), focal_point: t.toArray(), view_up: this.camera.up.toArray(),
             parallel_scale: this.halfHeight / this.camera.zoom };
  }

  setTheme(theme) { this.theme = theme; if (this.last) this.setScene(this.last); else this.render(); }

  clear(group) {
    group.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); } });
    group.clear();
  }

  setScene(d) {
    this.last = d;
    const P = PAL[this.theme], g = this.groups.scene;
    this.clear(g);
    this.labels = this.labels.filter((l) => l.group === "slice");
    const clip = d.clip ? CLIP : null;
    // the ball outline (equator and meridian)
    for (const L of splitLines(d.outline, false)) g.add(polyline(L.pts, P.outline));
    // axis rods coloured by pole type, with the flow cone and the label
    for (const a of d.axis || []) {
      const len = Math.abs(a.z1 - a.z0);
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(a.radius, a.radius, len, 20),
                                 new THREE.MeshPhongMaterial({ color: a.color, shininess: 40 }));
      rod.rotation.x = Math.PI / 2;
      rod.position.set(0, 0, (a.z0 + a.z1) / 2);
      g.add(rod);
      if (a.flow) g.add(coneAlong([0, 0, a.sign * 0.5], [0, 0, a.flow], 0.26, 0.08, a.color));
      this.labels.push({ pos: [0, 0, a.sign * 1.1], text: a.label, bg: a.color, group: "scene" });
    }
    if (d.clip) {      // the cut plane, faintly
      const disc = new THREE.Mesh(new THREE.CircleGeometry(1.0, 72),
        new THREE.MeshBasicMaterial({ color: P.cut, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
      disc.rotation.x = Math.PI / 2;
      g.add(disc);
    }
    // streamlines: coloured tubes, or thin grey lines while overlays are shown
    const sl = splitLines(d.streamlines);
    if (sl.length) {
      if (d.overlays) {
        for (const L of sl) { const ln = polyline(L.pts, P.outline, 0.5); ln.material.clippingPlanes = clip; g.add(ln); }
      } else {
        const mesh = new THREE.Mesh(tubes(sl, 0.007, 10),
          new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 30, clippingPlanes: clip }));
        g.add(mesh);
      }
    }
    // cones downstream: instanced, coloured like the line at that point
    if (d.arrows && d.arrows.n) {
      const pts = b64(d.arrows.points, Float32Array), dirs = b64(d.arrows.dirs, Float32Array);
      const cols = b64(d.arrows.colors, Uint8Array);
      const cone = new THREE.ConeGeometry(0.28 * 0.12, 0.12, 12);
      const inst = new THREE.InstancedMesh(cone, new THREE.MeshPhongMaterial({ clippingPlanes: clip, shininess: 30 }), d.arrows.n);
      const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0), v = new THREE.Vector3();
      const one = new THREE.Vector3(1, 1, 1), c = new THREE.Color();
      for (let i = 0; i < d.arrows.n; i++) {
        v.set(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
        q.setFromUnitVectors(up, v);
        m.compose(new THREE.Vector3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]), q, one);
        inst.setMatrixAt(i, m);
        inst.setColorAt(i, c.setRGB(cols[3 * i] / 255, cols[3 * i + 1] / 255, cols[3 * i + 2] / 255, THREE.SRGBColorSpace));
      }
      g.add(inst);
    }
    if (d.surface) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(b64(d.surface.points, Float32Array), 3));
      const cols = b64(d.surface.colors, Uint8Array), cf = new Float32Array(cols.length);
      for (let i = 0; i < cols.length; i++) cf[i] = cols[i] / 255;
      geo.setAttribute("color", new THREE.BufferAttribute(cf, 3));
      geo.setIndex(new THREE.BufferAttribute(b64(d.surface.triangles, Uint32Array), 1));
      geo.computeVertexNormals();
      g.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ vertexColors: true, transparent: true, opacity: d.surface.opacity,
        side: THREE.DoubleSide, depthWrite: false, clippingPlanes: clip })));
    }
    if (d.vortex && d.vortex.n) {
      g.add(new THREE.Mesh(tubes(splitLines(d.vortex, false), 0.006, 8),
        new THREE.MeshPhongMaterial({ color: "#2e9d3a", clippingPlanes: clip })));
    }
    if (d.selfsim && d.selfsim.n) {
      const ss = splitLines(d.selfsim, false), radii = [0.016, 0.006, 0.008, 0.011], fr = [0.85, 0.15, 0.4, 0.65];
      const names = ["X(s)", "0.6·X", "0.8·X", "1.25·X"];
      ss.slice(0, 4).forEach((L, i) => {
        g.add(new THREE.Mesh(tubes([L], radii[i], 10), new THREE.MeshPhongMaterial({ color: P.text })));
        const k = Math.floor(fr[i] * (L.pts.length / 3 - 1));
        this.labels.push({ pos: [L.pts[3 * k], L.pts[3 * k + 1], L.pts[3 * k + 2]], text: names[i], plain: true, group: "scene" });
      });
    }
    // origin ball and the swirl arcs
    const ball = new THREE.Mesh(new THREE.SphereGeometry(d.origin_r || 0.12, 24, 16),
                                new THREE.MeshPhongMaterial({ color: "#8a8c92", emissive: "#2a2a2c" }));
    g.add(ball);
    this.labels.push({ pos: [-0.75, 0, -0.3], text: "origin (cut out)", plain: true, group: "scene" });
    for (const a of d.arcs || []) {
      const pts = b64(a.points, Float32Array), k = pts.length / 3;
      g.add(new THREE.Mesh(tubes([{ pts, cols: null }], 0.012, 10), new THREE.MeshPhongMaterial({ color: "#e65100" })));
      const tip = [pts[3 * k - 3], pts[3 * k - 2], pts[3 * k - 1]], prev = [pts[3 * k - 12], pts[3 * k - 11], pts[3 * k - 10]];
      const dir = [tip[0] - prev[0], tip[1] - prev[1], tip[2] - prev[2]];
      const n = Math.hypot(...dir);
      g.add(coneAlong([tip[0] + 0.04 * dir[0] / n, tip[1] + 0.04 * dir[1] / n, tip[2] + 0.04 * dir[2] / n], dir, 0.11, 0.045, "#e65100", 20));
    }
    this.drawColorbar(d);
    this.render();
  }

  drawColorbar(d) {
    const cb = d.colorbar, el = this.colorbarDiv;
    const coloured = (d.streamlines && d.streamlines.n && !d.overlays) || d.surface;
    if (!cb || !coloured) { el.innerHTML = ""; return; }
    const [lo, hi] = cb.clim;
    const fmt = (v) => (Math.max(Math.abs(lo), Math.abs(hi)) >= 100 ? v.toFixed(0) : Number(v.toPrecision(3)).toString()).replace("-", "−");
    const ticks = [];
    for (let i = 0; i < 5; i++) {
      const t = i / 4;
      const v = cb.log ? lo * Math.pow(hi / lo, t) : lo + (hi - lo) * t;
      ticks.push(`<span style="left:${t * 100}%">${fmt(v)}</span>`);
    }
    el.innerHTML = `<div class="title">${cb.title}</div><div class="ticks">${ticks.join("")}</div>` +
      `<div class="ramp" style="background:linear-gradient(90deg,${cb.stops.join(",")})"></div>`;
  }

  // the slice plane in 3D: unlit texture with the 2D colours, its outline, normal and in-plane lines
  setSlice(s) {
    const g = this.groups.slice;
    this.clear(g);
    this.labels = this.labels.filter((l) => l.group !== "slice");
    if (!s) { this.render(); return; }
    const n = s.tex_n, tex = new THREE.DataTexture(b64(s.texture, Uint8Array), n, n, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    const e1 = new THREE.Vector3(...s.e1), e2 = new THREE.Vector3(...s.e2), nn = new THREE.Vector3(...s.normal);
    const c = new THREE.Vector3(...s.center);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2 * s.half, 2 * s.half),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    plane.matrixAutoUpdate = false;
    plane.matrix.makeBasis(e1, e2, nn).setPosition(c);
    g.add(plane);
    const d = s.center[0] * s.normal[0] + s.center[1] * s.normal[1] + s.center[2] * s.normal[2];
    if (Math.abs(d) < 1) {
      const rr = Math.sqrt(1 - d * d), pts = [];
      for (let i = 0; i <= 144; i++) {
        const th = (2 * Math.PI * i) / 144;
        const p = c.clone().addScaledVector(e1, rr * Math.cos(th)).addScaledVector(e2, rr * Math.sin(th));
        pts.push(p.x, p.y, p.z);
      }
      g.add(polyline(pts, "#f2c14e"));
    }
    const lift = nn.clone().multiplyScalar(0.006);
    for (const L of splitLines(s.lines, false)) {     // in-plane (s, t, 0) points, mapped onto the plane
      const k = L.pts.length / 3, pts = [];
      for (let i = 0; i < k; i++) {
        const p = c.clone().addScaledVector(e1, L.pts[3 * i]).addScaledVector(e2, L.pts[3 * i + 1]).add(lift);
        pts.push(p.x, p.y, p.z);
      }
      g.add(polyline(pts, "#111111"));
    }
    g.add(coneAlong(c.clone().addScaledVector(nn, 0.3).toArray(), s.normal, 0.1, 0.08 * 0.35, "#f2c14e"));
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.025 * 0.35, 0.025 * 0.35, 0.27, 12),
                                 new THREE.MeshPhongMaterial({ color: "#f2c14e" }));
    shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), nn);
    shaft.position.copy(c.clone().addScaledVector(nn, 0.135));
    g.add(shaft);
    this.render();
  }

  viewAlong(center, normal, e2) {
    const c = new THREE.Vector3(...center), n = new THREE.Vector3(...normal);
    this.setCamera({ position: c.clone().addScaledVector(n, 5).toArray(), focal_point: c.toArray(), view_up: e2 }, 1.35);
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = null;
      this.light.position.copy(this.camera.position);
      this.light.target.position.copy(this.controls.target);
      this.light.target.updateMatrixWorld();
      this.renderer.render(this.scene, this.camera);
      this.placeLabels();
    });
  }

  renderNow() {
    this.light.position.copy(this.camera.position);
    this.renderer.render(this.scene, this.camera);
    this.placeLabels();
  }

  placeLabels() {
    const el = this.labelsDiv, w = el.clientWidth, h = el.clientHeight;
    el.innerHTML = "";
    const v = new THREE.Vector3();
    for (const L of this.labels) {
      v.set(...L.pos).project(this.camera);
      if (Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1) continue;
      const d = document.createElement("div");
      d.className = "lab3d" + (L.plain ? " plain" : "");
      if (L.bg) d.style.background = L.bg;
      d.style.left = `${((v.x + 1) / 2) * w}px`;
      d.style.top = `${((1 - v.y) / 2) * h}px`;
      d.textContent = L.text;
      el.appendChild(d);
    }
  }
}
