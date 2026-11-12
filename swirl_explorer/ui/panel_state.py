"""Reading and restoring the state of the panels (sessions, path keyframes) and grabbing frames.

Kept out of the panels so they stay focused on drawing.  Every ``apply_*`` sets the
widgets with their signals blocked and then triggers one recompute, so loading a
session does not fire a cascade of intermediate solves.
"""
from __future__ import annotations

import io

import numpy as np

from ..core.paths import EXTRA_KEYS
from ..viz.scene_compute import SceneSettings
from ..viz.slices import PlaneSpec

SEED_KEYS = ("seed_mode", "n_seeds", "sphere_center", "sphere_radius", "line_p0", "line_p1")


def _set_combo(cb, data) -> None:
    i = cb.findData(data)
    if i >= 0:
        cb.blockSignals(True)
        cb.setCurrentIndex(i)
        cb.blockSignals(False)


def _set(w, value) -> None:
    w.blockSignals(True)
    if hasattr(w, "setChecked") and isinstance(value, bool):
        w.setChecked(value)
    else:
        w.setValue(value)
    w.blockSignals(False)


# ----------------------------------------------------------------------------- 3D view
def camera_state(view3d) -> dict | None:
    pl = view3d.plotter
    if pl is None:
        return None
    cam = pl.camera
    return {"position": tuple(map(float, cam.position)), "focal_point": tuple(map(float, cam.focal_point)),
            "view_up": tuple(map(float, cam.up)), "parallel_scale": float(cam.parallel_scale)}


def apply_camera(view3d, d: dict | None, render: bool = True) -> None:
    pl = view3d.plotter
    if pl is None or not d:
        return
    cam = pl.camera
    cam.position = tuple(d["position"])
    cam.focal_point = tuple(d["focal_point"])
    cam.up = tuple(d["view_up"])
    if d.get("parallel_scale"):
        cam.parallel_scale = float(d["parallel_scale"])
    pl.renderer.ResetCameraClippingRange()
    view3d._camera_set = True
    if render:
        pl.render()


def seeds_state(view3d) -> dict:
    s = view3d.settings
    return {k: getattr(s, k) for k in SEED_KEYS}


def apply_seeds(view3d, d: dict | None) -> None:
    if not d:
        return
    s = view3d.settings
    for k in SEED_KEYS:
        if k in d:
            setattr(s, k, tuple(d[k]) if isinstance(d[k], (list, tuple)) else d[k])
    _set_combo(view3d.seed_mode, s.seed_mode)
    _set(view3d.n_seeds, int(s.n_seeds))
    view3d._widgets_mode = None          # move the handle to the new seeds
    view3d._sync_widgets()
    view3d._debounce.start()


def scene_state(view3d) -> dict:
    view3d._read_controls()
    return {**view3d.settings.to_dict(), "show_handle": view3d.show_handle.isChecked(),
            "show_slice": view3d.chk_slice.isChecked()}


def apply_scene(view3d, d: dict | None) -> None:
    if not d:
        return
    s = SceneSettings.from_dict(d)
    view3d.settings = s
    _set_combo(view3d.seed_mode, s.seed_mode)
    _set(view3d.n_seeds, int(s.n_seeds))
    _set_combo(view3d.color_by, s.color_by)
    for k, c in view3d.chk.items():
        _set(c, bool(getattr(s, k)))
    _set_combo(view3d.iso_field, s.iso_field)
    _set(view3d.iso_level, int(round(s.iso_quantile * 100)))
    _set_combo(view3d.shape, s.domain_shape)
    _set(view3d.r_min, float(s.r_min))
    _set(view3d.axis_a, float(s.axis_a))
    _set(view3d.opacity, int(round(s.opacity * 100)))
    if "show_handle" in d:
        _set(view3d.show_handle, bool(d["show_handle"]))
    view3d._changed()                    # re-reads the widgets, moves the handle, recomputes
    if "show_slice" in d:
        view3d.chk_slice.setChecked(bool(d["show_slice"]))


# ----------------------------------------------------------------------------- slice
def plane_state(slice_ctrl) -> dict:
    return slice_ctrl.spec.to_dict()


def apply_plane(slice_panel, d: dict | None) -> None:
    if not d:
        return
    spec = PlaneSpec.from_dict(d)
    slice_panel._load_spec(spec)
    slice_panel.ctrl.set_spec(spec, source=slice_panel)


SLICE_CHECKS = ("chk_un", "chk_lines", "chk_lic", "chk_rho", "chk_log", "follow")


def slice_options(slice_panel) -> dict:
    d = {k: getattr(slice_panel, k).isChecked() for k in SLICE_CHECKS}
    d["density"] = slice_panel.density.value()
    return d


def apply_slice_options(slice_panel, d: dict | None) -> None:
    if not d:
        return
    for k in SLICE_CHECKS:
        if k in d:
            _set(getattr(slice_panel, k), bool(d[k]))
    if "density" in d:
        _set(slice_panel.density, float(d["density"]))
    slice_panel._display_changed()
    slice_panel._options_changed()


# ----------------------------------------------------------------------------- plots
def plots_state(plots) -> dict:
    return {"xvar": plots.xvar.currentData(), "mode": plots.mode.currentData(), "pole": plots.pole.currentData(),
            "quantities": plots.quantities(), "overlay": plots.overlay.isChecked()}


def apply_plots(plots, d: dict | None) -> None:
    if not d:
        return
    for cb, k in ((plots.xvar, "xvar"), (plots.mode, "mode"), (plots.pole, "pole")):
        if k in d:
            _set_combo(cb, d[k])
    if "quantities" in d:
        for k, c in plots.checks.items():
            _set(c, k in d["quantities"])
    if "overlay" in d:
        _set(plots.overlay, bool(d["overlay"]))
    plots._mode_changed()
    plots.redraw()


# ----------------------------------------------------------------------------- Σ-map
def sigma_state(sigma) -> dict:
    return {"x_name": sigma.x_name.currentData(), "y_name": sigma.y_name.currentData(),
            "x_range": [sigma.x_lo.value(), sigma.x_hi.value()],
            "y_range": [sigma.y_lo.value(), sigma.y_hi.value()],
            "grid": sigma.res_spin.value(), "mode": sigma.mode.currentData(),
            "edge": sigma.chk_edge.isChecked(), "mu": sigma.chk_mu.isChecked()}


def apply_sigma(sigma, d: dict | None) -> None:
    if not d:
        return
    if "x_name" in d and "y_name" in d:
        sigma._set_ranges(d["x_name"], d["y_name"], tuple(d.get("x_range", (-1, 1))),
                          tuple(d.get("y_range", (-1, 1))))
    if "grid" in d:
        _set(sigma.res_spin, int(d["grid"]))
    if "mode" in d:
        _set_combo(sigma.mode, d["mode"])
    for w, k in ((sigma.chk_edge, "edge"), (sigma.chk_mu, "mu")):
        if k in d:
            _set(w, bool(d[k]))
    sigma._update_fixed()
    sigma.res = None                       # computed again on the next visit (or now if visible)
    sigma.redraw()


# ----------------------------------------------------------------------------- keyframe extras
def capture_extras(win, keys) -> dict:
    out = {}
    if "camera" in keys:
        c = camera_state(win.view3d)
        if c:
            out["camera"] = c
    if "plane" in keys:
        out["plane"] = plane_state(win.slice_ctrl)
    if "seeds" in keys:
        out["seeds"] = seeds_state(win.view3d)
    return out


def apply_extras(win, extras: dict) -> None:
    if not extras:
        return
    if "seeds" in extras:
        apply_seeds(win.view3d, extras["seeds"])
    if "plane" in extras:
        apply_plane(win.slice, extras["plane"])
    if "camera" in extras:
        apply_camera(win.view3d, extras["camera"])


assert set(EXTRA_KEYS) == {"camera", "plane", "seeds"}


# ----------------------------------------------------------------------------- frame grabbing
def fig_to_array(fig, w: int, h: int) -> np.ndarray:
    """Render a matplotlib figure at exactly w × h pixels (RGB), then put its size back."""
    from PIL import Image
    old = fig.get_size_inches().copy()
    dpi = max(72.0, min(w, h) / 6.0)
    fig.set_size_inches(w / dpi, h / dpi, forward=False)
    buf = io.BytesIO()
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)      # tiny cells: constrained layout gives up
            fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    finally:
        fig.set_size_inches(*old, forward=False)
        fig.canvas.draw_idle()
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if im.size != (w, h):
        im = im.resize((w, h), Image.LANCZOS)
    return np.asarray(im)


def _grab_widget(widget) -> np.ndarray | None:
    try:
        qi = widget.grab().toImage().convertToFormat(widget.grab().toImage().Format.Format_RGB888)
        b = qi.constBits()
        arr = np.frombuffer(b, np.uint8, qi.sizeInBytes()).reshape(qi.height(), qi.bytesPerLine())
        return arr[:, :qi.width() * 3].reshape(qi.height(), qi.width(), 3).copy()
    except Exception:
        return None


def _placeholder(w: int, h: int, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (22, 24, 28))
    ImageDraw.Draw(im).text((w / 2, h / 2), text, fill=(150, 154, 163), anchor="mm")
    return np.asarray(im)


def grab_3d(view3d, w: int, h: int) -> np.ndarray | None:
    """Screenshot of the 3D view, rendered at up to 3× and scaled to fit w × h (no cropping)."""
    pl = view3d.plotter
    if pl is None:
        return None
    from PIL import Image
    nw, nh = pl.window_size
    scale = int(min(3, max(1, np.ceil(max(w / max(nw, 1), h / max(nh, 1))))))
    try:
        img = np.asarray(pl.screenshot(return_img=True, scale=scale))[..., :3]
    except Exception:                         # no current GL context (hidden / offscreen): grab the widget
        img = _grab_widget(pl.interactor)
    if img is None or img.size == 0 or img.std() == 0:
        img = _placeholder(w, h, "3D view not available (no OpenGL window)")
    im = Image.fromarray(np.ascontiguousarray(img))
    s = min(w / im.width, h / im.height)
    return np.asarray(im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS))
