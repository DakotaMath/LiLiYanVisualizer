"""Milestone 6: paths (keyframes, extras, events), sessions and MP4 export."""
import json
import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from swirl_explorer.core.params import Params
from swirl_explorer.core.paths import ParamPath, scan_path, interp_camera, interp_plane, frame_status
from swirl_explorer.io.session import Session, save_session, load_session, SessionError
from swirl_explorer.io.movie import compose, Caption, MovieWriter, cells, LAYOUTS


# ----------------------------------------------------------------------------- core
def lens_path():
    p = ParamPath()
    p.add(0.0, Params(gamma=-0.5, kappa=1.0))
    p.add(1.0, Params(gamma=-2.5, kappa=1.0))
    return p


def test_event_is_bisected_to_the_lens_edge():
    from swirl_explorer.core.edges import gamma_minus
    frames, events = scan_path(lens_path(), 21)
    leave = [e for e in events if e.kind == "leaves_global"]
    assert len(leave) == 1
    gm = gamma_minus(Params(kappa=1.0))                     # -1.7806
    t_edge = (-0.5 - gm) / 2.0
    assert leave[0].t == pytest.approx(t_edge, abs=2e-4)   # far finer than the frame spacing 0.05
    assert frames[leave[0].index - 1].is_global and not frames[leave[0].index].is_global
    assert "blows up" in leave[0].detail


def test_reentry_and_spline():
    p = ParamPath(interpolation="spline")
    p.add(0, Params(gamma=-1.0, kappa=1.0))
    p.add(1, Params(gamma=-2.6, kappa=1.0))
    p.add(2, Params(gamma=-1.0, kappa=1.0))
    frames, events = scan_path(p, 41)
    kinds = [e.kind for e in events if e.kind != "failure_change"]
    assert kinds == ["leaves_global", "enters_global"]
    # the natural spline passes through the keyframes
    assert p.at(1.0)[0].gamma == pytest.approx(-2.6)
    assert p.varying() == ["gamma"]


def test_camera_turns_on_a_great_circle():
    a = {"position": (4, 0, 0), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "parallel_scale": 1.0}
    b = {"position": (0, 2, 0), "focal_point": (0, 0, 0), "view_up": (0, 0, 1), "parallel_scale": 3.0}
    m = interp_camera(a, b, 0.5)
    pos = np.array(m["position"])
    assert np.linalg.norm(pos) == pytest.approx(3.0)        # distance linear, never through the focal point
    assert pos[0] == pytest.approx(pos[1])                  # halfway round
    assert m["parallel_scale"] == pytest.approx(2.0)
    assert np.dot(m["view_up"], pos) == pytest.approx(0.0, abs=1e-12)


def test_plane_azimuth_takes_the_short_way():
    m = interp_plane({"azimuth": 170.0, "elevation": 0.0, "offset": 0.0},
                     {"azimuth": -170.0, "elevation": 20.0, "offset": 0.4}, 0.5)
    assert abs(m["azimuth"]) == pytest.approx(180.0)
    assert m["elevation"] == pytest.approx(10.0) and m["offset"] == pytest.approx(0.2)


def test_extras_keyed_on_some_keyframes_only():
    p = lens_path()
    p.keyframes[0].extras["plane"] = {"azimuth": 0.0, "elevation": 0.0, "offset": 0.0}
    p.keyframes[1].extras["plane"] = {"azimuth": 90.0, "elevation": 0.0, "offset": 0.0}
    assert p.keyed("plane") and not p.keyed("camera")
    assert p.extra_at(0.5, "plane")["azimuth"] == pytest.approx(45.0)
    assert p.extra_at(0.5, "camera") is None


# ----------------------------------------------------------------------------- sessions
def test_session_roundtrip(tmp_path):
    p = lens_path()
    s = Session(Params(gamma=-1.6, kappa=1.0, A=0.3), {"scene": {"sphere_center": (0.1, 0.2, 0.3)},
                                                       "camera": {"x": float("nan")}, "future": {"k": 1}}, p)
    f = save_session(s, tmp_path / "s.json")
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["format"] == "swirl-explorer-session" and d["camera"]["x"] is None
    s2 = load_session(f)
    assert s2.params == s.params and s2.sections["future"] == {"k": 1}
    assert s2.path.at(0.25)[0].gamma == pytest.approx(-1.0)


def test_session_rejects_garbage(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{\"hello\": 1}", encoding="utf-8")
    with pytest.raises(SessionError):
        load_session(f)
    f.write_text("not json", encoding="utf-8")
    with pytest.raises(SessionError):
        load_session(f)


# ----------------------------------------------------------------------------- movie (no GUI)
def test_layout_cells_tile_the_frame():
    for k in LAYOUTS:
        cl, cap = cells(k, 1920, 1080)
        area = sum(w * h for _, (x, y, w, h) in cl)
        assert area == pytest.approx(1920 * (1080 - cap), rel=0.01)


def test_writer_makes_a_playable_mp4(tmp_path):
    import imageio.v2 as imageio
    imgs = {"3d": np.full((300, 400, 3), 90, np.uint8)}
    path = tmp_path / "m.mp4"
    with MovieWriter(path, 12, (321, 181)) as w:            # odd sizes are made even
        for i in range(5):
            w.append(compose(imgs, "3d", *w.size, Caption("t", "d", "global", i < 3, "" if i < 3 else "x")))
    r = imageio.get_reader(str(path))
    n = sum(1 for _ in r)
    assert n == 5 and r.get_meta_data()["size"] == (320, 180)


# ----------------------------------------------------------------------------- UI
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=20.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture(scope="module")
def win(app):
    from swirl_explorer.ui.main_window import MainWindow
    w = MainWindow()
    w.quiet = True
    w.resize(1400, 900)
    w.show()
    assert _wait(app, lambda: w.last is not None)
    yield w
    w.close()


def test_path_dock_scans_and_scrubs(app, win):
    pp = win.path_panel
    assert not win.dock_path.isVisible()
    win.path_btn.click()
    assert win.dock_path.isVisible()
    win.params.lock.setChecked(False)
    pp.frames_spin.setValue(21)
    pp.set_path(None)
    win.params.set_params(Params(gamma=-0.5, kappa=1.0))
    pp.add_current()
    pp.add_keyframe(Params(gamma=-2.5, kappa=1.0))
    assert len(pp.path) == 2 and pp.table.rowCount() == 2
    assert _wait(app, lambda: len(pp.frames) == 21)
    assert any(e.kind == "leaves_global" for e in pp.events)
    assert "not global" in pp.events_lab.text()
    pp.goto(20)
    assert _wait(app, lambda: win.last is not None and win.last[0].gamma == pytest.approx(-2.5))
    assert "NOT GLOBAL" in pp.frame_lab.text() and "blows up" in pp.frame_lab.text()
    # typing into the table moves the keyframe
    pp.table.item(1, 4).setText("-1.0")
    assert pp.path.keyframes[1].params.gamma == pytest.approx(-1.0)
    assert _wait(app, lambda: len(pp.frames) == 21 and all(f.is_global for f in pp.frames))
    # the Σ-map gets the path overlay; shift-click adds a keyframe
    ov = pp.overlay()
    assert len(ov["samples"]) == 21 and len(ov["keys"]) == 2
    win.sigma.keyframeRequested.emit(Params(gamma=-1.2, kappa=2.0))
    assert len(pp.path) == 3 and pp.path.keyframes[-1].t == pytest.approx(2.0)


def test_path_keyframes_the_plane(app, win):
    pp = win.path_panel
    pp.set_path(None)
    pp.keep["plane"].setChecked(True)
    from swirl_explorer.ui import panel_state as ps
    ps.apply_plane(win.slice, {"azimuth": 0.0, "elevation": 0.0, "offset": 0.0})
    pp.add_keyframe(Params(gamma=-1.0, kappa=1.0))
    ps.apply_plane(win.slice, {"azimuth": 90.0, "elevation": 0.0, "offset": 0.0})
    pp.add_keyframe(Params(gamma=-1.2, kappa=1.0))
    pp.frames_spin.setValue(11)
    pp.goto(5)
    assert _wait(app, lambda: win.slice_ctrl.spec.azimuth == pytest.approx(45.0))
    assert win.slice.az_spin.value() == pytest.approx(45.0)


def test_session_through_the_window(app, win, tmp_path):
    win.params.lock.setChecked(False)
    win.params.set_params(Params(gamma=-1.3, kappa=1.5, A=0.2))
    win.view3d.chk["show_glyphs"].setChecked(True)
    win.view3d.n_seeds.setValue(33)
    win.plots.checks["z"].setChecked(True)
    win.sigma.res_spin.setValue(41)
    f = win.save_session(path=str(tmp_path / "s.json"))
    assert f and os.path.exists(f)
    # change everything, then load
    win.params.set_params(Params())
    win.view3d.chk["show_glyphs"].setChecked(False)
    win.view3d.n_seeds.setValue(12)
    win.plots.checks["z"].setChecked(False)
    win.sigma.res_spin.setValue(101)
    win.path_panel.set_path(None)
    assert win.open_session(f)
    assert win.params.params().kappa == pytest.approx(1.5) and win.params.params().A == pytest.approx(0.2)
    assert win.view3d.chk["show_glyphs"].isChecked() and win.view3d.settings.n_seeds == 33
    assert win.plots.checks["z"].isChecked() and win.sigma.res_spin.value() == 41
    assert len(win.path_panel.path) == 2
    assert _wait(app, lambda: win.last is not None and win.last[0].kappa == pytest.approx(1.5))


def test_movie_export_end_to_end(app, win, tmp_path):
    import imageio.v2 as imageio
    from swirl_explorer.ui.movie_export import MovieExporter
    pp = win.path_panel
    pp.keep["plane"].setChecked(False)
    pp.set_path(None)
    pp.add_keyframe(Params(gamma=-1.0, kappa=1.0), extras={})
    pp.add_keyframe(Params(gamma=-2.4, kappa=1.0), extras={})   # leaves the lens near the end
    out = tmp_path / "path.mp4"
    msgs = []
    ex = MovieExporter(win, {"layout": "all", "size": (640, 360), "fps": 8, "frames": 6, "policy": "hold",
                             "path": str(out)})
    ex.finished.connect(msgs.append)
    ex.start()
    assert _wait(app, lambda: bool(msgs), timeout=120)
    assert "Wrote 6 frames" in msgs[0] and "held" in msgs[0]
    r = imageio.get_reader(str(out))
    frames = [f for f in r]
    assert len(frames) == 6 and frames[0].shape == (360, 640, 3)
    assert frames[0].std() > 10                              # not a blank picture
    assert not win.exporting
    # skip policy: fewer frames
    msgs.clear()
    ex = MovieExporter(win, {"layout": "3d", "size": (320, 240), "fps": 8, "frames": 6, "policy": "skip",
                             "path": str(tmp_path / "skip.mp4")})
    ex.finished.connect(msgs.append)
    ex.start()
    assert _wait(app, lambda: bool(msgs), timeout=120)
    n = sum(1 for _ in imageio.get_reader(str(tmp_path / "skip.mp4")))
    assert 0 < n < 6 and "left out" in msgs[0]
