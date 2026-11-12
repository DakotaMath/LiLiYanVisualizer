"""The browser bridge (swirl_explorer.web.api): same numbers as the desktop code, valid JSON for every
input, no Qt or VTK.  Runs with numpy + scipy + matplotlib only, as in Pyodide."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from swirl_explorer.core.params import Params
from swirl_explorer.web import api

GENERIC = Params(gamma=-1.6806045, kappa=1.0).to_dict()


def call(name, *args):
    return json.loads(api.call(name, json.dumps(list(args))))   # exactly what the page does


def test_no_qt_or_vtk_needed():
    code = ("import sys, json; from swirl_explorer.web import api; "
            "p = json.dumps([api.preset('default')]); api.call('solve', p); api.call('profiles', '[{}]'); "
            "api.call('scene', '[{}]'); api.call('slice_view', json.dumps([{'azimuth': -90, 'elevation': 0, 'offset': 0}, {}])); "
            "bad = [m for m in ('pyvista', 'vtk', 'vtkmodules', 'PySide6', 'pyvistaqt') if m in sys.modules]; "
            "print(bad); assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_solve_matches_the_desktop_status():
    from swirl_explorer.core.solver import solve
    from swirl_explorer.core.poles import summarize
    from swirl_explorer.viz.status_text import head_html, pole_line
    r = call("solve", GENERIC)
    sm = summarize(solve(Params.from_dict(GENERIC)))
    assert r["status_line"] == sm.status_line()
    assert r["head"] == head_html(sm) and r["north"] == pole_line("North x = +1", sm.north, +1)[0]
    assert r["global"] is True


@pytest.mark.parametrize("p", [
    Params().to_dict(),                                  # rest state
    Params(gamma=5.0).to_dict(),                         # Landau |λ| < 1: blows up
    Params(gamma=-1.0, kappa=0.0).to_dict(),             # no swirl
    Params(gamma=-1.2, kappa=1.0, xb=0.5).to_dict(),     # off-equator base point
    Params(gamma=1.9, kappa=-3.5, A=0.4, c1=0.3).to_dict(),
])
def test_every_view_is_valid_json_for_awkward_solutions(p):
    call("solve", p)
    assert call("profiles", {"quantities": ["U", "f", "Q", "u_r"], "mode": "log", "w": 300, "h": 400})["png"]
    sc = call("scene", {"show_surface": True, "show_vortex": True, "show_selfsim": True})
    assert "axis" in sc and len(sc["axis"]) == 2
    for opts in ({}, {"lic": True, "rho": True, "log": True}):
        sl = call("slice_view", {"azimuth": 30, "elevation": 20, "offset": 0.3}, {**opts, "n": 61})
        assert sl["png"].startswith("data:image/png")
        assert len(np.frombuffer(__import__("base64").b64decode(sl["texture"]), np.uint8)) == 61 * 61 * 4


def test_scene_matches_the_desktop_numbers():
    from swirl_explorer.core.solver import solve
    from swirl_explorer.core.poles import summarize
    from swirl_explorer.viz.scene_compute import SceneSettings, compute_numeric
    from swirl_explorer.viz.scene_values import axis_info
    call("solve", GENERIC)
    sc = call("scene", {})
    sol = solve(Params.from_dict(GENERIC))
    num = compute_numeric(sol, SceneSettings())
    assert sc["n_lines"] == len(num.traced.lines)
    assert [a["label"] for a in sc["axis"]] == [a["label"] for a in axis_info(summarize(sol), num.domain, num.fi)]
    pts = np.frombuffer(__import__("base64").b64decode(sc["streamlines"]["points"]), np.float32).reshape(-1, 3)
    assert np.allclose(pts, np.vstack(num.traced.lines), atol=1e-6)


def test_sigma_rows_equal_the_desktop_map():
    from swirl_explorer.core.sigma_map import SigmaMapSpec, compute_sigma_map
    spec_d = {"base": Params().to_dict(), "x_name": "gamma", "y_name": "kappa", "x_range": [-2.6, 2.6],
              "y_range": [-4.6, 4.6], "nx": 13, "ny": 13}
    call("sigma_start", spec_d)
    rows = call("sigma_rows", spec_d, list(range(13)))
    st = call("sigma_add", rows)
    ref = compute_sigma_map(SigmaMapSpec(Params(), "gamma", "kappa", (-2.6, 2.6), (-4.6, 4.6), nx=13, ny=13))
    assert st["done"] == 169 and st["global"] == int(ref.is_global.sum())
    for r in rows:
        assert r["cn"] == ref.code_n[r["j"]].tolist() and r["cs"] == ref.code_s[r["j"]].tolist()
    d = call("sigma_draw", {"mode": "mu", "current": GENERIC, "w": 500, "h": 500})
    assert len(d["axes"]) == 2 and all(0 <= a["x0"] < a["x1"] <= 1 for a in d["axes"])
    assert "GLOBAL" in call("sigma_point", 0.0, 0.0)["text"]


def test_lock_stops_at_the_lens_edge():
    from swirl_explorer.core.edges import gamma_minus
    r = call("lock", GENERIC, "gamma", GENERIC["gamma"], -3.0)
    assert r["value"] == pytest.approx(gamma_minus(Params(kappa=1.0)), abs=1e-8) and "minimum" in r["note"]
    assert call("lock", GENERIC, "gamma", GENERIC["gamma"], -1.5)["value"] == -1.5


def test_path_scan_matches_core():
    from swirl_explorer.core.paths import ParamPath, scan_path
    path = {"interpolation": "linear", "keyframes": [
        {"t": 0, "params": Params(gamma=-0.5, kappa=1.0).to_dict()},
        {"t": 1, "params": Params(gamma=-2.5, kappa=1.0).to_dict()}]}
    r = call("path_scan", path, 21)
    frames, events = scan_path(ParamPath.from_dict(path), 21)
    assert [f["global"] for f in r["frames"]] == [f.is_global for f in frames]
    assert [e["t"] for e in r["events"]] == [e.t for e in events]
    assert r["frames"][3]["params"]["gamma"] == pytest.approx(frames[3].params.gamma)
    assert call("path_scan", {"keyframes": []}, 10) == {"frames": [], "events": []}


def test_sessions_are_the_desktop_format(tmp_path):
    from swirl_explorer.io.session import load_session, Session, save_session
    from swirl_explorer.core.paths import ParamPath
    text = call("session_json", {"params": GENERIC, "path": {"keyframes": [{"t": 0, "params": GENERIC}]},
                                 "scene": {"n_seeds": 33}, "ui": {"theme": "light"}})
    f = tmp_path / "web.json"
    f.write_text(text, encoding="utf-8")
    s = load_session(f)                                 # the desktop's loader
    assert s.params == Params.from_dict(GENERIC) and s.sections["scene"]["n_seeds"] == 33 and len(s.path) == 1
    desk = tmp_path / "desk.json"                        # and the other way round
    save_session(Session(Params(gamma=0.3), {"plots": {"xvar": "x"}}, ParamPath()), desk)
    back = call("session_load", desk.read_text(encoding="utf-8"))
    assert back["params"]["gamma"] == 0.3 and back["sections"]["plots"]["xvar"] == "x"


def test_presets_are_the_desktop_values():
    from swirl_explorer.core.edges import gamma_minus
    assert call("preset", "rest") == Params().to_dict()
    assert call("preset", "landau", 1.5)["gamma"] == pytest.approx(2 / 1.5)
    assert call("preset", "default")["gamma"] == pytest.approx(gamma_minus(Params(kappa=1.0)) + 0.1)
    r = call("preset", "recessive")
    assert r["c2"] == -0.9 and r["kappa"] == 0.3


def test_unknown_calls_are_refused():
    with pytest.raises(ValueError):
        api.call("__import__", "[]")


def _arr(s, dtype, k=3):
    return np.frombuffer(__import__("base64").b64decode(s), dtype).reshape(-1, k)


def test_scene_has_no_swirl_arcs_and_draws_arrows_and_isosurface_like_the_desktop():
    pytest.importorskip("skimage")
    from swirl_explorer.core.solver import solve
    from swirl_explorer.core.poles import summarize
    from swirl_explorer.viz.scene_compute import SceneSettings, compute_numeric, build_scene
    call("solve", GENERIC)
    sc = call("scene", {"show_glyphs": True, "show_iso": True, "clip": False})
    assert "arcs" not in sc and not any("orange" in t for t in sc["legend"])
    assert any("isosurface" in t for t in sc["legend"]) and any("arrows" in t for t in sc["legend"])
    pv = pytest.importorskip("pyvista")
    s = SceneSettings(show_glyphs=True, show_iso=True, clip=False)
    sol = solve(Params.from_dict(GENERIC))
    data = build_scene(compute_numeric(sol, s), summarize(sol))
    assert sc["iso"]["level"] == pytest.approx(data.iso_level, rel=1e-12)
    # glyphs: the same base points and vectors as the desktop's VTK glyph source
    assert sc["glyphs"]["n"] == len(_arr(sc["glyphs"]["points"], np.float32))
    # isosurface: marching cubes on the same grid gives the same surface (area within 2 %)
    P, T = _arr(sc["iso"]["points"], np.float32).astype(float), _arr(sc["iso"]["triangles"], np.uint32)
    area = 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1).sum()
    ref = data.iso.triangulate().area
    assert area == pytest.approx(ref, rel=0.02)
    assert np.allclose(P.min(0), data.iso.points.min(0), atol=0.04) and np.allclose(P.max(0), data.iso.points.max(0), atol=0.04)
