"""Milestone 7: the entry point and the packaging files."""
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_version_option(capsys):
    from swirl_explorer import __version__
    from swirl_explorer.__main__ import main
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_option_is_rejected():
    from swirl_explorer.__main__ import main
    with pytest.raises(SystemExit):
        main(["--no-such-option"])


def test_packaging_files_present():
    for f in ("build_windows.ps1", "requirements.txt", "packaging/SwirlExplorer.spec", "packaging/launch.py",
              "packaging/swirl.ico"):
        assert (ROOT / f).exists(), f
    spec = (ROOT / "packaging/SwirlExplorer.spec").read_text(encoding="utf-8")
    assert "imageio_ffmpeg" in spec and "console=False" in spec
    # every run-time import of the package is in requirements.txt
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for name in ("numpy", "scipy", "matplotlib", "pyside6", "pyvista", "pyvistaqt", "vtk", "imageio", "pillow"):
        assert name in req, name


def test_launcher_calls_freeze_support_first():
    src = (ROOT / "packaging/launch.py").read_text(encoding="utf-8")
    assert src.index("freeze_support()") < src.index("from swirl_explorer")
