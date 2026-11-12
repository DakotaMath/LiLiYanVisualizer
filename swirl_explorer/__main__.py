"""Entry point:  python -m swirl_explorer   (also the entry point of the Windows build)

Options:
  SESSION.json        open a saved session at start-up
  --selftest DIR      run the end-to-end self-test, write DIR/selftest.txt, exit 0 / 1
  --version           print the version and exit
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()          # Σ-map process pools in the frozen build re-enter here
    from . import __version__
    ap = argparse.ArgumentParser(prog="SwirlExplorer", description="Swirl Explorer " + __version__)
    ap.add_argument("session", nargs="?", help="a saved session (.json) to open")
    ap.add_argument("--selftest", metavar="DIR", help="run the end-to-end self-test and write DIR/selftest.txt")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    args = ap.parse_args(argv)
    if args.version:
        _say(f"Swirl Explorer {__version__}")
        return 0
    if args.selftest:
        from .selftest import run
        return run(args.selftest)
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Swirl Explorer")
    app.setApplicationVersion(__version__)
    from .ui.theme import apply_theme
    apply_theme(app)
    win = MainWindow()
    win.resize(1680, 980)
    win.show()
    if args.session:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(300, lambda: win.open_session(args.session))
    return app.exec()


def _say(text: str) -> None:
    """Print, also from the windowed build (no console): fall back to a message box."""
    if sys.stdout is not None:
        print(text)
        return
    try:                                       # pragma: no cover - frozen, windowed
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "Swirl Explorer", 0)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
