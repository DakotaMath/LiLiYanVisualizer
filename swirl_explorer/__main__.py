"""Entry point:  python -m swirl_explorer"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()          # needed for Σ-map process pools in frozen builds
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("Swirl Explorer")
    from .ui.theme import apply_theme
    apply_theme(app)
    win = MainWindow()
    win.resize(1680, 980)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
