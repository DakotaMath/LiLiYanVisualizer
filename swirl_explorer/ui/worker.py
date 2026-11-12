"""Background solving: latest-request-wins worker so slider drags never block the UI."""
from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core.params import Params
from ..core.solver import solve
from ..core.poles import summarize


class SolveWorker(QObject):
    """Lives in its own QThread. ``submit`` may be called from the GUI thread at any rate."""

    finished = Signal(object, object, object, float)    # params, Solution, Summary, ms
    failed = Signal(object, str)
    _kick = Signal()

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._pending: Params | None = None
        self._kick.connect(self._run)

    def submit(self, p: Params) -> None:
        with self._lock:
            self._pending = p
        self._kick.emit()

    @Slot()
    def _run(self):
        while True:
            with self._lock:
                p, self._pending = self._pending, None
            if p is None:
                return
            t0 = time.perf_counter()
            try:
                sol = solve(p)
                sm = summarize(sol)
            except Exception as exc:  # report, never crash the UI
                self.failed.emit(p, f"{type(exc).__name__}: {exc}")
                continue
            with self._lock:
                stale = self._pending is not None
            if not stale:
                self.finished.emit(p, sol, sm, (time.perf_counter() - t0) * 1e3)


class SolveThread:
    def __init__(self):
        self.thread = QThread()
        self.worker = SolveWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()

    def stop(self):
        self.thread.quit()
        self.thread.wait(2000)
