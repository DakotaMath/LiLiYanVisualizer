"""Sessions: everything needed to come back to a view, as one JSON file.

The file holds plain sections that the UI fills and reads back:

    params   the six data and x₀                 lock     the "global only" lock
    scene    3D settings (seeds, layers, colours) camera   position / focal point / up / zoom
    plane    the slice plane                      slice    slice display options
    plots    profile plot selections              sigma    Σ-map axes, ranges, grid, view
    path     keyframes and interpolation          movie    frames, fps, layout, resolution
    ui       theme, which tab, 2D slice shown

Unknown sections are kept on load and written back on save, so a newer file opened by an
older build loses nothing.  This module has no GUI imports.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.params import Params
from ..core.paths import ParamPath

FORMAT = "swirl-explorer-session"
VERSION = 1


class SessionError(ValueError):
    pass


def _clean(o: Any) -> Any:
    """JSON-safe copy: tuples to lists, NaN/inf to None, numpy scalars to floats."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            pass
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


@dataclass
class Session:
    params: Params = field(default_factory=Params)
    sections: dict = field(default_factory=dict)       # every other section, as plain dicts
    path: ParamPath | None = None

    def to_dict(self) -> dict:
        d = {"format": FORMAT, "version": VERSION,
             "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "params": self.params.to_dict()}
        d.update(self.sections)
        if self.path is not None and len(self.path):
            d["path"] = self.path.to_dict()
        return _clean(d)

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            raise SessionError("not a Swirl Explorer session file")
        if int(d.get("version", 0)) > VERSION:
            raise SessionError(f"session version {d.get('version')} is newer than this build ({VERSION})")
        try:
            params = Params.from_dict(d["params"])
            path = ParamPath.from_dict(d["path"]) if d.get("path") else None
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionError(f"damaged session file: {exc}") from exc
        sections = {k: v for k, v in d.items() if k not in ("format", "version", "saved", "params", "path")}
        return cls(params, sections, path)


def save_session(s: Session, path: str | Path) -> Path:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(s.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                              # never leave a half-written file behind
    return path


def load_session(path: str | Path) -> Session:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionError(f"not valid JSON: {exc}") from exc
    return Session.from_dict(d)
