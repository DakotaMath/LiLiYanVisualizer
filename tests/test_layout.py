"""The repository layout: the shared packages stay usable by both front ends."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "swirl_explorer"
SHARED = ("core", "viz", "io")
FRONT_ENDS = ("ui", "web")


def _imports(path: Path) -> set[str]:
    """Absolute module names imported by a file (relative imports resolved against the package)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(PKG.parent).with_suffix("").parts      # ('swirl_explorer', 'core', 'solver')
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(rel[:len(rel) - node.level])
                out.add(".".join(base + ([node.module] if node.module else [])))
            elif node.module:
                out.add(node.module)
    return out


def test_shared_code_has_no_front_end_imports():
    bad = []
    for sub in SHARED:
        for f in (PKG / sub).rglob("*.py"):
            for m in _imports(f):
                top = m.split(".")[0]
                if top in ("PySide6", "PyQt5", "PyQt6", "pyvistaqt", "trame", "qtpy") or \
                        any(m.startswith(f"swirl_explorer.{fe}") for fe in FRONT_ENDS):
                    bad.append(f"{f.relative_to(PKG)} imports {m}")
    assert not bad, "\n".join(bad)


def test_front_ends_do_not_import_each_other():
    for a, b in (("ui", "web"), ("web", "ui")):
        for f in (PKG / a).rglob("*.py"):
            for m in _imports(f):
                assert not m.startswith(f"swirl_explorer.{b}"), f"{f.relative_to(PKG)} imports {m}"
