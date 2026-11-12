"""Colour palettes shared by the Qt panels, the matplotlib plots and the 3D view (no Qt imports).

Dark mode is the default. ``set_mode("light")`` switches every consumer that reads ``c()``.
"""
from __future__ import annotations

PALETTES = {
    "dark": dict(
        window="#1e1f22", panel="#26282c", base="#2b2d31", button="#34373c",
        text="#e8e6e3", muted="#b4b2ad", faint="#8c8a85", border="#3d4046",
        ok="#4cc38a", bad="#ff6b6b", accent="#4c9be8",
        plot_bg="#1f2125", plot_ink="#e8e6e3", plot_ink2="#a9a7a2", plot_grid="#33363b",
        view_bg="#16181c", view_bg_top="#23262c", view_text="#e8e6e3", outline="#6f737b",
        cut_plane="#8fa3bf", origin="#8a8c92",
    ),
    "light": dict(
        window="#f3f2ef", panel="#ffffff", base="#ffffff", button="#ecebe7",
        text="#1d1c1a", muted="#46453f", faint="#7a7974", border="#d9d8d3",
        ok="#1a8a5f", bad="#c0392b", accent="#2a78d6",
        plot_bg="#fcfcfb", plot_ink="#0b0b0b", plot_ink2="#52514e", plot_grid="#e4e3df",
        view_bg="#ffffff", view_bg_top="#ffffff", view_text="#1d1c1a", outline="#9a9ca3",
        cut_plane="#8fa3bf", origin="#8a8c92",
    ),
}
MODE = "dark"


def set_mode(mode: str) -> None:
    global MODE
    if mode not in PALETTES:
        raise ValueError(mode)
    MODE = mode


def c() -> dict:
    return PALETTES[MODE]


def mpl_rc() -> dict:
    """matplotlib rcParams for the current palette."""
    p = c()
    return {
        "figure.facecolor": p["plot_bg"], "axes.facecolor": p["plot_bg"], "savefig.facecolor": p["plot_bg"],
        "text.color": p["plot_ink"], "axes.labelcolor": p["plot_ink"], "axes.edgecolor": p["plot_ink2"],
        "xtick.color": p["plot_ink2"], "ytick.color": p["plot_ink2"], "legend.labelcolor": p["plot_ink"],
        "grid.color": p["plot_grid"],
    }
