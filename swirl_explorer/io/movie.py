"""Movie export: frame layout, compositing and MP4 encoding (imageio-ffmpeg).

The UI renders each panel (3D view, slice, profiles) to an RGB array; this module fits
them into the cells of a layout, adds a caption strip (time, data, global / pole
status) and streams the frames to an H.264 MP4.  No GUI imports.

Frames where the solution is not global are never passed off as ordinary frames: the
policy decides whether they are held (the last global picture with a red banner),
skipped, or rendered with the banner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# panel cells as fractions of the picture area (x0, y0, x1, y1), the caption strip is below
LAYOUTS: dict[str, tuple[str, list[tuple[str, tuple[float, float, float, float]]]]] = {
    "3d": ("3D view", [("3d", (0, 0, 1, 1))]),
    "slice": ("2D slice", [("slice", (0, 0, 1, 1))]),
    "plots": ("ODE profiles", [("plots", (0, 0, 1, 1))]),
    "3d+slice": ("3D view | 2D slice", [("3d", (0, 0, 0.5, 1)), ("slice", (0.5, 0, 1, 1))]),
    "3d+plots": ("3D view | profiles", [("3d", (0, 0, 0.62, 1)), ("plots", (0.62, 0, 1, 1))]),
    "all": ("3D view | slice over profiles",
            [("3d", (0, 0, 0.5, 1)), ("slice", (0.5, 0, 1, 0.5)), ("plots", (0.5, 0.5, 1, 1))]),
}

RESOLUTIONS = {"1280 × 720": (1280, 720), "1920 × 1080": (1920, 1080), "2560 × 1440": (2560, 1440),
               "1080 × 1080": (1080, 1080), "3840 × 2160": (3840, 2160)}

NON_GLOBAL_POLICIES = {
    "hold": "hold the last global frame, with a red banner",
    "mark": "render them anyway, with a red banner",
    "skip": "leave them out of the movie",
}

BG = (16, 17, 20)
INK = (226, 228, 233)
MUTED = (150, 154, 163)
OK = (63, 168, 119)
BAD = (214, 64, 69)


@dataclass
class Caption:
    title: str                 # e.g. "t = 0.412   frame 50 / 120"
    data: str                  # e.g. "(c₁, c₂, c₃, γ, A, κ) = (...)"
    status: str                # e.g. "global · N: G, μ = 0.83 · S: G, μ = 0.91"
    ok: bool = True
    banner: str = ""           # when set, every panel is dimmed and labelled with it (held / not global)
    strip: list | None = None  # per-frame RGB colours of the whole path (the dock's timeline in miniature)
    pos: int = 0               # this frame's index in `strip`
    keys: tuple = ()           # keyframe positions as fractions 0..1
    events: tuple = ()         # event positions as fractions 0..1
    held_at: float | None = None   # fraction 0..1 of the frame whose picture is shown (held frames)
    key: tuple = ()            # ((rgb, label), ...): colour key for the strip


def even(n: int) -> int:
    return max(2, int(n) // 2 * 2)


def cells(layout: str, W: int, H: int) -> tuple[list[tuple[str, tuple[int, int, int, int]]], int]:
    """[(panel, (x, y, w, h))] in pixels and the caption height, for a W × H frame."""
    cap = caption_height(H)
    ph = H - cap
    out = []
    for name, (x0, y0, x1, y1) in LAYOUTS[layout][1]:
        x, y = round(x0 * W), round(y0 * ph)
        out.append((name, (x, y, round(x1 * W) - x, round(y1 * ph) - y)))
    return out, cap


def caption_height(H: int) -> int:
    return max(48, round(0.095 * H))


def panels_needed(layout: str) -> list[str]:
    return [n for n, _ in LAYOUTS[layout][1]]


_FONTS: dict = {}


def _font(px: int, bold: bool = False):
    key = (px, bold)
    if key not in _FONTS:
        try:   # DejaVu ships with matplotlib and has Greek, subscripts and maths symbols
            import matplotlib
            fn = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / fn
            _FONTS[key] = ImageFont.truetype(str(path), px)
        except Exception:                                        # pragma: no cover
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def fit(img: np.ndarray, w: int, h: int, bg=BG) -> Image.Image:
    """Scale `img` (H×W×3/4 uint8) into a w × h cell keeping its aspect ratio, centred."""
    im = Image.fromarray(np.ascontiguousarray(img[..., :3]))
    s = min(w / im.width, h / im.height)
    nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    cell = Image.new("RGB", (w, h), bg)
    cell.paste(im, ((w - nw) // 2, (h - nh) // 2))
    return cell


def compose(images: dict[str, np.ndarray], layout: str, W: int, H: int, cap: Caption) -> np.ndarray:
    """One movie frame (H × W × 3 uint8)."""
    frame = Image.new("RGB", (W, H), BG)
    cl, ch = cells(layout, W, H)
    for name, (x, y, w, h) in cl:
        if name in images and images[name] is not None:
            frame.paste(fit(images[name], w, h), (x, y))
    d = ImageDraw.Draw(frame)
    if len(cl) > 1:                                         # thin separators between panels
        for _, (x, y, w, h) in cl:
            if x > 0:
                d.line([(x, y), (x, y + h)], fill=(52, 55, 62), width=2)
            if y > 0:
                d.line([(x, y), (x + w, y)], fill=(52, 55, 62), width=2)
    if cap.banner:
        _stamp(frame, cl, H, cap.banner)
        d = ImageDraw.Draw(frame)
    _caption(frame, d, W, H, ch, cap)
    return np.asarray(frame)


def _stamp(frame, cells_, H, text):
    """Dim every panel (the picture is not the solution at this t) and label each one."""
    f = _font(max(10, round(H * 0.019)), True)
    m = round(H * 0.008)
    for _, (x, y, w, h) in cells_:
        pic = frame.crop((x, y, x + w, y + h))
        frame.paste(Image.blend(pic, Image.new("RGB", pic.size, (0, 0, 0)), 0.5), (x, y))
        d = ImageDraw.Draw(frame)
        tw = d.textlength(text, font=f)
        lh = round(H * 0.019 * 1.5)
        x1 = x + w - m                                       # a tag in the top-right corner
        d.rounded_rectangle([x1 - tw - 2 * m, y + m, x1, y + m + lh], radius=round(m * 0.6), fill=BAD)
        d.text((x1 - m, y + m + lh / 2), text, font=f, fill=(255, 255, 255), anchor="rm")


def _caption(frame, d, W, H, ch, cap: Caption):
    y0 = H - ch
    d.rectangle([0, y0, W, H], fill=(24, 25, 29))
    sh = 0
    if cap.strip:
        # where we are on the path: the coloured timeline, numbered keyframes and a playhead
        n, sh = len(cap.strip), max(8, round(ch * 0.17))
        x0s, ws = sh * 1.5, W - 3 * sh                      # inset, so the end badges are not cut
        for i, col in enumerate(cap.strip):
            d.rectangle([x0s + i * ws / n, y0 + 2, x0s + (i + 1) * ws / n, y0 + sh], fill=tuple(col))
        for f in cap.events:                          # ▼ like the dock's timeline
            x = x0s + f * ws
            d.line([(x, y0 + 2), (x, y0 + sh)], fill=(242, 193, 78), width=2)
            d.polygon([(x - sh * 0.45, y0 - sh * 0.75), (x + sh * 0.45, y0 - sh * 0.75), (x, y0)],
                      fill=(242, 193, 78))
        kf = _font(max(8, round(sh * 0.85)), True)
        for k, f in enumerate(cap.keys):
            x = x0s + f * ws
            d.ellipse([x - sh * 0.6, y0 + sh / 2 - sh * 0.6, x + sh * 0.6, y0 + sh / 2 + sh * 0.6],
                      fill=(242, 193, 78))
            d.text((x, y0 + sh / 2), str(k + 1), font=kf, fill=(17, 17, 17), anchor="mm")
        if cap.held_at is not None:                 # where the held picture comes from: a hollow marker
            xh, r = x0s + cap.held_at * ws, sh * 0.75
            yc = y0 - r
            d.polygon([(xh, yc - r), (xh + r, yc), (xh, yc + r), (xh - r, yc)], outline=BAD, width=3)
        x = x0s + (cap.pos + 0.5) * ws / n
        d.polygon([(x - sh * 0.7, y0 - sh), (x + sh * 0.7, y0 - sh), (x, y0)], fill=(255, 255, 255))
        d.line([(x, y0), (x, y0 + sh)], fill=(255, 255, 255), width=3)
    else:
        d.line([(0, y0), (W, y0)], fill=OK if cap.ok else BAD, width=3)
    body = ch - sh
    yb = y0 + sh
    big, small = _font(max(11, round(body * 0.30)), True), _font(max(10, round(body * 0.25)))
    pad = round(body * 0.25)
    d.text((pad, yb + round(body * 0.12)), cap.title, font=big, fill=INK)
    d.text((pad, yb + round(body * 0.55)), cap.data, font=small, fill=MUTED)
    # status badge on the right
    tw = d.textlength(cap.status, font=big)
    if cap.key:                                    # key for the strip, left of the badge
        kfnt = _font(max(9, round(body * 0.21)))
        xk = W - pad - tw - 2 * pad - pad
        yk = yb + round(body * 0.30)
        s = round(body * 0.2)
        marks = [("tri", (242, 193, 78), "event"), ("dot", (242, 193, 78), "keyframe")]
        if cap.held_at is not None:
            marks.append(("dia", BAD, "picture held from"))
        for shape, col, lab in reversed(list(cap.key) and [("box", c, l) for c, l in cap.key] + marks):
            lw = d.textlength(lab, font=kfnt)
            xk -= lw
            d.text((xk, yk), lab, font=kfnt, fill=MUTED, anchor="lm")
            xk -= s + 4
            if shape == "box":
                d.rectangle([xk, yk - s / 2, xk + s, yk + s / 2], fill=tuple(col))
            elif shape == "tri":
                d.polygon([(xk, yk - s / 2), (xk + s, yk - s / 2), (xk + s / 2, yk + s / 2)], fill=col)
            elif shape == "dot":
                d.ellipse([xk, yk - s / 2, xk + s, yk + s / 2], fill=col)
            else:
                d.polygon([(xk + s / 2, yk - s / 2 - 1), (xk + s + 1, yk), (xk + s / 2, yk + s / 2 + 1),
                           (xk - 1, yk)], outline=col, width=2)
            xk -= round(pad * 0.8)
    bx1, by0 = W - pad, yb + round(body * 0.18)
    bx0, by1 = bx1 - tw - 2 * pad, H - round(body * 0.18)
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=round(body * 0.15), fill=OK if cap.ok else BAD)
    d.text((bx0 + pad, (by0 + by1) / 2), cap.status, font=big, fill=(255, 255, 255), anchor="lm")


class MovieWriter:
    """Streams RGB frames to an MP4 (H.264, yuv420p, plays everywhere)."""

    def __init__(self, path: str | Path, fps: int, size: tuple[int, int], quality: int = 8):
        import imageio.v2 as imageio
        self.path = Path(path)
        self.size = (even(size[0]), even(size[1]))
        self.n = 0
        self._w = imageio.get_writer(str(self.path), format="FFMPEG", mode="I", fps=fps, codec="libx264",
                                     quality=quality, pixelformat="yuv420p", macro_block_size=2,
                                     ffmpeg_log_level="error")

    def append(self, frame: np.ndarray) -> None:
        W, H = self.size
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = np.asarray(Image.fromarray(frame[..., :3]).resize((W, H), Image.LANCZOS))
        self._w.append_data(np.ascontiguousarray(frame[..., :3]))
        self.n += 1

    def close(self) -> None:
        self._w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
