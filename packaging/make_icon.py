"""Draw the app icon (a swirl over the global lens) and write packaging/swirl.ico.

Run once; the .ico is committed.  Needs only Pillow and numpy.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

S = 512
OUT = Path(__file__).with_name("swirl.ico")


def draw(size: int = S) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = size * 0.04
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=size * 0.2, fill=(22, 24, 28, 255))
    c = size / 2
    # the lens (the global set of Fig. 1.3), green
    lens = []
    for t in np.linspace(-1, 1, 80):
        lens.append((c + 0.30 * size * (1 - t * t) ** 0.85, c - 0.36 * size * t))
    for t in np.linspace(1, -1, 80):
        lens.append((c - 0.30 * size * (1 - t * t) ** 0.85, c - 0.36 * size * t))
    d.polygon(lens, fill=(63, 168, 119, 255))
    # a swirling streamline around the axis
    pts = []
    for k in range(400):
        s = k / 399
        r = size * (0.05 + 0.33 * s)
        a = 4.2 * math.pi * s
        pts.append((c + r * math.cos(a), c + 0.55 * r * math.sin(a) + size * 0.18 * (0.5 - s)))
    d.line(pts, fill=(242, 193, 78, 255), width=max(2, int(size * 0.035)), joint="curve")
    # the symmetry axis
    d.line([(c, size * 0.1), (c, size * 0.9)], fill=(76, 155, 232, 255), width=max(2, int(size * 0.03)))
    return im


if __name__ == "__main__":
    draw().save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    draw().resize((256, 256), Image.LANCZOS).save(OUT.with_suffix(".png"))
    print("wrote", OUT)
