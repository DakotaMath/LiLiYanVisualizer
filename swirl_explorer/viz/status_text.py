"""The status text for a solution (HTML), shared by the desktop status panel and the browser version."""
from __future__ import annotations

import math

from ..core.poles import Summary, PoleDiagnostics
from . import style


def _col(k):
    return style.c()[k]
STRATUM_GLOSS = {
    "G": "generic: U → −2+2μ, swirl bounded", "R": "recessive: U → −2−2μ, swirl unbounded",
    "L1": "logarithmic: U → −2 like 4/s, swirl bounded", "L2": "logarithmic: U → −2, swirl unbounded",
    "R0": "no-swirl recessive branch",
}
BP_GLOSS = {"II.5 (and III)": "Type II.5: u_r milder than d^(−2/3), u_θ, u_φ ~ c/d",
            "III": "Type III: d·|u| bounded", "beyond III": "beyond Type III: more singular than 1/d",
            "I or II": "Type I or II: bounded or logarithmic"}
CLASS_GLOSS = {"Z": "class Z — the swirl changes sign once", "N": "class N — the swirl keeps its sign",
               "face": "no genuine swirl (κ = 0)"}


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "+∞" if v > 0 else "−∞"
    if isinstance(v, float) and not math.isfinite(v):
        return "—"
    return f"{v:.{nd}g}".replace("-", "−")


def _bp_short(bp: str) -> str:
    return {"II.5 (and III)": "II.5 (u<sub>r</sub> mild)", "III": "III (d·|u| bounded)",
            "beyond III": "beyond III (worse than 1/d)", "I or II": "I/II (bounded or log)"}.get(bp, bp)


def pole_line(name: str, d: PoleDiagnostics, sign: int) -> tuple[str, str]:
    """(html line, tooltip text) for one pole."""
    if d.stratum == "blowup":
        return f"<b>{name}</b> &nbsp;<span style='color:{_col('bad')}'>not reached (blow-up first)</span>", d.note
    if d.stratum == "unsafe":
        return (f"<b>{name}</b> &nbsp;<span style='color:{_col('bad')}'>no admissible pole limit</span>",
                "the solution would blow up beyond the computed window (book's terminal test)")
    U = d.U_lim if sign > 0 else -d.U_lim
    pole = "U(1)" if sign > 0 else "U(−1)"
    unc = (f" <span style='color:{_col('bad')}'>(uncertain)</span>" if d.confidence < 0.9 else "")
    tg = {"G": "generic", "R": "recessive", "L1": "log, bounded swirl", "L2": "log, unbounded swirl",
          "R0": "no-swirl recessive"}.get(d.stratum, "")
    line = (f"<b>{name}</b> &nbsp;type <b>{d.stratum}</b> ({tg}){unc} · μ = {_fmt(d.mu)} · "
            f"{pole} = {_fmt(U)} · class <b>{_bp_short(d.bp)}</b>")
    ex = d.exponents or {}
    plain, html = [], []
    for k, lab, hl in (("u_r", "u_r", "u<sub>r</sub>"), ("u_theta", "u_θ", "u<sub>θ</sub>"),
                       ("u_phi", "u_φ", "u<sub>φ</sub>")):
        sl = ex.get(k, {}).get("slope", math.nan)
        if math.isfinite(sl):
            plain.append(f"{lab} ~ d^{_fmt(sl, 3)}")
            html.append(f"{hl} ~ d<sup>{_fmt(sl, 3)}</sup>")
        else:
            plain.append(f"{lab}: below rounding")
            html.append(f"{hl}: below rounding")
    sub = (f"<br><span style='color:{_col('muted')}'>&nbsp;&nbsp;&nbsp;f = {_fmt(d.f_lim)} · near the axis: "
           + ", ".join(html) + "</span>")
    tip = (f"{name}\n"
           f"pole type {d.stratum}: {STRATUM_GLOSS.get(d.stratum, '')}\n"
           f"μ = √(1 + Q/2) = {_fmt(d.mu, 6)} (type confidence {d.confidence:.2f})\n"
           f"Bradshaw–Palmer {BP_GLOSS.get(d.bp, d.bp)}; Li–Yan Type {d.li_yan}\n"
           f"(classes follow from the pole type and μ; fitted powers of d = sin θ: {', '.join(plain)})")
    return line + sub, tip


def head_html(sm: Summary) -> str:
    """The headline: global (with the swirl class) or not global (with where it blows up)."""
    if sm.is_global:
        return (f"<span style='color:{_col('ok')};font-size:16px'><b>● GLOBAL</b></span>"
                f"&nbsp;&nbsp; {CLASS_GLOSS.get(sm.cls, sm.cls)}")
    if sm.blowup_x is not None:
        what = "U → −∞" if sm.north.stratum == "blowup" else "U → +∞"
        where = f" &nbsp; blow-up: {what} at x = {sm.blowup_x:+.6f}".replace("-", "−")
    else:
        where = " &nbsp; a pole has no admissible limit"
    return f"<span style='color:{_col('bad')};font-size:16px'><b>● NOT GLOBAL</b></span>{where}"


