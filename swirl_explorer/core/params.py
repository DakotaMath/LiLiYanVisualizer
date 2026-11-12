"""Cauchy data of a solution and the symmetries of the Q-system.

Notation follows the monograph (Ch. 1-2).  A solution is fixed by its state
at a base point ``x_b`` (default the equator ``x_b = 0``):

========  =====================================  ===========================
book      meaning                                task brief name
========  =====================================  ===========================
c1,c2,c3  Taylor constants of Q at x_b (P_c)     c1, c2, c3
gamma     U(x_b) = u_theta sin(theta) at x_b     Gamma
A         f(x_b) = u_phi sin(theta) at x_b       A
kappa     f_x(x_b)  (normalisation kappa_x, #1)  gamma (swirl rate)
========  =====================================  ===========================

The book writes the meridional value as gamma and the swirl rate as kappa;
the task brief used Gamma and gamma.  We follow the book (docs/conventions.md).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math

PARAM_NAMES: tuple[str, ...] = ("c1", "c2", "c3", "gamma", "A", "kappa")

PARAM_LABELS: dict[str, str] = {
    "c1": "c₁", "c2": "c₂", "c3": "c₃",
    "gamma": "γ = U(0)", "A": "A = f(0)", "kappa": "κ = f_x(0)",
}


@dataclass(frozen=True)
class Params:
    """Six Cauchy data (c1, c2, c3, gamma, A, kappa) at the base point ``xb``."""

    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    gamma: float = 0.0
    A: float = 0.0
    kappa: float = 0.0
    xb: float = 0.0

    # ------------------------------------------------------------------ basics
    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.c1, self.c2, self.c3, self.gamma, self.A, self.kappa)

    @classmethod
    def from_tuple(cls, t, xb: float = 0.0) -> "Params":
        return cls(*map(float, t), xb=float(xb))

    def with_(self, **kw) -> "Params":
        return replace(self, **{k: float(v) for k, v in kw.items()})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(**{k: float(d[k]) for k in (*PARAM_NAMES, "xb") if k in d})

    def key(self, ndigits: int = 14) -> tuple:
        """Hashable cache key (rounded to suppress float noise)."""
        return tuple(round(v, ndigits) for v in (*self.as_tuple(), self.xb))

    # ---------------------------------------------------------- derived data
    @property
    def lam(self) -> float:
        """First-integral constant lambda = L[P_c] + A^2 = A^2 - 2c1 - 2c2 - 4c3 (Prop. 2.20)."""
        return self.A ** 2 - 2 * self.c1 - 2 * self.c2 - 4 * self.c3

    def Pc(self, x: float) -> float:
        return self.c1 * (1 - x) + self.c2 * (1 + x) + self.c3 * (1 - x * x)

    def Pc_x(self, x: float) -> float:
        return -self.c1 + self.c2 - 2 * self.c3 * x

    def jet(self) -> tuple[float, float, float]:
        """(Q, Q_x, Q_xx) at the base point (Remark 2.6)."""
        x = self.xb
        return self.Pc(x), self.Pc_x(x), -2.0 * self.c3

    @property
    def kappa_line(self) -> float:
        """f'(xi_b) for z(xi_b) = 1: kappa[xi_b] = (1 - x_b^2) kappa_x (Lemma 2.15(2))."""
        return (1.0 - self.xb ** 2) * self.kappa

    @property
    def xi_b(self) -> float:
        return math.atanh(self.xb)

    # ------------------------------------------------------------ symmetries
    def sigma1(self) -> "Params":
        """f -> -f :  (A, kappa) -> (-A, -kappa)  (Lemma 2.8)."""
        return replace(self, A=-self.A, kappa=-self.kappa)

    def sigma2(self) -> "Params":
        """x -> -x : (c1,c2,c3,gamma,A,kappa) -> (c2,c1,c3,-gamma,A,-kappa) at x_b = 0.

        For x_b != 0 the image has base point -x_b.
        """
        return replace(self, c1=self.c2, c2=self.c1, gamma=-self.gamma,
                       kappa=-self.kappa, xb=-self.xb)


def landau_params(L: float, A: float = 0.0) -> Params:
    """Rest-state/Landau data: c = 0, kappa = 0, gamma = U(0) = 2/L (Example 2.29)."""
    return Params(gamma=2.0 / L, A=A)
