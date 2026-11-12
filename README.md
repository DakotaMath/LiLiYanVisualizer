# Swirl Explorer

A research tool for the six-parameter family of axisymmetric, (−1)-homogeneous
solutions of the stationary Navier–Stokes equations in R³ \ {axis}. It follows the
monograph *Axisymmetric (−1)-homogeneous solutions of the stationary Navier–Stokes
equations with swirl*, which is in `docs/`.

**Status:** milestones 1 to 4 are done.

- Milestone 1: core solver, pole diagnostics, regression tests.
- Milestone 2: the Qt shell, with the parameter panel, the live status and stacked
  2D profile plots, including pins and PNG/SVG/CSV export.

- Milestone 3: the 3D view (PyVista). It shows:
  - streamlines from rings around the axis, a ball, a disc or a line rake, with a
    draggable handle (off by default);
  - direction cones, and the inflow/outflow along the axis at each pole;
  - stream surfaces, vortex lines, arrows and isosurfaces;
  - self-similar copies λX of a streamline;
  - colouring by |u|, u_φ, helicity or pressure, and a half-space cut-away.

  The theme is dark by default (View menu: light).

- Milestone 4: a slice plane next to the 3D view. It shows:
  - colour by u·n or ρ·u·n, on a linear or signed-log scale, shared with the 3D view;
  - projected streamlines and an optional LIC texture;
  - the axis and the cut-out around the origin, with a small orientation sketch;
  - presets, azimuth/elevation/offset sliders, a draggable plane in 3D, and
    "point the 3D camera along n".

The Σ-map comes next.

Start the app with:

```bash
python -m swirl_explorer
```

## The mathematics in one page

A flow u = r⁻¹(u_r e_r + u_θ e_θ + u_φ e_φ) with pressure p = r⁻²p(θ) is described by
x = cos θ ∈ (−1, 1) and three unknowns:

- the meridional circulation U = u_θ sin θ,
- the swirl f = u_φ sin θ,
- the potential Q.

Divergence-freeness gives u_r = U_x. These unknowns solve the **Q-system**:

    (1 − x²) U_x + 2x U + ½ U² = Q,
    (1 − x²) f_xx + U f_x = 0,
    (1 − x²) Q_xxx = −2 f f_x.

The last equation has the first integral L[Q] + f² = λ, with
L[g] = (1−x²)g_xx + 2xg_x − 2g.

**The line form.** Set ξ = artanh x, so the poles x = ±1 move to ξ = ±∞. Then

    z'' = W z,   W = 1 + Q/2,   U = 2(z'/z − tanh ξ),   f' = κ z⁻².

A solution is *global* (it exists on all of (−1, 1)) if and only if z > 0 on R.
Loss of globality means z → 0 at a finite point, where U → ∓∞.

**Data.** A solution is fixed by its six numbers at the equator:
(c₁, c₂, c₃, γ, A, κ). Here c are the Taylor constants of Q,
with P_c = c₁(1−x) + c₂(1+x) + c₃(1−x²); γ = U(0); A = f(0); κ = f_x(0).
The set Ω ⊂ R⁶ of global data is closed. Most of R⁶ is *not* in Ω.

**Poles.** Let q = Q(1) ≥ −2 and μ = √(1 + q/2). The north pole is of one of four types:

| type | U(1) | f(1) | approach |
|---|---|---|---|
| G | −2 + 2μ > −2 | finite | O(E), E = e^{−2ξ} |
| R (0 < μ < ½) | −2 − 2μ | ±∞ | O(E^{1−2μ}) |
| L1 (μ = 0) | −2 | finite | U + 2 ≈ 4/s, logarithmic |
| L2 (μ = 0) | −2 | ±∞ | O(E ln²) |

The south pole follows by the reflection x → −x.

Near the axis, with d = sin θ, a type-G pole has u_θ ≈ (2μ−2)/d,
u_φ ≈ f(1)/d and u_r ~ d^{2μ−2}. This gives the labels:

- Bradshaw–Palmer **II.5** for μ > 2/3,
- **III** for ½ ≤ μ ≤ 2/3,
- "beyond III" for μ < ½ and for the strata R, L1 and L2.

See [docs/conventions.md](docs/conventions.md) for normalisations and
[docs/open_questions.md](docs/open_questions.md) for readings we had to choose.

## Using the core from Python or Jupyter

```python
from swirl_explorer.core.params import Params
from swirl_explorer.core.solver import solve
from swirl_explorer.core.poles import summarize, diagnose_pole
from swirl_explorer.core.edges import gamma_minus

sol = solve(Params(gamma=-1.6806, kappa=1.0))     # c = 0, A = 0
print(summarize(sol).status_line())
pr = sol.profiles(sol.grid())                     # U, f, Q, z, W, h, u_r, u_θ, u_φ, ω, p, …
u = sol.evaluate_velocity([[0.3, 0.2, 0.5]])      # Cartesian velocity, NaN on the axis
print(gamma_minus(Params(kappa=1.0)))             # lens edge: -1.7806…
```

## Tests

```bash
python -m pytest -q
```

The suite contains the brief's seven regression tests:

- the Landau family to 10⁻⁹;
- Numerical observation 4.28 (types G, L1 and R);
- κ_max = 4.1608149;
- div u = 0 and the NS momentum equation by finite differences;
- the symmetries σ₁ and σ₂.

It also has unit tests for blow-up location, velocity evaluation, Σ-maps, path
events and solve speed. A solve takes under 50 ms (about 18 ms), and results are cached.
