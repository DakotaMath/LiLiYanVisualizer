# Task: build "Swirl Explorer", a Windows desktop research tool for axisymmetric (−1)-homogeneous Navier–Stokes solutions with swirl

You are building an interactive desktop program for a PDE researcher. It explores the 6-parameter family of axisymmetric, (−1)-homogeneous solutions of the stationary Navier–Stokes equations in R³ \ {axis}, as classified in the monograph *"Axisymmetric (−1)-homogeneous solutions of the stationary Navier–Stokes equations with swirl: a complete picture of the solution set"*. The user will put the PDF in `docs/`. Supporting papers may also be there:
- the Li–Yan survey (type definitions: Type 1/2/3);
- Bradshaw–Palmer, arXiv 2606.22291 (Type II.5).

**Read in the monograph before coding:** Ch. 1 (all), Ch. 2 (§2.1–2.6), §4.5 (pole strata, Table 4.1, Prop. 4.19, Cor. 4.24–4.25), Appendix C (numerical methods) and Appendix E (the book's code). If the book's code repository is available, reuse its solvers instead of rewriting them. The formulas below are a summary written from a text extraction of the PDF; **where they disagree with the book, the book wins.** The regression tests in §6 are the way to catch normalisation errors (factors of 2, sign conventions).

---

## 1. The mathematics you need

### Variables
- x = cos θ ∈ (−1, 1). The poles x = ±1 are the north and south rays of the symmetry axis.
- Line variable ξ = artanh x ∈ R. Near the north pole, E := e^(−2ξ) ≈ (1−x)/2.
- Velocity: u = r⁻¹ (u_r(θ) e_r + u_θ(θ) e_θ + u_φ(θ) e_φ).
- Unknowns: U = u_θ sin θ (meridional circulation), f = u_φ sin θ (swirl), and the potential Q.
- Velocity recovery: u_θ = U / sin θ, u_φ = f / sin θ, u_r = dU/dx. The last comes from div u = 0; **verify it numerically, including its sign**.

### The Q-system on (−1, 1)
```
(1 − x²) U_x + 2x U + ½ U² = Q
(1 − x²) f_xx + U f_x      = 0
(1 − x²) Q_xxx             = −2 f f_x
```
Sanity check: with Q ≡ 0 and f ≡ A, the Landau family U = 2(1 − x²)/(L + x), |L| > 1, solves the first equation exactly. This was checked by hand.

### Line / Schrödinger form (preferred for numerics near the poles)
```
z'' = W z,   W = 1 + ½ Q,   U = 2 (z'/z − tanh ξ),   f' = γ z⁻²   (' = d/dξ)
```
- The solution is **global** (defined on all of (−1, 1)) iff z > 0 on all of R.
- Loss of globality shows up as z → 0, i.e. U → ±∞, at a finite point.
- Write Q's third-order equation in ξ yourself, using dx/dξ = 1 − x² = sech² ξ, and check it against the x-form.

### Parameters (state at the equator x = 0)
θ = (c₁, c₂, c₃, Γ, A, γ) ∈ R⁶:
- c are the Taylor constants of Q at the base point. P_c(x) = c₁(1−x) + c₂(1+x) + c₃(1−x²) is the 2-jet of Q at x = 0, so:
  - Q(0) = c₁ + c₂ + c₃
  - Q_x(0) = c₂ − c₁
  - Q_xx(0) = −2c₃
- Γ = U(0), A = f(0), γ = f_x(0) (the book's swirl normalisation; check this).
- Initial data in ξ: z(0) = 1, z'(0) = Γ/2, f(0) = A, f'(0) = γ.

### Pole asymptotics (north pole; south pole by x → −x)
- Let q = Q(1) ≥ −2 and μ = √(1 + q/2).
- Pole types (genuine swirl, γ ≠ 0):

  | Type | Condition | U(1) | f(1) |
  |---|---|---|---|
  | G | dominant | −2 + 2μ > −2 | finite |
  | R | recessive, 0 < μ < ½ | −2 − 2μ ∈ (−3, −2) | ±∞ |
  | L1 | μ = 0 | −2 | finite |
  | L2 | μ = 0 | −2 | infinite |

- **L1 converges only logarithmically:** U + 2 ≈ 4 / ln(1/(1−x)). Report "estimated limit + confidence", not a hard label.
- Velocity rates at a pole. d = distance to the axis on the unit sphere, d² ≈ 2(1 − x). These are the user's own derivations, so treat them as hypotheses the tool should help test (overlay the predicted slopes on log–log plots):

  | Pole | u_θ | u_φ | u_r | Label |
  |---|---|---|---|---|
  | G, μ > 1 | (2μ−2)/d | f(1)/d | O(1) | Type III, II.5 |
  | G, μ = 1, f(1) = 0 | — | — | ~ κ ln(1/d), κ ∝ Q_x(1) | II if κ ≠ 0, I otherwise |
  | G, 2/3 < μ < 1 | (2μ−2)/d | f(1)/d | ~ d^(2μ−2) | II.5 |
  | G, ½ ≤ μ ≤ 2/3 | (2μ−2)/d | f(1)/d | ~ d^(2μ−2) | III |
  | G, μ < ½ | (2μ−2)/d | f(1)/d | ~ d^(2μ−2) | worse than III |
  | R | — | ~ d^(−1−2μ) | — | worse than III |
  | L1 | — | — | ~ −2/(d² ln²(1/d)) | worse than III |
  | L2 | — | ~ ln(1/d)/d | — | worse than III |

- Classification labels per pole: Li–Yan Type 1/2/3; Bradshaw–Palmer Type III (0 < limsup d|u| < ∞) and Type II.5 (V₃ = O(d^(−2/3+ε))); "beyond III".

---

## 2. Required features

### A. Core solver (pure Python/NumPy/SciPy, no GUI imports, fully unit-tested)
- `solve(params, base_point=0.0) -> Solution`.
  - Integrate toward both poles in ξ, with an adaptive high-order method (DOP853, rtol ≈ 1e-11) up to |ξ| ≈ 15–20.
  - Detect blow-up (z → 0) and report where it happens.
- `Solution` exposes, on dense output:
  - x, θ, ξ, s = −ln(1 ∓ x)
  - U, f, Q, z, W, h = z'/z
  - u_r, u_θ, u_φ
  - vorticity components, helicity density, and pressure if derivable (see book §2.1)
- Pole diagnostics for each pole:
  - q, μ, U(±1), f(±1)
  - stratum G/R/L1/L2 with confidence
  - Type labels from §1
  - fitted exponent of each velocity component from a log–log regression near the pole
- Speed target: < 50 ms per solve. Cache results keyed by parameters.
- `evaluate_velocity(X: (N,3) array) -> (N,3)`: Cartesian velocity at arbitrary points, using homogeneity u(X) = |X|⁻¹ u(X/|X|) and interpolation in θ. Return NaN on the axis or at the origin.

### B. 2D ODE plots (matplotlib embedded in Qt, or pyqtgraph)
- A plot panel graphing any selection of U, f, Q, z, W, h, u_r, u_θ, u_φ.
  - Horizontal axis selectable: x, θ, ξ, or s = −ln(1−x) / −ln(1+x).
  - Linear, log, or log–log near-pole modes.
- Overlays of the predicted pole limits and slopes (dashed), e.g. U(1) = −2 + 2μ, U + 2 ≈ 4/s for L1, the slope 2μ−2 for u_r.
- Multiple solutions can be overlaid for comparison (pin the current one, then change parameters).
- Export as PNG/SVG/CSV.

### C. 3D view (PyVista/VTK in Qt via pyvistaqt)
- Render a ball or box region around the origin (excluding a small ball and a thin cylinder around the axis), with the axis drawn as a highlighted line coloured by pole type.
- **Streamlines:**
  - The user places and drags seed points in 3D with a VTK point/sphere widget, or a line/disc seed rake with an adjustable count.
  - Integrate dX/dt = u(X) with RK45, step control, and termination near the axis, the origin and the boundary.
  - Colour by |u|, u_φ, helicity, or the pressure.
- Other 3D options:
  - isosurfaces of |u|, of the swirl f/sin θ and of the helicity
  - arrow glyphs on a sampled grid
  - vortex lines (streamlines of the vorticity)
  - stream surfaces from a seed line
  - a "self-similarity" toggle that shows a streamline and its rescalings
- A transparency/clipping control.

### D. Slice plane
- A plane the user moves through R³ with a VTK plane widget, plus Qt controls: normal direction (azimuth/elevation sliders or presets x-y, x-z, meridional) and offset slider.
- On the plane:
  - a colour map of the normal component u·n (diverging colormap, symmetric limits, red = outflow, blue = inflow, colourbar);
  - the in-plane projected field shown as streamlines of the tangential projection (evenly spaced seeding) and/or line-integral convolution.
- The plane can also be shown in a separate 2D panel (true 2D, not perspective), synced with the 3D one.

### E. Parameter space navigation
- A parameter panel with six sliders and numeric fields (c₁, c₂, c₃, Γ, A, γ), and editable ranges.
- A live status line: global yes/no (and the blow-up point if not), class Z / N / face, both pole strata, μ±, and Type labels.
- **Σ-slice map** (important: most of R⁶ gives non-global solutions):
  - Pick 2 of the 6 parameters, fix the others, and compute a grid in a background thread.
  - Colour each grid point by global/non-global, and then by pole stratum or μ₊, μ₋ contours (2/3, ½, 1).
  - Click the map to jump there, or drag a point across it.
  - Reproduce the lens picture of the book's Figure 1.3 as a sanity check.
- Presets: rest state, Landau (L slider), the test data in §6.

### F. Paths and movies
- Define a path in parameter space by keyframes, either clicked on the Σ-slice map or typed. Interpolation is linear or spline.
- Preview it as an animation, with the 3D view, slice view and 2D plots all updating.
- Mark frames where the solution leaves Σ (non-global) or the pole stratum changes, and report them instead of silently rendering garbage.
- Export an MP4 (imageio-ffmpeg) of a chosen view or a composited multi-panel layout, at a chosen resolution and fps.
- Camera, seeds and plane are held fixed or keyframed too.

### G. Sessions
- Save/load a session as JSON: parameters, seeds, plane, camera, path, and plot selections.

### Later milestone (not first version): effective coordinates
- The book's effective data (c̄, ρ, f₋, D) (Ch. 8) make the pole exponents explicit: μ₋ = √(1 + c̄₁), μ₊ = √(1 + c̄₂).
- Solutions are fixed points of a map T_D, which can be non-unique (folds, Ch. 9).
- Implement the fixed-point solver plus pseudo-arclength continuation for paths, with a fold indicator and a branch selector.
- Plan for it in the architecture.

---

## 3. Stack and structure
- Python 3.11+, PySide6 (or PyQt6), pyvista + pyvistaqt, vtk, numpy, scipy, matplotlib or pyqtgraph, imageio[ffmpeg].
- Package for Windows with PyInstaller (a one-folder build is fine). Include a `build_windows.ps1`.
- Layout:
  ```
  swirl_explorer/
    core/   (solver.py, poles.py, fields.py, sigma_map.py, paths.py)   # no GUI imports
    viz/    (scene3d.py, streamlines.py, slices.py, plots2d.py)
    ui/     (main_window.py, panels/*.py)
    io/     (session.py, movie.py)
  tests/
  docs/
  ```
- Long computations (Σ-maps, movie rendering) run in a worker thread with progress and cancel. The UI must never freeze.
- Code is typed and documented. The README explains the math in a page, with the notation above.

## 4. Numerical cautions
- Never integrate in x up to the poles; use ξ. A solution is near a pole when |ξ| is large; the book integrates to s = −ln(1−x) ≈ 14–30.
- Detect z → 0 robustly: monitor z and stop at a threshold; do not let the solver crash.
- Estimate pole limits by extrapolation, using the known rates: O(E) for type G, logarithmic for L1. Report uncertainty.
- Streamlines: the field is singular on the axis (like 1/d or worse) and at the origin. Use an exclusion radius and stop gracefully.

## 5. Milestones
Deliver each with tests passing and a short demo note.
1. Core solver + pole diagnostics + regression tests (§6).
2. Qt shell with the parameter panel, status line and 2D ODE plots.
3. 3D view with draggable streamline seeds and colouring options.
4. Slice plane with u·n colouring, projected streamlines and the synced 2D panel.
5. Σ-slice map (threaded) with click-to-jump.
6. Paths, preview and MP4 export; sessions.
7. Windows packaging.
8. (Later) Effective coordinates and continuation.

## 6. Regression tests (from the monograph; confirm the normalisation against the book)
1. **Rest state / Landau.** c = 0, f ≡ A (γ = 0):
   - U must equal 2(1 − x²)/(L + x) to 1e-9, with L chosen to match Γ = U(0) = 2/L.
   - Global iff |L| ≥ 1.
2. **Numerical observation 4.28.** Class Z, base x₀ = 0, f(0) = 0, c = 0, swirl rate 1. With Γ = −1.7806 + 0.1 the solution is global, the north pole is type G, and q ≈ −1.2334, μ ≈ 0.6191 (at s = 14).
3. **Edge of the lens.** With Γ ≈ −1.7806 (the lens edge) the north pole is L1:
   - q + 2 ≈ 7e-5
   - U + 2 ≈ 0.323 at s = 14
4. **Recessive edge.** c = (0, −0.9, 0), swirl rate 0.3, Γ ≈ −1.0186: the north pole is type R with μ ≈ 0.2234.
5. **Lens apex.** At c = 0, x₀ = 0, τ = 0, the largest swirl rate with a global solution is ≈ 4.1608 (Example 5.71).
6. **Divergence and momentum.** The reconstructed 3D field satisfies div u ≈ 0 and the stationary NS momentum equation (with the computed pressure) to discretisation accuracy at random points away from the axis.
7. **Symmetries.** x → −x and f → −f map solutions to solutions, with pole data swapped or kept as in the book.

If a test fails because of a normalisation mismatch, find the correct convention in the book, fix the code, and document it in `docs/conventions.md`. Do not loosen the tolerance.

## 7. Working style
- Commit in small steps with clear messages. Keep `core/` usable from a Jupyter notebook.
- When the book is ambiguous, write the question in `docs/open_questions.md` and choose the reading that passes the regression tests.
- Ask the user before large UI-design decisions. Make a quick mock-up first (a screenshot of the layout).
