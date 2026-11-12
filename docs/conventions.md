# Conventions (checked against the monograph)

Every item below was checked against the book and is enforced by a test in `tests/`.

## Parameter names — differs from the task brief

| book (used in code) | meaning | brief's name |
|---|---|---|
| `c1, c2, c3` | Taylor constants of Q at the base point: P_c = c₁(1−x)+c₂(1+x)+c₃(1−x²) | c₁, c₂, c₃ |
| `gamma` | γ = U(x_b) = u_θ sin θ at the base point | **Γ** |
| `A` | A = f(x_b) | A |
| `kappa` | κ = f_x(x_b), normalisation κ_x (#1 of Table 2.1) | **γ** ("swirl rate") |

The brief's Γ is the book's γ, and the brief's γ is the book's κ. The code uses the
book's letters throughout, so formulas can be compared line by line with the
monograph. The six-vector order (c₁, c₂, c₃, γ, A, κ) is the same in both.

## Initial data in the line variable

At ξ_b = artanh x_b we normalise z(ξ_b) = 1. Then:

- h(ξ_b) = z′/z = γ/2 + x_b  (from U = 2(h − tanh ξ))
- f′(ξ_b) = (1 − x_b²) κ_x  (Lemma 2.15(2); equal to κ at x_b = 0)
- Q = P_c(x_b) and Q′ = (1 − x_b²) ∂ₓP_c(x_b)

At x_b = 0 these are the brief's z(0)=1, z′(0)=Γ/2, f′(0)=γ. ✔

## The Q equation in ξ

We use the first integral L[Q] + f² = λ, with λ = A² − 2c₁ − 2c₂ − 4c₃ (Prop. 2.20).
It is exactly equivalent to (1−x²)Q_xxx = −2ff_x and is one order lower:

    Q'' = (λ − f² + 2Q)/cosh²ξ − 4 tanh ξ · Q'

Derivation: L[g] = cosh²ξ g″ + 4 sinh ξ cosh ξ g′ − 2g (Lemma 2.18(a)).
`test_q_system_x_form` checks all three x-form equations on the computed solution.

## Velocity and other fields

- u_r = +U_x. This follows from div u = r⁻²(u_r − U_x) (eq. 2.11), and the sign is
  checked by finite-difference divergence (`test_ur_sign_from_divergence`).
- u_θ = U/sin θ and u_φ = f/sin θ.
- Near the poles, U_x = 2(W − h² − sech²ξ)cosh²ξ suffers cancellation. Exponent
  fits therefore drop points where |dU/dξ| < 10⁻⁹.
- Pressure: p = −½Q_xx + U_x − (U²+f²)/(2(1−x²)), eq. (2.15).
- Vorticity: ω = r⁻²(−f_x e_r + sin θ U_xx e_φ), and ω_θ = 0.
- Helicity density: u·ω = r⁻³(−U_x f_x + f U_xx).
- `test_divergence_and_momentum` checks div u = 0 and
  −Δu + (u·∇)u + ∇p = 0 by Cartesian finite differences, to a relative residual
  below 10⁻⁵.

## Symmetries (Lemma 2.8)

- σ₁: (A, κ) → (−A, −κ). It leaves U and Q unchanged and flips f.
- σ₂: (c₁,c₂,c₃,γ,A,κ) → (c₂,c₁,c₃,−γ,A,−κ). It maps U(x) → −U(−x),
  f(x) → f(−x) and Q(x) → Q(−x), and it swaps the north and south pole data.

## Pole readings: "at s = 14" versus extrapolated

The book reads values at a finite window (s = −ln(1−x) = 14, i.e. ξ ≈ 7.35).
`diagnose_pole(..., s_eval=14)` returns both readings:

- `Q_end` is the raw value at the window. It reproduces Numerical observation 4.28
  exactly: −1.2334, −1.99993 and −1.9002.
- `q` is the Aitken-extrapolated limit. For type R the convergence is only
  O(E^{1−2μ}), so the extrapolated μ = 0.2229 differs from the raw-window value
  0.2234. The book's 0.2234 is a window reading, not the limit.

## Globality and the terminal test

A run is **global** when both of these hold:

1. there is no blow-up (|h| < 10⁵) in |ξ| ≤ ξ_max (default 18, i.e. s ≈ 35), and
2. the book's terminal test passes at both ends (Appendix C.2): D = 1 + q/2 ≥ 0 and
   h_side > −√D. We allow a tolerance of 10⁻⁹, so that the exactly recessive
   one-ray Landau solutions (|L| = 1) count as global, as the brief requires.

Without the terminal test the lens tip comes out near 4.27 instead of 4.1608 (Atlas 5.73).
