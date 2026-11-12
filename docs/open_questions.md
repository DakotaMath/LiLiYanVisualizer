# Open questions and readings chosen

## 1. Bradshaw–Palmer's Type II.5 example uses d ≈ 1+cos θ (it should be d² ≈ 2(1+cos θ))

Bradshaw–Palmer §3.5 write v_r = V_θ′(x) ≈ a₁α₀(1+cos θ)^{α₀−1}. They then say that
"1+cos θ is equivalent asymptotically to |x_h|", and conclude Type II.5 whenever
α₀ − 1 > −2/3. On the unit sphere, however, |x_h| = sin θ and 1 + cos θ ≈ sin²θ/2.
So v_r ~ |x_h|^{2(α₀−1)}.

With α₀ = μ (their α₀ = 1 − V_θ(−1)/2 and the book's U(−1) = 2 − 2μ), the condition
V₃ = O(|x_h|^{−2/3+}) becomes **2μ − 2 > −2/3, i.e. μ > 2/3**. Their wording gives μ > 1/3.

**Reading chosen:** μ > 2/3. This matches the brief's table ("G, 2/3 < μ < 1 → II.5").
The tool also shows the fitted slope of u_r against d, so the user can check either reading.

## 2. Classifying logarithmic strata in a finite window

A window of length ξ_max cannot distinguish L1 from type G with μ ≲ 1/ξ_max. The
book calls μ = 0.006 at s = 14 "L1". We report L1 when μ < 0.5/ξ_end, or when
μ < 0.05 and h > 3μ (h ~ 1/ξ in L1). The confidence is 1 − μ/h, and the note shows
q + 2, ξ·h (→ 1 for L1, → 0 for L2), and U + 2 against 4/s.

## 3. Rate tables are hypotheses

The u_r rates in the brief's §1 table are the user's derivations. The tool overlays
the predicted slopes and reports the fitted slopes. For type G with 1/2 < μ < 1 the
fit agrees with 2μ − 2 to about 10⁻² (`test_fitted_exponents_type_G`).

The cases μ = 1 with f(1) = 0 (log u_r, κ ∝ Q_x(1)) and the L1 rate
u_r ~ −2/(d² ln²(1/d)) are untested so far. They are shown as predictions only.

## 4. Rate of u_r at a type-R pole (not in the brief's table)

Derivation: Table 4.1 gives R = Q − q = O(E^{1−2μ}). Then h + μ = O(E^{1−2μ}),
U′ = 2h′ = O(E^{1−2μ}), and u_r = U_x = U′cosh²ξ ~ E^{−2μ} ~ **d^{−4μ}**.

Evidence: at the R edge of Numerical observation 4.28 (μ = 0.2229) the fitted slope
of |u_r| against d is −0.90, while −4μ = −0.89.

The tool reports this as a *derived candidate*. It is not a proved rate, and the
leading constant could vanish. Since −4μ > −2 for μ < ½, u_r stays less singular
than d^{−2}. It is more singular than 1/d once μ > ¼.
