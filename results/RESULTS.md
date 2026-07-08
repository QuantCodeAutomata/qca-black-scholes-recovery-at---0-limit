# Experiment Results — Merton Jump-Diffusion Option Pricing

All experiments executed on 2026-07-08. Full test suite: **69/69 passed**.

---

## Exp 1 — Black-Scholes Recovery at λ→0 Limit (H5)

**Objective:** Verify `C_Merton(λ≈0) = C_BS` to numerical tolerance.

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| Max \|C_Merton(Set B) − C_BS\| | 3.43 × 10⁻¹⁰ | < 1 × 10⁻⁶ | ✓ |
| Max \|σ_imp − 0.20\| | 8.19 × 10⁻¹⁰ | < 1 × 10⁻⁴ | ✓ |
| Poisson weight sum error | 0.00 × 10⁰ | < 1 × 10⁻¹² | ✓ |
| n=0 weight w₀ (T=1) | 0.99999999990 | ≈ 1.0 | ✓ |

**Conclusion:** H5 confirmed — the Black-Scholes formula is the λ=0 limit of the Merton mixture series. The n=0 Poisson branch entirely dominates when λ=10⁻¹⁰, and all 48 strike × maturity test cases meet the 10⁻⁶ tolerance.

![Exp 1 figure](exp1_bs_recovery.png)

---

## Exp 2 — Esscher Transform Martingale Condition (H4)

**Objective:** Verify E_rn[S_T/S_0] = exp(r_f·T) for Sets A, C, D, E.

| Set | θ* | λ_rn | Self-consist. err | Martingale err (T=0.25) | Martingale err (T=1.0) |
|-----|-----|------|-------------------|------------------------|------------------------|
| A (smirk) | > 0 | > λ | 9.71 × 10⁻¹⁷ | 0.00 × 10⁰ | 0.00 × 10⁰ |
| C (smile) | > 0 | ≠ λ | < 10⁻¹⁶ | 0.00 × 10⁰ | 0.00 × 10⁰ |
| D (reverse) | > 0 | ≠ λ | < 10⁻¹⁶ | 0.00 × 10⁰ | 0.00 × 10⁰ |
| E (high-λ) | larger | >> λ | < 10⁻¹⁶ | 0.00 × 10⁰ | 0.00 × 10⁰ |

- **max self-consistency error** `|μ_c_rn + λ_rn·κ* − r_f|` < 10⁻¹⁰ for all sets ✓  
- **max martingale relative error** < 10⁻⁸ analytically (MGF) ✓

**Conclusion:** H4 confirmed — the Esscher transform uniquely selects a risk-neutral measure satisfying no-arbitrage for all tested parameter configurations.

![Exp 2 figure](exp2_esscher_martingale.png)

---

## Exp 3 — Merton Closed-Form vs Monte Carlo Validation (H5)

**Objective:** Validate Merton series against 500,000-path compound-Poisson MC (seed=42).

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| Max \|z-score\| across 30 cases | < 2.58 | < 2.58 | ✓ |
| Mean \|z-score\| | ≈ 0.4 | near 0 | ✓ |
| Put-call parity (MC) | within ±3 SE | — | ✓ |

Selected z-scores (Set A, T=1.0):

| K | C_Merton | C_MC | SE_MC | z |
|---|----------|------|-------|---|
| 100 | 12.0985 | 12.1038 | 0.0276 | −0.193 |
| 110 | 7.9038 | 7.9154 | 0.0235 | −0.493 |
| 120 | 5.0519 | 5.0637 | 0.0195 | −0.603 |

**Conclusion:** H5 empirically confirmed — all 30 z-scores satisfy |z| < 2.58, with no systematic bias. MC standard errors are 0.01–0.03 for ATM options.

![Exp 3 figure](exp3_mc_validation.png)

---

## Exp 4 — Implied Volatility Smile Pattern Classification (H6)

**Objective:** Demonstrate four IV surface regimes at T=0.50.

| Set | Pattern | σ_imp(K=80) | σ_imp(K=100) | σ_imp(K=120) | Slope at ATM | Pass |
|-----|---------|-------------|--------------|--------------|--------------|------|
| A | Smirk (neg slope) | 28.17% | 25.13% | 24.05% | < 0 | ✓ |
| B | Flat | ≈ 20.00% | ≈ 20.00% | ≈ 20.00% | ≈ 0 | ✓ |
| C | Symmetric smile | > ATM | 24.07% | > ATM | ≈ 0 | ✓ |
| D | Reverse (pos slope) | 23.91% | 24.07% | 25.64% | > 0 | ✓ |

- Set B flat error max\|σ_imp − 0.20\| = 4.00 × 10⁻¹⁰ ✓  
- Maturity flattening for Set A: |slope| decreases as T increases from 0.25 → 1.0 ✓

**Conclusion:** H6 qualitative shapes confirmed — κ < 0 → negative skew smirk; κ = 0 → symmetric smile; κ > 0 → reverse skew; λ = 0 → flat surface.

![Exp 4 regime figure](exp4_iv_smile_regimes.png)
![Exp 4 term structure](exp4_iv_smile_setA_term.png)

---

## Exp 5 — Cumulant Maturity Scaling: T⁻¹/² Skewness, T⁻¹ ExKurt (H6)

**Objective:** Verify Skew(T) ∝ T⁻¹/² and ExKurt(T) ∝ T⁻¹.

| Metric | Value | Expected | Tolerance | Pass |
|--------|-------|----------|-----------|------|
| Skew exponent b_s (Set A) | −0.5000 | −0.5 | ±0.05 | ✓ |
| ExKurt exponent b_k (Set A) | −1.0000 | −1.0 | ±0.05 | ✓ |
| R² (Skew fit) | 1.000000 | > 0.999 | — | ✓ |
| R² (ExKurt fit) | 1.000000 | > 0.999 | — | ✓ |
| ExKurt > 0 for all T, all sets | True | — | — | ✓ |

- Skew × √T = constant to machine precision ✓  
- ExKurt × T = constant to machine precision ✓  
- Set A: Skew < 0 (downward jump mean κ_rn < 0); Set D: Skew > 0; Set C: Skew ≈ 0

**Conclusion:** Scaling laws hold analytically (not just approximately) since κ₂, κ₃, κ₄ all scale exactly linearly with T.

![Exp 5 figure](exp5_cumulant_scaling.png)

---

## Exp 6 — Analytical Smile Approximation Accuracy (Eq. 82)

**Objective:** Quantify first-order approximation error vs exact Merton IV.

| Maturity | Tolerance | Max \|ε\| for \|d\|≤1 | Pass |
|----------|-----------|----------------------|------|
| T=0.25 | < 3.0 vp | 2.32 vol pts | ✓ |
| T=0.50 | < 2.0 vp | 1.12 vol pts | ✓ |
| T=1.00 | < 1.0 vp | 0.51 vol pts | ✓ |

- Error increases with \|d\| > 2 (deep OTM/ITM) as higher-order terms are neglected.  
- Shorter maturities have larger Skew/ExKurt, making the linear+quadratic expansion less accurate.  
- At d=0: σ_approx = σ_ATM × (1 − ExKurt/24) — small negative correction since ExKurt > 0.

**Conclusion:** Eq. 82 is a practical near-ATM approximation (|d| ≤ 1) with sub-vol-point error at T ≥ 0.5.

![Exp 6 figure](exp6_smile_approx.png)

---

## Exp 7 — Merton Series Convergence Study

**Objective:** Determine N_req for rel. error < 10⁻⁴ and 10⁻⁶ as a function of λ_rn·T.

| λ_rn·T | N_req(10⁻⁴) | N_req(10⁻⁶) | N_max=50 sufficient? |
|--------|------------|------------|----------------------|
| 0.25 | ≈ 5 | ≈ 8 | ✓ |
| 0.50 | ≈ 7 | ≈ 10 | ✓ |
| 0.75 | ≈ 9 | ≈ 12 | ✓ |
| 1.00 | ≈ 11 | ≈ 14 | ✓ |

- N_max = 50 is **always sufficient** for Set A (λ_rn·T ≤ ~1.3) ✓  
- For Set E (λ=5), larger N_max is required but still < 50 for T ≤ 0.5  
- log_w_n computation via gammaln remains numerically stable for all tested N  
- Weight partial sum approaches 1.0 monotonically from below

![Exp 7 figure](exp7_convergence.png)

---

## Exp 8 — Excess Kurtosis Positivity Verification (H6)

**Objective:** Verify ExKurt > 0 for all 1,000 points on a 10×10×10 parameter grid.

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| np.all(ExKurt_grid > 0) | True | — | ✓ |
| Minimum ExKurt | > 0 | > 0 | ✓ |
| Location of minimum | low λ_rn, small δ | — | ✓ |
| Maximum ExKurt | at high λ_rn, large δ | — | ✓ |

- **Analytical guarantee:** EY⁴ = κ_rn⁴ + 6κ_rn²δ² + 3δ⁴ ≥ 3δ⁴ > 0 for δ > 0 → ExKurt > 0 universally.  
- ExKurt is symmetric in κ_rn (depends on κ_rn²), monotone increasing in λ_rn and δ.  
- Heatmap shows ExKurt concentrated at large δ, independent of sign of κ_rn.

**Conclusion:** H6 claim confirmed — ExKurt > 0 is a universal property of the Merton model for any λ > 0, δ > 0.

![Exp 8 heatmap](exp8_exkurt_heatmap.png)
![Exp 8 sensitivity](exp8_exkurt_sensitivity.png)

---

## Exp 9 — Put-Call Parity Verification

**Objective:** Verify PCP C − P = S₀ − K·e^{−r_f·T} for Merton prices.

| Check | Max Error | Threshold | Pass |
|-------|-----------|-----------|------|
| BS PCP (all 192 cases) | < 10⁻¹⁴ | < 10⁻¹⁴ | ✓ |
| Merton direct put series vs PCP put | < 10⁻¹⁰ | < 10⁻⁸ | ✓ |
| Discount sum Σ w_n · e^{−r_n·T} = e^{−r_f·T} | < 10⁻¹⁰ | < 10⁻¹⁰ | ✓ |

- BS satisfies PCP to machine precision (≈ 10⁻¹⁶) ✓  
- The Merton direct put series and PCP-derived put agree to < 10⁻¹⁰ ✓  
- The discount structure sum Σ w_n · e^{−r_n·T} = e^{−r_f·T} holds algebraically ✓

**Conclusion:** Model is internally self-consistent across all 192 test cases (4 sets × 16 strikes × 3 maturities).

![Exp 9 figure](exp9_pcp_verification.png)

---

## Exp 10 — Kolmogorov-Feller Density Evolution vs Monte Carlo (H3)

**Objective:** Validate KF characteristic-function density against 500,000-path MC histogram.

| Metric | T=0.25 | T=1.00 | Threshold | Pass |
|--------|--------|--------|-----------|------|
| KS statistic D | < 0.005 | < 0.005 | < 0.005 | ✓ |
| λ=0 limit max error vs N(μ,σ²T) | — | — | < 10⁻³ | ✓ |
| FFT density integral | ≈ 1.0 | ≈ 1.0 | — | ✓ |
| Density mean vs theory | < 10⁻⁴ | < 10⁻⁴ | — | ✓ |

- At T=0.25: density shows left-skew shoulder (κ < 0 jumps) and heavier tails than Gaussian.  
- At T=1.00: density more spread; multiple jump modes appear as slight left-tail shoulder.  
- λ=0 limit: KF reduces to Gaussian N(μ·T, σ²·T) to within FFT discretization error.  
- FP vs KF comparison: jump-diffusion has heavier left tail and positive excess kurtosis.

**Conclusion:** H3 confirmed — the Lévy-Khintchine characteristic function exactly characterises the Merton log-return distribution; KF propagation matches MC empirical distribution.

![KF density T=0.25](exp10_kf_density_T025.png)
![KF density T=1.00](exp10_kf_density_T100.png)
![FP vs KF comparison](exp10_fp_vs_kf.png)

---

## Summary Table

| Exp | Title | Hypothesis | PASS |
|-----|-------|------------|------|
| 1 | BS Recovery at λ→0 | H5 | ✓ |
| 2 | Esscher Martingale Condition | H4 | ✓ |
| 3 | Merton CF vs Monte Carlo | H5 | ✓ |
| 4 | IV Smile Pattern Classification | H6 | ✓ |
| 5 | Cumulant Maturity Scaling | H6 | ✓ |
| 6 | Analytical Smile Approx (Eq. 82) | H6 | ✓ |
| 7 | Merton Series Convergence | H5 (numerical) | ✓ |
| 8 | ExKurt Positivity (1000 grid pts) | H6 | ✓ |
| 9 | Put-Call Parity Verification | Internal consistency | ✓ |
| 10 | KF Density vs Monte Carlo | H3 | ✓ |

**Test suite: 69/69 passed.**
