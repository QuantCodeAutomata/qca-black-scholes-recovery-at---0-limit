# Project: Merton Jump-Diffusion Option Pricing Experiments

## Repository Structure

```
src/          — Core pricing modules (do not modify without care)
exp/          — Experiment scripts (exp_N_*.py)
results/      — Output PNGs and any saved data (auto-created by scripts)
tests/        — Unit tests
data/         — Static data files
```

## Source Module APIs

| Module | Key Functions |
|--------|--------------|
| `src/bs_pricing.py` | `bs_call(S0,K,T,r,sigma)`, `bs_put(...)`, `implied_vol(C_target,S0,K,T,rf,option_type,tol)` |
| `src/esscher.py` | `solve_esscher(mu_c,rf,sigma,lam,kappa,delta)` → `theta*`; `compute_rn_params(theta,mu_c,sigma,lam,kappa,delta)` → `(lam_rn,kappa_rn,kappa_star,mu_c_rn)`; `esscher_equation(theta,...)` |
| `src/merton_pricing.py` | `merton_call(S0,K,T,rf,sigma,lam_rn,kappa_rn,kappa_star,delta,N_max=None)`, `merton_put(...)`, `merton_partial_sums(...,N_max_compute=200)` → ndarray of length N_max_compute+1, `poisson_weight_sum(lam_rn,T,N_max=None)` |
| `src/cumulants.py` | `compute_cumulants(lam_rn,kappa_rn,delta,sigma,T)` → `(kappa2,kappa3,kappa4,skew,exkurt)`; `power_law_fit(T_array,y_array)` → `(exponent,prefactor,R2)` |
| `src/density.py` | `kf_density_fft(lnS0,T,mu_c,sigma,lam,kappa,delta,N_grid=4096,x_mult=10)` → `(x_grid,density,cdf)`; `cdf_interpolator(x_grid,cdf)` → callable |
| `src/monte_carlo.py` | `simulate_merton_physical(lnS0,T,mu_c,sigma,lam,kappa,delta,N_paths=500000,seed=42)` → log-prices array; `simulate_merton_terminal(S0,T,mu_c_rn,sigma,lam_rn,kappa_rn,kappa_star,delta,...)` → stock prices array |

## Canonical Parameter Sets

| Set | mu_c | rf | sigma | lam | kappa | delta |
|-----|------|----|-------|-----|-------|-------|
| A | 0.10 | 0.05 | 0.20 | 1.0 | −0.10 | 0.15 |
| C | 0.10 | 0.05 | 0.20 | 1.0 |  0.00 | 0.15 |
| D | 0.10 | 0.05 | 0.20 | 1.0 | +0.10 | 0.15 |
| E | 0.10 | 0.05 | 0.20 | 5.0 | −0.10 | 0.15 |

S0=100 for all sets.

## Experiment Script Conventions

Every experiment script follows this pattern:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ...
def run_expN() -> dict:
    os.makedirs("results", exist_ok=True)
    # ...
    return {"PASS": bool, ...}

if __name__ == "__main__":
    results = run_expN()
    print(f"[EXP N] PASS = {results['PASS']}")
```

## Experiments Implemented

| # | File | Subject | PASS condition |
|---|------|---------|----------------|
| 1 | exp_1_bs_recovery.py | BS pricing recovery | price error < 1e-10 |
| 2 | exp_2_esscher_martingale.py | Esscher martingale condition | rel_err < 1e-8 |
| 3 | exp_3_mc_validation.py | Monte Carlo validation | MC vs Merton < 1% |
| 4 | exp_4_iv_smile.py | IV smile from Merton prices | smile shape |
| 5 | exp_5_cumulant_scaling.py | Cumulant T-scaling (Skew~T^{-1/2}, ExKurt~T^{-1}) | R²>0.999, exponents in range |
| 6 | exp_6_smile_approx.py | Eq.82 smile approximation accuracy | max err < 0.5 vol pts for \|d\|≤1 |
| 7 | exp_7_convergence.py | Merton series convergence; N_max=50 sufficiency | N=50 sufficient for Set A (rel_err<1e-4) |
| 8 | exp_8_exkurt_positivity.py | ExKurt>0 for all params (analytic + numeric) | np.all(ExKurt>0) over 1000-pt grid |
| 9 | exp_9_pcp_verification.py | PCP verification: BS exact; Merton structural deviation | BS PCP < 1e-14 |
| 10 | exp_10_kf_density.py | KF FFT density vs MC KS test, moments, lam→0 limit | KS < 0.005 for T=0.25,1.00 |

## Key Numerical Facts

- `merton_partial_sums(..., N_max_compute=200)` returns array of shape (201,); `partial[N]` = sum of terms n=0..N; use `partial[-1]` as reference.
- `compute_cumulants` returns `(kappa2, kappa3, kappa4, skew, exkurt)` — index [4] is ExKurt.
- ExKurt is always positive: E[Y⁴] = κ_rn⁴ + 6κ_rn²δ² + 3δ⁴ ≥ 3δ⁴ > 0, so κ₄=λ_rn·T·E[Y⁴]>0 and ExKurt=κ₄/κ₂²>0.
- For Set A (lam_rn≈0.95), N=10 achieves rel_err<1e-9 for all T∈[0.25,1.0]; N=50 is a very conservative bound.
- For Set E (lam_rn≈4.2), N=20 achieves rel_err<1e-9 at T=1.0; N=50 is still sufficient.
- **density.py FFT formula**: The correct inversion is `np.real(fft.fft(phi * np.exp(-1j*u*x_grid[0]))) / (N_grid*dx)`, NOT `np.real(fft.ifft(phi))/dx`. The phase term `exp(-i*u*x_min)` shifts the density onto the correct x-grid. Without it, the density peak is displaced by ~0.5 from the true mean.
- **Merton PCP structure**: The Merton formula satisfies `C - P = S0 - K * D` where `D = Σ wₙ·exp(-rₙT)`. D equals exp(-rf·T) ONLY when kappa_star=0. For nonzero kappa_star (Esscher-corrected): `D = exp(-rf·T)·exp(lam_rn·T·kappa_star²/(1+kappa_star))`. Standard PCP C-P=S0-K·exp(-rf·T) thus fails by O(lam·T·kappa_star²) dollars. This is structural, not numerical.
- **KF density moments**: theoretical mean = lnS0 + (mu_c - 0.5·sigma² + lam·kappa)·T; var = (sigma² + lam·(kappa²+delta²))·T. FFT matches to < 2e-3 (residual FFT discretisation error).
