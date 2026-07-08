# Merton Jump-Diffusion Option Pricing

A quantitative finance project implementing and validating the Merton (1976) jump-diffusion option pricing model, including Black-Scholes recovery, Esscher transform risk-neutral measure changes, and implied volatility surface analysis.

## Research Hypotheses Tested

| Hypothesis | Description | Experiment |
|------------|-------------|------------|
| H3 | KF equation propagates jump-diffusion density correctly | Exp 10 |
| H4 | Esscher transform satisfies no-arbitrage martingale condition | Exp 2 |
| H5 | Merton series reduces to BS at λ→0; CF formula matches MC | Exp 1, 3 |
| H6 | IV smile shapes driven by κ_rn; cumulant scaling T⁻¹/², T⁻¹ | Exp 4, 5, 6, 8 |

## Repository Structure

```
project/
├── src/                    # Core model implementations
│   ├── bs_pricing.py       # Black-Scholes call/put pricing and IV inversion
│   ├── esscher.py          # Esscher transform: solve θ*, compute RN parameters
│   ├── merton_pricing.py   # Merton jump-diffusion series pricing
│   ├── cumulants.py        # Cumulant computation and power-law fitting
│   ├── monte_carlo.py      # Compound-Poisson Monte Carlo simulation
│   └── density.py          # FFT-based Lévy-Khintchine density computation
├── exp/                    # Experiment scripts
│   ├── exp_1_bs_recovery.py         # H5: BS = Merton(λ=0)
│   ├── exp_2_esscher_martingale.py  # H4: Esscher martingale condition
│   ├── exp_3_mc_validation.py       # H5: Merton CF vs Monte Carlo
│   ├── exp_4_iv_smile.py            # H6: IV smile regimes
│   ├── exp_5_cumulant_scaling.py    # H6: T⁻¹/² skew, T⁻¹ kurtosis
│   ├── exp_6_smile_approx.py        # H6: Eq. 82 approximation accuracy
│   ├── exp_7_convergence.py         # Series convergence vs N_max
│   ├── exp_8_exkurt_positivity.py   # H6: ExKurt > 0 universally
│   ├── exp_9_pcp_verification.py    # Put-call parity internal check
│   └── exp_10_kf_density.py         # H3: KF density vs MC histogram
├── tests/
│   ├── test_src_modules.py          # Unit tests for src modules
│   └── test_experiments.py          # Integration tests for all experiments
├── results/
│   ├── RESULTS.md                   # Summary of all experiment outcomes
│   └── *.png                        # Generated figures
├── data/                            # Data directory (synthetic data only)
├── AGENTS.md                        # Repository knowledge for AI agents
└── README.md
```

## Installation

```bash
pip install numpy scipy matplotlib
# All other required libraries pre-installed in the project environment
```

## Running Experiments

```bash
# Run a single experiment
python -m exp.exp_1_bs_recovery
python -m exp.exp_2_esscher_martingale
# ... etc.

# Run full test suite
pytest tests/ -v

# Run all experiments
for exp in exp_1_bs_recovery exp_2_esscher_martingale exp_3_mc_validation \
           exp_4_iv_smile exp_5_cumulant_scaling exp_6_smile_approx \
           exp_7_convergence exp_8_exkurt_positivity exp_9_pcp_verification \
           exp_10_kf_density; do
    python -m exp.$exp
done
```

## Parameter Sets

| Set | Description | λ | κ | δ | σ | r_f | μ_c |
|-----|-------------|---|---|---|---|-----|-----|
| A | Equity smirk | 1.0 | −0.10 | 0.15 | 0.20 | 0.05 | 0.10 |
| B | No-jump (flat) | 1×10⁻¹⁰ | −0.10 | 0.15 | 0.20 | 0.05 | 0.10 |
| C | Symmetric smile | 1.0 | 0.00 | 0.15 | 0.20 | 0.05 | 0.10 |
| D | Reverse skew | 1.0 | +0.10 | 0.15 | 0.20 | 0.05 | 0.10 |
| E | High intensity | 5.0 | −0.10 | 0.15 | 0.20 | 0.05 | 0.10 |

## Key Results

- **69/69 tests pass**
- **H3 ✓** — KS statistic D < 0.005 at both T=0.25 and T=1.0
- **H4 ✓** — Martingale error < 10⁻⁸; self-consistency < 10⁻¹⁰
- **H5 ✓** — Max|C_Merton(λ≈0) − C_BS| = 3.4×10⁻¹⁰; all z-scores < 2.58
- **H6 ✓** — Skew exponent = −0.500, ExKurt exponent = −1.000, R² = 1.000

See [results/RESULTS.md](results/RESULTS.md) for full tables and figures.

## Methodology Notes

### Black-Scholes
Standard log-normal formula with scipy.stats.norm.cdf for N(·). Edge case handled for σ√T < 10⁻¹².

### Esscher Transform
Solves the transcendental equation F(θ) = μ_c + θσ² + λ·exp(θκ + ½θ²δ²)·[exp(κ + θδ² + ½δ²) − 1] − r_f = 0 via scipy.optimize.brentq with tolerance 10⁻¹².

### Merton Series
Weighted sum of BS prices with Poisson weights. Log-space computation for numerical stability. N_max = max(50, poisson_quantile(1−10⁻¹², λ_rn·T) + 10).

### Monte Carlo
Exact compound-Poisson simulation: total jump count N_i ~ Poisson(λ_rn·T), jump sizes ~ N(κ_rn, δ²). No time-stepping required for constant parameters.

### Cumulant Scaling
κ₂(T) = σ²T + λ_rn·T·E[Y²], κ₃(T) = λ_rn·T·E[Y³], κ₄(T) = λ_rn·T·E[Y⁴]. All scale exactly as T → Skew ∝ T⁻¹/², ExKurt ∝ T⁻¹.
