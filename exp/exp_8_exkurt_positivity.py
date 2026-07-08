"""Experiment 8: Excess Kurtosis Positivity in the Merton Jump-Diffusion Model.

Demonstrates analytically and verifies numerically that ExKurt > 0 for all
valid Merton parameters.

Analytical proof
----------------
For Y ~ N(κ_rn, δ²), the raw fourth moment is:
    E[Y⁴] = κ_rn⁴ + 6·κ_rn²·δ² + 3·δ⁴   ≥  3·δ⁴  >  0

Hence the fourth cumulant κ₄ = λ_rn·T·E[Y⁴] > 0  (since λ_rn, T, E[Y⁴] > 0)
and the second cumulant κ₂ = σ²·T + λ_rn·T·(κ_rn² + δ²) > 0.
Therefore ExKurt = κ₄ / κ₂²  >  0  for every admissible parameter vector.

This experiment:
  1. Sweeps a 10×10×10 grid of (λ_rn, κ_rn, δ) and asserts ExKurt > 0.
  2. Reports the minimum, maximum and location of the minimum.
  3. Performs additional sweeps for Sets A, C, D across λ ∈ {0.5,1,2,5}.
  4. Produces sensitivity plots (a–c) and a κ_rn × δ heatmap.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.esscher import solve_esscher, compute_rn_params
from src.cumulants import compute_cumulants


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
SIGMA = 0.20
T = 1.0

lam_rn_grid  = np.linspace(0.1,  5.0,  10)
kappa_rn_grid = np.linspace(-0.5, 0.5,  10)
delta_grid   = np.linspace(0.05, 0.50, 10)

# Physical parameter sets for additional sweeps
PARAM_SETS = {
    "A": dict(mu_c=0.10, rf=0.05, sigma=0.20, kappa=-0.10, delta=0.15),
    "C": dict(mu_c=0.10, rf=0.05, sigma=0.20, kappa=0.0,   delta=0.15),
    "D": dict(mu_c=0.10, rf=0.05, sigma=0.20, kappa=+0.10, delta=0.15),
}
LAM_SWEEP = [0.5, 1.0, 2.0, 5.0]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp8() -> dict:
    """Run Experiment 8 and return a result dictionary.

    Returns
    -------
    dict with keys:
        min_exkurt   : float — minimum ExKurt found in the 3-D grid sweep
        max_exkurt   : float — maximum ExKurt found in the 3-D grid sweep
        all_positive : bool  — True iff every grid point gives ExKurt > 0
        PASS         : bool  — same as all_positive
    """
    print("=" * 65)
    print("EXP 8: Excess Kurtosis Positivity")
    print("=" * 65)

    # Analytical proof reminder (see module docstring for derivation):
    #   E[Y⁴] = κ_rn⁴ + 6·κ_rn²·δ² + 3·δ⁴  ≥  3·δ⁴  >  0
    #   κ₄ = λ_rn·T·E[Y⁴]  >  0
    #   κ₂ = σ²·T + λ_rn·T·(κ_rn² + δ²)  >  0
    #   ExKurt = κ₄/κ₂²  >  0   □

    os.makedirs("results", exist_ok=True)

    # -----------------------------------------------------------------------
    # 3-D grid sweep: 10 × 10 × 10 = 1 000 parameter combinations
    # -----------------------------------------------------------------------
    n_pts = len(lam_rn_grid) * len(kappa_rn_grid) * len(delta_grid)
    ExKurt_flat = np.empty(n_pts)
    idx = 0
    min_ek = np.inf
    min_loc = (0, 0, 0)   # indices into (lam_rn_grid, kappa_rn_grid, delta_grid)

    for i, lam_rn in enumerate(lam_rn_grid):
        for j, kappa_rn in enumerate(kappa_rn_grid):
            for k, delta in enumerate(delta_grid):
                _, _, _, _, exkurt = compute_cumulants(
                    lam_rn, kappa_rn, delta, SIGMA, T
                )
                ExKurt_flat[idx] = exkurt
                if exkurt < min_ek:
                    min_ek = exkurt
                    min_loc = (i, j, k)
                idx += 1

    all_positive = bool(np.all(ExKurt_flat > 0))
    assert all_positive, (
        f"ExKurt is not strictly positive at some grid point! "
        f"min={ExKurt_flat.min():.6e}"
    )
    min_exkurt = float(ExKurt_flat.min())
    max_exkurt = float(ExKurt_flat.max())

    il, jl, kl = min_loc
    print(f"\n3-D grid sweep ({n_pts} points: λ_rn × κ_rn × δ):")
    print(f"  All ExKurt > 0: {all_positive} ✓")
    print(f"  min ExKurt = {min_exkurt:.6f}  at  "
          f"λ_rn={lam_rn_grid[il]:.3f},  "
          f"κ_rn={kappa_rn_grid[jl]:.3f},  "
          f"δ={delta_grid[kl]:.3f}")
    print(f"  max ExKurt = {max_exkurt:.6f}")

    # Analytical lower bound: when κ_rn=0, E[Y⁴] = 3δ⁴, so
    # ExKurt_min ≥ lam_rn·T·3·δ_min⁴ / (σ²·T + lam_rn·T·δ_min²)²
    delta_min = float(delta_grid[0])
    EY4_min   = 3.0 * delta_min**4
    kappa2_floor = SIGMA**2 * T + lam_rn_grid[0] * T * delta_min**2
    lb = lam_rn_grid[0] * T * EY4_min / kappa2_floor**2
    print(f"\n  Analytical lower bound (κ_rn=0, δ=δ_min={delta_min}):")
    print(f"    E[Y⁴]_min = 3·δ_min⁴ = {EY4_min:.6e}")
    print(f"    ExKurt ≥ {lb:.6e}   (> 0  confirmed)")

    # -----------------------------------------------------------------------
    # Additional sweeps: Sets A, C, D for λ ∈ {0.5, 1.0, 2.0, 5.0}
    # -----------------------------------------------------------------------
    print("\nAdditional sweeps — Sets A, C, D,  T=1.0:")
    print(f"  {'Set':>5}  {'λ_phys':>8}  {'λ_rn':>8}  {'κ_rn':>8}  "
          f"{'κ₂':>10}  {'κ₄':>10}  {'ExKurt':>10}")
    for set_name, p in PARAM_SETS.items():
        for lam in LAM_SWEEP:
            theta = solve_esscher(
                p["mu_c"], p["rf"], p["sigma"], lam, p["kappa"], p["delta"]
            )
            lam_rn, kappa_rn, kappa_star, _ = compute_rn_params(
                theta, p["mu_c"], p["sigma"], lam, p["kappa"], p["delta"]
            )
            k2, _, k4, _, exkurt = compute_cumulants(
                lam_rn, kappa_rn, p["delta"], p["sigma"], T
            )
            print(f"  {set_name:>5}  {lam:8.1f}  {lam_rn:8.4f}  {kappa_rn:8.4f}  "
                  f"{k2:10.6f}  {k4:10.6f}  {exkurt:10.6f}")

    # -----------------------------------------------------------------------
    # Sensitivity plots (a) (b) (c)
    # -----------------------------------------------------------------------
    kappa_sweep = np.linspace(-0.5,  0.5,  200)
    delta_sweep = np.linspace(0.01, 0.50, 200)
    lam_sweep   = np.linspace(0.1,  10.0, 200)

    ek_vs_kappa = np.array([
        compute_cumulants(1.0, kr, 0.15, SIGMA, T)[4] for kr in kappa_sweep
    ])
    ek_vs_delta = np.array([
        compute_cumulants(1.0, 0.0, d, SIGMA, T)[4]   for d  in delta_sweep
    ])
    ek_vs_lam = np.array([
        compute_cumulants(lr, -0.1, 0.15, SIGMA, T)[4] for lr in lam_sweep
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(kappa_sweep, ek_vs_kappa, color="steelblue", linewidth=2)
    axes[0].axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("κ_rn  (mean log-jump)", fontsize=11)
    axes[0].set_ylabel("Excess Kurtosis", fontsize=11)
    axes[0].set_title("(a) ExKurt vs κ_rn\nλ_rn=1, δ=0.15", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(delta_sweep, ek_vs_delta, color="seagreen", linewidth=2)
    axes[1].axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("δ  (log-jump std dev)", fontsize=11)
    axes[1].set_ylabel("Excess Kurtosis", fontsize=11)
    axes[1].set_title("(b) ExKurt vs δ\nλ_rn=1, κ_rn=0", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lam_sweep, ek_vs_lam, color="darkorange", linewidth=2)
    axes[2].axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel("λ_rn  (jump intensity)", fontsize=11)
    axes[2].set_ylabel("Excess Kurtosis", fontsize=11)
    axes[2].set_title("(c) ExKurt vs λ_rn\nκ_rn=−0.1, δ=0.15", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Exp 8: Excess Kurtosis Sensitivity  (σ={SIGMA}, T={T})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("results/exp8_exkurt_sensitivity.png", dpi=150)
    plt.close()
    print("\nSaved: results/exp8_exkurt_sensitivity.png")

    # -----------------------------------------------------------------------
    # Heatmap: fixed λ_rn=1.0, varying κ_rn (x) and δ (y)
    # -----------------------------------------------------------------------
    KK, DD = np.meshgrid(kappa_rn_grid, delta_grid)
    EK_heat = np.zeros_like(KK)
    for i_d, delta in enumerate(delta_grid):
        for j_k, kappa_rn in enumerate(kappa_rn_grid):
            _, _, _, _, ek = compute_cumulants(1.0, kappa_rn, delta, SIGMA, T)
            EK_heat[i_d, j_k] = ek

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    pcm = ax2.pcolormesh(KK, DD, EK_heat, cmap="viridis", shading="auto")
    cbar = plt.colorbar(pcm, ax=ax2)
    cbar.set_label("Excess Kurtosis", fontsize=11)
    ax2.set_xlabel("κ_rn  (mean log-jump)", fontsize=11)
    ax2.set_ylabel("δ  (log-jump std dev)", fontsize=11)
    ax2.set_title(
        f"Exp 8: ExKurt Heatmap  (λ_rn=1.0,  σ={SIGMA},  T={T})",
        fontsize=12,
    )
    # Annotate the minimum
    min_ek_heat = float(EK_heat.min())
    min_pos = np.unravel_index(np.argmin(EK_heat), EK_heat.shape)
    ax2.scatter(
        kappa_rn_grid[min_pos[1]], delta_grid[min_pos[0]],
        color="crimson", marker="*", s=150, zorder=5,
        label=f"min = {min_ek_heat:.4f}",
    )
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/exp8_exkurt_heatmap.png", dpi=150)
    plt.close()
    print("Saved: results/exp8_exkurt_heatmap.png")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    if all_positive:
        print("\nExKurt positivity CONFIRMED for all 1 000 grid points ✓")

    print(f"\n[EXP 8] PASS = {all_positive}")
    return {
        "min_exkurt":   min_exkurt,
        "max_exkurt":   max_exkurt,
        "all_positive": all_positive,
        "PASS":         all_positive,
    }


if __name__ == "__main__":
    results = run_exp8()
    print(f"[EXP 8] PASS = {results['PASS']}")
