"""Experiment 1: Black-Scholes Recovery at λ→0 Limit.

Verifies that the Merton jump-diffusion formula exactly reduces to Black-Scholes
when λ→0 (Set B: λ=1e-10), testing hypothesis H5.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bs_pricing import bs_call, implied_vol
from src.esscher import solve_esscher, compute_rn_params
from src.merton_pricing import merton_call, poisson_weight_sum


# Set B — No-Jump Limit parameters
S0 = 100.0
rf = 0.05
mu_c = 0.10
sigma = 0.20
lam_B = 1e-10    # near-zero intensity
kappa = -0.10    # irrelevant when lam≈0 but needed for module
delta = 0.15

K_GRID = np.array([60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 140, 150], dtype=np.float64)
T_GRID = np.array([0.25, 0.50, 1.00])


def run_exp1() -> dict:
    """Run Experiment 1 and return result dictionary."""
    print("=" * 60)
    print("EXP 1: Black-Scholes Recovery at λ→0 Limit")
    print("=" * 60)

    # Solve Esscher for Set B (lambda ≈ 0)
    theta_B = solve_esscher(mu_c=mu_c, rf=rf, sigma=sigma,
                            lam=lam_B, kappa=kappa, delta=delta)
    lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
        theta_B, mu_c, sigma, lam_B, kappa, delta
    )
    print(f"theta* Set B = {theta_B:.8f}  (expect ≈ -1.25)")
    print(f"lambda_rn    = {lam_rn:.4e}  (expect ≈ 1e-10)")

    # Poisson weight sum verification (uses lam_prime = lam_rn*(1+kappa_star))
    lam_prime = lam_rn * (1.0 + kappa_star)
    ws = poisson_weight_sum(lam_rn, T=1.0, kappa_star=kappa_star)
    print(f"lam_prime (T=1)          = {lam_prime:.4e}  (≈ lam_rn since lambda≈0)")
    print(f"Poisson weight sum (T=1) = {ws:.15f}  |1 - ws| = {abs(1-ws):.2e}")
    assert abs(1.0 - ws) < 1e-12, f"Poisson weight sum error: {abs(1-ws)}"

    n0_weight = np.exp(-lam_prime * 1.0)
    print(f"n=0 weight w_0 (T=1) = {n0_weight:.15f}  (expect ≈ 1.0)")

    # Compute discrepancy for all (K, T)
    discrepancy = np.zeros((len(K_GRID), len(T_GRID)))
    iv_surface = np.zeros((len(K_GRID), len(T_GRID)))

    max_ws_err = 0.0
    for j, T in enumerate(T_GRID):
        ws_T = poisson_weight_sum(lam_rn, T=T, kappa_star=kappa_star)
        max_ws_err = max(max_ws_err, abs(1.0 - ws_T))
        assert abs(1.0 - ws_T) < 1e-12, f"Weight sum error at T={T}: {abs(1-ws_T)}"
        for i, K in enumerate(K_GRID):
            C_merton = merton_call(S0, K, T, rf, sigma, lam_rn, kappa_rn, kappa_star, delta)
            C_bs = bs_call(S0, K, T, rf, sigma)
            discrepancy[i, j] = abs(C_merton - C_bs)

            # Implied vol check
            iv = implied_vol(C_merton, S0, K, T, rf)
            iv_surface[i, j] = iv

    max_disc = float(np.nanmax(discrepancy))
    print(f"\nMax |C_Merton(Set B) - C_BS| = {max_disc:.4e}  (must be < 1e-6)")
    assert max_disc < 1e-6, f"Max discrepancy {max_disc} exceeds 1e-6!"

    iv_errors = np.abs(iv_surface - 0.20)
    max_iv_err = float(np.nanmax(iv_errors[~np.isnan(iv_errors)]))
    print(f"Max |sigma_imp - 0.20|       = {max_iv_err:.4e}  (must be < 1e-4)")
    assert max_iv_err < 1e-4, f"Max IV error {max_iv_err} exceeds 1e-4!"

    print("\nHypothesis H5 CONFIRMED: Merton(λ=0) = BS to within 1e-6 ✓")

    # --- Figure ---
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Discrepancy vs K for each T
    for j, T in enumerate(T_GRID):
        axes[0].semilogy(K_GRID, discrepancy[:, j] + 1e-20,
                         label=f"T={T}", linewidth=2)
    axes[0].axhline(1e-6, color="red", linestyle="--", label="Tolerance 1e-6")
    axes[0].set_xlabel("Strike K")
    axes[0].set_ylabel("|C_Merton(λ≈0) - C_BS|")
    axes[0].set_title("Exp 1: Merton(λ≈0) vs BS Discrepancy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: IV surface (should be flat at 0.20)
    for j, T in enumerate(T_GRID):
        valid = ~np.isnan(iv_surface[:, j])
        axes[1].plot(K_GRID[valid] / S0, iv_surface[valid, j] * 100,
                     label=f"T={T}", linewidth=2)
    axes[1].axhline(20.0, color="black", linestyle="--", label="σ=20%")
    axes[1].set_xlabel("K/S₀")
    axes[1].set_ylabel("Implied Volatility (%)")
    axes[1].set_title("Exp 1: IV Surface (Set B, λ≈0)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/exp1_bs_recovery.png", dpi=150)
    plt.close()

    return {
        "theta_B": theta_B,
        "lam_rn_B": lam_rn,
        "max_discrepancy": max_disc,
        "max_iv_error": max_iv_err,
        "weight_sum_ok": max_ws_err < 1e-12,
        "discrepancy": discrepancy,
        "iv_surface": iv_surface,
        "PASS": max_disc < 1e-6 and max_iv_err < 1e-4,
    }


if __name__ == "__main__":
    results = run_exp1()
    print(f"\n[EXP 1] PASS = {results['PASS']}")
