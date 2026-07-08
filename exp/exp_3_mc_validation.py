"""Experiment 3: Merton Closed-Form Option Price Validation Against Monte Carlo.

Validates the Merton jump-diffusion pricing formula against Monte Carlo
simulation under the risk-neutral measure. Tests hypothesis H5.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bs_pricing import bs_call, bs_put
from src.esscher import solve_esscher, compute_rn_params
from src.merton_pricing import merton_call
from src.monte_carlo import simulate_merton_terminal, mc_option_price


PARAM_SETS = {
    "A": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,  kappa=-0.10, delta=0.15),
    "C": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,  kappa=0.0,   delta=0.15),
    "D": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,  kappa=+0.10, delta=0.15),
}
K_GRID = np.array([80, 90, 100, 110, 120], dtype=np.float64)
T_GRID = np.array([0.25, 1.00])
N_PATHS = 500_000
SEED = 42
Z_THRESHOLD = 2.58   # 99% CI


def run_exp3() -> dict:
    """Run Experiment 3 and return result dictionary."""
    print("=" * 60)
    print("EXP 3: Merton Closed-Form vs Monte Carlo Validation")
    print("=" * 60)

    all_rows = []
    all_z_pass = True
    max_abs_z = 0.0

    for name, p in PARAM_SETS.items():
        print(f"\n--- Set {name} ---")
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"],
                              p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )

        for T in T_GRID:
            # Simulate risk-neutral paths once per (set, T)
            S_T = simulate_merton_terminal(
                p["S0"], T, mu_c_rn, p["sigma"],
                lam_rn, kappa_rn, kappa_star, p["delta"],
                N_paths=N_PATHS, seed=SEED,
            )

            for K in K_GRID:
                C_cf = merton_call(p["S0"], K, T, p["rf"], p["sigma"],
                                   lam_rn, kappa_rn, kappa_star, p["delta"])
                C_mc, SE_mc = mc_option_price(S_T, K, T, p["rf"], "call")
                P_mc, SP_mc = mc_option_price(S_T, K, T, p["rf"], "put")

                z = (C_cf - C_mc) / SE_mc if SE_mc > 0 else 0.0

                # Put-call parity in MC
                pcp_mc = abs(C_mc - P_mc - (p["S0"] - K * np.exp(-p["rf"] * T)))
                pcp_tol = 5 * max(SE_mc, SP_mc)

                z_ok = abs(z) < Z_THRESHOLD
                if not z_ok:
                    all_z_pass = False
                max_abs_z = max(max_abs_z, abs(z))

                row = {
                    "set": name, "K": K, "T": T,
                    "C_Merton": C_cf, "C_MC": C_mc, "SE_MC": SE_mc,
                    "z_score": z, "z_ok": z_ok,
                    "pcp_mc": pcp_mc, "pcp_ok": pcp_mc < pcp_tol,
                }
                all_rows.append(row)
                mark = "✓" if z_ok else "✗"
                print(f"  K={K:5.0f}, T={T:.2f}: C_Merton={C_cf:.4f}, "
                      f"C_MC={C_mc:.4f}±{SE_mc:.4f}, z={z:+.3f} {mark}")

    print(f"\nMax |z-score| = {max_abs_z:.4f}  (must < {Z_THRESHOLD})")
    if all_z_pass:
        print("All 30 z-scores within 99% CI ✓")
    else:
        print("WARNING: Some z-scores exceed threshold!")

    # Compare Merton ATM price to BS at σ=0.20
    print("\n--- ATM Merton vs BS comparison ---")
    for name, p in PARAM_SETS.items():
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"],
                              p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )
        for T in T_GRID:
            C_M = merton_call(100, 100, T, p["rf"], p["sigma"],
                              lam_rn, kappa_rn, kappa_star, p["delta"])
            C_B = bs_call(100, 100, T, p["rf"], p["sigma"])
            print(f"  Set {name}, T={T}: C_Merton={C_M:.4f}, C_BS={C_B:.4f}, diff={C_M-C_B:+.4f}")

    # --- Figure ---
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, len(PARAM_SETS), figsize=(16, 5), sharey=False)

    for ax, (name, p) in zip(axes, PARAM_SETS.items()):
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"],
                              p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )
        T = 1.00
        S_T = simulate_merton_terminal(
            p["S0"], T, mu_c_rn, p["sigma"],
            lam_rn, kappa_rn, kappa_star, p["delta"],
            N_paths=N_PATHS, seed=SEED,
        )
        C_cf_arr = [merton_call(p["S0"], K, T, p["rf"], p["sigma"],
                                lam_rn, kappa_rn, kappa_star, p["delta"])
                    for K in K_GRID]
        C_mc_arr = [mc_option_price(S_T, K, T, p["rf"], "call")[0] for K in K_GRID]
        SE_arr = [mc_option_price(S_T, K, T, p["rf"], "call")[1] for K in K_GRID]

        ax.plot(K_GRID, C_cf_arr, "o-", label="Merton CF", linewidth=2)
        ax.errorbar(K_GRID, C_mc_arr, yerr=[2 * s for s in SE_arr],
                    fmt="s--", label="MC ±2SE", linewidth=1.5, capsize=4)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Call Price")
        ax.set_title(f"Set {name} (T=1.0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 3: Merton Closed-Form vs Monte Carlo")
    plt.tight_layout()
    plt.savefig("results/exp3_mc_validation.png", dpi=150)
    plt.close()

    return {"rows": all_rows, "max_abs_z": max_abs_z, "PASS": all_z_pass}


if __name__ == "__main__":
    results = run_exp3()
    print(f"\n[EXP 3] PASS = {results['PASS']}")
