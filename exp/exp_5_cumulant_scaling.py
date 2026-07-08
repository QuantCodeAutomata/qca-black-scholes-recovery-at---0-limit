"""Experiment 5: Cumulant Maturity Scaling — Skew ~ T^{-1/2}, ExKurt ~ T^{-1}.

Verifies power-law maturity scaling of standardized cumulants as predicted
by equation (81) of hypothesis H6.
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
from src.cumulants import compute_cumulants, power_law_fit


PARAM_SETS = {
    "A": dict(mu_c=0.10, rf=0.05, sigma=0.20, lam=1.0,  kappa=-0.10, delta=0.15),
    "C": dict(mu_c=0.10, rf=0.05, sigma=0.20, lam=1.0,  kappa=0.0,   delta=0.15),
    "D": dict(mu_c=0.10, rf=0.05, sigma=0.20, lam=1.0,  kappa=+0.10, delta=0.15),
    "E": dict(mu_c=0.10, rf=0.05, sigma=0.20, lam=5.0,  kappa=-0.10, delta=0.15),
}
T_GRID = np.array([0.10, 0.25, 0.50, 1.00, 2.00, 5.00])
EXPONENT_RANGE_SKEW = (-0.55, -0.45)
EXPONENT_RANGE_KURT = (-1.05, -0.95)


def run_exp5() -> dict:
    """Run Experiment 5 and return result dictionary."""
    print("=" * 60)
    print("EXP 5: Cumulant Maturity Scaling")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    all_pass = True
    results_all = {}

    for name, p in PARAM_SETS.items():
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"],
                              p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )

        data = []
        for T in T_GRID:
            k2, k3, k4, skew, exkurt = compute_cumulants(
                lam_rn, kappa_rn, p["delta"], p["sigma"], T
            )
            # Scaling constants (should be constant across T)
            skew_sq = skew * np.sqrt(T)
            ekurt_T = exkurt * T
            data.append((T, k2, k3, k4, skew, exkurt, skew_sq, ekurt_T))

        T_arr = np.array([d[0] for d in data])
        skew_arr = np.array([d[4] for d in data])
        exkurt_arr = np.array([d[5] for d in data])
        skew_sq_arr = np.array([d[6] for d in data])
        ekurt_T_arr = np.array([d[7] for d in data])

        # Power-law fit
        b_s, A_s, r2_s = power_law_fit(T_arr, skew_arr)
        b_k, A_k, r2_k = power_law_fit(T_arr, exkurt_arr)

        # Constancy of scaling product
        skew_sq_std = float(np.std(skew_sq_arr) / abs(np.mean(skew_sq_arr)))
        ekurt_T_std = float(np.std(ekurt_T_arr) / abs(np.mean(ekurt_T_arr)))

        # ExKurt positivity
        exkurt_positive = bool(np.all(exkurt_arr > 0))

        s_ok = EXPONENT_RANGE_SKEW[0] <= b_s <= EXPONENT_RANGE_SKEW[1]
        k_ok = EXPONENT_RANGE_KURT[0] <= b_k <= EXPONENT_RANGE_KURT[1]
        r2_ok = r2_s > 0.999 and r2_k > 0.999
        set_pass = s_ok and k_ok and r2_ok and exkurt_positive
        if not set_pass:
            all_pass = False

        print(f"\nSet {name}:")
        print(f"  Skew exponent b_s = {b_s:.6f}  (expect ≈ -0.5) {'✓' if s_ok else '✗'}")
        print(f"  ExKurt exponent b_k = {b_k:.6f} (expect ≈ -1.0) {'✓' if k_ok else '✗'}")
        print(f"  R² Skew = {r2_s:.8f}, R² ExKurt = {r2_k:.8f}  {'✓' if r2_ok else '✗'}")
        print(f"  Skew*√T CoV = {skew_sq_std:.2e}  (expect ≈ 0)")
        print(f"  ExKurt*T CoV = {ekurt_T_std:.2e}  (expect ≈ 0)")
        print(f"  ExKurt > 0 for all T: {exkurt_positive} {'✓' if exkurt_positive else '✗'}")

        # Tabulate
        print(f"\n  {'T':>5} {'κ₂':>10} {'κ₃':>12} {'κ₄':>12} {'Skew':>10} {'ExKurt':>10} {'Skew*√T':>12} {'ExKurt*T':>12}")
        for row in data:
            print(f"  {row[0]:5.2f} {row[1]:10.6f} {row[2]:12.6f} {row[3]:12.6f} "
                  f"{row[4]:10.6f} {row[5]:10.6f} {row[6]:12.8f} {row[7]:12.8f}")

        results_all[name] = {
            "T_arr": T_arr, "skew_arr": skew_arr, "exkurt_arr": exkurt_arr,
            "b_s": b_s, "b_k": b_k, "r2_s": r2_s, "r2_k": r2_k,
        }

    # --- Figure 3: Log-log scaling plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, res in results_all.items():
        T_arr = res["T_arr"]
        axes[0].loglog(T_arr, np.abs(res["skew_arr"]), "o-", label=f"Set {name}")
        axes[1].loglog(T_arr, res["exkurt_arr"], "o-", label=f"Set {name}")

    # Reference lines
    T_ref = np.array([0.1, 5.0])
    axes[0].loglog(T_ref, 0.5 * T_ref**(-0.5), "k--", linewidth=2, label="T^{-1/2}")
    axes[1].loglog(T_ref, 0.5 * T_ref**(-1.0), "k--", linewidth=2, label="T^{-1}")

    axes[0].set_xlabel("Maturity T", fontsize=12)
    axes[0].set_ylabel("|Skewness(T)|", fontsize=12)
    axes[0].set_title("Exp 5: |Skew| ∝ T^{-1/2}", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("Maturity T", fontsize=12)
    axes[1].set_ylabel("Excess Kurtosis(T)", fontsize=12)
    axes[1].set_title("Exp 5: ExKurt ∝ T^{-1}", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/exp5_cumulant_scaling.png", dpi=150)
    plt.close()

    if all_pass:
        print("\nHypothesis H6 cumulant scaling CONFIRMED ✓")

    # Expose Set A top-level keys for test assertions
    res_A = results_all["A"]
    return {
        "results_all": results_all,
        "PASS": all_pass,
        "skew_exponent": res_A["b_s"],
        "kurt_exponent": res_A["b_k"],
        "r2_skew": res_A["r2_s"],
        "r2_kurt": res_A["r2_k"],
    }


if __name__ == "__main__":
    results = run_exp5()
    print(f"\n[EXP 5] PASS = {results['PASS']}")
