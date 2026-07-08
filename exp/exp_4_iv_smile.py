"""Experiment 4: Implied Volatility Smile Pattern Classification.

Demonstrates four distinct IV regimes: smirk (κ<0), flat (λ=0),
symmetric smile (κ=0), reverse skew (κ>0). Tests hypothesis H6.
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
from src.merton_pricing import merton_call


PARAM_SETS = {
    "A (Smirk)": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,    kappa=-0.10, delta=0.15),
    "B (Flat)":  dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1e-10,  kappa=-0.10, delta=0.15),
    "C (Smile)": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,    kappa=0.0,   delta=0.15),
    "D (Reverse)":dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,   kappa=+0.10, delta=0.15),
}
K_GRID = np.array([60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 140, 150], dtype=np.float64)
T_PRIMARY = 0.50
T_SET_A = [0.25, 0.50, 1.00]   # for maturity term structure of Set A

LINESTYLES = ["-", "--", ":", "-."]
COLORS = ["steelblue", "darkorange", "green", "crimson"]


def compute_smile(p: dict, T: float) -> np.ndarray:
    """Return IV array for all K at given T and parameter set."""
    theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"],
                          p["lam"], p["kappa"], p["delta"])
    lam_rn, kappa_rn, kappa_star, _ = compute_rn_params(
        theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
    )
    ivs = []
    for K in K_GRID:
        C_M = merton_call(p["S0"], K, T, p["rf"], p["sigma"],
                          lam_rn, kappa_rn, kappa_star, p["delta"])
        iv = implied_vol(C_M, p["S0"], K, T, p["rf"])
        ivs.append(iv)
    return np.array(ivs)


def slope_at_atm(iv_arr: np.ndarray, K_arr: np.ndarray, S0: float, dk_frac: float = 0.01) -> float:
    """Finite-difference slope dσ_imp/d(K/S0) at K=S0."""
    K_atm = S0
    K_lo = K_atm * (1 - dk_frac)
    K_hi = K_atm * (1 + dk_frac)

    # Interpolate IV at K_lo and K_hi
    from scipy.interpolate import interp1d
    valid = ~np.isnan(iv_arr)
    if np.sum(valid) < 2:
        return np.nan
    f = interp1d(K_arr[valid] / S0, iv_arr[valid], kind="linear",
                 fill_value="extrapolate")
    return float((f(K_hi / S0) - f(K_lo / S0)) / (2 * dk_frac))


def run_exp4() -> dict:
    """Run Experiment 4 and return result dictionary."""
    print("=" * 60)
    print("EXP 4: Implied Volatility Smile Pattern Classification")
    print("=" * 60)

    smiles = {}
    for name, p in PARAM_SETS.items():
        print(f"\nComputing smile for Set {name} at T={T_PRIMARY}...")
        smiles[name] = compute_smile(p, T_PRIMARY)

    # Slope analysis
    S0 = 100.0
    slopes = {}
    for name, iv_arr in smiles.items():
        p = PARAM_SETS[name]
        slopes[name] = slope_at_atm(iv_arr, K_GRID, p["S0"])
        print(f"  ATM slope dσ/d(K/S0) at T={T_PRIMARY}: {slopes[name]:+.5f}")

    # Helper: find IV at target strike via interpolation
    from scipy.interpolate import interp1d

    def iv_at(iv_arr: np.ndarray, K_target: float) -> float:
        valid = ~np.isnan(iv_arr)
        if valid.sum() < 2:
            return np.nan
        f = interp1d(K_GRID[valid], iv_arr[valid], kind="linear", fill_value="extrapolate")
        return float(f(K_target))

    S0 = 100.0

    # Set A: σ_imp(K=80) > σ_imp(K=100) > σ_imp(K=120) — downward smirk
    ivA = smiles["A (Smirk)"]
    smirk_ok = iv_at(ivA, 80) > iv_at(ivA, 100) > iv_at(ivA, 120)
    print(f"\nSet A σ_imp(80)>{iv_at(ivA,80)*100:.3f}% > σ_imp(100)>{iv_at(ivA,100)*100:.3f}% > σ_imp(120)>{iv_at(ivA,120)*100:.3f}%: {smirk_ok} {'✓' if smirk_ok else '✗'}")

    # Set B: flat within 1e-4
    ivB = smiles["B (Flat)"]
    valid_B = ~np.isnan(ivB)
    flat_err = float(np.max(np.abs(ivB[valid_B] - 0.20))) if valid_B.any() else np.nan
    flat_ok = flat_err < 1e-4
    print(f"Set B max |σ_imp - 0.20| = {flat_err:.2e} {'✓' if flat_ok else '✗'}")

    # Set C: slope near zero at ATM
    smile_ok = abs(slopes["C (Smile)"]) < abs(slopes["A (Smirk)"]) * 0.5
    print(f"Set C ATM slope near zero (|C| < 50% |A|): {smile_ok} {'✓' if smile_ok else '✗'}")

    # Set D: σ_imp(K=80) < σ_imp(K=100) < σ_imp(K=120) — upward reverse skew
    # (Expected outcome: "σ_imp(K=80) < σ_imp(K=100) < σ_imp(K=120)")
    ivD = smiles["D (Reverse)"]
    reverse_ok = iv_at(ivD, 80) < iv_at(ivD, 100) < iv_at(ivD, 120)
    print(f"Set D σ_imp(80)={iv_at(ivD,80)*100:.3f}% < σ_imp(100)={iv_at(ivD,100)*100:.3f}% < σ_imp(120)={iv_at(ivD,120)*100:.3f}%: {reverse_ok} {'✓' if reverse_ok else '✗'}")

    all_pass = smirk_ok and flat_ok and smile_ok and reverse_ok

    # Maturity term structure for Set A
    print("\n--- Set A maturity term structure ---")
    set_A_params = PARAM_SETS["A (Smirk)"]
    theta_A = solve_esscher(set_A_params["mu_c"], set_A_params["rf"],
                            set_A_params["sigma"], set_A_params["lam"],
                            set_A_params["kappa"], set_A_params["delta"])
    smiles_A_mat = {}
    for T in T_SET_A:
        smiles_A_mat[T] = compute_smile(set_A_params, T)
        s = slope_at_atm(smiles_A_mat[T], K_GRID, set_A_params["S0"])
        print(f"  T={T}: ATM slope = {s:+.5f}")

    # --- Figures ---
    os.makedirs("results", exist_ok=True)

    # Figure 1: All 4 smiles
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, iv_arr) in enumerate(smiles.items()):
        valid = ~np.isnan(iv_arr)
        ax.plot(K_GRID[valid] / S0, iv_arr[valid] * 100,
                linestyle=LINESTYLES[i], color=COLORS[i],
                linewidth=2, label=f"Set {name}")
    ax.axhline(20.0, color="gray", linestyle=":", alpha=0.5, label="σ=20%")
    ax.set_xlabel("K/S₀", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title(f"Exp 4: Implied Volatility Smile Regimes (T={T_PRIMARY})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.6)
    plt.tight_layout()
    plt.savefig("results/exp4_iv_smile_regimes.png", dpi=150)
    plt.close()

    # Figure 2: Set A term structure
    fig, ax = plt.subplots(figsize=(9, 5))
    for T, ls in zip(T_SET_A, ["-", "--", ":"]):
        iv_arr = smiles_A_mat[T]
        valid = ~np.isnan(iv_arr)
        ax.plot(K_GRID[valid] / S0, iv_arr[valid] * 100,
                linestyle=ls, linewidth=2, label=f"T={T}")
    ax.axhline(20.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("K/S₀", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title("Exp 4: Set A Smirk — Maturity Term Structure", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp4_iv_smile_setA_term.png", dpi=150)
    plt.close()

    return {
        "smiles": smiles,
        "slopes": slopes,
        "flat_err": flat_err,
        "smirk_ok": smirk_ok,
        "smile_ok": smile_ok,
        "reverse_ok": reverse_ok,
        "flat_ok": flat_ok,
        # Aliased keys expected by tests
        "set_A_smirk_ok": smirk_ok,
        "set_B_flat_ok": flat_ok,
        "set_D_reverse_ok": reverse_ok,
        "PASS": all_pass,
    }


if __name__ == "__main__":
    results = run_exp4()
    print(f"\n[EXP 4] PASS = {results['PASS']}")
