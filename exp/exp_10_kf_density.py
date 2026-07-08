"""Experiment 10: Kolmogorov-Feller Density Verification.

Validates the FFT-based KF density against Monte Carlo simulation under
the physical measure (Set A parameters) using a KS test and moment checks.
Also verifies the lambda→0 limit recovers a Gaussian (Fokker-Planck) density.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

from src.density import kf_density_fft, cdf_interpolator
from src.monte_carlo import simulate_merton_physical


# Set A physical parameters
S0 = 100.0
LN_S0 = np.log(S0)
MU_C = 0.10
SIGMA = 0.20
LAM = 1.0
KAPPA = -0.10
DELTA = 0.15
T_GRID = [0.25, 1.00]
N_GRID = 4096
X_MULT = 12
N_PATHS = 500_000
SEED = 42


def run_exp10() -> dict:
    """Run Experiment 10 and return result dictionary."""
    print("=" * 60)
    print("EXP 10: KF Density Verification via Monte Carlo")
    print("=" * 60)

    ks_stats: dict[float, float] = {}
    mc_samples: dict[float, np.ndarray] = {}
    fft_results: dict[float, tuple] = {}

    # ------------------------------------------------------------------ #
    # Steps 2–4: KF density, MC simulation, and KS test
    # ------------------------------------------------------------------ #
    print("\n--- KS Test and Moment Checks ---")
    for T in T_GRID:
        x_grid, dens, cdf = kf_density_fft(
            LN_S0, T, MU_C, SIGMA, LAM, KAPPA, DELTA,
            N_grid=N_GRID, x_mult=X_MULT,
        )
        fft_results[T] = (x_grid, dens, cdf)

        cdf_fn = cdf_interpolator(x_grid, cdf)
        lnS_mc = simulate_merton_physical(
            LN_S0, T, MU_C, SIGMA, LAM, KAPPA, DELTA,
            N_paths=N_PATHS, seed=SEED,
        )
        mc_samples[T] = lnS_mc

        stat, pval = kstest(lnS_mc, cdf_fn)
        ks_stats[T] = float(stat)

        # Moment check
        mean_fft = float(np.trapz(x_grid * dens, x_grid))
        var_fft  = float(np.trapz((x_grid - mean_fft)**2 * dens, x_grid))
        mean_th  = LN_S0 + (MU_C - 0.5 * SIGMA**2 + LAM * KAPPA) * T
        var_th   = (SIGMA**2 + LAM * (KAPPA**2 + DELTA**2)) * T

        print(f"\n  T = {T:.2f}")
        print(f"    KS statistic : {stat:.6f}  p-value = {pval:.4f}  "
              f"(pass D<0.005: {stat < 0.005})")
        print(f"    Mean — FFT: {mean_fft:.6f}  theory: {mean_th:.6f}  "
              f"err: {abs(mean_fft - mean_th):.2e}")
        print(f"    Var  — FFT: {var_fft:.6f}  theory: {var_th:.6f}  "
              f"err: {abs(var_fft - var_th):.2e}")
        print(f"    MC mean: {np.mean(lnS_mc):.6f}  MC std: {np.std(lnS_mc):.6f}")

    ks_stat_025 = ks_stats[0.25]
    ks_stat_100 = ks_stats[1.00]

    assert ks_stat_025 < 0.005, f"KS stat T=0.25: {ks_stat_025:.6f} >= 0.005"
    assert ks_stat_100 < 0.005, f"KS stat T=1.00: {ks_stat_100:.6f} >= 0.005"
    print("\n  Both KS statistics < 0.005 ✓")

    # ------------------------------------------------------------------ #
    # Step 7: Lambda=0 limit — should recover Gaussian (Fokker-Planck)
    # ------------------------------------------------------------------ #
    print("\n--- Lambda=0 Limit Check (Fokker-Planck vs KF) ---")
    T_fp = 0.25
    x_grid_fp, dens_lam0, _ = kf_density_fft(
        LN_S0, T_fp, MU_C, SIGMA, 1e-10, KAPPA, DELTA,
        N_grid=N_GRID, x_mult=X_MULT,
    )
    mean_fp = LN_S0 + (MU_C - 0.5 * SIGMA**2) * T_fp
    std_fp  = SIGMA * np.sqrt(T_fp)
    gauss_fp = norm.pdf(x_grid_fp, mean_fp, std_fp)
    lam0_max_err = float(np.max(np.abs(dens_lam0 - gauss_fp)))
    print(f"  lam=1e-10 vs Gaussian: max|p_KF - p_Gauss| = {lam0_max_err:.4e}")
    print(f"  (< 1e-4: {lam0_max_err < 1e-4})")

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 8: FP vs KF comparison plot at T=0.25
    # ------------------------------------------------------------------ #
    fig_fp, ax_fp = plt.subplots(figsize=(8, 5))
    x025, dens025, _ = fft_results[0.25]
    ax_fp.plot(x025, dens025, "b-", linewidth=2, label=r"KF density ($\lambda=1.0$)")
    ax_fp.plot(x_grid_fp, gauss_fp, "r--", linewidth=2,
               label=r"Fokker-Planck ($\lambda\to 0$, Gaussian)")
    ax_fp.set_xlabel("ln S")
    ax_fp.set_ylabel("Density p(x, T=0.25)")
    ax_fp.set_title("Exp 10: Fokker-Planck vs Kolmogorov-Feller Density (T=0.25)")
    ax_fp.legend()
    ax_fp.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp10_fp_vs_kf.png", dpi=150)
    plt.close()
    print("\nFigure saved: results/exp10_fp_vs_kf.png")

    # ------------------------------------------------------------------ #
    # Steps 9: MC histogram vs KF density (T=0.25 and T=1.00)
    # ------------------------------------------------------------------ #
    fig_mc, axes_mc = plt.subplots(1, 2, figsize=(14, 5))

    for ax, T in zip(axes_mc, T_GRID):
        x_grid_t, dens_t, _ = fft_results[T]
        lnS_t = mc_samples[T]

        bin_edges = np.linspace(x_grid_t[0], x_grid_t[-1], 120)
        ax.hist(lnS_t, bins=bin_edges, density=True,
                alpha=0.45, color="steelblue", label=f"MC ({N_PATHS:,} paths)")
        ax.plot(x_grid_t, dens_t, "r-", linewidth=2, label="KF FFT density")

        mean_mc = np.mean(lnS_t)
        mean_th = LN_S0 + (MU_C - 0.5 * SIGMA**2 + LAM * KAPPA) * T
        ax.axvline(mean_mc, color="steelblue", linestyle="--", alpha=0.7, label=f"MC mean={mean_mc:.3f}")
        ax.axvline(mean_th, color="red", linestyle=":", alpha=0.8, label=f"Theory mean={mean_th:.3f}")

        ks_t = ks_stats[T]
        ax.set_xlabel("ln S")
        ax.set_ylabel("Density")
        ax.set_title(f"T={T:.2f} | KS stat = {ks_t:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig_mc.suptitle("Exp 10: MC Histogram vs KF FFT Density (Set A, Physical Measure)")
    plt.tight_layout()

    save_path_025 = "results/exp10_kf_density_T025.png"
    save_path_100 = "results/exp10_kf_density_T100.png"

    # Save the two-panel figure with both T-values as the T=0.25 file,
    # then save individual panels
    fig_mc.savefig(save_path_025, dpi=150)
    plt.close()

    # Individual panel for T=0.25
    fig_t025, ax_t025 = plt.subplots(figsize=(8, 5))
    x025, dens025, _ = fft_results[0.25]
    bin_edges025 = np.linspace(x025[0], x025[-1], 120)
    ax_t025.hist(mc_samples[0.25], bins=bin_edges025, density=True,
                 alpha=0.45, color="steelblue", label=f"MC ({N_PATHS:,} paths)")
    ax_t025.plot(x025, dens025, "r-", linewidth=2, label="KF FFT density")
    ax_t025.set_xlabel("ln S"); ax_t025.set_ylabel("Density")
    ax_t025.set_title(f"Exp 10: KF Density T=0.25 | KS = {ks_stat_025:.4f}")
    ax_t025.legend(); ax_t025.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_025, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path_025}")

    # Individual panel for T=1.00
    fig_t100, ax_t100 = plt.subplots(figsize=(8, 5))
    x100, dens100, _ = fft_results[1.00]
    bin_edges100 = np.linspace(x100[0], x100[-1], 120)
    ax_t100.hist(mc_samples[1.00], bins=bin_edges100, density=True,
                 alpha=0.45, color="steelblue", label=f"MC ({N_PATHS:,} paths)")
    ax_t100.plot(x100, dens100, "r-", linewidth=2, label="KF FFT density")
    ax_t100.set_xlabel("ln S"); ax_t100.set_ylabel("Density")
    ax_t100.set_title(f"Exp 10: KF Density T=1.00 | KS = {ks_stat_100:.4f}")
    ax_t100.legend(); ax_t100.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_100, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path_100}")

    PASS = ks_stat_025 < 0.005 and ks_stat_100 < 0.005
    return {
        "ks_stat_025": ks_stat_025,
        "ks_stat_100": ks_stat_100,
        "lam0_max_err": lam0_max_err,
        "PASS": PASS,
    }


if __name__ == "__main__":
    results = run_exp10()
    print(f"\n[EXP 10] PASS = {results['PASS']}")
    print(f"         ks_stat_025  = {results['ks_stat_025']:.6f}")
    print(f"         ks_stat_100  = {results['ks_stat_100']:.6f}")
    print(f"         lam0_max_err = {results['lam0_max_err']:.4e}")
