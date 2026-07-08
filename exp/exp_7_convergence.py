"""Experiment 7: Merton Series Convergence Analysis.

Verifies that the Merton call-price series converges rapidly and that
N_max=50 is sufficient for practical accuracy across Sets A (lam=1) and
E (lam=5).  For each (parameter set, maturity) pair we:

  1. Compute partial sums up to N=200 via merton_partial_sums.
  2. Treat C_exact = partial[200] as the reference price.
  3. Measure rel_err(N) = |C_N - C_exact| / |C_exact| for N in 1..200.
  4. Report the first N achieving rel_err < 1e-4 (N_req_1e4) and < 1e-6
     (N_req_1e6) and compare them to Poisson quantiles at 0.9999 and
     0.999999.
  5. Confirm that N=50 is always sufficient for Set A (rel_err < 1e-4).
  6. Report the Poisson weight sum poisson_weight_sum(lam_rn, T) → 1.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson

from src.esscher import solve_esscher, compute_rn_params
from src.merton_pricing import merton_partial_sums, poisson_weight_sum


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K_ATM = 100.0
T_GRID = np.array([0.25, 0.50, 0.75, 1.00])

PARAM_SETS = {
    "A": dict(S0=100.0, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0, kappa=-0.10, delta=0.15),
    "E": dict(S0=100.0, rf=0.05, mu_c=0.10, sigma=0.20, lam=5.0, kappa=-0.10, delta=0.15),
}

N_MAX_COMPUTE = 200       # reference truncation (C_exact = partial[N_MAX_COMPUTE])
N_MAX_SUFFICIENT = 50     # practical cutoff to verify for Set A
N_CHECK = [1, 2, 5, 10, 20, 50, 100]  # specific N values reported in the table
TOL_COARSE = 1e-4
TOL_FINE = 1e-6


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _first_n_below(partial: np.ndarray, tol: float) -> int:
    """Return the smallest N ≥ 1 such that rel_err(N) < tol.

    Searches the full partial-sums array.  Returns N_MAX_COMPUTE if the
    tolerance is never reached (which should not happen for realistic params).
    """
    C_exact = partial[-1]
    for n in range(1, len(partial)):
        if abs(partial[n] - C_exact) / abs(C_exact) < tol:
            return n
    return len(partial) - 1


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp7() -> dict:
    """Run Experiment 7 and return a result dictionary.

    Returns
    -------
    dict with keys:
        rows  : list of per-(set, T) result dicts
        PASS  : bool — N_max=50 is sufficient (rel_err < 1e-4) for Set A
                at all maturities in T_GRID
    """
    print("=" * 65)
    print("EXP 7: Merton Series Convergence Analysis")
    print("=" * 65)

    os.makedirs("results", exist_ok=True)

    rows = []
    set_a_pass = True

    # One convergence curve per (set, T) combination → 8 curves total
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10
    color_cycle = [cmap(i) for i in range(len(PARAM_SETS) * len(T_GRID))]
    line_styles = ["-", "--"]    # solid for Set A, dashed for Set E
    color_idx = 0

    for s_idx, (set_name, p) in enumerate(PARAM_SETS.items()):
        # --- Solve Esscher once per parameter set ---
        theta = solve_esscher(
            p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )
        print(f"\nSet {set_name}: theta*={theta:.6f},  lam_rn={lam_rn:.4f},  "
              f"kappa_rn={kappa_rn:.4f},  kappa_star={kappa_star:.6f}")
        print(f"  {'T':>5}  {'λT':>6}  {'C_exact':>10}  "
              f"{'N_req(1e-4)':>12}  {'N_req(1e-6)':>12}  "
              f"{'Q(0.9999)':>10}  {'Q(1-1e-6)':>10}  "
              f"{'wt_sum':>10}  {'rel_err(50)':>12}")

        for T in T_GRID:
            lam_T = lam_rn * T

            # --- Compute partial sums up to N=200 ---
            partial = merton_partial_sums(
                S0, K_ATM, T, p["rf"], p["sigma"],
                lam_rn, kappa_rn, kappa_star, p["delta"],
                N_max_compute=N_MAX_COMPUTE,
            )
            C_exact = partial[-1]

            # --- N_req at each tolerance ---
            n_req_4 = _first_n_below(partial, TOL_COARSE)
            n_req_6 = _first_n_below(partial, TOL_FINE)

            # --- Poisson quantile benchmarks ---
            q_9999   = int(poisson.ppf(0.9999,   lam_T))
            q_999999 = int(poisson.ppf(0.999999, lam_T))

            # --- Poisson weight sums (use kappa_star so lam_prime is used) ---
            wt_sum_default = poisson_weight_sum(lam_rn, T, kappa_star=kappa_star)
            wt_sum_50      = poisson_weight_sum(lam_rn, T, kappa_star=kappa_star, N_max=N_MAX_SUFFICIENT)

            # --- Rel error at N=50 ---
            rel_err_50 = abs(partial[N_MAX_SUFFICIENT] - C_exact) / abs(C_exact)

            # Check PASS condition for Set A
            if set_name == "A" and rel_err_50 >= TOL_COARSE:
                set_a_pass = False

            # --- Store row ---
            row = {
                "set":          set_name,
                "T":            T,
                "lam_rn":       lam_rn,
                "lam_rn_T":     lam_T,
                "C_exact":      C_exact,
                "n_req_1e4":    n_req_4,
                "n_req_1e6":    n_req_6,
                "quant_9999":   q_9999,
                "quant_999999": q_999999,
                "wt_sum":       wt_sum_default,
                "wt_sum_50":    wt_sum_50,
                "rel_err_N50":  rel_err_50,
            }
            rows.append(row)

            print(f"  {T:5.2f}  {lam_T:6.3f}  {C_exact:10.6f}  "
                  f"{n_req_4:12d}  {n_req_6:12d}  "
                  f"{q_9999:10d}  {q_999999:10d}  "
                  f"{wt_sum_default:10.8f}  {rel_err_50:12.2e}  "
                  f"{'✓' if rel_err_50 < TOL_COARSE else '✗'}")

            # --- Rel errors at specific N_CHECK values ---
            check_str = "  N-check: " + "  ".join(
                f"N={nc}:{abs(partial[nc] - C_exact) / abs(C_exact):.1e}"
                for nc in N_CHECK
            )
            print(check_str)

            # --- Convergence curve ---
            N_array = np.arange(1, N_MAX_COMPUTE + 1)
            rel_errs = np.array([
                abs(partial[n] - C_exact) / abs(C_exact) for n in N_array
            ])
            rel_errs_plot = np.maximum(rel_errs, 1e-16)  # floor for log scale

            label = f"Set {set_name}, T={T:.2f}  (λT={lam_T:.2f})"
            ax.semilogy(
                N_array, rel_errs_plot,
                linestyle=line_styles[s_idx],
                color=color_cycle[color_idx],
                linewidth=1.6,
                label=label,
            )
            color_idx += 1

    # --- Reference lines and annotations ---
    ax.axvline(N_MAX_SUFFICIENT, color="crimson", linestyle=":",
               linewidth=2.0, label=f"N={N_MAX_SUFFICIENT} practical cutoff")
    ax.axhline(TOL_COARSE, color="dimgray", linestyle="-.",
               linewidth=1.2, label="tol = 1 × 10⁻⁴")
    ax.axhline(TOL_FINE,   color="silver",  linestyle="-.",
               linewidth=1.2, label="tol = 1 × 10⁻⁶")

    ax.set_xlabel("Truncation order N", fontsize=12)
    ax.set_ylabel("Relative error  |C_N − C_exact| / |C_exact|", fontsize=12)
    ax.set_title("Exp 7: Merton Series Convergence (log-linear)\n"
                 "Solid = Set A (λ=1), Dashed = Set E (λ=5)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xlim(0, 120)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp7_convergence.png", dpi=150)
    plt.close()

    # --- Summary ---
    if set_a_pass:
        print(f"\nN_max={N_MAX_SUFFICIENT} is sufficient for Set A at all maturities "
              f"(rel_err < {TOL_COARSE:.0e}) ✓")
    else:
        print(f"\nN_max={N_MAX_SUFFICIENT} is NOT sufficient for Set A at some maturity ✗")

    print(f"\n[EXP 7] PASS = {set_a_pass}")
    return {"rows": rows, "PASS": set_a_pass}


if __name__ == "__main__":
    results = run_exp7()
    print(f"[EXP 7] PASS = {results['PASS']}")
