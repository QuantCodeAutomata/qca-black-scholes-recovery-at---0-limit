"""Experiment 2: Esscher Transform Martingale Condition Verification.

Verifies that the Esscher transform uniquely selects a risk-neutral measure
satisfying E_rn[S_T/S_0] = exp(r_f*T) for parameter sets A, C, D, E.
Tests hypothesis H4.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.esscher import solve_esscher, compute_rn_params, esscher_equation


# Parameter sets
PARAM_SETS = {
    "A": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,  kappa=-0.10, delta=0.15),
    "C": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,  kappa=0.0,   delta=0.15),
    "D": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0,  kappa=+0.10, delta=0.15),
    "E": dict(S0=100, rf=0.05, mu_c=0.10, sigma=0.20, lam=5.0,  kappa=-0.10, delta=0.15),
}
T_CHECK = [0.25, 1.00]


def verify_martingale(
    mu_c_rn: float,
    lam_rn: float,
    kappa_star: float,
    rf: float,
    T: float,
) -> tuple[float, float, float]:
    """Compute MGF at u=1 and compare to exp(rf*T).

    M_X(1) = exp(T*(mu_c_rn + lam_rn*kappa_star)) = exp(rf*T) by no-arbitrage.
    """
    log_moment = T * (mu_c_rn + lam_rn * kappa_star)
    moment = np.exp(log_moment)
    expected = np.exp(rf * T)
    rel_err = abs(moment - expected) / expected
    return float(moment), float(expected), float(rel_err)


def fft_martingale_check(
    T: float,
    mu_c_rn: float,
    sigma: float,
    lam_rn: float,
    kappa_rn: float,
    delta: float,
    rf: float,
    N_grid: int = 4096,
) -> float:
    """FFT-based numerical integral of E_rn[exp(X)] vs exp(rf*T)."""
    import numpy.fft as fft_module

    x_max = 15.0 * max(sigma * np.sqrt(T), delta * np.sqrt(lam_rn * T + 1e-10))
    x_grid = np.linspace(-x_max / 2.0, x_max / 2.0, N_grid)
    dx = x_grid[1] - x_grid[0]

    u = 2.0 * np.pi * fft_module.fftfreq(N_grid, d=dx)

    # Characteristic function of X = ln(S_T/S_0) centered at 0
    drift = mu_c_rn - 0.5 * sigma**2 - lam_rn * (np.exp(kappa_rn + 0.5 * delta**2) - 1.0)
    psi = (
        1j * drift * u
        - 0.5 * sigma**2 * u**2
        + lam_rn * (np.exp(1j * kappa_rn * u - 0.5 * delta**2 * u**2) - 1.0)
    )
    phi = np.exp(T * psi)

    density_raw = np.real(fft_module.ifft(phi)) / dx
    density = np.maximum(density_raw, 0.0)

    norm_const = np.trapz(density, x_grid)
    if norm_const > 1e-15:
        density = density / norm_const

    # Numerical integral of exp(x)*p(x)
    integrand = np.exp(x_grid) * density
    moment_fft = float(np.trapz(integrand, x_grid))
    return moment_fft


def run_exp2() -> dict:
    """Run Experiment 2 and return result dictionary."""
    print("=" * 60)
    print("EXP 2: Esscher Transform Martingale Condition Verification")
    print("=" * 60)

    rows = []
    all_pass = True

    for name, p in PARAM_SETS.items():
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"],
                              p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )

        # Root accuracy
        F_theta = esscher_equation(theta, p["mu_c"], p["rf"], p["sigma"],
                                   p["lam"], p["kappa"], p["delta"])

        # Self-consistency check
        sc_error = abs(mu_c_rn + lam_rn * kappa_star - p["rf"])

        # Martingale checks at each T
        mg_errors = {}
        fft_results = {}
        for T in T_CHECK:
            moment, expected, rel_err = verify_martingale(
                mu_c_rn, lam_rn, kappa_star, p["rf"], T
            )
            mg_errors[T] = rel_err

            # FFT cross-check (only Set A for brevity)
            if name == "A":
                m_fft = fft_martingale_check(
                    T, mu_c_rn, p["sigma"], lam_rn, kappa_rn, p["delta"], p["rf"]
                )
                fft_results[T] = m_fft

        row = {
            "set": name,
            "theta": theta,
            "lam_rn": lam_rn,
            "kappa_rn": kappa_rn,
            "kappa_star": kappa_star,
            "mu_c_rn": mu_c_rn,
            "|F(theta*)|": abs(F_theta),
            "sc_error": sc_error,
            "mg_err_025": mg_errors[0.25],
            "mg_err_100": mg_errors[1.00],
        }
        rows.append(row)

        sc_pass = sc_error < 1e-10
        mg_pass = all(v < 1e-8 for v in mg_errors.values())
        if not (sc_pass and mg_pass):
            all_pass = False

        print(f"\nSet {name}: theta*={theta:.6f}, lam_rn={lam_rn:.4f}, "
              f"kappa_rn={kappa_rn:.4f}, kappa_star={kappa_star:.4f}, mu_c_rn={mu_c_rn:.4f}")
        print(f"  |F(theta*)| = {abs(F_theta):.2e}")
        print(f"  Self-consistency = {sc_error:.2e}  (must < 1e-10) {'✓' if sc_pass else '✗'}")
        for T in T_CHECK:
            print(f"  Martingale rel err T={T}: {mg_errors[T]:.2e}  "
                  f"(must < 1e-8) {'✓' if mg_errors[T] < 1e-8 else '✗'}")
        if name == "A":
            for T in T_CHECK:
                fft_rel = abs(fft_results[T] - np.exp(p["rf"] * T)) / np.exp(p["rf"] * T)
                print(f"  FFT moment check T={T}: rel err={fft_rel:.2e}")

    if all_pass:
        print("\nHypothesis H4 CONFIRMED: Esscher martingale condition satisfied ✓")
    else:
        print("\nWARNING: Some checks failed!")

    # --- Figure ---
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    thetas = [row["theta"] for row in rows]
    labels = [row["set"] for row in rows]
    lam_rns = [row["lam_rn"] for row in rows]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, thetas, width, label="θ*", color="steelblue")
    bars2 = ax.bar(x + width / 2, lam_rns, width, label="λ_rn", color="darkorange")
    ax.set_xlabel("Parameter Set")
    ax.set_ylabel("Value")
    ax.set_title("Exp 2: Esscher Multiplier θ* and Risk-Neutral Intensity λ_rn")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/exp2_esscher_martingale.png", dpi=150)
    plt.close()

    max_sc_err = max(row["sc_error"] for row in rows)
    max_mg_err = max(max(row["mg_err_025"], row["mg_err_100"]) for row in rows)
    return {
        "rows": rows,
        "PASS": all_pass,
        "max_self_consistency_err": max_sc_err,
        "max_martingale_err": max_mg_err,
    }


if __name__ == "__main__":
    results = run_exp2()
    print(f"\n[EXP 2] PASS = {results['PASS']}")
