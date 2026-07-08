"""Monte Carlo simulation for Merton jump-diffusion model.

Uses the exact compound Poisson representation for efficiency.
"""
from __future__ import annotations

import numpy as np


def simulate_merton_terminal(
    S0: float,
    T: float,
    mu_c_rn: float,
    sigma: float,
    lam_rn: float,
    kappa_rn: float,
    kappa_star: float,
    delta: float,
    N_paths: int = 500_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate terminal stock prices under risk-neutral measure.

    Uses the exact compound Poisson decomposition (no time-stepping):
        ln(S_T) = ln(S_0)
                  + (mu_c_rn - lam_rn*kappa_star - 0.5*sigma^2)*T
                  + sigma*sqrt(T)*Z
                  + sum_{j=1}^{N} Y_j    (compound Poisson)

    Parameters
    ----------
    S0 : float
        Initial spot price.
    T : float
        Time to maturity.
    mu_c_rn : float
        Risk-neutral drift of the continuous component.
    sigma : float
        Diffusion volatility.
    lam_rn : float
        Risk-neutral jump intensity.
    kappa_rn : float
        Risk-neutral mean log-jump.
    kappa_star : float
        exp(kappa_rn + 0.5*delta^2) - 1.
    delta : float
        Std-dev of log-jump.
    N_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of terminal stock prices S_T, shape (N_paths,).
    """
    rng = np.random.default_rng(seed)

    # Continuous (Brownian) part.
    # No-arbitrage: mu_c_rn + lam_rn*kappa_star = rf, so mu_c_rn = rf - lam_rn*kappa_star.
    # The correct log-drift is mu_c_rn - 0.5*sigma^2 = rf - lam_rn*kappa_star - 0.5*sigma^2.
    drift_log = mu_c_rn - 0.5 * sigma**2
    log_S_cont = drift_log * T + sigma * np.sqrt(T) * rng.standard_normal(N_paths)

    # Compound Poisson part
    n_jumps = rng.poisson(lam_rn * T, N_paths)
    log_S_jump = np.zeros(N_paths, dtype=np.float64)

    unique_counts = np.unique(n_jumps[n_jumps > 0])
    for n in unique_counts:
        mask = n_jumps == n
        n_paths_n = int(mask.sum())
        jump_sizes = rng.normal(kappa_rn, delta, (n_paths_n, int(n)))
        log_S_jump[mask] = jump_sizes.sum(axis=1)

    log_ST = np.log(S0) + log_S_cont + log_S_jump
    return np.exp(log_ST)


def simulate_merton_physical(
    lnS0: float,
    T: float,
    mu_c: float,
    sigma: float,
    lam: float,
    kappa: float,
    delta: float,
    N_paths: int = 500_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate log-prices under the physical measure.

    Parameters
    ----------
    lnS0 : float
        Initial log-price.
    T : float
        Time horizon.
    mu_c : float
        Physical drift.
    sigma : float
        Diffusion volatility.
    lam : float
        Physical jump intensity.
    kappa : float
        Physical mean log-jump.
    delta : float
        Std-dev of log-jump.
    N_paths : int
        Number of paths.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Log-prices ln(S_T), shape (N_paths,).
    """
    rng = np.random.default_rng(seed)
    log_S_cont = (mu_c - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(N_paths)
    n_jumps = rng.poisson(lam * T, N_paths)
    log_S_jump = np.zeros(N_paths, dtype=np.float64)

    unique_counts = np.unique(n_jumps[n_jumps > 0])
    for n in unique_counts:
        mask = n_jumps == n
        n_paths_n = int(mask.sum())
        jump_sizes = rng.normal(kappa, delta, (n_paths_n, int(n)))
        log_S_jump[mask] = jump_sizes.sum(axis=1)

    return lnS0 + log_S_cont + log_S_jump


def mc_option_price(
    S_T: np.ndarray,
    K: float,
    T: float,
    rf: float,
    option_type: str = "call",
) -> tuple[float, float]:
    """Compute discounted Monte Carlo option price and standard error.

    Parameters
    ----------
    S_T : np.ndarray
        Terminal stock prices.
    K : float
        Strike price.
    T : float
        Maturity.
    rf : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    tuple of (price, standard_error)
    """
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    disc = np.exp(-rf * T)
    price = disc * float(np.mean(payoff))
    se = disc * float(np.std(payoff, ddof=1)) / np.sqrt(len(S_T))
    return price, se
