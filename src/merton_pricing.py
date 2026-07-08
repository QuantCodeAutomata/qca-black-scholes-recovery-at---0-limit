"""Merton (1976) jump-diffusion option pricing module.

Implements the Merton call and put pricing series, series convergence utilities,
and put-call parity verification.
Custom implementation — Context7 found no library equivalent (paper Eq. 57).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from .bs_pricing import bs_call, bs_put


def _nmax(lam_T: float) -> int:
    """Compute truncation order N_max for the Merton series."""
    if lam_T < 1e-15:
        return 1
    return max(50, int(poisson.ppf(1 - 1e-12, max(lam_T, 1e-10))) + 10)


def merton_call(
    S0: float,
    K: float,
    T: float,
    rf: float,
    sigma: float,
    lam_rn: float,
    kappa_rn: float,
    kappa_star: float,
    delta: float,
    N_max: int | None = None,
) -> float:
    """Merton jump-diffusion call price via the infinite series (truncated).

    Correct Merton (1976) formula:
        C_Merton = sum_{n=0}^{N_max} w'_n * C_BS(r_n, sigma_n)

    where w'_n  = Poisson(n; lambda_prime * T),
          lambda_prime = lambda_rn * exp(kappa_rn + 0.5*delta^2) = lambda_rn*(1+kappa_star),
          sigma_n      = sqrt(sigma^2 + n*delta^2/T),
          r_n          = rf - lambda_rn*kappa_star + n*(kappa_rn + 0.5*delta^2)/T.

    The algebraic identity w'_n * exp(-r_n*T) = w_n * exp(-rf*T)
    (where w_n = Poisson(n; lam_rn*T)) guarantees that the series equals
    e^{-rf*T} * E_Q[max(S_T - K, 0)] exactly.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike.
    T : float
        Maturity.
    rf : float
        Risk-free rate.
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
    N_max : int or None
        Series truncation order. If None, determined automatically.

    Returns
    -------
    float
        Merton call price.
    """
    # lambda_prime is the intensity governing the Poisson weights in the Merton formula.
    # lam_prime = lam_rn * (1 + kappa_star) = lam_rn * exp(kappa_rn + 0.5*delta^2)
    lam_prime = lam_rn * (1.0 + kappa_star)
    lam_prime_T = lam_prime * T

    if N_max is None:
        N_max = _nmax(max(lam_prime_T, lam_rn * T))

    log_lam_prime_T = np.log(lam_prime_T + 1e-300)
    total = 0.0

    for n in range(N_max + 1):
        if n == 0:
            log_w = -lam_prime_T
        else:
            log_w = n * log_lam_prime_T - lam_prime_T - gammaln(n + 1)
        w = np.exp(log_w)

        sigma_n = float(np.sqrt(sigma**2 + n * delta**2 / T))
        r_n = float(rf - lam_rn * kappa_star + n * (kappa_rn + 0.5 * delta**2) / T)
        C_n = bs_call(S0, K, T, r_n, sigma_n)
        total += w * C_n

    return float(total)


def merton_put(
    S0: float,
    K: float,
    T: float,
    rf: float,
    sigma: float,
    lam_rn: float,
    kappa_rn: float,
    kappa_star: float,
    delta: float,
    N_max: int | None = None,
) -> float:
    """Merton put price via direct series (uses lam_prime weights, same as merton_call).

    P_Merton = sum_{n=0}^{N_max} w'_n * P_BS(r_n, sigma_n)

    Parameters are identical to merton_call.
    """
    lam_prime = lam_rn * (1.0 + kappa_star)
    lam_prime_T = lam_prime * T
    if N_max is None:
        N_max = _nmax(max(lam_prime_T, lam_rn * T))

    log_lam_prime_T = np.log(lam_prime_T + 1e-300)
    total = 0.0

    for n in range(N_max + 1):
        if n == 0:
            log_w = -lam_prime_T
        else:
            log_w = n * log_lam_prime_T - lam_prime_T - gammaln(n + 1)
        w = np.exp(log_w)
        sigma_n = float(np.sqrt(sigma**2 + n * delta**2 / T))
        r_n = float(rf - lam_rn * kappa_star + n * (kappa_rn + 0.5 * delta**2) / T)
        P_n = bs_put(S0, K, T, r_n, sigma_n)
        total += w * P_n

    return float(total)


def merton_partial_sums(
    S0: float,
    K: float,
    T: float,
    rf: float,
    sigma: float,
    lam_rn: float,
    kappa_rn: float,
    kappa_star: float,
    delta: float,
    N_max_compute: int = 200,
) -> np.ndarray:
    """Return array of partial Merton call sums C_N for N = 0, ..., N_max_compute.

    C_N = sum_{n=0}^{N} w'_n * C_BS(r_n, sigma_n)   [w' uses lam_prime weights]

    Useful for convergence analysis.
    """
    lam_prime = lam_rn * (1.0 + kappa_star)
    lam_prime_T = lam_prime * T
    log_lam_prime_T = np.log(lam_prime_T + 1e-300)
    partial = np.zeros(N_max_compute + 1, dtype=np.float64)
    cumulative = 0.0

    for n in range(N_max_compute + 1):
        if n == 0:
            log_w = -lam_prime_T
        else:
            log_w = n * log_lam_prime_T - lam_prime_T - gammaln(n + 1)
        w = np.exp(log_w)
        sigma_n = float(np.sqrt(sigma**2 + n * delta**2 / T))
        r_n = float(rf - lam_rn * kappa_star + n * (kappa_rn + 0.5 * delta**2) / T)
        C_n = bs_call(S0, K, T, r_n, sigma_n)
        cumulative += w * C_n
        partial[n] = cumulative

    return partial


def poisson_weight_sum(
    lam_rn: float,
    T: float,
    kappa_star: float = 0.0,
    N_max: int | None = None,
) -> float:
    """Sum of Merton Poisson weights sum_{n=0}^{N_max} P(N=n; lam_prime*T).

    Uses lam_prime = lam_rn*(1+kappa_star) consistent with merton_call.
    When kappa_star=0 (default), reduces to the plain lam_rn version.
    """
    lam_prime = lam_rn * (1.0 + kappa_star)
    lam_prime_T = lam_prime * T
    if N_max is None:
        N_max = _nmax(max(lam_prime_T, lam_rn * T))
    log_lam_prime_T = np.log(lam_prime_T + 1e-300)
    s = 0.0
    for n in range(N_max + 1):
        if n == 0:
            log_w = -lam_prime_T
        else:
            log_w = n * log_lam_prime_T - lam_prime_T - gammaln(n + 1)
        s += np.exp(log_w)
    return float(s)
