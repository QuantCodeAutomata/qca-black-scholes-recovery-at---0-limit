"""Cumulant computations for Merton jump-diffusion model.

Implements analytic cumulant formulas (eqs. 76-80) and standardized
skewness/excess kurtosis.
Custom implementation — Context7 found no library equivalent (paper Eqs. 76-80).
"""
from __future__ import annotations

import numpy as np


def compute_cumulants(
    lam_rn: float,
    kappa_rn: float,
    delta: float,
    sigma: float,
    T: float,
) -> tuple[float, float, float, float, float]:
    """Compute cumulants of log-price increment X = ln(S_T/S_0).

    Formulas (eqs. 76-80):
        kappa_2 = sigma^2*T + lambda_rn*T*(kappa_rn^2 + delta^2)
        kappa_3 = lambda_rn*T * kappa_rn*(kappa_rn^2 + 3*delta^2)
        kappa_4 = lambda_rn*T * (kappa_rn^4 + 6*kappa_rn^2*delta^2 + 3*delta^4)
        Skew = kappa_3 / kappa_2^(3/2)
        ExKurt = kappa_4 / kappa_2^2

    Parameters
    ----------
    lam_rn : float
        Risk-neutral jump intensity.
    kappa_rn : float
        Risk-neutral mean log-jump.
    delta : float
        Std-dev of log-jump.
    sigma : float
        Diffusion volatility.
    T : float
        Time horizon.

    Returns
    -------
    tuple of (kappa2, kappa3, kappa4, skew, exkurt)
    """
    EY2 = kappa_rn**2 + delta**2          # E[Y^2] for Y ~ N(kappa_rn, delta^2)
    EY3 = kappa_rn * (kappa_rn**2 + 3.0 * delta**2)
    EY4 = kappa_rn**4 + 6.0 * kappa_rn**2 * delta**2 + 3.0 * delta**4

    kappa2 = sigma**2 * T + lam_rn * T * EY2
    kappa3 = lam_rn * T * EY3
    kappa4 = lam_rn * T * EY4

    skew = kappa3 / kappa2**1.5
    exkurt = kappa4 / kappa2**2

    return float(kappa2), float(kappa3), float(kappa4), float(skew), float(exkurt)


def power_law_fit(
    T_array: np.ndarray, y_array: np.ndarray
) -> tuple[float, float, float]:
    """Fit a power law y = A * T^b via linear regression in log-log space.

    Parameters
    ----------
    T_array : array-like
        Maturity values.
    y_array : array-like
        Values to fit (must be positive; sign handled externally).

    Returns
    -------
    tuple of (exponent b, prefactor A, R^2)
    """
    log_T = np.log(np.asarray(T_array, dtype=np.float64))
    log_y = np.log(np.abs(np.asarray(y_array, dtype=np.float64)))
    coeffs = np.polyfit(log_T, log_y, 1)
    b, log_A = coeffs
    y_pred = log_A + b * log_T
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(b), float(np.exp(log_A)), float(r2)
