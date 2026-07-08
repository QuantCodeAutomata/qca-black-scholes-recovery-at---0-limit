"""Black-Scholes option pricing module.

Implements BS call/put pricing, put-call parity, and implied volatility inversion.
Custom implementation — Context7 found no library equivalent for the exact
Merton-compatible BS branch pricing needed here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Volatility (annualised).

    Returns
    -------
    float
        Call price.
    """
    S0, K, T, r, sigma = float(S0), float(K), float(T), float(r), float(sigma)
    denom = sigma * np.sqrt(T)
    if denom < 1e-12:
        return float(max(S0 - K * np.exp(-r * T), 0.0))
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    return float(S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2))


def bs_put(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price via put-call parity.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.

    Returns
    -------
    float
        Put price.
    """
    return bs_call(S0, K, T, r, sigma) - S0 + K * np.exp(-r * T)


def implied_vol(
    C_target: float,
    S0: float,
    K: float,
    T: float,
    rf: float,
    option_type: str = "call",
    tol: float = 1e-10,
) -> float:
    """Compute implied volatility by numerical inversion of the BS formula.

    Parameters
    ----------
    C_target : float
        Observed option price.
    S0 : float
        Spot price.
    K : float
        Strike.
    T : float
        Maturity.
    rf : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.
    tol : float
        Brentq tolerance.

    Returns
    -------
    float
        Implied volatility, or np.nan if inversion fails.
    """
    if option_type == "call":
        intrinsic = max(S0 - K * np.exp(-rf * T), 0.0)
        price_fn = lambda s: bs_call(S0, K, T, rf, s)
    else:
        intrinsic = max(K * np.exp(-rf * T) - S0, 0.0)
        price_fn = lambda s: bs_put(S0, K, T, rf, s)

    if C_target <= intrinsic * (1 + 1e-8):
        return np.nan
    if C_target >= S0 * (1 - 1e-8):
        return np.nan

    def residual(sigma: float) -> float:
        return price_fn(sigma) - C_target

    try:
        return float(brentq(residual, 1e-4, 20.0, xtol=tol, maxiter=500))
    except ValueError:
        return np.nan
