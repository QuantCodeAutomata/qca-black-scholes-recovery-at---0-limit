"""Esscher transform module.

Solves the Esscher multiplier theta* for the Merton jump-diffusion model
and computes risk-neutral parameters.
Custom implementation — Context7 found no library equivalent (paper Eq. F(theta)=0).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def esscher_equation(
    theta: float,
    mu_c: float,
    rf: float,
    sigma: float,
    lam: float,
    kappa: float,
    delta: float,
) -> float:
    """Transcendental equation F(theta) = 0 whose root gives the Esscher multiplier.

    F(theta) = mu_c + theta*sigma^2
               + lambda*exp(theta*kappa + 0.5*theta^2*delta^2)
                       * [exp(kappa + theta*delta^2 + 0.5*delta^2) - 1]
               - r_f

    Parameters
    ----------
    theta : float
        Candidate Esscher multiplier.
    mu_c : float
        Physical drift of the continuous component.
    rf : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Physical jump intensity.
    kappa : float
        Mean log-jump size.
    delta : float
        Std-dev of log-jump size.

    Returns
    -------
    float
        Value of F(theta).
    """
    log_amp = theta * kappa + 0.5 * theta**2 * delta**2
    if log_amp > 700:
        return float(np.sign(theta)) * np.inf
    if log_amp < -700:
        amp = 0.0
    else:
        amp = np.exp(log_amp)
    jump_inner = np.exp(kappa + theta * delta**2 + 0.5 * delta**2) - 1.0
    return mu_c + theta * sigma**2 + lam * amp * jump_inner - rf


def solve_esscher(
    mu_c: float,
    rf: float,
    sigma: float,
    lam: float,
    kappa: float,
    delta: float,
    bracket: tuple[float, float] = (-50.0, 50.0),
) -> float:
    """Find the Esscher multiplier theta* using Brent's method.

    Parameters
    ----------
    mu_c : float
        Physical drift.
    rf : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Physical jump intensity.
    kappa : float
        Mean log-jump size.
    delta : float
        Std-dev of log-jump size.
    bracket : tuple of float
        Initial bracket (low, high) for Brent's method.

    Returns
    -------
    float
        Esscher multiplier theta*.
    """
    if lam < 1e-15:  # no-jump limit
        return (rf - mu_c) / sigma**2

    f = lambda th: esscher_equation(th, mu_c, rf, sigma, lam, kappa, delta)

    lo, hi = bracket
    fl, fh = f(lo), f(hi)

    if fl * fh > 0:
        scan = np.linspace(-500.0, 500.0, 1000)
        fvals = np.array([f(x) for x in scan])
        sign_changes = np.where(np.diff(np.sign(fvals)))[0]
        assert len(sign_changes) > 0, "No sign change found in [-500, 500]"
        idx = int(sign_changes[0])
        lo, hi = float(scan[idx]), float(scan[idx + 1])

    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=1000))


def compute_rn_params(
    theta: float,
    mu_c: float,
    sigma: float,
    lam: float,
    kappa: float,
    delta: float,
) -> tuple[float, float, float, float]:
    """Compute risk-neutral parameters from the Esscher multiplier.

    Parameters
    ----------
    theta : float
        Esscher multiplier theta*.
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

    Returns
    -------
    tuple of (lam_rn, kappa_rn, kappa_star, mu_c_rn)
        Risk-neutral parameters.
    """
    lam_rn = lam * np.exp(theta * kappa + 0.5 * theta**2 * delta**2)
    kappa_rn = kappa + theta * delta**2
    kappa_star = np.exp(kappa_rn + 0.5 * delta**2) - 1.0
    mu_c_rn = mu_c + theta * sigma**2
    return float(lam_rn), float(kappa_rn), float(kappa_star), float(mu_c_rn)
