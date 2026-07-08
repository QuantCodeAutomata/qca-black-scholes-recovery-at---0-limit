"""Kolmogorov-Feller density propagation via FFT.

Computes the Levy-Khintchine characteristic function and inverts via IFFT
to obtain the probability density of log-price under physical or risk-neutral
measure.
Custom implementation — Context7 found no library equivalent (paper Eq. 39).
"""
from __future__ import annotations

import numpy as np
import numpy.fft as fft
from scipy.interpolate import interp1d


def kf_char_exponent(
    u: np.ndarray,
    mu_c: float,
    sigma: float,
    lam: float,
    kappa: float,
    delta: float,
) -> np.ndarray:
    """Levy-Khintchine characteristic exponent Psi(u) for Merton jump-diffusion.

    Psi(u) = i*(mu_c - 0.5*sigma^2)*u
             - 0.5*sigma^2*u^2
             + lambda*(exp(i*kappa*u - 0.5*delta^2*u^2) - 1)

    Parameters
    ----------
    u : np.ndarray
        Frequency grid.
    mu_c : float
        Drift of log-price (physical or risk-neutral).
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity.
    kappa : float
        Mean log-jump.
    delta : float
        Std-dev of log-jump.

    Returns
    -------
    np.ndarray
        Complex characteristic exponent values.
    """
    drift = mu_c - 0.5 * sigma**2
    psi = (
        1j * drift * u
        - 0.5 * sigma**2 * u**2
        + lam * (np.exp(1j * kappa * u - 0.5 * delta**2 * u**2) - 1.0)
    )
    return psi


def kf_density_fft(
    lnS0: float,
    T: float,
    mu_c: float,
    sigma: float,
    lam: float,
    kappa: float,
    delta: float,
    N_grid: int = 4096,
    x_mult: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KF density via IFFT of the characteristic function.

    Parameters
    ----------
    lnS0 : float
        Initial log-price.
    T : float
        Time horizon.
    mu_c : float
        Drift (physical or risk-neutral as required).
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity.
    kappa : float
        Mean log-jump.
    delta : float
        Std-dev of log-jump.
    N_grid : int
        Number of FFT grid points (power of 2 recommended).
    x_mult : float
        Support half-width multiplier.

    Returns
    -------
    tuple of (x_grid, density, cdf)
        x_grid : log-price grid
        density : probability density p(x, T)
        cdf : cumulative distribution function
    """
    x_max = x_mult * max(sigma * np.sqrt(T), delta * np.sqrt(lam * T + 1e-10))
    x_grid = np.linspace(lnS0 - x_max / 2.0, lnS0 + x_max / 2.0, N_grid)
    dx = x_grid[1] - x_grid[0]

    u = 2.0 * np.pi * fft.fftfreq(N_grid, d=dx)

    psi = kf_char_exponent(u, mu_c, sigma, lam, kappa, delta)
    phi = np.exp(1j * lnS0 * u + T * psi)

    density_raw = np.real(fft.fft(phi * np.exp(-1j * u * x_grid[0]))) / (N_grid * dx)
    density = np.maximum(density_raw, 0.0)

    norm_const = np.trapz(density, x_grid)
    if norm_const > 1e-15:
        density = density / norm_const

    # CDF via cumulative trapz
    cdf = np.zeros(N_grid)
    for i in range(1, N_grid):
        cdf[i] = cdf[i - 1] + 0.5 * (density[i] + density[i - 1]) * (x_grid[i] - x_grid[i - 1])
    if cdf[-1] > 1e-15:
        cdf = cdf / cdf[-1]

    return x_grid, density, cdf


def cdf_interpolator(x_grid: np.ndarray, cdf: np.ndarray):
    """Return an interpolating function for the CDF.

    Parameters
    ----------
    x_grid : np.ndarray
        Log-price grid.
    cdf : np.ndarray
        Cumulative distribution values.

    Returns
    -------
    callable
        CDF function that accepts array of x values.
    """
    return interp1d(x_grid, cdf, bounds_error=False, fill_value=(0.0, 1.0))
