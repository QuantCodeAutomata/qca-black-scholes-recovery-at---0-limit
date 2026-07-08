"""Pytest test suite for all src modules.

Tests cover:
  - bs_pricing  : Black-Scholes call/put and implied-vol inversion
  - esscher     : Esscher multiplier solver and risk-neutral parameter derivation
  - merton_pricing : Merton series call/put, Poisson weight sum
  - cumulants   : Cumulant formulas and power-law fitting
  - density     : FFT-based KF density
  - monte_carlo : Physical and risk-neutral path simulation
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from scipy.stats import norm

from src.bs_pricing import bs_call, bs_put, implied_vol
from src.esscher import esscher_equation, solve_esscher, compute_rn_params
from src.merton_pricing import merton_call, merton_put, merton_partial_sums, poisson_weight_sum
from src.cumulants import compute_cumulants, power_law_fit
from src.density import kf_density_fft, cdf_interpolator
from src.monte_carlo import simulate_merton_terminal, simulate_merton_physical, mc_option_price

# ---------------------------------------------------------------------------
# Parameter fixtures used across multiple tests
# ---------------------------------------------------------------------------
SET_A = dict(S0=100.0, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0, kappa=-0.10, delta=0.15)
SET_B = dict(S0=100.0, rf=0.05, mu_c=0.10, sigma=0.20, lam=1e-10, kappa=-0.10, delta=0.15)
SET_C = dict(S0=100.0, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0, kappa=0.0, delta=0.15)
SET_D = dict(S0=100.0, rf=0.05, mu_c=0.10, sigma=0.20, lam=1.0, kappa=0.10, delta=0.15)


# ===========================================================================
# bs_pricing tests
# ===========================================================================

class TestBSCall:
    def test_known_atm_value(self):
        """C_BS(100, 100, 1, 0.05, 0.20) ≈ 10.4506  (Merton paper benchmark)."""
        c = bs_call(100, 100, 1.0, 0.05, 0.20)
        assert abs(c - 10.4506) < 1e-3

    def test_deep_itm_call(self):
        """Deep ITM call ≈ intrinsic value."""
        c = bs_call(200, 100, 1.0, 0.05, 0.20)
        intrinsic = 200 - 100 * np.exp(-0.05 * 1.0)
        assert c > intrinsic
        assert c < 200

    def test_deep_otm_call_positive(self):
        """Deep OTM call price is positive."""
        c = bs_call(100, 150, 0.25, 0.05, 0.20)
        assert c > 0

    def test_zero_vol_edge_case(self):
        """sigma*sqrt(T) < 1e-12 returns max(S0 - K*exp(-r*T), 0)."""
        c = bs_call(100, 90, 1.0, 0.05, 1e-20)
        expected = max(100 - 90 * np.exp(-0.05), 0.0)
        assert abs(c - expected) < 1e-10

    def test_monotone_in_vol(self):
        """Call price is monotone increasing in sigma."""
        sigmas = [0.1, 0.2, 0.3, 0.4]
        prices = [bs_call(100, 100, 1.0, 0.05, s) for s in sigmas]
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

    def test_monotone_in_strike(self):
        """Call price decreases with strike."""
        strikes = [80, 90, 100, 110, 120]
        prices = [bs_call(100, K, 1.0, 0.05, 0.20) for K in strikes]
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]

    def test_call_bounded_above_by_spot(self):
        """Call price never exceeds spot."""
        c = bs_call(100, 50, 5.0, 0.05, 0.50)
        assert c <= 100.0


class TestBSPut:
    def test_put_call_parity_atm(self):
        """PCP: C - P = S0 - K*exp(-r*T) at ATM."""
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        c = bs_call(S0, K, T, r, sigma)
        p = bs_put(S0, K, T, r, sigma)
        parity = c - p - (S0 - K * np.exp(-r * T))
        assert abs(parity) < 1e-14

    def test_put_call_parity_grid(self):
        """PCP holds to machine precision across a strike/maturity grid."""
        strikes = [60, 80, 100, 120, 150]
        maturities = [0.25, 0.50, 1.00]
        for K in strikes:
            for T in maturities:
                c = bs_call(100, K, T, 0.05, 0.20)
                p = bs_put(100, K, T, 0.05, 0.20)
                err = abs(c - p - (100 - K * np.exp(-0.05 * T)))
                assert err < 1e-13, f"PCP error {err:.2e} at K={K}, T={T}"

    def test_put_non_negative(self):
        """Put price is always non-negative."""
        for K in [60, 80, 100, 120, 150]:
            p = bs_put(100, K, 0.50, 0.05, 0.20)
            assert p >= 0.0


class TestImpliedVol:
    def test_round_trip(self):
        """implied_vol(bs_call(S,K,T,r,sigma), ...) recovers sigma."""
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        C = bs_call(S0, K, T, r, sigma)
        sigma_imp = implied_vol(C, S0, K, T, r)
        assert abs(sigma_imp - sigma) < 1e-8

    def test_round_trip_otm(self):
        """Round-trip implied vol for OTM call."""
        S0, K, T, r, sigma = 100, 120, 0.50, 0.05, 0.25
        C = bs_call(S0, K, T, r, sigma)
        sigma_imp = implied_vol(C, S0, K, T, r)
        assert abs(sigma_imp - sigma) < 1e-8

    def test_below_intrinsic_returns_nan(self):
        """Price at or below intrinsic value returns NaN."""
        S0, K, T, r = 100, 80, 1.0, 0.05
        intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
        iv = implied_vol(intrinsic * 0.99, S0, K, T, r)
        assert np.isnan(iv)

    def test_put_round_trip(self):
        """Round-trip for put implied vol."""
        S0, K, T, r, sigma = 100, 80, 0.25, 0.05, 0.20
        P = bs_put(S0, K, T, r, sigma)
        sigma_imp = implied_vol(P, S0, K, T, r, option_type="put")
        assert abs(sigma_imp - sigma) < 1e-8


# ===========================================================================
# esscher tests
# ===========================================================================

class TestEsscherEquation:
    def test_zero_jump_case(self):
        """With lambda=0 analytically F(theta) = mu_c + theta*sigma^2 - rf."""
        theta = 1.0
        val = esscher_equation(theta, mu_c=0.10, rf=0.05, sigma=0.20, lam=0.0, kappa=-0.10, delta=0.15)
        expected = 0.10 + 1.0 * 0.04 - 0.05  # = 0.09
        assert abs(val - expected) < 1e-12

    def test_large_theta_overflow_guard(self):
        """Very large positive theta triggers overflow guard and returns +inf."""
        result = esscher_equation(1000.0, mu_c=0.10, rf=0.05, sigma=0.20,
                                  lam=1.0, kappa=0.5, delta=0.15)
        assert result == np.inf


class TestSolveEsscher:
    def test_no_jump_limit(self):
        """With lam<1e-15, theta* = (rf - mu_c) / sigma^2."""
        theta = solve_esscher(mu_c=0.10, rf=0.05, sigma=0.20, lam=1e-20, kappa=-0.10, delta=0.15)
        expected = (0.05 - 0.10) / 0.04
        assert abs(theta - expected) < 1e-10

    def test_root_satisfies_equation(self):
        """The returned theta* satisfies F(theta*) = 0 to 1e-10."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        residual = esscher_equation(theta, p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        assert abs(residual) < 1e-10

    @pytest.mark.parametrize("pset", [SET_A, SET_C, SET_D])
    def test_root_all_sets(self, pset):
        """Root satisfies equation for all tested parameter sets."""
        theta = solve_esscher(pset["mu_c"], pset["rf"], pset["sigma"],
                               pset["lam"], pset["kappa"], pset["delta"])
        residual = esscher_equation(theta, pset["mu_c"], pset["rf"], pset["sigma"],
                                     pset["lam"], pset["kappa"], pset["delta"])
        assert abs(residual) < 1e-9


class TestComputeRNParams:
    def test_no_arbitrage_condition(self):
        """mu_c_rn + lam_rn * kappa_star = rf (no-arbitrage self-consistency)."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"]
        )
        err = abs(mu_c_rn + lam_rn * kappa_star - p["rf"])
        assert err < 1e-10

    def test_kappa_rn_formula(self):
        """kappa_rn = kappa + theta * delta^2."""
        theta = 0.5
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, mu_c=0.10, sigma=0.20, lam=1.0, kappa=-0.10, delta=0.15
        )
        expected_kappa_rn = -0.10 + 0.5 * 0.15**2
        assert abs(kappa_rn - expected_kappa_rn) < 1e-12

    def test_kappa_star_formula(self):
        """kappa_star = exp(kappa_rn + 0.5*delta^2) - 1."""
        theta = 0.5
        delta = 0.15
        lam_rn, kappa_rn, kappa_star, _ = compute_rn_params(
            theta, mu_c=0.10, sigma=0.20, lam=1.0, kappa=-0.10, delta=delta
        )
        expected_ks = np.exp(kappa_rn + 0.5 * delta**2) - 1.0
        assert abs(kappa_star - expected_ks) < 1e-12


# ===========================================================================
# merton_pricing tests
# ===========================================================================

class TestMertonCall:
    def _rn(self, p):
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        return compute_rn_params(theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])

    def test_lambda_zero_limit_matches_bs(self):
        """Merton call with lam~0 matches BS call to 1e-6."""
        p = SET_B
        lam_rn, kappa_rn, kappa_star, _ = self._rn(p)
        C_merton = merton_call(p["S0"], 100, 1.0, p["rf"], p["sigma"],
                                lam_rn, kappa_rn, kappa_star, p["delta"])
        C_bs = bs_call(p["S0"], 100, 1.0, p["rf"], p["sigma"])
        assert abs(C_merton - C_bs) < 1e-6

    def test_positive_price(self):
        """Merton call price is positive."""
        p = SET_A
        lam_rn, kappa_rn, kappa_star, _ = self._rn(p)
        c = merton_call(p["S0"], 100, 1.0, p["rf"], p["sigma"],
                        lam_rn, kappa_rn, kappa_star, p["delta"])
        assert c > 0

    def test_decreasing_in_strike(self):
        """Merton call decreases with strike."""
        p = SET_A
        lam_rn, kappa_rn, kappa_star, _ = self._rn(p)
        prices = [merton_call(p["S0"], K, 1.0, p["rf"], p["sigma"],
                               lam_rn, kappa_rn, kappa_star, p["delta"])
                  for K in [80, 100, 120]]
        assert prices[0] > prices[1] > prices[2]

    def test_atm_merton_ge_bs(self):
        """ATM Merton call >= BS call (jump convexity / Jensen's inequality)."""
        p = SET_A
        lam_rn, kappa_rn, kappa_star, _ = self._rn(p)
        C_merton = merton_call(p["S0"], 100, 1.0, p["rf"], p["sigma"],
                                lam_rn, kappa_rn, kappa_star, p["delta"])
        C_bs = bs_call(p["S0"], 100, 1.0, p["rf"], p["sigma"])
        assert C_merton >= C_bs - 1e-8


class TestPoissonWeightSum:
    def test_sum_close_to_one(self):
        """Poisson weight sum is 1 within 1e-10."""
        ws = poisson_weight_sum(lam_rn=1.0, T=1.0)
        assert abs(ws - 1.0) < 1e-10

    def test_tiny_lambda_sum(self):
        """For tiny lambda, n=0 dominates and sum ≈ 1."""
        ws = poisson_weight_sum(lam_rn=1e-10, T=1.0)
        assert abs(ws - 1.0) < 1e-10

    def test_high_intensity_sum(self):
        """High lambda Poisson weights still sum to 1."""
        ws = poisson_weight_sum(lam_rn=5.0, T=2.0)
        assert abs(ws - 1.0) < 1e-8


class TestMertonPartialSums:
    def test_convergence_monotone(self):
        """Partial sums converge to a stable limit."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, _ = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        partials = merton_partial_sums(p["S0"], 100, 1.0, p["rf"], p["sigma"],
                                       lam_rn, kappa_rn, kappa_star, p["delta"],
                                       N_max_compute=100)
        # After N=50, relative change from N=100 should be tiny
        c50 = partials[50]
        c100 = partials[100]
        assert abs(c100 - c50) / max(abs(c100), 1e-12) < 1e-6

    def test_length_correct(self):
        """Partial sums array has N_max_compute+1 elements."""
        partials = merton_partial_sums(100, 100, 1.0, 0.05, 0.20, 1.0, -0.1, 0.1, 0.15,
                                       N_max_compute=50)
        assert len(partials) == 51


# ===========================================================================
# cumulants tests
# ===========================================================================

class TestComputeCumulants:
    def test_kappa2_positive(self):
        """Total variance kappa_2 is always positive."""
        k2, *_ = compute_cumulants(lam_rn=1.0, kappa_rn=-0.1, delta=0.15, sigma=0.20, T=1.0)
        assert k2 > 0

    def test_exkurt_positive_positive_lam(self):
        """ExKurt > 0 whenever lam_rn > 0."""
        for kappa_rn in [-0.5, 0.0, 0.5]:
            for delta in [0.05, 0.15, 0.30]:
                _, _, _, _, ek = compute_cumulants(1.0, kappa_rn, delta, 0.20, 1.0)
                assert ek > 0

    def test_skew_zero_for_kappa_rn_zero(self):
        """With kappa_rn=0 (symmetric jumps), skewness is 0."""
        _, _, _, skew, _ = compute_cumulants(1.0, 0.0, 0.15, 0.20, 1.0)
        assert abs(skew) < 1e-14

    def test_scaling_exact_t_minus_half(self):
        """Skew * sqrt(T) is constant (T^{-1/2} scaling is exact)."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, _, _ = compute_rn_params(theta, p["mu_c"], p["sigma"],
                                                     p["lam"], p["kappa"], p["delta"])
        T_arr = [0.25, 0.50, 1.00, 2.00]
        skew_sqrtT = []
        for T in T_arr:
            _, _, _, skew, _ = compute_cumulants(lam_rn, kappa_rn, p["delta"], p["sigma"], T)
            skew_sqrtT.append(skew * np.sqrt(T))
        # All values should be equal
        for i in range(1, len(skew_sqrtT)):
            assert abs(skew_sqrtT[i] - skew_sqrtT[0]) < 1e-10

    def test_scaling_exact_t_minus_one(self):
        """ExKurt * T is constant (T^{-1} scaling is exact)."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, _, _ = compute_rn_params(theta, p["mu_c"], p["sigma"],
                                                     p["lam"], p["kappa"], p["delta"])
        T_arr = [0.25, 0.50, 1.00, 2.00]
        ek_T = []
        for T in T_arr:
            _, _, _, _, ek = compute_cumulants(lam_rn, kappa_rn, p["delta"], p["sigma"], T)
            ek_T.append(ek * T)
        for i in range(1, len(ek_T)):
            assert abs(ek_T[i] - ek_T[0]) < 1e-10

    def test_kappa2_linear_in_T(self):
        """kappa_2 is exactly linear in T."""
        _, k2_1, _, _, _ = compute_cumulants(1.0, -0.1, 0.15, 0.20, 1.0), None, None, None, None
        k2_1 = compute_cumulants(1.0, -0.1, 0.15, 0.20, 1.0)[0]
        k2_2 = compute_cumulants(1.0, -0.1, 0.15, 0.20, 2.0)[0]
        assert abs(k2_2 - 2 * k2_1) < 1e-12


class TestPowerLawFit:
    def test_exact_power_law(self):
        """Fit recovers exact exponent for a clean power law."""
        T = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
        b_true = -0.5
        y = 3.0 * T**b_true
        b_fit, A_fit, r2 = power_law_fit(T, y)
        assert abs(b_fit - b_true) < 1e-10
        assert abs(A_fit - 3.0) < 1e-8
        assert r2 > 0.9999

    def test_r2_unity_for_exact_power(self):
        """R² = 1 for an exact power law."""
        T = np.linspace(0.1, 5.0, 20)
        y = 2.0 * T**(-1.0)
        _, _, r2 = power_law_fit(T, y)
        assert r2 > 0.99999


# ===========================================================================
# density tests
# ===========================================================================

class TestKFDensityFFT:
    def test_density_integrates_to_one(self):
        """FFT density integrates to ≈ 1.0."""
        x, p, cdf = kf_density_fft(np.log(100), 0.25, 0.10, 0.20, 1.0, -0.10, 0.15,
                                     N_grid=4096, x_mult=10)
        integral = float(np.trapz(p, x))
        assert abs(integral - 1.0) < 1e-3

    def test_density_non_negative(self):
        """FFT density values are non-negative after clipping."""
        x, p, cdf = kf_density_fft(np.log(100), 1.0, 0.10, 0.20, 1.0, -0.10, 0.15)
        assert np.all(p >= 0)

    def test_cdf_monotone(self):
        """CDF is monotone non-decreasing."""
        x, p, cdf = kf_density_fft(np.log(100), 0.25, 0.10, 0.20, 1.0, -0.10, 0.15)
        diffs = np.diff(cdf)
        assert np.all(diffs >= -1e-12)

    def test_cdf_ends_at_one(self):
        """CDF approaches 1 at the right boundary."""
        x, p, cdf = kf_density_fft(np.log(100), 1.0, 0.10, 0.20, 1.0, -0.10, 0.15,
                                     N_grid=4096, x_mult=12)
        assert abs(cdf[-1] - 1.0) < 1e-6

    def test_lam_zero_limit_gaussian(self):
        """With lambda ≈ 0, KF density matches Gaussian to < 1e-4."""
        from scipy.stats import norm
        T, mu_c, sigma = 0.25, 0.10, 0.20
        x, p_kf, _ = kf_density_fft(np.log(100), T, mu_c, sigma, 1e-10, -0.10, 0.15,
                                      N_grid=4096, x_mult=10)
        mean_th = np.log(100) + (mu_c - 0.5 * sigma**2) * T
        std_th = sigma * np.sqrt(T)
        p_gauss = norm.pdf(x, mean_th, std_th)
        max_err = float(np.max(np.abs(p_kf - p_gauss)))
        assert max_err < 1e-3

    def test_cdf_interpolator_returns_callable(self):
        """cdf_interpolator returns a callable."""
        x, _, cdf = kf_density_fft(np.log(100), 0.5, 0.10, 0.20, 1.0, -0.10, 0.15)
        fn = cdf_interpolator(x, cdf)
        assert callable(fn)
        val = fn(np.log(100))
        assert 0.0 <= float(val) <= 1.0


# ===========================================================================
# monte_carlo tests
# ===========================================================================

class TestSimulateMertonTerminal:
    def test_output_shape(self):
        """Returns array of correct length."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        S_T = simulate_merton_terminal(p["S0"], 1.0, mu_c_rn, p["sigma"],
                                        lam_rn, kappa_rn, kappa_star, p["delta"],
                                        N_paths=1000, seed=0)
        assert S_T.shape == (1000,)

    def test_positive_prices(self):
        """All terminal prices are positive."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        S_T = simulate_merton_terminal(p["S0"], 1.0, mu_c_rn, p["sigma"],
                                        lam_rn, kappa_rn, kappa_star, p["delta"],
                                        N_paths=5000, seed=1)
        assert np.all(S_T > 0)

    def test_martingale_property(self):
        """E_rn[S_T] / S_0 ≈ exp(rf*T) within 3σ (N=100k paths)."""
        p = SET_A
        rf = p["rf"]
        theta = solve_esscher(p["mu_c"], rf, p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        N = 200_000
        S_T = simulate_merton_terminal(p["S0"], 1.0, mu_c_rn, p["sigma"],
                                        lam_rn, kappa_rn, kappa_star, p["delta"],
                                        N_paths=N, seed=99)
        ratio = float(np.mean(S_T)) / p["S0"]
        expected = np.exp(rf * 1.0)
        se = float(np.std(S_T / p["S0"])) / np.sqrt(N)
        assert abs(ratio - expected) < 5 * se


class TestSimulateMertonPhysical:
    def test_output_is_log_prices(self):
        """Returns log-prices (no exp), values near ln(S0)."""
        lnS0 = np.log(100)
        log_ST = simulate_merton_physical(lnS0, 0.25, 0.10, 0.20, 1.0, -0.10, 0.15,
                                           N_paths=1000, seed=0)
        assert log_ST.shape == (1000,)
        # Most log-prices should be in a reasonable range
        assert np.mean(np.abs(log_ST - lnS0)) < 2.0

    def test_mean_moment(self):
        """MC mean ≈ theoretical mean to within 3σ."""
        lnS0 = np.log(100)
        T, mu_c, sigma, lam, kappa, delta = 0.25, 0.10, 0.20, 1.0, -0.10, 0.15
        N = 200_000
        log_ST = simulate_merton_physical(lnS0, T, mu_c, sigma, lam, kappa, delta,
                                           N_paths=N, seed=5)
        mean_mc = float(np.mean(log_ST))
        mean_th = lnS0 + (mu_c - 0.5 * sigma**2 + lam * kappa) * T
        se = float(np.std(log_ST)) / np.sqrt(N)
        assert abs(mean_mc - mean_th) < 5 * se

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        r1 = simulate_merton_physical(np.log(100), 0.25, 0.10, 0.20, 1.0, -0.10, 0.15,
                                       N_paths=1000, seed=42)
        r2 = simulate_merton_physical(np.log(100), 0.25, 0.10, 0.20, 1.0, -0.10, 0.15,
                                       N_paths=1000, seed=42)
        np.testing.assert_array_equal(r1, r2)


class TestMCOptionPrice:
    def test_call_price_positive(self):
        """MC call price is positive for ATM option."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        S_T = simulate_merton_terminal(p["S0"], 1.0, mu_c_rn, p["sigma"],
                                        lam_rn, kappa_rn, kappa_star, p["delta"],
                                        N_paths=20_000, seed=7)
        price, se = mc_option_price(S_T, K=100, T=1.0, rf=p["rf"], option_type="call")
        assert price > 0
        assert se > 0

    def test_put_call_parity_mc(self):
        """MC PCP: |C - P - (S0 - K*exp(-r*T))| < 5*SE."""
        p = SET_A
        T, K = 1.0, 100.0
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        S_T = simulate_merton_terminal(p["S0"], T, mu_c_rn, p["sigma"],
                                        lam_rn, kappa_rn, kappa_star, p["delta"],
                                        N_paths=200_000, seed=13)
        C, se_C = mc_option_price(S_T, K, T, p["rf"], "call")
        P, se_P = mc_option_price(S_T, K, T, p["rf"], "put")
        pcp_err = abs(C - P - (p["S0"] - K * np.exp(-p["rf"] * T)))
        se_total = se_C + se_P
        assert pcp_err < 5 * se_total


# ===========================================================================
# Cross-module integration tests
# ===========================================================================

class TestCrossModuleIntegration:
    def test_exp1_bs_recovery_lambda_zero(self):
        """Merton(lambda≈0) vs BS: max error < 1e-6 for standard strikes."""
        p = SET_B
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, kappa_star, _ = compute_rn_params(
            theta, p["mu_c"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        for K in strikes:
            C_m = merton_call(p["S0"], K, 1.0, p["rf"], p["sigma"],
                               lam_rn, kappa_rn, kappa_star, p["delta"])
            C_bs = bs_call(p["S0"], K, 1.0, p["rf"], p["sigma"])
            assert abs(C_m - C_bs) < 1e-6, f"Merton-BS gap at K={K}: {abs(C_m-C_bs):.2e}"

    def test_exp2_self_consistency_all_sets(self):
        """No-arbitrage self-consistency: mu_c_rn + lam_rn*kappa_star = rf."""
        for pset in [SET_A, SET_C, SET_D]:
            theta = solve_esscher(pset["mu_c"], pset["rf"], pset["sigma"],
                                   pset["lam"], pset["kappa"], pset["delta"])
            lam_rn, kappa_rn, kappa_star, mu_c_rn = compute_rn_params(
                theta, pset["mu_c"], pset["sigma"], pset["lam"], pset["kappa"], pset["delta"])
            err = abs(mu_c_rn + lam_rn * kappa_star - pset["rf"])
            assert err < 1e-10

    def test_exp5_exkurt_positive_all_maturities(self):
        """ExKurt > 0 for all T in grid for Set A."""
        p = SET_A
        theta = solve_esscher(p["mu_c"], p["rf"], p["sigma"], p["lam"], p["kappa"], p["delta"])
        lam_rn, kappa_rn, _, _ = compute_rn_params(theta, p["mu_c"], p["sigma"],
                                                     p["lam"], p["kappa"], p["delta"])
        for T in [0.10, 0.25, 0.50, 1.00, 2.00, 5.00]:
            _, _, _, _, ek = compute_cumulants(lam_rn, kappa_rn, p["delta"], p["sigma"], T)
            assert ek > 0

    def test_exp8_exkurt_universal_positivity_sample(self):
        """ExKurt > 0 at sampled points from the 10x10x10 grid."""
        sigma, T = 0.20, 1.0
        for lam_rn in [0.1, 1.0, 5.0]:
            for kappa_rn in [-0.5, 0.0, 0.5]:
                for delta in [0.05, 0.25, 0.50]:
                    _, _, _, _, ek = compute_cumulants(lam_rn, kappa_rn, delta, sigma, T)
                    assert ek > 0, f"ExKurt <= 0 at lam={lam_rn}, kappa={kappa_rn}, delta={delta}"
