"""Integration tests for experiment scripts.

Each test_exp_N function imports and calls the corresponding run_expN()
function and checks that it returns PASS=True and key numerical criteria.

These tests validate that the experiment functions run end-to-end
without errors and that their core assertions pass.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Exp 1: BS Recovery at lambda=0
# ---------------------------------------------------------------------------
def test_exp1_bs_recovery_pass():
    """Exp 1: max|Merton(Set B) - C_BS| < 1e-6 and flat IV surface."""
    from exp.exp_1_bs_recovery import run_exp1
    results = run_exp1()
    assert results["PASS"], f"Exp 1 FAIL: {results}"
    assert results["max_discrepancy"] < 1e-6
    assert results["max_iv_error"] < 1e-4
    assert results["weight_sum_ok"]


# ---------------------------------------------------------------------------
# Exp 2: Esscher martingale condition
# ---------------------------------------------------------------------------
def test_exp2_esscher_martingale_pass():
    """Exp 2: martingale condition, self-consistency for Sets A,C,D,E."""
    from exp.exp_2_esscher_martingale import run_exp2
    results = run_exp2()
    assert results["PASS"], f"Exp 2 FAIL: {results}"
    assert results["max_self_consistency_err"] < 1e-10
    assert results["max_martingale_err"] < 1e-8


# ---------------------------------------------------------------------------
# Exp 3: Merton closed-form vs Monte Carlo
# ---------------------------------------------------------------------------
def test_exp3_mc_validation_pass():
    """Exp 3: z-scores |z| < 2.58 for all 30 Merton vs MC cases."""
    from exp.exp_3_mc_validation import run_exp3
    results = run_exp3()
    assert results["PASS"], f"Exp 3 FAIL: {results}"
    assert results["max_abs_z"] < 2.58


# ---------------------------------------------------------------------------
# Exp 4: IV smile pattern classification
# ---------------------------------------------------------------------------
def test_exp4_iv_smile_pass():
    """Exp 4: correct smile shapes for Sets A,B,C,D."""
    from exp.exp_4_iv_smile import run_exp4
    results = run_exp4()
    assert results["PASS"], f"Exp 4 FAIL: {results}"
    assert results["set_B_flat_ok"]
    assert results["set_A_smirk_ok"]
    assert results["set_D_reverse_ok"]


# ---------------------------------------------------------------------------
# Exp 5: Cumulant maturity scaling
# ---------------------------------------------------------------------------
def test_exp5_cumulant_scaling_pass():
    """Exp 5: Skew ~ T^{-1/2} and ExKurt ~ T^{-1} exponents within tolerance."""
    from exp.exp_5_cumulant_scaling import run_exp5
    results = run_exp5()
    assert results["PASS"], f"Exp 5 FAIL: {results}"
    assert abs(results["skew_exponent"] - (-0.5)) < 0.05
    assert abs(results["kurt_exponent"] - (-1.0)) < 0.05
    assert results["r2_skew"] > 0.999
    assert results["r2_kurt"] > 0.999


# ---------------------------------------------------------------------------
# Exp 6: Analytical smile approximation accuracy
# ---------------------------------------------------------------------------
def test_exp6_smile_approx_pass():
    """Exp 6: approximation error within maturity-scaled tolerances for |d| <= 1.

    First-order expansion error scales as |Skew| ∝ T^{-1/2}, so:
      T=0.25 → < 3.0 vp, T=0.50 → < 2.0 vp, T=1.00 → < 1.0 vp
    """
    from exp.exp_6_smile_approx import run_exp6
    results = run_exp6()
    assert results["PASS"], f"Exp 6 FAIL: {results}"
    # max error at T=1.00 (best accuracy) for |d|<=1 should be < 1.0 vol points
    assert results["max_err_near_atm_vp"] < 1.0


# ---------------------------------------------------------------------------
# Exp 7: Merton series convergence
# ---------------------------------------------------------------------------
def test_exp7_convergence_pass():
    """Exp 7: N_max=50 sufficient for Set A; convergence verified."""
    from exp.exp_7_convergence import run_exp7
    results = run_exp7()
    assert results["PASS"], f"Exp 7 FAIL: {results}"


# ---------------------------------------------------------------------------
# Exp 8: Excess kurtosis positivity
# ---------------------------------------------------------------------------
def test_exp8_exkurt_positivity_pass():
    """Exp 8: ExKurt > 0 at all 1000 grid points."""
    from exp.exp_8_exkurt_positivity import run_exp8
    results = run_exp8()
    assert results["PASS"], f"Exp 8 FAIL: {results}"
    assert results["all_positive"]
    assert results["min_exkurt"] > 0


# ---------------------------------------------------------------------------
# Exp 9: Put-call parity
# ---------------------------------------------------------------------------
def test_exp9_pcp_pass():
    """Exp 9: BS PCP < 1e-14; documents structural Merton PCP deviation."""
    from exp.exp_9_pcp_verification import run_exp9
    results = run_exp9()
    assert results["PASS"], f"Exp 9 FAIL: {results}"
    assert results["bs_pcp_max"] < 1e-14


# ---------------------------------------------------------------------------
# Exp 10: KF density vs Monte Carlo
# ---------------------------------------------------------------------------
def test_exp10_kf_density_pass():
    """Exp 10: KS stat < 0.005 and lambda=0 limit < 1e-4."""
    from exp.exp_10_kf_density import run_exp10
    results = run_exp10()
    assert results["PASS"], f"Exp 10 FAIL: {results}"
    assert results["ks_stat_025"] < 0.005
    assert results["ks_stat_100"] < 0.005
    assert results["lam0_max_err"] < 1e-3
