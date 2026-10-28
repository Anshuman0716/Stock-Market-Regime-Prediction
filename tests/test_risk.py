"""
Tests for the daily VaR / CVaR estimators (src/backtest/risk.py).
Losses are reported as positive numbers (0.02 = a 2% one-day loss).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.backtest.risk import (
    compute_cornish_fisher_var, compute_historical_cvar, compute_historical_var,
)


def test_historical_var_and_cvar_hand_computed():
    # 101 evenly spaced returns from -5% to +5%: the 5th percentile is -4.5%,
    # and the six returns at or below it (-5.0% ... -4.5%) average -4.75%.
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    assert np.isclose(compute_historical_var(r, 0.95), 0.045)
    assert np.isclose(compute_historical_cvar(r, 0.95), 0.0475)


def test_cvar_is_never_smaller_than_var():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(df=3, size=2000) * 0.01)
    for level in (0.95, 0.99):
        assert compute_historical_cvar(r, level) >= compute_historical_var(r, level)


def test_cornish_fisher_equals_normal_var_without_skew_or_fat_tails():
    # Symmetric, normal-like sample: the Cornish-Fisher correction should
    # vanish and leave the plain normal VaR.
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 200_000))
    expected = -(r.mean() + norm.ppf(0.05) * r.std())
    assert np.isclose(compute_cornish_fisher_var(r, 0.95), expected, rtol=0.02)


def test_cornish_fisher_raises_var_for_fat_left_tail():
    rng = np.random.default_rng(2)
    calm = rng.normal(0.0, 0.01, 5000)
    crashes = rng.normal(-0.05, 0.02, 100)  # occasional big losses: negative skew, fat tails
    r = pd.Series(np.concatenate([calm, crashes]))
    normal_var = -(r.mean() + norm.ppf(0.01) * r.std())
    assert compute_cornish_fisher_var(r, 0.99) > normal_var


def test_empty_or_all_missing_returns_give_nan():
    for r in (pd.Series([], dtype=float), pd.Series([np.nan, np.nan])):
        assert np.isnan(compute_historical_var(r))
        assert np.isnan(compute_historical_cvar(r))
        assert np.isnan(compute_cornish_fisher_var(r))
