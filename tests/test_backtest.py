"""
Tests for the backtest engine, metrics and baselines: execution lag, costs,
hand-computed metrics and causality of the baselines.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.baselines import buy_and_hold_weights, dma200_weights, vol_target_weights
from src.backtest.engine import run_backtest
from src.backtest.metrics import get_all_metrics


def test_max_drawdown_hand_computed():
    # Equity 1.0 -> 1.0 -> 0.5 -> 1.0: the worst fall from a peak is exactly 50%.
    returns = pd.Series([0.0, -0.5, 1.0])
    metrics = get_all_metrics(returns, pd.Series([1.0, 1.0, 1.0]), ann_factor=252, asset_name="test")
    assert np.isclose(metrics["Max DD"], -0.5)


def test_weight_held_is_target_from_two_days_earlier():
    """A weight decided at day t's close is traded at t+1's close and earns
    day t+2's return. Checked on every day, not just one."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.Series(rng.normal(0, 0.01, 30), index=dates)
    target = pd.Series(rng.uniform(0, 1, 30), index=dates)

    res = run_backtest(pd.DataFrame({"returns": returns}), target, cost_bps=0.0)

    expected_weight = target.shift(2).fillna(0.0)
    np.testing.assert_allclose(res["executed_weight"], expected_weight)
    np.testing.assert_allclose(res["net_return"], expected_weight * returns)


def test_oracle_signal_cannot_dodge_a_known_crash():
    """A target that 'knows' day 2 crashes only changes the weight from day 4,
    so the crash is still taken. A look-ahead bug would avoid it."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.Series([0.1, 0.1, -0.5, 0.1, 0.1], index=dates)
    target = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0], index=dates)  # flat exactly on the crash day
    res = run_backtest(pd.DataFrame({"returns": returns}), target, cost_bps=0.0)
    assert res["net_return"].iloc[2] == -0.5


def test_cost_is_turnover_times_rate():
    dates = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame({"returns": [0.0, 0.0, 0.0, 0.0]}, index=dates)
    target = pd.Series([1.0, 0.0, 1.0, 0.0], index=dates)

    # Held weights [0, 0, 1, 0] -> turnover [0, 0, 1, 1] -> cost at 100 bps.
    res = run_backtest(df, target, cost_bps=100.0)
    np.testing.assert_allclose(res["cost"].values, [0.0, 0.0, 0.01, 0.01])

    free = run_backtest(df, target, cost_bps=0.0)
    assert free["cost"].sum() == 0.0
    assert free["net_return"].sum() > res["net_return"].sum()


def test_cost_argument_is_required():
    dates = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(TypeError):
        run_backtest(pd.DataFrame({"returns": [0.0] * 3}, index=dates), pd.Series(1.0, index=dates))


def test_buy_and_hold_is_always_fully_invested():
    idx = pd.bdate_range("2020-01-01", periods=10)
    assert (buy_and_hold_weights(idx) == 1.0).all()


def test_dma200_is_flat_during_warm_up_and_causal():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=400)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)), index=idx)
    w = dma200_weights(close)

    # No average exists for the first 199 days: stay flat rather than guess.
    assert (w.iloc[:199] == 0).all()
    assert set(w.unique()) <= {0.0, 1.0}

    # Changing future prices must not change any earlier weight.
    shocked = close.copy()
    shocked.iloc[300:] *= 3
    pd.testing.assert_series_equal(dma200_weights(shocked).iloc[:300], w.iloc[:300])


def test_uninvested_capital_earns_cash():
    dates = pd.bdate_range("2020-01-01", periods=6)
    df = pd.DataFrame({"returns": [0.01] * 6}, index=dates)
    cash = pd.Series(0.0002, index=dates)
    target = pd.Series([0.25] * 6, index=dates)
    res = run_backtest(df, target, cost_bps=0.0, cash_returns=cash)
    # Flat for the first two days (lag): all cash. Then 25% asset, 75% cash.
    np.testing.assert_allclose(res["net_return"].iloc[:2], 0.0002)
    np.testing.assert_allclose(res["net_return"].iloc[2:], 0.25 * 0.01 + 0.75 * 0.0002)
    assert (res["cash_return"] == 0.0002).all()


def test_vol_target_scales_down_only_when_volatility_is_above_normal():
    vol = pd.Series([np.nan, 0.01, 0.01, 0.01, 0.04, 0.005])
    w = vol_target_weights(vol)
    assert w.iloc[0] == 0.0                         # no estimate yet: flat
    np.testing.assert_allclose(w.iloc[1:4], 1.0)    # at its own norm: fully invested
    # Day 4: norm = mean(0.01, 0.01, 0.01, 0.04) = 0.0175 -> 0.0175 / 0.04
    assert w.iloc[4] == pytest.approx(0.0175 / 0.04)
    assert w.iloc[5] == 1.0                          # calmer than normal: capped at 100%, no leverage


def test_vol_target_is_causal():
    rng = np.random.default_rng(5)
    vol = pd.Series(rng.uniform(0.005, 0.03, 300))
    shocked = vol.copy()
    shocked.iloc[200:] *= 4
    pd.testing.assert_series_equal(vol_target_weights(shocked).iloc[:200], vol_target_weights(vol).iloc[:200])
