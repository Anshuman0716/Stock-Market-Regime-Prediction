"""
Tests for the v2.0 portfolio backtest (src/backtest/portfolio.py):
a hand-checked example, no look-ahead in any weight rule or the engine,
weight limits, calendar alignment, and agreement with the v1.0 engine.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.portfolio import (
    align_signals,
    equal_weights,
    inverse_vol_weights,
    membership,
    regime_aware_weights,
    run_portfolio_backtest,
    stock_calendar_returns,
)

ASSETS = ["spx", "gold", "bitcoin"]
COSTS = {"spx": 1.5, "gold": 2.5, "bitcoin": 5.5}


def _random_inputs(n=300, seed=0):
    """Returns, regime exposures and membership for three assets; Bitcoin joins on day 150."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    returns = pd.DataFrame(rng.normal(0, [0.01, 0.008, 0.04], (n, 3)), index=dates, columns=ASSETS)
    exposures = pd.DataFrame(rng.uniform(0, 1, (n, 3)), index=dates, columns=ASSETS)
    members = membership(dates, {"spx": dates[70], "gold": dates[70], "bitcoin": dates[150]})
    return returns, exposures, members


def _all_weights(returns, exposures, members):
    rp = inverse_vol_weights(returns, members)
    return {"equal": equal_weights(members), "risk_parity": rp,
            "regime": regime_aware_weights(rp, exposures)}


# ── Hand-checked example ──────────────────────────────────────────────────

def test_hand_checked_two_asset_example():
    """Five days, two assets, every number worked out by hand.

    Targets: 50/50 on days 0-1, then 100% A. Held (two-day delay):
    day0 0/0, day1 0/0, day2 .5/.5, day3 .5/.5, day4 1/0.
    Costs: A 10 bps, B 20 bps. Cash earns 0.1% a day.

    day0, day1: all cash -> +0.1%, no trades.
    day2: buy .5 A and .5 B -> cost .5*.001 + .5*.002 = .0015;
          gross = .5*10% + .5*(-10%) = 0 -> net -0.0015.
    day3: yesterday's .5/.5 drifted to .55/.45 (A rose 10%, B fell 10%,
          portfolio flat); back to .5/.5 -> trade .05 each ->
          cost .05*.001 + .05*.002 = .00015; gross 0 -> net -0.00015.
    day4: no move on day3, so drifted is .5/.5; go to 1/0 -> trade .5 each
          -> cost .0015; gross = 1*2% = .02 -> net .0185.
    """
    dates = pd.bdate_range("2021-01-04", periods=5)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.10, 0.0, 0.02],
                            "B": [0.0, 0.0, -0.10, 0.0, 0.01]}, index=dates)
    target = pd.DataFrame({"A": [0.5, 0.5, 1.0, 1.0, 1.0],
                           "B": [0.5, 0.5, 0.0, 0.0, 0.0]}, index=dates)
    cash = pd.Series(0.001, index=dates)

    res = run_portfolio_backtest(returns, target, {"A": 10.0, "B": 20.0}, cash_returns=cash)

    np.testing.assert_allclose(res.held_weights["A"], [0, 0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(res.held_weights["B"], [0, 0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(res.turnover["A"], [0, 0, 0.5, 0.05, 0.5], atol=1e-12)
    np.testing.assert_allclose(res.turnover["B"], [0, 0, 0.5, 0.05, 0.5], atol=1e-12)
    np.testing.assert_allclose(res.daily["cost"], [0, 0, 0.0015, 0.00015, 0.0015], atol=1e-12)
    np.testing.assert_allclose(res.daily["gross_return"], [0.001, 0.001, 0.0, 0.0, 0.02], atol=1e-12)
    np.testing.assert_allclose(res.daily["net_return"], [0.001, 0.001, -0.0015, -0.00015, 0.0185], atol=1e-12)
    np.testing.assert_allclose(res.daily["invested"], [0, 0, 1, 1, 1])


def test_part_invested_portfolio_earns_cash_on_the_rest():
    dates = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.05]}, index=dates)
    target = pd.DataFrame({"A": [0.4, 0.4, 0.4]}, index=dates)
    cash = pd.Series(0.01, index=dates)
    res = run_portfolio_backtest(returns, target, {"A": 0.0}, cash_returns=cash)
    # Day 2: 40% earns 5%, 60% earns 1%.
    assert np.isclose(res.daily["gross_return"].iloc[2], 0.4 * 0.05 + 0.6 * 0.01)


# ── No look-ahead ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [100, 160, 250])
def test_weights_on_day_t_do_not_change_when_later_data_changes(cut):
    """Scramble every return and exposure after day `cut`: no weight up to
    and including `cut` may move. A centred window, a backward fill or a
    full-sample estimate would fail this."""
    returns, exposures, members = _random_inputs()
    rng = np.random.default_rng(99)
    returns2, exposures2 = returns.copy(), exposures.copy()
    returns2.iloc[cut + 1:] = rng.normal(0, 0.05, returns2.iloc[cut + 1:].shape)
    exposures2.iloc[cut + 1:] = rng.uniform(0, 1, exposures2.iloc[cut + 1:].shape)

    before = _all_weights(returns, exposures, members)
    after = _all_weights(returns2, exposures2, members)
    for name in before:
        pd.testing.assert_frame_equal(before[name].iloc[:cut + 1], after[name].iloc[:cut + 1], obj=name)


def test_portfolio_returns_on_day_t_do_not_change_when_later_data_changes():
    returns, exposures, members = _random_inputs()
    cut = 200
    returns2, exposures2 = returns.copy(), exposures.copy()
    returns2.iloc[cut + 1:] *= -3
    exposures2.iloc[cut + 1:] = 1 - exposures2.iloc[cut + 1:]

    for name in ("equal", "risk_parity", "regime"):
        a = run_portfolio_backtest(returns, _all_weights(returns, exposures, members)[name], COSTS)
        b = run_portfolio_backtest(returns2, _all_weights(returns2, exposures2, members)[name], COSTS)
        pd.testing.assert_frame_equal(a.daily.iloc[:cut + 1], b.daily.iloc[:cut + 1], obj=name)


def test_a_signal_that_knows_the_crash_still_takes_it():
    """Exposure drops to 0 exactly on the crash day; with the two-day delay
    the portfolio still holds the asset that day."""
    dates = pd.bdate_range("2021-01-04", periods=5)
    returns = pd.DataFrame({"A": [0.0, 0.0, -0.5, 0.0, 0.0]}, index=dates)
    target = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
    res = run_portfolio_backtest(returns, target, {"A": 0.0})
    assert res.daily["gross_return"].iloc[2] == -0.5


def test_calendar_alignment_does_not_look_ahead():
    cal = pd.bdate_range("2021-01-04", periods=40)
    btc_days = pd.date_range("2021-01-01", periods=60)
    rng = np.random.default_rng(1)
    price = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.03, 60))) * 100, index=btc_days)
    signal = pd.Series(rng.uniform(0, 1, 60), index=btc_days)
    cut = cal[20]
    price2, signal2 = price.copy(), signal.copy()
    price2[price2.index > cut] *= 2
    signal2[signal2.index > cut] = 1 - signal2[signal2.index > cut]

    r1 = stock_calendar_returns({"bitcoin": price}, cal)
    r2 = stock_calendar_returns({"bitcoin": price2}, cal)
    s1 = align_signals({"bitcoin": signal}, cal)
    s2 = align_signals({"bitcoin": signal2}, cal)
    pd.testing.assert_frame_equal(r1.loc[:cut], r2.loc[:cut])
    pd.testing.assert_frame_equal(s1.loc[:cut], s2.loc[:cut])


# ── Weight limits ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weights_are_never_negative_and_never_sum_above_one(seed):
    returns, exposures, members = _random_inputs(seed=seed)
    for name, w in _all_weights(returns, exposures, members).items():
        assert (w >= 0).all().all(), name
        assert (w.sum(axis=1) <= 1.0 + 1e-12).all(), name


def test_baselines_are_fully_invested_once_assets_are_in():
    returns, exposures, members = _random_inputs()
    w = _all_weights(returns, exposures, members)
    active = members.any(axis=1)
    for name in ("equal", "risk_parity"):
        np.testing.assert_allclose(w[name].loc[active].sum(axis=1), 1.0)
        assert (w[name].loc[~active] == 0).all().all()


def test_engine_rejects_borrowing_and_shorting():
    dates = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    with pytest.raises(ValueError, match="more than 100%"):
        run_portfolio_backtest(returns, pd.DataFrame({"A": 0.6, "B": 0.6}, index=dates), {"A": 1, "B": 1})
    with pytest.raises(ValueError, match="negative"):
        run_portfolio_backtest(returns, pd.DataFrame({"A": -0.1, "B": 0.5}, index=dates), {"A": 1, "B": 1})


def test_engine_requires_a_cost_for_every_asset():
    dates = pd.bdate_range("2021-01-04", periods=3)
    returns = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    with pytest.raises(KeyError, match="B"):
        run_portfolio_backtest(returns, pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates), {"A": 1.0})


def test_engine_rejects_a_held_asset_without_a_return():
    dates = pd.bdate_range("2021-01-04", periods=4)
    returns = pd.DataFrame({"A": [0.0, 0.0, np.nan, 0.0]}, index=dates)
    with pytest.raises(ValueError, match="no return"):
        run_portfolio_backtest(returns, pd.DataFrame({"A": 1.0}, index=dates), {"A": 1.0})


# ── Weight rules ──────────────────────────────────────────────────────────

def test_equal_weight_switches_from_halves_to_thirds_when_bitcoin_joins():
    dates = pd.bdate_range("2020-12-28", periods=10)
    members = membership(dates, {"spx": dates[0], "gold": dates[0], "bitcoin": "2021-01-01"})
    w = equal_weights(members)
    joined = dates[dates >= "2021-01-01"][0]
    np.testing.assert_allclose(w.loc[:joined - pd.Timedelta(days=1)].values,
                               [[0.5, 0.5, 0.0]] * len(w.loc[:joined - pd.Timedelta(days=1)]))
    np.testing.assert_allclose(w.loc[joined:].values, 1 / 3)


def test_inverse_vol_gives_twice_as_volatile_asset_half_the_weight():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=100)
    base = rng.normal(0, 0.01, 100)
    returns = pd.DataFrame({"calm": base, "wild": 2 * base}, index=dates)
    members = membership(dates, {"calm": dates[70], "wild": dates[70]})
    w = inverse_vol_weights(returns, members)
    np.testing.assert_allclose(w.iloc[70:]["calm"], 2 / 3)
    np.testing.assert_allclose(w.iloc[70:]["wild"], 1 / 3)


def test_inverse_vol_refuses_a_member_without_enough_history():
    dates = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 80)}, index=dates)
    members = membership(dates, {"A": dates[10]})
    with pytest.raises(ValueError, match="63-day volatility"):
        inverse_vol_weights(returns, members)


def test_regime_aware_equals_risk_parity_at_full_exposure_and_cash_at_zero():
    returns, exposures, members = _random_inputs()
    rp = inverse_vol_weights(returns, members)
    pd.testing.assert_frame_equal(regime_aware_weights(rp, exposures * 0 + 1), rp)
    assert (regime_aware_weights(rp, exposures * 0) == 0).all().all()


def test_regime_aware_sends_freed_weight_to_cash_not_other_assets():
    dates = pd.bdate_range("2021-01-04", periods=1)
    rp = pd.DataFrame({"spx": [0.6], "gold": [0.4]}, index=dates)
    e = pd.DataFrame({"spx": [0.0], "gold": [1.0]}, index=dates)
    w = regime_aware_weights(rp, e)
    assert w.loc[dates[0], "gold"] == 0.4      # unchanged, not scaled up
    assert w.sum(axis=1).iloc[0] == 0.4         # the other 60% is cash


def test_missing_exposure_means_no_position():
    dates = pd.bdate_range("2021-01-04", periods=2)
    rp = pd.DataFrame({"A": [1.0, 1.0]}, index=dates)
    w = regime_aware_weights(rp, pd.DataFrame({"A": [np.nan, 0.5]}, index=dates))
    np.testing.assert_allclose(w["A"], [0.0, 0.5])


# ── Calendar ──────────────────────────────────────────────────────────────

def test_bitcoin_weekend_move_lands_on_monday():
    # Fri 100, Sat 110, Sun 120, Mon 132: Monday's return is 132/100 - 1.
    btc = pd.Series([100.0, 110.0, 120.0, 132.0], index=pd.date_range("2021-01-08", periods=4))
    cal = pd.DatetimeIndex(["2021-01-08", "2021-01-11"])
    r = stock_calendar_returns({"bitcoin": btc}, cal)
    assert np.isclose(r.loc["2021-01-11", "bitcoin"], 0.32)


def test_missing_price_is_a_zero_return_not_a_gap():
    cal = pd.bdate_range("2021-01-04", periods=4)
    price = pd.Series([100.0, 101.0, 103.02], index=cal[[0, 1, 3]])  # day 2 missing
    r = stock_calendar_returns({"gold": price}, cal)["gold"]
    assert np.isnan(r.iloc[0])
    np.testing.assert_allclose(r.iloc[1:], [0.01, 0.0, 0.02])


def test_signal_is_carried_forward_up_to_five_days_then_zero():
    cal = pd.bdate_range("2021-01-04", periods=10)
    signal = pd.Series([0.7], index=[cal[0]])
    s = align_signals({"gold": signal}, cal)["gold"]
    # Days 0-5: 0 to 5 days stale -> kept; day 6 onwards: too stale -> 0.
    np.testing.assert_allclose(s.iloc[:6], 0.7)
    np.testing.assert_allclose(s.iloc[6:], 0.0)


def test_signal_is_nan_before_its_first_value_and_uses_weekend_values():
    cal = pd.DatetimeIndex(["2021-01-08", "2021-01-11"])            # Fri, Mon
    signal = pd.Series([0.2, 0.9], index=pd.DatetimeIndex(["2021-01-09", "2021-01-10"]))  # Sat, Sun
    s = align_signals({"bitcoin": signal}, cal)["bitcoin"]
    assert np.isnan(s.iloc[0])          # nothing known on Friday yet
    assert s.iloc[1] == 0.9             # Monday uses Sunday's signal


# ── Agreement with v1.0 ───────────────────────────────────────────────────

def test_single_asset_all_in_or_out_matches_the_v1_engine():
    """With one asset and weights of only 0 or 1, holdings cannot drift, so
    the portfolio engine must reproduce src/backtest/engine.py exactly."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=120)
    r = pd.Series(rng.normal(0, 0.01, 120), index=dates)
    target = pd.Series((rng.uniform(size=120) > 0.4).astype(float), index=dates)
    cash = pd.Series(0.0001, index=dates)

    v1 = run_backtest(pd.DataFrame({"returns": r}), target, cost_bps=3.0, cash_returns=cash)
    v2 = run_portfolio_backtest(r.to_frame("A"), target.to_frame("A"), {"A": 3.0}, cash_returns=cash)

    np.testing.assert_allclose(v2.daily["net_return"], v1["net_return"], atol=1e-15)
    np.testing.assert_allclose(v2.daily["cost"], v1["cost"], atol=1e-15)
    np.testing.assert_allclose(v2.held_weights["A"], v1["executed_weight"])
