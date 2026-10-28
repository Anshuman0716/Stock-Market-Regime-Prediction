"""
Tests for out-of-sample evaluation, annualization and the shared Sharpe.

The key property: no strategy may be scored on dates before the HMM's first
real prediction, and every strategy for an asset is scored on the same dates.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import get_ann_factor
from src.backtest.evaluation import evaluate_asset, out_of_sample_start, hmm_target_weights
from src.backtest.metrics import compute_sharpe, compute_cagr, compute_sortino
from src.stats.significance import block_bootstrap_sharpe_diff, run_significance_suite

CFG = {
    "model": {"smoothing_window": 5},
    "backtest": {
        "annualization": {"default": 252, "bitcoin": 365},
        "costs": {"commission_bps": 0.5,
                  "spread_bps": {"default": 2.0, "spx": 1.0, "bitcoin": 5.0}},
    },
}


def _synthetic(n=600, oos_from=300, seed=0):
    """Features + probabilities where predictions only exist from row `oos_from`."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(0.0004, 0.01, n)
    close = 100 * np.cumprod(1 + rets)
    features = pd.DataFrame({"returns": rets, "Close": close}, index=idx)
    features["volatility_20"] = features["returns"].rolling(20).std()
    probs = pd.DataFrame(np.nan, index=idx, columns=["Bull", "Transition", "Crisis"])
    p_bull = rng.uniform(0, 1, n - oos_from)
    probs.iloc[oos_from:, 0] = p_bull
    probs.iloc[oos_from:, 1] = 1 - p_bull
    probs.iloc[oos_from:, 2] = 0.0
    return features, probs


def test_ann_factor_per_asset():
    assert get_ann_factor(CFG, "bitcoin") == 365
    assert get_ann_factor(CFG, "spx") == 252
    assert get_ann_factor({"model": {}}, "bitcoin") == 252  # no table -> default


def test_out_of_sample_start_is_first_prediction():
    features, probs = _synthetic(oos_from=300)
    assert out_of_sample_start(probs) == features.index[300]


def test_every_strategy_scored_only_on_oos_window():
    features, probs = _synthetic(oos_from=300)
    ev = evaluate_asset(features, probs, CFG, "spx")
    oos_index = features.index[300:]

    assert ev["oos_start"] == oos_index[0]
    for name, bt in ev["backtests"].items():
        # Same dates for all strategies, none from the training window.
        assert bt.index.equals(oos_index), name
        # Everyone starts the window flat (2-day execution lag) ...
        assert (bt["executed_weight"].iloc[:2] == 0).all(), name
    # ... and buy & hold then pays exactly one entry trade.
    bh = ev["backtests"]["buy_hold"]
    assert bh["turnover"].sum() == pytest.approx(1.0)
    assert ev["metrics"]["hmm"]["Years"] == pytest.approx(len(oos_index) / 252)


def test_pre_oos_crash_does_not_affect_results():
    """A crash inside the training window must not change any metric.

    This is the exact bug the evaluation window fixes: previously the HMM
    'sat in cash' through pre-model crashes and got credit for it.
    """
    features, probs = _synthetic(oos_from=300)
    crashed = features.copy()
    crashed.iloc[100:120, crashed.columns.get_loc("returns")] = -0.05

    base = evaluate_asset(features, probs, CFG, "spx")["metrics"]
    after = evaluate_asset(crashed, probs, CFG, "spx")["metrics"]
    for strat in ("hmm", "buy_hold"):
        assert after[strat]["Max DD"] == pytest.approx(base[strat]["Max DD"])
        assert after[strat]["CAGR"] == pytest.approx(base[strat]["CAGR"])


def test_hmm_weights_are_trailing_only():
    """Changing a future probability must not change today's weight."""
    _, probs = _synthetic(oos_from=300)
    w1 = hmm_target_weights(probs, 5)
    probs2 = probs.copy()
    probs2.iloc[400:, 0] = 0.0
    w2 = hmm_target_weights(probs2, 5)
    pd.testing.assert_series_equal(w1.iloc[:400], w2.iloc[:400])


def test_bitcoin_annualization_changes_cagr():
    features, probs = _synthetic()
    spx = evaluate_asset(features, probs, CFG, "spx")["metrics"]["buy_hold"]
    btc = evaluate_asset(features, probs, CFG, "bitcoin")["metrics"]["buy_hold"]
    # Same rows, but 365 rows/yr means the window spans fewer years.
    # (Sharpe/CAGR also differ slightly because Bitcoin's spread is wider.)
    assert btc["Years"] == pytest.approx(spx["Years"] * 252 / 365)
    assert btc["CAGR"] != pytest.approx(spx["CAGR"])


def test_sharpe_hand_computed():
    r = pd.Series([0.01, -0.005, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert compute_sharpe(r, 252) == pytest.approx(expected)


def test_sortino_uses_downside_deviation_over_all_days():
    r = pd.Series([0.02, -0.01, 0.01, -0.01])
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert compute_sortino(r, 252) == pytest.approx(r.mean() / dd * np.sqrt(252))


def test_cagr_hand_computed():
    r = pd.Series([0.0] * 251 + [0.10])  # +10% over exactly one year
    assert compute_cagr(r, 252) == pytest.approx(0.10)


def test_significance_uses_reported_sharpe():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.0005, 0.01, 500))
    b = pd.Series(rng.normal(0.0002, 0.01, 500))
    res = block_bootstrap_sharpe_diff(a, b, 252, n_bootstraps=50)
    assert res["diff"] == pytest.approx(compute_sharpe(a, 252) - compute_sharpe(b, 252))


def test_bootstrap_pvalue_behaves():
    """Identical strategies -> p near 1; a large real gap -> small p.

    Also guards the old bug where a small positive difference could be
    reported as p = 1.0 purely because of where the bootstrap was centred.
    """
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0.0003, 0.01, 2000))

    # Exactly the same returns: zero difference, nothing to reject.
    same = block_bootstrap_sharpe_diff(base, base, 252, n_bootstraps=300)
    assert same["p_value"] == pytest.approx(1.0)

    # A big, persistent daily edge must be detected.
    gap = block_bootstrap_sharpe_diff(base + 0.002, base, 252, n_bootstraps=300)
    assert 0 < gap["p_value"] < 0.01

    # A small but non-zero difference must not be pinned at the 1.0 cap.
    other = pd.Series(rng.normal(0.0003, 0.01, 2000))
    small = block_bootstrap_sharpe_diff(base, other, 252, n_bootstraps=300)
    assert small["p_value"] < 1.0


def test_significance_suite_holm_is_not_smaller_than_raw():
    rng = np.random.default_rng(2)
    idx = pd.RangeIndex(400)
    rets = {
        asset: {s: pd.Series(rng.normal(0.0003, 0.01, 400), index=idx)
                for s in ("hmm", "buy_hold", "dma200")}
        for asset in ("spx", "gold")
    }
    out = run_significance_suite(rets, {"spx": 252, "gold": 252}, n_bootstraps=50)
    assert len(out) == 4
    assert (out["p_value_holm"] >= out["p_value"] - 1e-12).all()


# ── Traded instrument and cash ───────────────────────────────────────────────

def test_trading_returns_use_the_instrument_and_yesterdays_cash_rate():
    from src.backtest.evaluation import trading_returns
    idx = pd.bdate_range("2020-01-01", periods=4)
    asset_df = pd.DataFrame({"Close": [100.0, 110.0, 121.0, 133.1],          # index (+10%/day)
                             "Trade_Close": [50.0, 51.0, 52.02, 53.0604],      # fund (+2%/day)
                             "Cash_Yield": [5.04, 2.52, 2.52, 2.52]}, index=idx)
    r, cash = trading_returns(asset_df, idx, ann_factor=252)
    np.testing.assert_allclose(r.iloc[1:], 0.02)            # P&L follows the fund, not the index
    assert np.isnan(cash.iloc[0])                           # no rate known before the first day
    assert cash.iloc[1] == pytest.approx(5.04 / 100 / 252)  # day 1 earns day 0's quoted rate
    assert cash.iloc[2] == pytest.approx(2.52 / 100 / 252)


def test_missing_instrument_prices_in_the_window_are_an_error():
    features, probs = _synthetic(oos_from=300)
    fund = features["returns"].copy()
    fund.iloc[400] = np.nan  # e.g. the fund had not launched yet
    with pytest.raises(ValueError, match="no price for the traded instrument"):
        evaluate_asset(features, probs, CFG, "spx", asset_returns=fund)


def test_sharpe_is_measured_above_cash():
    features, probs = _synthetic(oos_from=300)
    cash = pd.Series(0.0001, index=features.index)
    ev = evaluate_asset(features, probs, CFG, "spx", cash_returns=cash)
    bh = ev["backtests"]["buy_hold"]
    expected = compute_sharpe(bh["net_return"] - bh["cash_return"], 252)
    assert ev["metrics"]["buy_hold"]["Sharpe"] == pytest.approx(expected)
    assert set(ev["backtests"]) == {"hmm", "buy_hold", "dma200", "vol_target"}

    # A strategy that is always in cash earns the cash rate exactly, so its
    # excess return - and its Sharpe - is zero, however high the rate.
    always_cash = probs.copy()
    always_cash.loc[always_cash["Bull"].notna(), ["Bull", "Transition"]] = [0.0, 1.0]
    flat = evaluate_asset(features, always_cash, CFG, "spx", cash_returns=cash)
    assert flat["metrics"]["hmm"]["Sharpe"] == 0.0
    assert flat["metrics"]["hmm"]["CAGR"] > 0
