"""
Out-of-sample evaluation: score the HMM strategy and its baselines on the
SAME dates, and only on dates where the HMM genuinely existed.

Why this module exists
----------------------
Walk-forward training needs an initial training window (~9 years). During
that window there is no model, so there is no prediction. An earlier
version of the pipeline filled those missing predictions with P(Bull) = 0,
i.e. "sit in cash", and then measured the strategy over the full history.
That quietly credited the HMM with sitting out the 2000-2002 and 2008
crashes — years in which it did not exist — while buy-and-hold was charged
for them. The headline drawdown reduction was largely that artefact.

What is traded: the strategy's signal comes from the index, but the P&L
comes from the tradable instrument (config backtest.instruments: SPY,
ONEQ, GLD, BTC-USD) and uninvested capital earns the T-bill rate
(backtest.cash_ticker). See trading_returns().

The rule enforced here:
    Signals may use any PAST data (e.g. the 200-day average at the first
    out-of-sample date uses the 200 days before it). But P&L, costs and
    every metric are computed only from the first out-of-sample date
    onwards, and every strategy starts that window flat and pays to enter.
"""

from src.data.loader import get_ann_factor, get_cost_bps
from src.backtest.engine import run_backtest
from src.models.signal import compute_signal
import pandas as pd

from src.backtest.baselines import buy_and_hold_weights, dma200_weights, vol_target_weights
from src.backtest.metrics import get_all_metrics

STRATEGIES = ("hmm", "buy_hold", "dma200", "vol_target")
BASELINES = STRATEGIES[1:]


def trading_returns(asset_df, index, ann_factor):
    """
    Daily returns actually earned: on the traded instrument, and on cash.

    Parameters
    ----------
    asset_df : DataFrame from download_prices for one asset. Uses
        "Trade_Close" (the instrument) if present, else the index "Close";
        uses "Cash_Yield" (% a year) if present, else cash earns 0.
    index : the dates to return (the feature index).
    ann_factor : rows per year, to turn the annual yield into a daily rate.

    Returns
    -------
    (asset_returns, cash_returns) as Series on `index`.

    Why the yield is shifted by one day: the rate known at yesterday's close
    is what today's cash earns. Using today's quote would be a (tiny)
    look-ahead.
    """
    price = asset_df["Trade_Close"] if "Trade_Close" in asset_df else asset_df["Close"]
    asset_returns = price.pct_change(fill_method=None).reindex(index)
    if "Cash_Yield" in asset_df:
        cash = (asset_df["Cash_Yield"] / 100.0 / ann_factor).shift(1).reindex(index)
    else:
        cash = pd.Series(0.0, index=index)
    return asset_returns, cash


def out_of_sample_start(probs):
    """
    First date with a real out-of-sample prediction.

    walk_forward_predict leaves rows NaN for every date inside the initial
    training window, so the first row with any non-NaN probability is the
    first day the model could have been used.
    """
    has_pred = probs.notna().any(axis=1)
    if not has_pred.any():
        raise ValueError("No out-of-sample predictions found (all probabilities are NaN).")
    return has_pred.idxmax()


def build_target_weights(features, signal):
    """Target weights (decided at each day's close) for every strategy.

    Every signal is computed from the index and the model features, using
    data up to that day only.
    """
    return {
        "hmm": signal["target_weight"],
        "buy_hold": buy_and_hold_weights(features.index),
        "dma200": dma200_weights(features["Close"]),
        "vol_target": vol_target_weights(features["volatility_20"]),
    }


def evaluate_asset(features, probs, config, asset_name, asset_returns=None, cash_returns=None):
    """
    Backtest the HMM and every baseline over the identical out-of-sample window.

    Parameters
    ----------
    features : pd.DataFrame
        Feature frame for one asset; must contain 'returns' and 'Close'.
    probs : pd.DataFrame
        Filtered regime probabilities from walk_forward_predict (NaN before
        the out-of-sample start).
    config : dict
        Project config.
    asset_name : str
        Asset key, e.g. "spx". Drives costs and annualization.
    asset_returns, cash_returns : pd.Series or None
        What an invested and an uninvested dollar earn each day (see
        trading_returns). Default: the index's own returns and 0% cash.

    Returns
    -------
    dict with keys:
        signal : DataFrame from compute_signal (what the HMM strategy trades)
        oos_start, oos_end : pd.Timestamp
        ann_factor : int
        backtests : dict[strategy -> DataFrame from run_backtest]
        metrics : dict[strategy -> metrics dict]
    """
    for col in ("returns", "Close", "volatility_20"):
        if col not in features.columns:
            raise ValueError(f"features must contain '{col}'.")
    if not features.index.equals(probs.index):
        raise ValueError("features and probs must share the same index.")

    ann_factor = get_ann_factor(config, asset_name)
    cost_bps = get_cost_bps(config, asset_name)
    oos_start = out_of_sample_start(probs)
    signal = compute_signal(probs, config["model"]["smoothing_window"])

    if asset_returns is None:
        asset_returns = features["returns"]
    if cash_returns is None:
        cash_returns = pd.Series(0.0, index=features.index)
    oos = pd.DataFrame({"returns": asset_returns.reindex(features.index)}).loc[oos_start:]
    cash = cash_returns.reindex(features.index).loc[oos_start:]
    # The first out-of-sample day's return is never earned (the 2-day lag
    # keeps every strategy flat on it), so only later gaps matter.
    gaps = oos["returns"].iloc[1:].isna()
    if gaps.any():
        raise ValueError(
            f"{asset_name}: no price for the traded instrument on {int(gaps.sum())} out-of-sample day(s), "
            f"first {gaps[gaps].index[0].date()}. Did it launch after the out-of-sample start?")
    oos["returns"] = oos["returns"].fillna(0.0)

    backtests, metrics = {}, {}
    for name, weights in build_target_weights(features, signal).items():
        # Slicing the weights to the window means each strategy enters the
        # window flat: the engine's 2-day lag then makes it pay the entry
        # trade inside the window, exactly as a real launch would.
        bt = run_backtest(oos, weights.loc[oos_start:], cost_bps=cost_bps, cash_returns=cash)
        backtests[name] = bt
        metrics[name] = get_all_metrics(bt["net_return"], bt["executed_weight"], ann_factor, asset_name,
                                        cash_returns=bt["cash_return"])

    return {
        "signal": signal,
        "oos_start": oos_start,
        "oos_end": oos.index[-1],
        "ann_factor": ann_factor,
        "backtests": backtests,
        "metrics": metrics,
    }
