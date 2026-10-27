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

The rule enforced here:
    Signals may use any PAST data (e.g. the 200-day average at the first
    out-of-sample date uses the 200 days before it). But P&L, costs and
    every metric are computed only from the first out-of-sample date
    onwards, and every strategy starts that window flat and pays to enter.
"""
import pandas as pd

from src.data.loader import get_ann_factor
from src.backtest.engine import run_backtest
from src.backtest.risk import compute_target_weight_proba
from src.backtest.baselines import buy_and_hold_weights, dma200_weights
from src.backtest.metrics import get_all_metrics

STRATEGIES = ("hmm", "buy_hold", "dma200")


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


def hmm_target_weights(probs, smoothing_window):
    """
    Target weight for the HMM strategy = trailing mean of P(Bull).

    Why a trailing mean: raw daily P(Bull) can flip back and forth, and each
    flip costs a trade. Averaging over the last `smoothing_window` days damps
    that churn. It is trailing (past and current days only), never centered,
    so no future day leaks into today's weight.

    min_periods=1 lets the signal start on the first out-of-sample day
    (averaging however many real predictions exist so far) instead of
    forcing ~20 days of cash while the window fills up. Days before the
    out-of-sample start have no prediction and get weight 0, but those days
    are excluded from evaluation anyway.
    """
    p_bull = probs["Bull"].rolling(window=smoothing_window, min_periods=1).mean()
    return compute_target_weight_proba(pd.DataFrame({"Bull": p_bull}))


def build_target_weights(features, probs, config):
    """Target weights (decided at each day's close) for every strategy."""
    return {
        "hmm": hmm_target_weights(probs, config["model"]["smoothing_window"]),
        "buy_hold": buy_and_hold_weights(features.index),
        "dma200": dma200_weights(features["Close"]),
    }


def evaluate_asset(features, probs, config, asset_name):
    """
    Backtest the HMM and both baselines over the identical out-of-sample window.

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

    Returns
    -------
    dict with keys:
        oos_start, oos_end : pd.Timestamp
        ann_factor : int
        backtests : dict[strategy -> DataFrame from run_backtest]
        metrics : dict[strategy -> metrics dict]
    """
    for col in ("returns", "Close"):
        if col not in features.columns:
            raise ValueError(f"features must contain '{col}'.")
    if not features.index.equals(probs.index):
        raise ValueError("features and probs must share the same index.")

    ann_factor = get_ann_factor(config, asset_name)
    oos_start = out_of_sample_start(probs)
    oos = features.loc[oos_start:]

    backtests, metrics = {}, {}
    for name, weights in build_target_weights(features, probs, config).items():
        # Slicing the weights to the window means each strategy enters the
        # window flat: the engine's 2-day lag then makes it pay the entry
        # trade inside the window, exactly as a real launch would.
        bt = run_backtest(oos, weights.loc[oos_start:], asset_name=asset_name, use_dynamic_costs=True)
        backtests[name] = bt
        metrics[name] = get_all_metrics(bt["net_return"], bt["executed_weight"], ann_factor, asset_name)

    return {
        "oos_start": oos_start,
        "oos_end": oos.index[-1],
        "ann_factor": ann_factor,
        "backtests": backtests,
        "metrics": metrics,
    }
