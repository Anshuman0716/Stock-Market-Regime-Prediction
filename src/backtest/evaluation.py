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

from src.data.loader import get_ann_factor, get_cost_bps
from src.backtest.engine import run_backtest
from src.models.signal import compute_signal
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
    Target weight for the HMM strategy, taken from the shared signal
    definition (src/models/signal.py) so the backtest trades exactly what
    the dashboard shows and what paper trading would trade.
    """
    return compute_signal(probs, smoothing_window)["target_weight"]


def build_target_weights(features, signal):
    """Target weights (decided at each day's close) for every strategy."""
    return {
        "hmm": signal["target_weight"],
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
        signal : DataFrame from compute_signal (what the HMM strategy trades)
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
    cost_bps = get_cost_bps(config, asset_name)
    oos_start = out_of_sample_start(probs)
    oos = features.loc[oos_start:]
    signal = compute_signal(probs, config["model"]["smoothing_window"])

    backtests, metrics = {}, {}
    for name, weights in build_target_weights(features, signal).items():
        # Slicing the weights to the window means each strategy enters the
        # window flat: the engine's 2-day lag then makes it pay the entry
        # trade inside the window, exactly as a real launch would.
        bt = run_backtest(oos, weights.loc[oos_start:], cost_bps=cost_bps)
        backtests[name] = bt
        metrics[name] = get_all_metrics(bt["net_return"], bt["executed_weight"], ann_factor, asset_name)

    return {
        "signal": signal,
        "oos_start": oos_start,
        "oos_end": oos.index[-1],
        "ann_factor": ann_factor,
        "backtests": backtests,
        "metrics": metrics,
    }
