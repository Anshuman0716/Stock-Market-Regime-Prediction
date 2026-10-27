"""
The trading signal: ONE definition, used by the backtest, the SQLite store,
the dashboard and paper trading.

Why a single module
-------------------
Before this, the backtest traded on a 21-day mean of P(Bull), the README
described a 21-day rolling MODE of labels, the dashboard showed RAW labels,
and paper trading used RAW P(Bull). Four definitions meant the regime on
the dashboard was not the regime that was backtested, and the live trader
did not trade what was backtested. Now every consumer calls compute_signal().

Definition
----------
1. Start from the HMM's FILTERED probabilities P(Bull), P(Transition),
   P(Crisis) for each day (each uses only data up to that day).
2. Smooth each probability with a TRAILING mean over `window` days.
   Trailing (past and current days only) so no future day leaks in; a
   centered window would. Averaging probabilities rather than taking the
   mode of labels keeps the confidence information: three days at
   P(Bull)=0.51 and three at 0.99 are not the same thing.
3. Smoothed regime = the label with the highest smoothed probability.
4. Target weight = smoothed P(Bull), clipped to [0, 1], with positions
   under 0.1% set to 0. This is the share of the portfolio held in the
   asset, decided at the day's close.

The averages of rows that each sum to 1 also sum to 1, so the smoothed
probabilities stay a valid distribution.
"""
import numpy as np
import pandas as pd

REGIME_LABELS = ["Bull", "Transition", "Crisis"]

# Smallest position worth holding: 0.1% of the portfolio. Smoothed P(Bull)
# often decays to values like 1e-7 rather than exactly 0. A position that
# small is worth cents, nobody would trade it, and floating-point noise
# flips it between "zero" and "not zero" from run to run, which made
# metrics such as time-in-market and hit rate unstable. Below this
# threshold the weight is 0.
MIN_WEIGHT = 1e-3


def smooth_probabilities(probs, window):
    """Trailing `window`-day mean of each regime probability.

    min_periods=1 lets the signal start on the first day a prediction
    exists instead of waiting `window` days. Rows before the first
    prediction stay NaN.
    """
    return probs[REGIME_LABELS].rolling(window=window, min_periods=1).mean()


def regime_labels(probs):
    """Most likely regime per row; NaN where there is no prediction."""
    has_pred = probs[REGIME_LABELS].notna().any(axis=1)
    labels = pd.Series(np.nan, index=probs.index, dtype="object")
    labels[has_pred] = probs.loc[has_pred, REGIME_LABELS].idxmax(axis=1)
    return labels


def target_weight(smoothed_probs, min_weight=MIN_WEIGHT):
    """Portfolio weight = smoothed P(Bull), clipped to [0, 1].

    0 where there is no prediction, and 0 below `min_weight` (see MIN_WEIGHT).
    """
    w = smoothed_probs["Bull"].fillna(0.0).clip(lower=0.0, upper=1.0)
    return w.where(w >= min_weight, 0.0)


def compute_signal(probs, window):
    """
    Turn filtered regime probabilities into the full signal frame.

    Parameters
    ----------
    probs : pd.DataFrame
        Filtered probabilities with columns Bull / Transition / Crisis
        (NaN before the first out-of-sample prediction).
    window : int
        Trailing smoothing window in rows (config: model.smoothing_window).

    Returns
    -------
    pd.DataFrame with columns:
        raw_label, p_bull, p_transition, p_crisis           (unsmoothed)
        label, sp_bull, sp_transition, sp_crisis            (smoothed)
        target_weight
    """
    smoothed = smooth_probabilities(probs, window)
    return pd.DataFrame({
        "raw_label": regime_labels(probs),
        "p_bull": probs["Bull"],
        "p_transition": probs["Transition"],
        "p_crisis": probs["Crisis"],
        "label": regime_labels(smoothed),
        "sp_bull": smoothed["Bull"],
        "sp_transition": smoothed["Transition"],
        "sp_crisis": smoothed["Crisis"],
        "target_weight": target_weight(smoothed),
    }, index=probs.index)
