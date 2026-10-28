"""
Baseline strategies the HMM must beat.

Each baseline is defined as a TARGET-WEIGHT function (decided at the close of
day t, using only data up to t). The weights then go through the exact same
engine as the HMM — same two-day execution lag, same cost model, same
evaluation window — so any difference in results comes from the signal, not
from the plumbing.
"""
import pandas as pd


def buy_and_hold_weights(index):
    """Always 100% invested."""
    return pd.Series(1.0, index=index)


def dma200_weights(close, window=200):
    """
    Long (100%) when Close > its trailing 200-day average, flat otherwise.

    The average is a trailing window, so it only uses past and current
    closes. During the first `window` days the average does not exist yet;
    we stay flat rather than guess.
    """
    dma = close.rolling(window=window).mean()
    weights = (close > dma).astype(float)
    weights[dma.isna()] = 0.0
    return weights


def vol_target_weights(volatility, max_weight=1.0):
    """
    Volatility targeting: hold less of the asset when it is unusually volatile.

        weight_t = min(1, long-run average volatility up to t / volatility_t)

    `volatility` is the trailing 20-day volatility (the model's
    volatility_20 feature). The "normal" level is its expanding mean, so
    both numbers use only data up to day t. At normal or calm volatility
    the strategy is fully invested; when recent volatility is twice its
    long-run norm it holds half. Capped at 100%: no borrowing.

    Why this baseline: the HMM mostly steps aside in volatile periods. If a
    one-line volatility rule does the same job, the HMM's extra machinery
    has to beat it, not just buy-and-hold.
    """
    vol = pd.Series(volatility, dtype=float)
    norm = vol.expanding().mean()
    weights = (norm / vol.where(vol > 0)).clip(upper=max_weight)
    return weights.fillna(0.0)  # flat until a volatility estimate exists
