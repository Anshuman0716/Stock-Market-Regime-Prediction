"""
Baseline strategies the HMM must beat.

Each baseline is defined as a TARGET-WEIGHT function (decided at the close of
day t, using only data up to t). The weights then go through the exact same
engine as the HMM — same two-day execution lag, same cost model, same
evaluation window — so any difference in results comes from the signal, not
from the plumbing.
"""
import pandas as pd

from src.backtest.engine import run_backtest


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


# --- Convenience wrappers (kept for notebooks that call them directly) ------

def get_buy_and_hold(df, cost_bps):
    """Run the buy-and-hold baseline through the backtest engine."""
    return run_backtest(df, buy_and_hold_weights(df.index), cost_bps=cost_bps)


def get_200dma_filter(df, cost_bps):
    """Run the 200-day moving-average baseline through the backtest engine."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' to compute 200DMA.")
    return run_backtest(df, dma200_weights(df["Close"]), cost_bps=cost_bps)
