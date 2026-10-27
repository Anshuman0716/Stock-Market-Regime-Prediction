"""
Performance metrics for daily return series.

Every function takes `ann_factor` (rows per year: 252 for equities/gold,
365 for Bitcoin) instead of assuming 252, because annualizing with the
wrong factor silently mis-states every crypto number.

Sharpe definition — why arithmetic mean / std:
    Sharpe = mean(daily r) / std(daily r) * sqrt(ann_factor)
This is the standard definition and the one the significance tests in
src/stats/significance.py bootstrap. An earlier version used CAGR / vol,
which is a different statistic (it penalises volatility twice, because
CAGR already contains a volatility drag), so the reported Sharpe and the
Sharpe being tested for significance did not match. One definition, used
everywhere, avoids that.

Risk-free rate is taken as 0 throughout, for both the strategy and the
baselines, so comparisons between them are unaffected.
"""
import numpy as np
import pandas as pd

from src.backtest.risk import (
    compute_historical_var, compute_historical_cvar,
    compute_cornish_fisher_var,
)


def _clean(returns):
    """Float numpy array with NaNs removed (works for Series or arrays)."""
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def compute_cagr(returns, ann_factor):
    """Compound Annual Growth Rate: the constant yearly return that turns $1
    into the strategy's final value over the same number of years."""
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    years = len(r) / ann_factor
    growth = np.prod(1 + r)
    if growth <= 0:  # total wipe-out; CAGR is -100%
        return -1.0
    return growth ** (1 / years) - 1


def compute_volatility(returns, ann_factor):
    """Annualized volatility: daily std (ddof=1) scaled by sqrt(rows/year)."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    return np.std(r, ddof=1) * np.sqrt(ann_factor)


def compute_sharpe(returns, ann_factor, risk_free_rate=0.0):
    """Annualized Sharpe ratio = mean / std * sqrt(ann_factor).

    risk_free_rate is an annual rate; it is converted to a per-row rate
    before subtracting.
    """
    r = _clean(returns) - risk_free_rate / ann_factor
    if len(r) < 2:
        return 0.0
    std = np.std(r, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(r) / std * np.sqrt(ann_factor)


def compute_sortino(returns, ann_factor, risk_free_rate=0.0):
    """Annualized Sortino ratio = mean / downside deviation * sqrt(ann_factor).

    Downside deviation is sqrt(mean(min(r, 0)^2)) over ALL days, not the std
    of only the negative days. Taking the std of just the losing days would
    measure how much losses vary among themselves, not how big they are,
    and would ignore how often they happen.
    """
    r = _clean(returns) - risk_free_rate / ann_factor
    if len(r) == 0:
        return 0.0
    downside_dev = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside_dev == 0:
        return 0.0
    return np.mean(r) / downside_dev * np.sqrt(ann_factor)


def compute_drawdown_series(returns):
    """Drawdown at each date: equity / running peak - 1 (0 = at a new high)."""
    equity = (1 + pd.Series(returns).fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1


def compute_drawdowns(returns):
    """Max drawdown and the longest stretch (in rows) spent below a prior peak."""
    drawdowns = compute_drawdown_series(returns)
    if drawdowns.empty:
        return 0.0, 0
    max_dd = drawdowns.min()

    # Each time equity is back at a peak, a new "episode" id starts; summing
    # the underwater flags within an episode gives that episode's length.
    is_dd = drawdowns < 0
    duration = is_dd.astype(int).groupby((~is_dd).cumsum()).sum()
    max_duration = int(duration.max()) if not duration.empty else 0
    return max_dd, max_duration


def compute_calmar(returns, ann_factor):
    """Calmar ratio = CAGR / |max drawdown|: growth per unit of worst pain."""
    cagr = compute_cagr(returns, ann_factor)
    max_dd, _ = compute_drawdowns(returns)
    if max_dd == 0:
        return 0.0
    return cagr / abs(max_dd)


def compute_hit_rate(returns):
    """Share of days with a positive return."""
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def compute_time_in_market(weights):
    """Share of days with a non-zero position."""
    w = _clean(weights)
    if len(w) == 0:
        return 0.0
    return float((w != 0).mean())


def compute_avg_exposure(weights):
    """Average position size (0 = always in cash, 1 = always fully invested).

    For probability-weighted sizing this is more informative than time in
    market: a weight of 0.03 counts as "in the market" but is nearly cash.
    """
    w = _clean(weights)
    if len(w) == 0:
        return 0.0
    return float(w.mean())


def compute_turnover(weights, ann_factor):
    """Annualized turnover: total absolute weight traded per year.
    Going fully in and then fully out once a year counts as 2.0."""
    w = pd.Series(_clean(weights))
    if len(w) == 0:
        return 0.0
    years = len(w) / ann_factor
    return w.diff().abs().sum() / years


def get_all_metrics(returns, weights, ann_factor, asset_name):
    """Full metrics dictionary for one strategy on one asset."""
    max_dd, max_duration = compute_drawdowns(returns)
    return {
        "Asset": asset_name,
        "CAGR": compute_cagr(returns, ann_factor),
        "Ann. Vol": compute_volatility(returns, ann_factor),
        "Sharpe": compute_sharpe(returns, ann_factor),
        "Sortino": compute_sortino(returns, ann_factor),
        "Calmar": compute_calmar(returns, ann_factor),
        "Max DD": max_dd,
        "Max DD Duration (days)": max_duration,
        "Hit Rate": compute_hit_rate(returns),
        "Ann. Turnover": compute_turnover(weights, ann_factor),
        "Time in Market": compute_time_in_market(weights),
        "Avg. Exposure": compute_avg_exposure(weights),
        "VaR 95% (Hist)": compute_historical_var(returns, 0.95),
        "CVaR 95% (Hist)": compute_historical_cvar(returns, 0.95),
        "VaR 99% (Hist)": compute_historical_var(returns, 0.99),
        "CVaR 99% (Hist)": compute_historical_cvar(returns, 0.99),
        "VaR 95% (CF)": compute_cornish_fisher_var(returns, 0.95),
        "VaR 99% (CF)": compute_cornish_fisher_var(returns, 0.99),
        "Years": len(_clean(returns)) / ann_factor,
    }
