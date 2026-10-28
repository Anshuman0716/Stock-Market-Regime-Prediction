"""
Multi-asset portfolio backtest for v2.0 (docs/v2_preregistration.md).

v1.0 traded each asset on its own: one asset, the rest in cash. v2.0 holds
several assets at once and asks whether the per-asset regime signals help
a PORTFOLIO beat simple portfolio rules. This module has four parts:

1. Putting every asset on ONE calendar (US stock trading days), both its
   returns and its signal (stock_calendar_returns, align_signals).
2. Which assets are in the portfolio on each day (membership).
3. The three weight rules being compared (equal_weights,
   inverse_vol_weights, regime_aware_weights).
4. The engine that turns target weights into daily returns, with costs per
   asset and cash on whatever is not invested (run_portfolio_backtest).

Every weight on day t uses only data up to and including day t. The engine
then applies the same two-day delay as v1.0 (decide at t, trade at t+1,
earn from t+2), so the weight rules never need to worry about timing.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Pre-registered values (docs/v2_preregistration.md, Section 13). Phase 3
# moves them into config.yaml; they stay here as the defaults the tests use.
RISK_PARITY_LOOKBACK = 63   # ~one quarter of stock trading days
MAX_SIGNAL_STALENESS = 5    # stock trading days before a signal counts as 0
EXECUTION_DELAY = 2         # decide at t, trade at t+1 close, earn from t+2

# Floating-point slack when checking that weights add up to at most 100%.
_SUM_TOLERANCE = 1e-9


# ── 1. One calendar ───────────────────────────────────────────────────────

def stock_calendar_returns(prices, calendar):
    """
    Daily returns of each asset measured on the stock calendar.

    Parameters
    ----------
    prices : dict[str, pd.Series]
        asset -> close price of its traded instrument, on its OWN dates
        (Bitcoin has weekends, GLD has a few days SPY does not, etc.).
    calendar : pd.DatetimeIndex
        The stock trading days to measure on.

    Why carry the last price forward instead of dropping days: Bitcoin
    moves on Saturday and Sunday too. Taking Monday's price over Friday's
    price puts the whole weekend move into Monday's return, so no move is
    lost; dropping weekends would silently throw those returns away. The
    same rule makes a missing price a 0% return for that day. Only PAST
    prices are carried forward, so day t never sees a later price.

    Returns NaN before an asset's first price.
    """
    out = {}
    for asset, price in prices.items():
        price = price.dropna().sort_index()
        # method="ffill": the last price dated on or before each calendar day.
        on_calendar = price.reindex(calendar, method="ffill")
        out[asset] = on_calendar.pct_change(fill_method=None)
    return pd.DataFrame(out, index=calendar)


def align_signals(signals, calendar, max_staleness=MAX_SIGNAL_STALENESS):
    """
    Each asset's regime exposure (v1.0 target_weight) on the stock calendar.

    The value on stock day t is the asset's latest signal dated on or
    before t. If that signal is more than `max_staleness` stock trading
    days old, the exposure is 0 (cash) rather than a stale bet.

    Why a staleness limit: carrying a signal over a holiday or a missing
    futures day is reasonable; carrying it over weeks of missing data would
    mean trading on information the model no longer has.

    Staleness counts the stock trading days AFTER the signal's date, up to
    and including t: a signal from today is 0 days stale, Friday's signal
    on Monday is 1 day stale, and a Sunday Bitcoin signal on Monday is 1.

    Returns NaN where an asset has no signal yet (before its first value).
    """
    out = {}
    positions = np.arange(len(calendar))
    for asset, signal in signals.items():
        signal = signal.dropna().sort_index()
        if signal.empty:
            out[asset] = pd.Series(np.nan, index=calendar)
            continue
        value = signal.reindex(calendar, method="ffill")
        # Date of the signal being used on each calendar day.
        source_date = pd.Series(signal.index, index=signal.index).reindex(calendar, method="ffill")
        # Index of the first calendar day strictly after the source date.
        first_after = np.searchsorted(calendar.values, source_date.values, side="right")
        staleness = positions - first_after + 1
        value = value.where(~((staleness > max_staleness) & value.notna()), 0.0)
        out[asset] = value
    return pd.DataFrame(out, index=calendar)


# ── 2. Membership ─────────────────────────────────────────────────────────

def membership(calendar, entry_dates):
    """
    Which assets are in the portfolio on each day (boolean frame).

    An asset joins on the first calendar day on or after its entry date
    (its first out-of-sample signal) and stays in. Every portfolio uses the
    same membership, so on any day all three hold the same set of assets
    and are compared like for like.
    """
    return pd.DataFrame({asset: calendar >= pd.Timestamp(start) for asset, start in entry_dates.items()},
                        index=calendar)


# ── 3. Weight rules ───────────────────────────────────────────────────────

def equal_weights(members):
    """Equal-weight baseline: 1/N for each of the N member assets.

    Needs no estimate at all, so it cannot be wrong about the future; it is
    the standard hard-to-beat benchmark for any allocation rule. Days with
    no members are all cash.
    """
    members = members.astype(bool)
    n = members.sum(axis=1)
    return members.astype(float).div(n.where(n > 0), axis=0).fillna(0.0)


def inverse_vol_weights(returns, members, lookback=RISK_PARITY_LOOKBACK):
    """
    Plain risk parity: weights proportional to 1 / volatility, fully invested.

        sigma_i(t) = std (ddof=1) of asset i's returns over the `lookback`
                     days ending at t (inclusive)
        w_i(t)     = (1 / sigma_i(t)) / sum over members of (1 / sigma_j(t))

    Why: the assets differ a lot in volatility (Bitcoin is several times as
    volatile as the S&P 500). With equal weights the most volatile asset
    dominates the portfolio's risk; inverse volatility gives each asset a
    roughly equal share of it. "Plain" because correlations are ignored,
    which keeps the rule simple and avoids estimating a covariance matrix
    from a few months of data.

    The rolling window is trailing (past and current days only). A member
    without a full window of history raises an error rather than quietly
    getting no weight: the pre-registration says this never happens, so if
    it does, the input data is wrong.
    """
    members = members.astype(bool).reindex(index=returns.index, columns=returns.columns, fill_value=False)
    sigma = returns.rolling(window=lookback, min_periods=lookback).std(ddof=1)

    missing = members & ~(sigma > 0)
    if missing.to_numpy().any():
        day, asset = missing.stack().loc[lambda s: s].index[0]
        raise ValueError(f"{asset} is in the portfolio on {day.date()} but has no {lookback}-day "
                         "volatility estimate (too little history or zero volatility).")

    inv = (1.0 / sigma).where(members, 0.0).fillna(0.0)
    total = inv.sum(axis=1)
    return inv.div(total.where(total > 0), axis=0).fillna(0.0)


def regime_aware_weights(base_weights, exposures):
    """
    The strategy under test: base (risk-parity) weights scaled by each
    asset's own regime exposure, the rest in cash.

        w_i(t) = base_i(t) * e_i(t),   e_i(t) in [0, 1]

    Why scale instead of re-spreading: when the model steps out of one
    asset, its slice goes to cash. Moving it into the other assets would
    concentrate the portfolio in whatever is left exactly when the model
    sees stress. And because the result equals risk parity whenever every
    exposure is 100%, comparing it with risk parity measures only what the
    regime signals add.

    A missing exposure (no signal) counts as 0: no signal, no position.
    """
    e = exposures.reindex(index=base_weights.index, columns=base_weights.columns)
    return base_weights * e.fillna(0.0).clip(lower=0.0, upper=1.0)


# ── 4. Engine ─────────────────────────────────────────────────────────────

@dataclass
class PortfolioBacktest:
    """Result of run_portfolio_backtest.

    daily : one row per day with gross_return, net_return, cost, turnover
        (sum over assets), cash_return and invested (sum of held weights).
    held_weights : weight held in each asset during each day's return.
    turnover : per-asset trade size on each day.
    """
    daily: pd.DataFrame
    held_weights: pd.DataFrame
    turnover: pd.DataFrame


def _check_weights(weights):
    if (weights < 0).to_numpy().any():
        raise ValueError("Target weights must not be negative (no shorting).")
    if (weights.sum(axis=1) > 1.0 + _SUM_TOLERANCE).any():
        raise ValueError("Target weights must not add up to more than 100% (no borrowing).")


def run_portfolio_backtest(returns, target_weights, cost_bps, cash_returns=None, delay=EXECUTION_DELAY):
    """
    Turn daily target weights for several assets into daily portfolio returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return of each asset's instrument (columns = assets).
    target_weights : pd.DataFrame
        Weight decided at each day's close, same index and columns.
    cost_bps : dict[str, float]
        Cost per unit of turnover for each asset, in basis points
        (src.data.loader.get_cost_bps). Required for every asset, so a
        backtest can never silently run cost-free on one of them.
    cash_returns : pd.Series or None
        Daily return on uninvested capital (None = 0).
    delay : int
        Days between deciding a weight and earning with it (2, as in v1.0).

    Mechanics (same conventions as src/backtest/engine.py)
    ------------------------------------------------------
    * Held weight on day t = target weight from day t - delay. Before the
      first target takes effect the portfolio is in cash, so every
      strategy starts flat and pays to enter.
    * Return = sum of held weight x asset return + the rest x cash return.
    * Trades are measured from DRIFTED weights: after day t-1's returns,
      an asset that rose is a bigger share of the portfolio. Rebalancing to
      day t's held weight means trading |held_t - drifted_{t-1}| of each
      asset. Comparing target to target would make constant-weight rules
      such as equal weight look free to hold, when they really trade every
      day to stay balanced.
    * Cost = trade size x that asset's cost, summed over assets, charged on
      day t (the trade happens at the close of t-1). Cash moves are free.
      Drift uses gross returns; the effect of the few basis points of cost
      on next day's weights is second-order and ignored.
    """
    if not target_weights.index.equals(returns.index):
        raise ValueError("returns and target_weights must cover the same dates.")
    assets = list(returns.columns)
    missing_costs = [a for a in assets if a not in cost_bps]
    if missing_costs:
        raise KeyError(f"No trading cost given for: {missing_costs}.")

    target = target_weights.reindex(columns=assets).fillna(0.0)
    _check_weights(target)

    held = target.shift(delay).fillna(0.0)
    cash = (pd.Series(0.0, index=returns.index) if cash_returns is None
            else pd.Series(cash_returns, index=returns.index).fillna(0.0))

    # An asset with no return on a day it is held is a data error; an asset
    # with no return on a day it is NOT held (before it exists) is fine.
    bad = held.gt(0) & returns.isna()
    if bad.to_numpy().any():
        day, asset = bad.stack().loc[lambda s: s].index[0]
        raise ValueError(f"{asset} is held on {day.date()} but has no return that day.")
    r = returns.fillna(0.0)

    invested = held.sum(axis=1)
    gross = (held * r).sum(axis=1) + (1.0 - invested) * cash

    # Weights at the end of each day after prices moved, then lagged one day:
    # what yesterday's holdings had become just before today's trade.
    grown = held * (1.0 + r)
    drifted = grown.div(1.0 + gross, axis=0).shift(1).fillna(0.0)
    turnover = (held - drifted).abs()

    costs = turnover.mul(pd.Series({a: cost_bps[a] for a in assets}) / 10_000.0, axis=1).sum(axis=1)

    daily = pd.DataFrame({
        "gross_return": gross,
        "net_return": gross - costs,
        "cost": costs,
        "turnover": turnover.sum(axis=1),
        "cash_return": cash,
        "invested": invested,
    }, index=returns.index)
    return PortfolioBacktest(daily=daily, held_weights=held, turnover=turnover)
