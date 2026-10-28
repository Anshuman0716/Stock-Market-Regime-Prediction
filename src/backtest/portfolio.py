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

evaluate_portfolio() puts the four parts together for the pipeline.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import get_all_metrics

# Pre-registered values (docs/v2_preregistration.md, Section 13). Phase 3
# moves them into config.yaml; they stay here as the defaults the tests use.
RISK_PARITY_LOOKBACK = 63   # ~one quarter of stock trading days
MAX_SIGNAL_STALENESS = 5    # stock trading days before a signal counts as 0
EXECUTION_DELAY = 2         # decide at t, trade at t+1 close, earn from t+2
PORTFOLIO_ANN_FACTOR = 252  # the portfolio lives on the stock calendar

# How the portfolio is stored: as one extra "asset" next to the real ones,
# with its own three strategies. The strategy under test comes first.
PORTFOLIO_ID = "portfolio"
PORTFOLIO_STRATEGY = "regime_aware"
PORTFOLIO_BASELINES = ("equal_weight", "risk_parity")
PORTFOLIO_STRATEGIES = (PORTFOLIO_STRATEGY,) + PORTFOLIO_BASELINES


@dataclass(frozen=True)
class PortfolioSettings:
    """The pre-registered portfolio settings in one place.

    The defaults ARE the pre-registration (Section 13); config.yaml's
    `portfolio:` section repeats them and tests/test_config.py checks the
    two agree. They live in their own section so v1.0's config fingerprint
    does not change (src/db/store.py::get_config_hash).
    """
    assets: tuple = ("spx", "gold", "bitcoin")
    calendar_asset: str = "spx"   # whose traded instrument (SPY) sets the trading days and cash
    risk_parity_lookback: int = RISK_PARITY_LOOKBACK
    max_signal_staleness: int = MAX_SIGNAL_STALENESS
    execution_delay: int = EXECUTION_DELAY
    ann_factor: int = PORTFOLIO_ANN_FACTOR
    # Sharpe-difference test, the same method and settings as v1.0.
    block_size: int = 60
    n_bootstraps: int = 2000
    seed: int = 42
    alpha: float = 0.05
    # No forward verdict before this many forward trading days (~1 year).
    forward_min_days: int = 252

    @classmethod
    def from_config(cls, cfg):
        """Settings from config.yaml's `portfolio:` section, or None if it has none.

        Unknown keys are an error rather than ignored: a misspelt key would
        otherwise silently fall back to the default and the run would use a
        setting nobody wrote down.
        """
        section = (cfg or {}).get("portfolio")
        if section is None:
            return None
        section = dict(section)
        sig = dict(section.pop("significance", {}) or {})
        known = {f for f in cls.__dataclass_fields__}
        flat = {**section, **sig}
        unknown = sorted(set(flat) - known)
        if unknown:
            raise KeyError(f"Unknown setting(s) in config portfolio section: {unknown}")
        if "assets" in flat:
            flat["assets"] = tuple(flat["assets"])
        return cls(**flat)

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


# ── 5. Putting it together ────────────────────────────────────────────────

def portfolio_metrics(bt, ann_factor=PORTFOLIO_ANN_FACTOR):
    """The usual metrics dictionary for one portfolio backtest.

    Uses the same functions as the per-asset results, with one change:
    "Ann. Turnover" comes from the engine's real per-asset trades. The
    generic version measures changes in ONE weight, which for a portfolio
    would only see the invested total and miss every switch between assets.
    """
    m = get_all_metrics(bt.daily["net_return"], bt.daily["invested"], ann_factor, PORTFOLIO_ID,
                        cash_returns=bt.daily["cash_return"])
    years = len(bt.daily) / ann_factor
    m["Ann. Turnover"] = float(bt.daily["turnover"].sum() / years) if years > 0 else 0.0
    return m


def slice_backtest(bt, start=None, end=None):
    """The same backtest restricted to [start, end] (both inclusive).

    Used for the historical window (up to the v2.0 freeze) and the forward
    window (after it). The days themselves are unchanged: the portfolio is
    NOT restarted at `start`, so the forward window continues the positions
    built before it, as a live portfolio would.
    """
    return PortfolioBacktest(daily=bt.daily.loc[start:end], held_weights=bt.held_weights.loc[start:end],
                             turnover=bt.turnover.loc[start:end])


def as_backtest_frame(bt):
    """The daily frame in the single-asset layout (executed_weight = invested
    share), so per-asset helpers such as period_breakdown and the store's
    daily writer can be reused unchanged."""
    return bt.daily.rename(columns={"invested": "executed_weight"})


def evaluate_portfolio(prices, signals, entry_dates, calendar, cash_returns, cost_bps,
                       settings=PortfolioSettings()):
    """
    Build and backtest the three pre-registered portfolios.

    Parameters
    ----------
    prices : dict[str, pd.Series]
        asset -> traded instrument's close on the asset's own dates.
    signals : dict[str, pd.Series]
        asset -> v1.0 HMM target_weight, out-of-sample days only (exactly
        what the per-asset HMM strategy trades).
    entry_dates : dict[str, Timestamp]
        asset -> first out-of-sample day; the asset joins on the first
        calendar day on or after it.
    calendar : pd.DatetimeIndex
        Stock trading days (SPY dates), including history before the window
        so the risk-parity volatility has data from day one.
    cash_returns : pd.Series
        Daily cash return on the calendar.
    cost_bps : dict[str, float]
        asset -> cost per unit of turnover.

    Returns
    -------
    dict with oos_start, oos_end, backtests (strategy -> PortfolioBacktest),
    metrics (strategy -> metrics dict), exposures (aligned regime exposures).

    The window starts on the first calendar day any asset is in. Weights
    are cut to the window before the engine runs, so (as in v1.0) every
    portfolio starts flat and pays to get in.
    """
    assets = list(settings.assets)
    calendar = pd.DatetimeIndex(calendar).sort_values()
    returns = stock_calendar_returns({a: prices[a] for a in assets}, calendar)
    exposures = align_signals({a: signals[a] for a in assets}, calendar, settings.max_signal_staleness)
    members = membership(calendar, {a: entry_dates[a] for a in assets})

    rp = inverse_vol_weights(returns, members, settings.risk_parity_lookback)
    targets = {
        PORTFOLIO_STRATEGY: regime_aware_weights(rp, exposures),
        "equal_weight": equal_weights(members),
        "risk_parity": rp,
    }

    in_window = members.any(axis=1)
    if not in_window.any():
        raise ValueError("No asset enters the portfolio within the calendar.")
    start = in_window.idxmax()
    window_returns = returns.loc[start:]
    cash = pd.Series(cash_returns).reindex(calendar).loc[start:]

    backtests = {name: run_portfolio_backtest(window_returns, w.loc[start:], cost_bps, cash_returns=cash,
                                              delay=settings.execution_delay)
                 for name, w in targets.items()}
    return {
        "oos_start": start,
        "oos_end": calendar[-1],
        "backtests": backtests,
        "metrics": {name: portfolio_metrics(bt, settings.ann_factor) for name, bt in backtests.items()},
        "exposures": exposures.loc[start:],
    }
