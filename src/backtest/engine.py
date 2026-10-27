"""
Backtest engine: turns target weights into daily strategy returns.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_backtest(df, target_weights, cost_bps):
    """
    Core backtest engine.

    CRITICAL SIGNAL LAG MECHANICS:
    A signal derived at the close of day t cannot be traded at the close of day t.
    We assume execution occurs at the close of day t+1.
    This means a signal generated on Monday (t) causes a trade execution on Tuesday
    close (t+1), and the portfolio begins earning the asset's return on Wednesday (t+2).

    Therefore, the weight held during day t's return is the target_weight from day t-2.
    Getting this wrong is the #1 cause of fake Sharpe ratios in backtests.

    Parameters
    ----------
    df : pd.DataFrame
        Asset data containing a 'returns' column.
    target_weights : pd.Series or np.ndarray
        The desired portfolio weights aligned to df.index (computed at day t close).
    cost_bps : float
        Cost in basis points per unit of turnover (spread + commission).
        Required, with no default, so a backtest can never silently run
        cost-free. Look it up with src.data.loader.get_cost_bps(config, asset).

    Returns
    -------
    pd.DataFrame
        Frame containing strategy returns, executed weights, and transaction costs.
    """
    weights = pd.Series(target_weights, index=df.index)

    # The target decided at t is executed at t+1 close, so the weight in
    # effect FOR day t's return is target_weight[t-2].
    executed_weights = weights.shift(2).fillna(0.0)

    # Turnover at t = change in executed weight (the trade made at t-1 close
    # to hold executed_weight[t]).
    turnover = executed_weights.diff().fillna(0.0).abs()

    # Transaction costs (in decimal form), paid on the day the turnover occurs.
    if cost_bps == 0.0:
        logger.warning("Backtest running with ZERO transaction costs.")
    t_costs = turnover * (cost_bps / 10000.0)

    gross_returns = executed_weights * df['returns']
    net_returns = gross_returns - t_costs

    return pd.DataFrame({
        'gross_return': gross_returns,
        'net_return': net_returns,
        'executed_weight': executed_weights,
        'turnover': turnover,
        'cost': t_costs
    }, index=df.index)
