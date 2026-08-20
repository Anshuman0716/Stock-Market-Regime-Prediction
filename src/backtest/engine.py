import logging
import pandas as pd
import numpy as np

from src.backtest.risk import estimate_transaction_costs

logger = logging.getLogger(__name__)

def compute_target_weight(regime_labels):
    """
    Interface for sizing positions based on regimes.
    Currently implements binary exposure (1 for Bull, 0 otherwise).
    """
    return np.where(regime_labels == 'Bull', 1.0, 0.0)

def run_backtest(df, target_weights, asset_name="spx", use_dynamic_costs=True, cost_bps=0.0):
    """
    Core walk-forward backtest engine.
    
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
        Asset data containing 'returns' column.
    target_weights : pd.Series or np.ndarray
        The desired portfolio weights aligned to df.index (computed at day t close).
    asset_name : str
        Name of the asset, used to estimate transaction costs (spread + commission).
    use_dynamic_costs : bool
        If True, ignores cost_bps and uses risk.estimate_transaction_costs.
    cost_bps : float
        Fallback flat transaction cost in basis points per rebalance (if dynamic costs off).
        
    Returns
    -------
    pd.DataFrame
        Frame containing strategy returns, executed weights, and transaction costs.
    """
    weights = pd.Series(target_weights, index=df.index)
    
    # Lag 1: Executed weight. The target decided at t is executed at t+1 close.
    # So the weight *held* entering day t+2 is target_weight[t].
    # Therefore, the executed_weight in effect FOR the return of day t is target_weight[t-2].
    executed_weights = weights.shift(2).fillna(0.0)
    
    # Calculate turnover (change in executed weight)
    # The trade happens at t-1 close to achieve executed_weight for day t.
    # So turnover at t is the difference between executed_weight[t] and executed_weight[t-1].
    turnover = executed_weights.diff().fillna(0.0).abs()
    
    # Transaction costs (in decimal form)
    if use_dynamic_costs:
        t_costs = estimate_transaction_costs(turnover, asset_name)
    else:
        if cost_bps == 0.0:
            logger.warning("\nLOUD WARNING: Backtest running with ZERO transaction costs!\n")
        t_costs = turnover * (cost_bps / 10000.0)
    
    # Strategy gross return
    # Asset returns for day t * weight held during day t
    gross_returns = executed_weights * df['returns']
    
    # Strategy net return (costs are paid on the day the turnover occurs)
    net_returns = gross_returns - t_costs
    
    return pd.DataFrame({
        'gross_return': gross_returns,
        'net_return': net_returns,
        'executed_weight': executed_weights,
        'turnover': turnover,
        'cost': t_costs
    }, index=df.index)
