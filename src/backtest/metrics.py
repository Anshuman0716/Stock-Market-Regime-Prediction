import numpy as np
import pandas as pd

def compute_cagr(returns, ann_factor):
    """Compound Annual Growth Rate"""
    if len(returns) == 0: return 0.0
    cum_ret = (1 + returns).prod()
    years = len(returns) / ann_factor
    return cum_ret ** (1 / years) - 1 if years > 0 else 0.0

def compute_volatility(returns, ann_factor):
    """Annualized Volatility"""
    if len(returns) == 0: return 0.0
    return returns.std() * np.sqrt(ann_factor)

def compute_sharpe(returns, ann_factor, risk_free_rate=0.0):
    """Annualized Sharpe Ratio"""
    vol = compute_volatility(returns, ann_factor)
    if vol == 0: return 0.0
    cagr = compute_cagr(returns, ann_factor)
    return (cagr - risk_free_rate) / vol

def compute_sortino(returns, ann_factor, risk_free_rate=0.0):
    """Annualized Sortino Ratio"""
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(ann_factor)
    if downside_vol == 0 or pd.isna(downside_vol): return 0.0
    cagr = compute_cagr(returns, ann_factor)
    return (cagr - risk_free_rate) / downside_vol

def compute_drawdowns(returns):
    """Compute drawdown series and max drawdown"""
    cum_rets = (1 + returns).cumprod()
    rolling_max = cum_rets.cummax()
    drawdowns = cum_rets / rolling_max - 1
    max_dd = drawdowns.min()
    
    # Calculate drawdown duration
    is_dd = drawdowns < 0
    duration = is_dd.astype(int).groupby((~is_dd).cumsum()).sum()
    max_duration = duration.max() if not duration.empty else 0
    
    return max_dd, max_duration

def compute_calmar(returns, ann_factor):
    """Annualized Calmar Ratio"""
    cagr = compute_cagr(returns, ann_factor)
    max_dd, _ = compute_drawdowns(returns)
    if max_dd == 0: return 0.0
    return cagr / abs(max_dd)

def compute_hit_rate(returns):
    """Percentage of positive trading days"""
    if len(returns) == 0: return 0.0
    return (returns > 0).sum() / len(returns)

def compute_time_in_market(weights):
    """Percentage of days with non-zero weight"""
    if len(weights) == 0: return 0.0
    return (weights != 0).sum() / len(weights)

def compute_turnover(weights, ann_factor):
    """Annualized Turnover"""
    if len(weights) == 0: return 0.0
    total_turnover = weights.diff().abs().sum()
    years = len(weights) / ann_factor
    return total_turnover / years if years > 0 else 0.0

from src.backtest.risk import (
    compute_historical_var, compute_historical_cvar,
    compute_cornish_fisher_var, compute_cornish_fisher_cvar
)

def get_all_metrics(returns, weights, ann_factor, asset_name):
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
        "VaR 95% (Hist)": compute_historical_var(returns, 0.95),
        "CVaR 95% (Hist)": compute_historical_cvar(returns, 0.95),
        "VaR 99% (Hist)": compute_historical_var(returns, 0.99),
        "CVaR 99% (Hist)": compute_historical_cvar(returns, 0.99),
        "VaR 95% (CF)": compute_cornish_fisher_var(returns, 0.95),
        "VaR 99% (CF)": compute_cornish_fisher_var(returns, 0.99)
    }
