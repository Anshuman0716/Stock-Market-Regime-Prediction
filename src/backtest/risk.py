"""
Daily tail-risk metrics: historical and Cornish-Fisher VaR / CVaR.

Losses are returned as positive numbers (0.02 = a 2% one-day loss).
Missing returns are dropped first, so an empty or all-missing series gives
NaN instead of an error.
"""
import numpy as np
from scipy.stats import norm

# --- Risk Metrics (VaR / CVaR) ---
# Note: Horizon is DAILY.
# To scale to T days, one might use sqrt(T), but this understates risk under volatility clustering.

def compute_historical_var(returns, confidence_level=0.95):
    """Historical Daily Value at Risk (VaR): the loss exceeded on (1 - level) of days."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return -np.percentile(returns, 100 * (1 - confidence_level))

def compute_historical_cvar(returns, confidence_level=0.95):
    """Historical Daily Conditional Value at Risk (CVaR): the average loss on the days beyond VaR."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    var = compute_historical_var(returns, confidence_level)
    tail_losses = returns[returns <= -var]
    if len(tail_losses) == 0:
        return np.nan
    return -tail_losses.mean()

def compute_cornish_fisher_var(returns, confidence_level=0.95):
    """
    Cornish-Fisher Daily Value at Risk (VaR), adjusting for skewness and kurtosis.
    Useful because strategy returns are often skewed and fat-tailed.
    Divergence from historical VaR indicates significant non-normality.
    """
    returns = returns.dropna()
    if len(returns) < 4:
        return np.nan

    z = norm.ppf(1 - confidence_level)
    mean = returns.mean()
    std = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis() # pandas kurtosis is excess kurtosis

    if std == 0:
        return np.nan

    # Cornish-Fisher expansion
    z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * (skew**2) / 36

    return -(mean + z_cf * std)

def compute_cornish_fisher_cvar(returns, confidence_level=0.95):
    """
    Cornish-Fisher Daily CVaR. Approximated using the tail of the Cornish-Fisher VaR.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    var_cf = compute_cornish_fisher_var(returns, confidence_level)
    tail_losses = returns[returns <= -var_cf]
    if len(tail_losses) == 0:
        return np.nan
    return -tail_losses.mean()


# Position sizing (target weight = smoothed P(Bull)) is defined once, in
# src/models/signal.py.


# Transaction costs live in config.yaml (backtest.costs) and are looked up
# with src.data.loader.get_cost_bps, so there is one place to change them.
