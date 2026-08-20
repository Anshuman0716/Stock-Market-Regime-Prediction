"""
Risk metrics, position sizing, and transaction cost modeling.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# --- Risk Metrics (VaR / CVaR) ---
# Note: Horizon is DAILY.
# To scale to T days, one might use sqrt(T), but this understates risk under volatility clustering.

def compute_historical_var(returns, confidence_level=0.95):
    """Historical Daily Value at Risk (VaR)."""
    if len(returns) == 0:
        return np.nan
    return -np.percentile(returns.dropna(), 100 * (1 - confidence_level))

def compute_historical_cvar(returns, confidence_level=0.95):
    """Historical Daily Conditional Value at Risk (CVaR)."""
    if len(returns) == 0:
        return np.nan
    var = compute_historical_var(returns, confidence_level)
    tail_losses = returns.dropna()[returns.dropna() <= -var]
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


# --- Position Sizing ---

def compute_target_weight_proba(prob_df):
    """
    Probability-weighted position sizing.
    Replaces binary exposure. 
    Weight = P(Bull)
    """
    weight = pd.Series(0.0, index=prob_df.index)
    if 'Bull' in prob_df.columns:
        weight += prob_df['Bull'].fillna(0.0)
    
    # Cap at 1.0 just in case
    return weight.clip(lower=0.0, upper=1.0)


# --- Transaction Costs ---

def estimate_transaction_costs(turnover, asset_name):
    """
    Model transaction costs as spread + commission, not a flat percentage.
    Spreads vary by asset liquidity.
    """
    asset = str(asset_name).lower()
    
    # Spread estimates (bps)
    if 'spx' in asset or 'gspc' in asset:
        spread_bps = 1.0
    elif 'nasdaq' in asset or 'ixic' in asset:
        spread_bps = 1.0
    elif 'gold' in asset or 'gc' in asset:
        spread_bps = 2.0
    elif 'btc' in asset or 'bitcoin' in asset:
        spread_bps = 5.0
    else:
        spread_bps = 2.0
        
    # Flat commission (e.g. 0.5 bps equivalent)
    commission_bps = 0.5
    
    total_cost_bps = spread_bps + commission_bps
    
    return turnover * (total_cost_bps / 10000.0)
