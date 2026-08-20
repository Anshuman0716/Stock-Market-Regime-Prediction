import numpy as np
import pandas as pd

def block_bootstrap_sharpe_diff(returns_a, returns_b, ann_factor, block_size=60, n_bootstraps=2000, seed=42):
    """
    Computes a moving-block bootstrap confidence interval and p-value for the difference in Sharpe ratios.
    
    Why block bootstrap: Daily returns are autocorrelated and volatility clusters. An i.i.d. bootstrap
    would destroy this temporal structure, resulting in artificially narrow confidence intervals.
    We use block_size=60 (approx 1 quarter) to preserve short/medium-term macroeconomic autocorrelation.
    
    Parameters
    ----------
    returns_a : pd.Series or np.ndarray
        Returns of strategy A (e.g., HMM)
    returns_b : pd.Series or np.ndarray
        Returns of strategy B (e.g., Baseline)
    ann_factor : int
        Annualization factor (252 for equities, 365 for crypto)
        
    Returns
    -------
    dict
        diff: observed difference (A - B)
        ci_lower: 2.5th percentile
        ci_upper: 97.5th percentile
        p_value: two-sided p-value
    """
    rng = np.random.default_rng(seed)
    
    # Ensure inputs are numpy arrays
    ra = np.asarray(returns_a)
    rb = np.asarray(returns_b)
    n = len(ra)
    
    # Observed Sharpe difference
    def calc_sharpe(r):
        std = np.std(r, ddof=1)
        if std == 0: return 0.0
        # Annualized
        return (np.mean(r) / std) * np.sqrt(ann_factor)

    obs_diff = calc_sharpe(ra) - calc_sharpe(rb)
    
    # Block bootstrap
    diffs = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        # Sample starting indices for blocks
        # We sample n // block_size + 1 blocks to cover the array
        num_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n - block_size + 1, size=num_blocks)
        
        # Build the bootstrap sample
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        
        r_a_boot = ra[idx]
        r_b_boot = rb[idx]
        
        diffs[i] = calc_sharpe(r_a_boot) - calc_sharpe(r_b_boot)
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    # Two-sided p-value calculation
    # If the observed difference is positive, how many bootstrap samples are <= 0?
    if obs_diff > 0:
        p_val = 2 * np.mean(diffs <= 0)
    else:
        p_val = 2 * np.mean(diffs >= 0)
        
    p_val = min(p_val, 1.0)
    
    return {
        "diff": obs_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val
    }

def ledoit_wolf_sharpe_diff(returns_a, returns_b, ann_factor):
    """
    Parametric cross-check for Sharpe ratio difference based on the robust Delta method
    with HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors, 
    similar in spirit to the Ledoit-Wolf (2008) robust Sharpe test.
    """
    ra = np.asarray(returns_a)
    rb = np.asarray(returns_b)
    n = len(ra)
    
    mu_a, mu_b = np.mean(ra), np.mean(rb)
    var_a, var_b = np.var(ra, ddof=0), np.var(rb, ddof=0)
    
    sharpe_a = (mu_a / np.sqrt(var_a)) if var_a > 0 else 0
    sharpe_b = (mu_b / np.sqrt(var_b)) if var_b > 0 else 0
    
    obs_diff = (sharpe_a - sharpe_b) * np.sqrt(ann_factor)
    
    # Simplified HAC Delta Method
    # Define moment conditions m_t = [r_{a,t} - mu_a, r_{b,t} - mu_b, (r_{a,t}-mu_a)^2 - var_a, (r_{b,t}-mu_b)^2 - var_b]
    m = np.column_stack([
        ra - mu_a,
        rb - mu_b,
        (ra - mu_a)**2 - var_a,
        (rb - mu_b)**2 - var_b
    ])
    
    # Estimate long-run covariance of moments (Newey-West style, lag=5)
    lag = 5
    omega = np.dot(m.T, m) / n
    for j in range(1, lag + 1):
        gamma_j = np.dot(m[j:].T, m[:-j]) / n
        weight = 1 - j / (lag + 1)
        omega += weight * (gamma_j + gamma_j.T)
        
    # Jacobian of Sharpe wrt moments
    # f(mu, var) = mu / sqrt(var)
    # df/dmu = 1/sqrt(var), df/dvar = -0.5 * mu / var^(3/2)
    # The gradient is [1/sqrt(var_a), -1/sqrt(var_b), -0.5*mu_a/var_a^1.5, 0.5*mu_b/var_b^1.5]
    if var_a > 0 and var_b > 0:
        grad = np.array([
            1 / np.sqrt(var_a),
            -1 / np.sqrt(var_b),
            -0.5 * mu_a / (var_a ** 1.5),
            0.5 * mu_b / (var_b ** 1.5)
        ])
    else:
        grad = np.zeros(4)
        
    # Variance of the difference in Sharpe (daily)
    var_diff = np.dot(grad.T, np.dot(omega, grad))
    
    # Annualize standard error
    se_diff = np.sqrt(var_diff / n) * np.sqrt(ann_factor)
    
    # Z-test
    if se_diff > 0:
        z_stat = obs_diff / se_diff
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0
        
    return {
        "diff": obs_diff,
        "p_value": p_value,
        "se": se_diff
    }

def adjust_pvalues(p_values, method='holm'):
    """
    Adjust p-values for multiple testing.
    We are testing 4 assets x 2 baselines = 8 comparisons.
    Manually implemented Holm-Bonferroni to avoid statsmodels dependency.
    """
    p_values = np.array(p_values)
    m = len(p_values)
    
    # Sort indices
    order = np.argsort(p_values)
    sorted_pvals = p_values[order]
    
    adjusted = np.zeros(m)
    current_max = 0.0
    
    for i, p in enumerate(sorted_pvals):
        adj_p = p * (m - i)
        current_max = min(max(current_max, adj_p), 1.0)
        adjusted[order[i]] = current_max
        
    return adjusted
