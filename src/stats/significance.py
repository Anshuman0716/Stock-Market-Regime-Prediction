import numpy as np
import pandas as pd
from scipy.stats import norm

# Same Sharpe definition as the reported metrics, so the number we test is
# the number we show.
from src.backtest.metrics import compute_sharpe

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

    if len(ra) != len(rb):
        raise ValueError("returns_a and returns_b must cover the same dates.")
    if n < 2 * block_size:
        raise ValueError(f"Need at least {2 * block_size} rows for block_size={block_size}, got {n}.")

    def calc_sharpe(r):
        return compute_sharpe(r, ann_factor)

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

    # Two-sided p-value for H0: "no Sharpe difference".
    #
    # Why centre the bootstrap distribution: the bootstrap draws show how the
    # Sharpe difference VARIES around its own centre, which is roughly the
    # observed value, not zero. To ask "how unusual is our result if the
    # true difference were 0?", we shift the draws to mean 0 and count how
    # often a shifted draw is at least as far from 0 as the observed value.
    #
    # An earlier version counted draws on the "wrong side" of zero and
    # doubled the count. That breaks when the draws are not centred exactly
    # on the observed value: with a small positive difference and a slightly
    # lower bootstrap centre, more than half the draws fall below zero and
    # the p-value hit the 1.0 cap.
    #
    # The +1 in numerator and denominator stops a finite number of draws
    # from ever reporting p = 0 exactly.
    centred = diffs - diffs.mean()
    p_val = (1 + np.sum(np.abs(centred) >= abs(obs_diff))) / (1 + n_bootstraps)

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
    # Define moment conditions
    # m_t = [r_{a,t} - mu_a, r_{b,t} - mu_b, (r_{a,t}-mu_a)^2 - var_a, (r_{b,t}-mu_b)^2 - var_b]
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
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0

    return {
        "diff": obs_diff,
        "p_value": p_value,
        "se": se_diff
    }

def adjust_pvalues(p_values):
    """
    Holm-Bonferroni adjustment for a family of tests (here: every asset x
    every baseline). Implemented by hand to avoid a statsmodels dependency.
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


def run_significance_suite(strategy_returns, ann_factors, strategy="hmm",
                           baselines=("buy_hold", "dma200", "vol_target"), n_bootstraps=2000,
                           family="asset", block_size=60, seed=42):
    """
    Test the strategy against every baseline on every asset, then correct for
    multiple testing.

    Why Holm across ALL comparisons: with several assets and baselines we
    run many tests at once (4 x 3 = 12 here). At alpha = 0.05, the chance of
    at least one false "significant" result among 12 independent tests is
    about 46%. Holm-Bonferroni controls that family-wise error rate while
    being less conservative than plain Bonferroni.

    Pass EXCESS returns (above the cash rate) so the Sharpe ratios tested are
    the ones reported. Baselines missing from an asset are skipped.

    One call = one Holm family. `family` names it in the output, so tests
    corrected together can be told apart from tests corrected separately
    ("asset" for v1.0, "portfolio" for the v2.0 portfolio).

    Parameters
    ----------
    strategy_returns : dict[str, dict[str, pd.Series]]
        asset -> strategy name -> daily net returns. All strategies of one
        asset must share the same (out-of-sample) index.
    ann_factors : dict[str, int]
        asset -> rows per year.

    Returns
    -------
    pd.DataFrame
        One row per (asset, baseline) with sharpe_diff, bootstrap CI and
        p-value, the HAC delta-method p-value as a cross-check, and the
        Holm-adjusted bootstrap p-value.
    """
    rows = []
    for asset, by_strategy in strategy_returns.items():
        ra = by_strategy[strategy]
        for baseline in baselines:
            if baseline not in by_strategy:
                continue
            rb = by_strategy[baseline]
            if not ra.index.equals(rb.index):
                raise ValueError(f"{asset}: {strategy} and {baseline} cover different dates.")
            boot = block_bootstrap_sharpe_diff(ra, rb, ann_factors[asset], block_size=block_size,
                                               n_bootstraps=n_bootstraps, seed=seed)
            hac = ledoit_wolf_sharpe_diff(ra, rb, ann_factors[asset])
            rows.append({
                "asset": asset,
                "family": family,
                "strategy": strategy,
                "baseline": baseline,
                "sharpe_diff": boot["diff"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
                "p_value": boot["p_value"],
                "p_value_hac": hac["p_value"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_holm"] = adjust_pvalues(out["p_value"].values)
    return out
