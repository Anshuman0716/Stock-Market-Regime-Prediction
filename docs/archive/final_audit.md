# Final Risk Audit Report

**Date:** 2026-08-20
**Reviewer:** Quant Risk / Model Validation Desk
**Verdict:** **REJECTED FOR PRODUCTION (Needs Remediation)**

The walk-forward implementation successfully eliminates the most egregious look-ahead biases (e.g., centered smoothing and full-sample `predict_proba`). However, severe structural flaws and execution mismatches remain.

## Findings by Severity

### 1. HIGH SEVERITY: Duplicated Logic & Bypassed Smoothing in Production DB
- **Issue:** The backfill script (`scripts/backfill_db.py`) bypasses the `smooth_regime_labels()` function explicitly built in `src/models/labeling.py`. Instead, it manually computes a rolling average on the probabilities for backtesting, but writes the strictly **unsmoothed** `raw_labels` directly to the `regimes` table in the database.
- **Impact:** Any downstream data scientist or dashboard querying the SQLite database for regime state transitions will observe chaotic, rapidly flipping states that contradict the backtest logic.
- **Remediation:** Remove the duplicated `.rolling()` pandas logic in `backfill_db.py`. Ensure `smooth_regime_labels` is imported from `src/` and applied to the labels *before* database insertion.

### 2. HIGH SEVERITY: Infinite-Decay Truncation Mismatch in Features
- **Issue:** `src/features/engineering.py` computes Exponential Moving Averages (`ta.ema`, `ta.macd`) on the *entire* historical dataframe before the walk-forward split.
- **Impact:** While this is not strictly "look-ahead" (it doesn't peek into the future), it creates a catastrophic production mismatch. EMA is an infinite-impulse response filter. If the backtest starts computing EMA from year 2000, its value on Jan 1, 2020 will slightly differ from a production container that boots up and calculates EMA starting from year 2015. This violates production parity.
- **Remediation:** Enforce a strict "warmup" window cutoff during feature engineering, or compute features strictly on an expanding window matching the HMM.

### 3. MEDIUM SEVERITY: Gaussian HMM on Fat-Tailed Returns
- **Issue:** The model relies on `hmmlearn.hmm.GaussianHMM`, assuming normally distributed emissions.
- **Impact:** Financial returns are definitively non-Gaussian (excess kurtosis, fat tails). By forcing a Gaussian emission, the model mathematically underestimates the probability of tail events, ironically forcing the transition matrix to aggressively flip states during crises just to accommodate the outliers.
- **Remediation:** Switch to a Student-t HMM or a Gaussian Mixture HMM.

### 4. MEDIUM SEVERITY: Post-Hoc Hyperparameter Tuning (Meta-Lookahead)
- **Issue:** The hyperparameters `smoothing_window: 21` and `n_states: 4` were locked into `config.yaml` *after* observing their impact on the final out-of-sample Sharpe ratios.
- **Impact:** While the daily step-by-step walk-forward prevents data leakage into the *model weights*, tuning hyperparameters based on the terminal backtest performance leaks future data into the *architecture*. The reported metrics in `README.md` are validation scores, not true out-of-sample scores.

### 5. LOW SEVERITY: Dependency on Unmaintained `pandas-ta`
- **Issue:** `requirements.txt` introduces `pandas-ta`.
- **Impact:** This is an abandoned repository. Relying on unmaintained 3rd party libraries for critical trading signals (RSI/MACD) introduces security and compatibility risks when upgrading core Pandas/NumPy versions.
- **Remediation:** Rewrite the ~5 indicators used in `src/features/engineering.py` using native pandas `.ewm()` and `.rolling()`.

---

## The Three Unanswered Interview Questions

If presented in a quant interview, the candidate must be prepared to answer:
1. *"Since you admit the HMM systematically underperforms Buy & Hold, what is the mathematical justification for using an HMM at all, rather than a simple rolling volatility filter which is vastly more computationally efficient and transparent?"*
2. *"How do you isolate whether the drawdown protection came from the HMM's state detection, or simply from the fact that the 21-day smoothing window forced the model to be in cash 50% of the time, halving beta exposure?"*
3. *"You modeled slippage and commissions statically (e.g., 1.5 bps). If you are trading between Regimes in a $100M book, how would you model the order-book impact of dumping equities during a 'Crisis' signal when liquidity evaporates?"*
