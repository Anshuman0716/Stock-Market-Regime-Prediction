# Market Regime Detection System

**A quantitative data science & risk project**

This project labels each trading day as **Bull, Transition or Crisis** using a 4-state Gaussian Hidden Markov Model (HMM) trained on 8 technical and macro features. It then tests whether moving in and out of the market based on those labels beats simple baselines after trading costs. It covers the S&P 500, NASDAQ, Gold and Bitcoin.

The project's priority is an **honest evaluation**: no look-ahead bias anywhere in the pipeline, strictly out-of-sample results, the same costs and execution lag for every strategy, and statistical tests that correct for multiple comparisons.

> **Status:** Step 1 of 5 of the improvement roadmap is complete (see [Roadmap](#9-roadmap)). All numbers below come from `scripts/backfill_db.py` run on data up to **2026-10-26** and are read straight from `data/regime_store.db`.

---

## 1. Results (out-of-sample, after costs)

**Question:** Can an unsupervised HMM detect market regimes early enough to improve a portfolio, and does any edge survive trading costs?

**Answer: no, not in its current form.** On every asset the HMM strategy has a lower Sharpe ratio than buy-and-hold, and on the S&P 500, NASDAQ and Bitcoin it is also below a plain 200-day moving-average filter. It does cut drawdowns on the two equity indices, but it does so mainly by being about half invested (average exposure ~50%), not by timing the market well: it gives up far more return than the exposure cut alone explains.

| Asset (window) | Strategy | CAGR | Ann. Vol | Sharpe | Max DD | Avg. Exposure | 95% VaR (1d) | 99% CVaR (1d) |
|---|---|---|---|---|---|---|---|---|
| **S&P 500** (2010–2026-10) | Buy & Hold | 12.05% | 17.14% | 0.75 | -33.92% | 100% | 1.63% | 4.37% |
|  | 200-day MA filter | 7.51% | 11.74% | 0.68 | -23.30% | 83% | 1.25% | 2.95% |
|  | **HMM regime** | 2.72% | 9.20% | 0.34 | -19.12% | 52% | 0.86% | 2.79% |
| **NASDAQ** (2010–2026-10) | Buy & Hold | 15.57% | 20.44% | 0.81 | -36.40% | 100% | 2.06% | 4.79% |
|  | 200-day MA filter | 10.84% | 14.97% | 0.76 | -23.82% | 82% | 1.62% | 3.60% |
|  | **HMM regime** | 3.85% | 10.55% | 0.41 | -18.78% | 49% | 1.06% | 3.08% |
| **Gold** (2010–2026-10) | Buy & Hold | 7.95% | 17.21% | 0.53 | -44.36% | 100% | 1.70% | 4.26% |
|  | 200-day MA filter | 6.48% | 14.01% | 0.52 | -32.43% | 64% | 1.38% | 3.87% |
|  | **HMM regime** | 7.69% | 16.89% | 0.52 | -46.58% | 95% | 1.66% | 4.26% |
| **Bitcoin** (2021–2026-10) | Buy & Hold | 16.96% | 56.59% | 0.56 | -76.63% | 100% | 4.54% | 10.78% |
|  | 200-day MA filter | 15.99% | 41.31% | 0.56 | -54.40% | 55% | 3.18% | 8.38% |
|  | **HMM regime** | -1.36% | 50.49% | 0.23 | -76.63% | 75% | 4.01% | 10.62% |

*Sharpe = mean / std of daily net returns × √(rows per year), risk-free rate 0. Rows per year are 252 for equities and gold, and 365 for Bitcoin (it trades every day). VaR and CVaR are historical, 1-day, shown as positive losses.*

### Is the difference statistically significant?

Moving-block bootstrap (60-day blocks, 2,000 resamples) of the Sharpe difference HMM − baseline, cross-checked with a HAC delta-method test. The Holm adjustment corrects for running 8 comparisons.

| Asset | HMM vs | Sharpe diff | 95% CI | p (bootstrap) | p (HAC) | p (Holm-adjusted) |
|---|---|---|---|---|---|---|
| S&P 500 | Buy & Hold | -0.41 | [-0.71, -0.08] | 0.011 | 0.033 | 0.077 |
| S&P 500 | 200-day MA | -0.34 | [-0.66, +0.03] | 0.074 | 0.073 | 0.296 |
| NASDAQ | Buy & Hold | -0.40 | [-0.77, -0.03] | 0.033 | 0.053 | 0.198 |
| NASDAQ | 200-day MA | -0.35 | [-0.72, -0.00] | 0.050 | 0.074 | 0.250 |
| Gold | Buy & Hold | -0.01 | [-0.08, +0.07] | 0.784 | 0.856 | 1.000 |
| Gold | 200-day MA | +0.00 | [-0.24, +0.29] | 0.780 | 0.974 | 1.000 |
| Bitcoin | Buy & Hold | -0.33 | [-0.63, -0.12] | 0.001 | 0.023 | 0.008 |
| Bitcoin | 200-day MA | -0.34 | [-1.09, +0.35] | 0.344 | 0.323 | 1.000 |

**Reading this:** none of the comparisons shows the HMM *beating* a baseline. On Bitcoin the HMM is significantly *worse* than buy-and-hold even after the Holm correction (p = 0.008). On Gold the HMM is almost always labeled Bull (95% average exposure), so it behaves like buy-and-hold.

### Why the results look like this (findings so far)

- **Bitcoin, 2022:** from the November 2021 peak to the November 2022 low (−77%), the HMM labeled every single day **Bull** and stayed 100% invested. The models for 2021 and 2022 were trained on Bitcoin's history since 2014, when even its "calm" periods were very volatile, so the lowest-volatility state, which the labeling rule calls Bull, still covered a crash. This is a weakness of the volatility-based labeling rule, not a coding error.
- **Equities:** being ~50% invested on average would by itself shrink drawdowns roughly in half, and the HMM's lower drawdowns are in line with that. But the return it gave up (2.7% vs 12.1% CAGR on the S&P) is far larger than half, so the timing itself has hurt.
- **Gold:** most states get labeled Bull, so the strategy is close to buy-and-hold.

### Correction to earlier versions of this README

An earlier version claimed the HMM "slashed S&P 500 max drawdown from ~56% to ~22%". That number was an **evaluation bug**. The walk-forward model only starts predicting after ~9 years of training data (2010 for equities, 2021 for Bitcoin), but the backtest started in 2000 and treated the missing predictions as "hold cash". So the HMM got credit for sitting out the 2000–02 and 2008 crashes, years in which the model did not exist yet, while buy-and-hold was charged for them. Bitcoin metrics were also annualized with 252 days instead of 365. Both are fixed: every strategy is now scored only on the out-of-sample window, starting flat, with per-asset annualization (see [Section 3](#3-evaluation-methodology)).

---

## 2. Core Concepts & Statistical Integrity

### A. Walk-forward training
The model is never fit on the full dataset. It uses an **expanding window refit once a year**: to predict year *Y*, the HMM and its `StandardScaler` are fit only on data before *Y*. The first prediction year is the first one with at least `min_train_days` (2,268 ≈ 9 years) of history before it.

### B. Filtered, not smoothed, probabilities
`hmmlearn`'s `predict()` (Viterbi) and `predict_proba()` (forward-backward) condition on the **whole** sequence passed in, so day 1's label would depend on day 250's data. `src/models/hmm.py::predict_proba_filtered` runs only the forward pass and normalizes it, which gives P(state at t | data up to t).

### C. State labeling per fold
HMM state numbers are arbitrary and change on every refit. In each fold, states are ranked using **training data only**: lowest volatility → Bull, highest volatility → Crisis, and the middle states by annualized Sharpe (> 0.2 → Bull, < −0.1 → Crisis, otherwise Transition).

### D. Trailing-only signal smoothing
The strategy's target weight is the **trailing 21-day mean of P(Bull)**. It is never a centered window, because a centered window lets future days into today's value. This trades some responsiveness at regime turns for zero look-ahead.

---

## 3. Evaluation Methodology

Implemented in `src/backtest/evaluation.py`.

1. **One window per asset.** The out-of-sample window starts on the first date with a real walk-forward prediction and ends on the last available date. Every strategy for that asset (HMM, buy-and-hold, 200-day MA) is scored on exactly these dates.
2. **Signals may look back, P&L may not start early.** The 200-day MA on the first out-of-sample day uses the 200 days before it (that information was available then). But returns, costs and metrics are only counted inside the window.
3. **Everyone starts flat.** Each strategy enters the window with no position and pays the cost of its first trade.
4. **Same plumbing for everyone.** The same execution lag, cost model and metric functions apply to every strategy, so differences come from the signal alone.
5. **Per-asset annualization** from `config.yaml` (`backtest.annualization`): 252 rows per year for equities and gold, 365 for Bitcoin.
6. **One Sharpe definition.** The same `compute_sharpe` is used for the results table and inside the significance tests.

---

## 4. Architecture & Data Flow

```text
[yfinance loader] → [8 features] → [walk-forward HMM] → [labeling] → [evaluation: HMM + baselines, risk, significance] → [SQLite store] → [Streamlit app / notebooks]
```

Notebooks and `app.py` only **read** from `src/` and the store; they never recompute pipeline logic.

### Directory structure
```
├── config/config.yaml            # Tickers, dates, HMM settings, assets, annualization
├── src/
│   ├── data/loader.py            # yfinance download, parquet cache, config + annualization helpers
│   ├── features/engineering.py   # The 8 features + look-ahead property check
│   ├── models/hmm.py             # Walk-forward fit/predict, filtered probabilities
│   ├── models/labeling.py        # Per-fold state → regime mapping, trailing smoothing
│   ├── backtest/engine.py        # Backtest with 2-day execution lag and costs
│   ├── backtest/baselines.py     # Buy & hold and 200-day MA target weights
│   ├── backtest/evaluation.py    # Out-of-sample window; scores all strategies on it
│   ├── backtest/risk.py          # Cost model, probability sizing, VaR / CVaR
│   ├── backtest/metrics.py       # CAGR, Sharpe, Sortino, Calmar, drawdowns, exposure
│   ├── stats/significance.py     # Block bootstrap, HAC test, Holm correction
│   └── db/schema.sql, store.py   # SQLite schema (versioned) and idempotent writers
├── scripts/
│   ├── train.py                  # Train walk-forward models → models/ + manifest.json
│   ├── backfill_db.py            # Run everything and write it to data/regime_store.db
│   └── paper_trade.py            # Alpaca paper-trading scaffold (dry run by default)
├── tests/                        # pytest: look-ahead, lag, costs, evaluation window, metrics
├── notebooks/                    # EDA, model development, backtest, SQL analysis
├── models/                       # Persisted fold artifacts (model + scaler + state map)
├── data/regime_store.db          # SQLite store read by the dashboard
└── app.py                        # Streamlit dashboard
```

### SQLite tables
`assets`, `prices`, `features` (long format), `backtest_runs`, `regimes` (out-of-sample labels and probabilities), `backtest_daily` and `backtest_metrics` (both with a `strategy` column: `hmm` / `buy_hold` / `dma200`), `evaluation_windows`, `significance_tests`. The schema is versioned with `PRAGMA user_version`; an out-of-date database is rebuilt automatically.

---

## 5. Features (the 8 inputs)

Computed with the `ta` library. Every indicator uses only past and current rows, and warm-up rows are dropped, never back-filled.

| Group | Feature | Description |
|---|---|---|
| Momentum | `returns` | Daily % change |
| | `rsi` | 14-day Relative Strength Index |
| Trend | `macd` | MACD line (12/26) |
| | `adx` | 14-day Average Directional Index (trend strength) |
| Volatility | `volatility_20` | 20-day rolling std of returns |
| | `atr` | 14-day Average True Range |
| Volume / macro | `obv_ratio` | On-Balance Volume ÷ its 20-day EMA |
| | `vix` | CBOE VIX close, forward-filled onto each asset's calendar |

> Known issue (fixed in Step 2): `atr` and `macd` are in price units, so they drift as prices rise over the years, and `obv_ratio` can blow up when OBV crosses zero.

---

## 6. Backtest Realism

- **Execution lag:** a weight decided at day *t*'s close is traded at day *t+1*'s close and earns returns from day *t+2*. Tested in `tests/test_backtest.py`.
- **Transaction costs:** spread + commission per unit of turnover: 1.5 bps for S&P 500 and NASDAQ, 2.5 bps for Gold, 5.5 bps for Bitcoin.
- **Probability-weighted sizing:** HMM exposure = trailing 21-day mean of P(Bull), between 0 and 1.
- **Risk reporting:** historical and Cornish-Fisher VaR, historical CVaR, drawdown depth and duration.

---

## 7. Limitations

| Limitation | Explanation |
|---|---|
| **Volatility-based labeling** | The calmest state is always called Bull. When a training window is very volatile overall (Bitcoin since 2014), the calmest state can still include crashes. |
| **Gaussian emissions** | `hmmlearn`'s Gaussian HMM assumes normal returns within each state. Real returns have fat tails. |
| **Detection lag** | The 21-day trailing smoothing delays reactions to regime changes by up to ~3 weeks. That's the price of zero look-ahead. |
| **Yearly forward-pass restart** | Each test year's forward pass starts from the model's initial state probabilities, so on January 1 the probabilities jump. (Planned fix in Step 2.) |
| **EM convergence warnings** | Some folds report small log-likelihood decreases during EM. Re-running gives very slightly different p-values in the 3rd decimal. (Multiple restarts planned in Step 2.) |
| **Index-level backtest** | `^GSPC` / `^IXIC` are traded as if they were investable, which ignores ETF tracking error and fees. |
| **Private `hmmlearn` API** | Filtered probabilities use `hmmlearn._hmmc.forward_log`, so `hmmlearn` is pinned to `>=0.3,<0.4`. |
| **Notebooks and older docs** | `notebooks/` and the phase docs in `docs/` were written before the Step 1 evaluation fix and may still show old numbers. They'll be refreshed in Steps 3 and 5. |

---

## 8. How to Run Locally

Requires **Python 3.10–3.12**. `requirements.txt` pins `numpy<2.0`, which has no wheels for 3.13.

```bash
git clone https://github.com/Anshuman0716/Stock-Market-Regime-Prediction.git
cd Stock-Market-Regime-Prediction

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt

python scripts/train.py          # walk-forward models → models/  (~1 min)
python scripts/backfill_db.py    # evaluation → data/regime_store.db  (~1–2 min)
pytest tests/                    # test suite
streamlit run app.py             # dashboard at http://localhost:8501
```

`python scripts/backfill_db.py --check-idempotency` runs the backfill twice and fails if any table's row count changes.

---

## 9. Roadmap

| Step | Scope | Status |
|---|---|---|
| 1 | Out-of-sample-only evaluation window, per-asset annualization, one Sharpe definition, baselines + significance stored in the database | ✅ Done |
| 2 | One signal definition shared by backtest, store, dashboard and paper trading; scale-free features; continuous forward pass; EM restarts | ⏳ Next |
| 3 | A single pipeline script that trains once and writes everything (incl. per-cycle metrics and fold info) | Planned |
| 4 | Dashboard rebuild: overview, regime timeline, performance vs baselines, risk, model insight, statistical evidence, paper trading | Planned |
| 5 | Repo clean-up, refreshed notebooks/docs, more tests, CI updates | Planned |

---

## 10. Tech Stack

- **Data science:** Python, `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Modeling:** `hmmlearn` (Gaussian HMM)
- **Features:** `ta`
- **Data & persistence:** `yfinance`, SQLite, `joblib`, parquet
- **UI:** `streamlit`, `matplotlib`
- **Testing / CI:** `pytest`, GitHub Actions

---

## 11. Deploy on Streamlit Community Cloud

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) choose **New app** → this repo, branch `main`, main file `app.py`.
3. The app reads `data/regime_store.db` and `models/manifest.json`, so commit both after running `train.py` and `backfill_db.py` locally.
4. Deploy.
