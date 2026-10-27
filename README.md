# Market Regime Detection System

**A quantitative data science & risk project**

This project labels each trading day as **Bull, Transition or Crisis** using a 4-state Gaussian Hidden Markov Model (HMM) trained on 8 technical and macro features. It then tests whether moving in and out of the market based on those labels beats simple baselines after trading costs. It covers the S&P 500, NASDAQ, Gold and Bitcoin.

The project's priority is an **honest evaluation**: no look-ahead bias anywhere in the pipeline, strictly out-of-sample results, the same costs and execution lag for every strategy, and statistical tests that correct for multiple comparisons.

> **Status:** Steps 1–2 of 5 of the improvement roadmap are complete (see [Roadmap](#9-roadmap)). All numbers below come from `scripts/backfill_db.py` run on data up to **2026-10-26** and are read straight from `data/regime_store.db`. Two separate runs of the pipeline agree to within 10⁻¹⁰ on every metric.

---

## 1. Results (out-of-sample, after costs)

**Question:** Can an unsupervised HMM detect market regimes early enough to improve a portfolio, and does any edge survive trading costs?

**Answer: not proven.** The HMM strategy gives a smoother ride on every asset: lower volatility, smaller drawdowns, smaller tail losses. It has a *higher* Sharpe ratio than buy-and-hold on Gold and Bitcoin and a *lower* one on the S&P 500 and NASDAQ. **None of these differences is statistically significant**, even before correcting for multiple tests, so the honest conclusion is that the data cannot tell the HMM apart from simply holding the asset.

| Asset (window) | Strategy | CAGR | Ann. Vol | Sharpe | Max DD | Avg. Exposure | 95% VaR (1d) | 99% CVaR (1d) |
|---|---|---|---|---|---|---|---|---|
| **S&P 500** (2010–2026-10) | Buy & Hold | 12.05% | 17.14% | 0.75 | -33.92% | 100% | 1.63% | 4.37% |
|  | 200-day MA filter | 7.51% | 11.74% | 0.68 | -23.30% | 83% | 1.25% | 2.95% |
|  | **HMM regime** | 4.81% | 9.44% | 0.55 | -20.83% | 60% | 0.94% | 2.67% |
| **NASDAQ** (2010–2026-10) | Buy & Hold | 15.57% | 20.44% | 0.81 | -36.40% | 100% | 2.06% | 4.79% |
|  | 200-day MA filter | 10.84% | 14.97% | 0.76 | -23.82% | 82% | 1.62% | 3.60% |
|  | **HMM regime** | 9.26% | 13.52% | 0.72 | -23.03% | 73% | 1.46% | 3.42% |
| **Gold** (2010–2026-10) | Buy & Hold | 7.95% | 17.21% | 0.53 | -44.36% | 100% | 1.70% | 4.26% |
|  | 200-day MA filter | 6.48% | 14.01% | 0.52 | -32.43% | 64% | 1.38% | 3.87% |
|  | **HMM regime** | 5.04% | 9.52% | 0.56 | -37.80% | 49% | 0.89% | 2.68% |
| **Bitcoin** (2021–2026-10) | Buy & Hold | 16.96% | 56.59% | 0.56 | -76.63% | 100% | 4.54% | 10.78% |
|  | 200-day MA filter | 15.99% | 41.31% | 0.56 | -54.40% | 55% | 3.18% | 8.38% |
|  | **HMM regime** | 24.46% | 32.17% | 0.84 | -45.71% | 46% | 2.11% | 7.27% |

*Sharpe = mean / std of daily net returns × √(rows per year), risk-free rate 0. Rows per year are 252 for equities and gold, and 365 for Bitcoin (it trades every day). VaR and CVaR are historical, 1-day, shown as positive losses. Bitcoin's window is only 5.8 years, so its numbers are the least reliable.*

### Is the difference statistically significant?

Moving-block bootstrap (60-day blocks, 2,000 resamples) of the Sharpe difference HMM − baseline, with the bootstrap distribution centred on zero to test "no difference", cross-checked with a HAC delta-method test. The Holm adjustment corrects for running 8 comparisons.

| Asset | HMM vs | Sharpe diff | 95% CI | p (bootstrap) | p (HAC) | p (Holm-adjusted) |
|---|---|---|---|---|---|---|
| S&P 500 | Buy & Hold | -0.20 | [-0.53, +0.11] | 0.195 | 0.271 | 1.000 |
| S&P 500 | 200-day MA | -0.13 | [-0.43, +0.18] | 0.412 | 0.403 | 1.000 |
| NASDAQ | Buy & Hold | -0.09 | [-0.38, +0.24] | 0.580 | 0.596 | 1.000 |
| NASDAQ | 200-day MA | -0.04 | [-0.32, +0.25] | 0.791 | 0.789 | 1.000 |
| Gold | Buy & Hold | +0.03 | [-0.32, +0.31] | 0.835 | 0.859 | 1.000 |
| Gold | 200-day MA | +0.05 | [-0.32, +0.40] | 0.803 | 0.832 | 1.000 |
| Bitcoin | Buy & Hold | +0.28 | [-0.33, +1.01] | 0.424 | 0.399 | 1.000 |
| Bitcoin | 200-day MA | +0.28 | [-0.48, +1.11] | 0.526 | 0.440 | 1.000 |

**Reading this:** every 95% confidence interval contains zero, and the lowest raw p-value is 0.195. The bootstrap and HAC p-values agree closely, which is a good sign the tests themselves are behaving.

### What changed in Step 2, and why the numbers moved

Step 2 fixed the model's inputs and plumbing, not the strategy idea. Same out-of-sample windows and costs as Step 1:

| HMM Sharpe | Step 1 | Step 2 |
|---|---|---|
| S&P 500 | 0.34 | 0.55 |
| NASDAQ | 0.41 | 0.72 |
| Gold | 0.52 | 0.56 |
| Bitcoin | 0.23 | 0.84 |

- **Scale-free features** ([Section 5](#5-features-the-8-inputs)): the old ATR and MACD were in price points, so recent years looked like permanent extreme volatility to a model trained on 2000–09 prices.
- **5 EM starts per fold:** single starts often settled in poor local optima.
- **Continuous forward pass:** probabilities no longer restart every 1 January.
- **Bitcoin, 2022:** from the November 2021 peak to the November 2022 low (−77% for buy-and-hold), the Step 1 model labeled every day Bull and stayed 100% invested. The Step 2 model's average exposure over the same stretch was 11%.

The bootstrap p-value calculation was also corrected in Step 2. The old version counted draws on the "wrong side" of zero, which broke when the bootstrap distribution wasn't centred exactly on the observed value: it reported Gold's +0.03 Sharpe difference as p = 1.000. The new version centres the draws on zero first. The conclusion (nothing significant) is the same under both methods.

### Caveats
- **Lower exposure explains part of the risk reduction.** The HMM is 46–73% invested on average, so some of its lower volatility and drawdown simply comes from holding less.
- **Gold and Bitcoin never get a Transition regime.** Every fold maps their 4 states to Bull or Crisis only (see [Limitations](#7-limitations)).

### Correction to earlier versions of this README

An earlier version claimed the HMM "slashed S&P 500 max drawdown from ~56% to ~22%". That number was an **evaluation bug**. The walk-forward model only starts predicting after ~9 years of training data (2010 for equities, 2021 for Bitcoin), but the backtest started in 2000 and treated the missing predictions as "hold cash". So the HMM got credit for sitting out the 2000–02 and 2008 crashes, years in which the model did not exist yet, while buy-and-hold was charged for them. Bitcoin metrics were also annualized with 252 days instead of 365. Both are fixed: every strategy is now scored only on the out-of-sample window, starting flat, with per-asset annualization (see [Section 3](#3-evaluation-methodology)).

---

## 2. Core Concepts & Statistical Integrity

### A. Walk-forward training
The model is never fit on the full dataset. It uses an **expanding window refit once a year**: to predict year *Y*, the HMM and its `StandardScaler` are fit only on data before *Y*. The first prediction year is the first one with at least `min_train_days` (2,268 ≈ 9 years) of history before it.

Each fold is trained **from 5 random starting points** (`model.n_init`), and the one with the highest *training* log-likelihood is kept. EM only climbs to a nearby local optimum, and in most folds a later start beat the first one (by up to ~1,300 log-likelihood points), so a single start often settled for a poor fit.

### B. Filtered, not smoothed, probabilities
`hmmlearn`'s `predict()` (Viterbi) and `predict_proba()` (forward-backward) condition on the **whole** sequence passed in, so day 1's label would depend on day 250's data. `src/models/hmm.py::predict_proba_filtered` runs only the forward pass and normalizes it, which gives P(state at t | data up to t).

The forward pass runs **continuously over the full history** up to the end of each test year, and only the test-year rows are kept. Because row *t* depends only on rows 1..*t*, this adds no look-ahead. It just means the model's belief on 2 January carries over from 31 December instead of restarting every year. `tests/test_signal.py` checks the whole walk-forward is causal: deleting the last year of data leaves every earlier prediction unchanged.

### C. State labeling per fold
HMM state numbers are arbitrary and change on every refit. In each fold, states are ranked using **training data only**: lowest volatility → Bull, highest volatility → Crisis, and the middle states by annualized Sharpe (> 0.2 → Bull, < −0.1 → Crisis, otherwise Transition).

### D. One signal, defined once
`src/models/signal.py::compute_signal` is the **only** definition of the trading signal. The backtest, the SQLite store, the dashboard and paper trading all call it:

1. Take the filtered P(Bull), P(Transition), P(Crisis) for each day.
2. Smooth each with a **trailing 21-day mean** (`model.smoothing_window`), never a centered window, which would let future days into today's value.
3. **Regime** = the label with the highest smoothed probability.
4. **Target weight** = smoothed P(Bull), between 0 and 1.

Averaging probabilities, rather than taking the most common label, keeps the confidence information: a run of 51% Bull days and a run of 99% Bull days are not the same. `tests/test_signal.py` also checks that paper trading's path reproduces the backtest's probabilities exactly.

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
[yfinance loader] → [8 features] → [walk-forward HMM] → [labeling] → [signal] → [evaluation: HMM + baselines, risk, significance] → [SQLite store] → [Streamlit app / notebooks]
```

Notebooks and `app.py` only **read** from `src/` and the store; they never recompute pipeline logic.

### Directory structure
```
├── config/config.yaml            # Tickers, dates, HMM settings, assets, annualization
├── src/
│   ├── data/loader.py            # yfinance download, parquet cache, config + annualization helpers
│   ├── features/engineering.py   # The 8 features + look-ahead property check
│   ├── models/hmm.py             # Walk-forward fit/predict, filtered probabilities
│   ├── models/labeling.py        # Per-fold state → regime mapping, fold artifacts
│   ├── models/signal.py          # THE signal: smoothed probabilities, regime, target weight
│   ├── backtest/engine.py        # Backtest with 2-day execution lag and costs
│   ├── backtest/baselines.py     # Buy & hold and 200-day MA target weights
│   ├── backtest/evaluation.py    # Out-of-sample window; scores all strategies on it
│   ├── backtest/risk.py          # Cost model, VaR / CVaR
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
`assets`, `prices`, `features` (long format), `backtest_runs`, `regimes` (the full signal per out-of-sample day: raw and smoothed probabilities and labels, target weight), `backtest_daily` and `backtest_metrics` (both with a `strategy` column: `hmm` / `buy_hold` / `dma200`), `evaluation_windows`, `significance_tests`. The schema is versioned with `PRAGMA user_version`; an out-of-date database is rebuilt automatically.

---

## 5. Features (the 8 inputs)

Computed with the `ta` library and pandas. Every indicator uses only past and current rows, and warm-up rows are dropped, never back-filled.

**Every feature is scale-free**: multiplying all prices by 10 leaves every feature unchanged (`tests/test_features.py`). This matters because the scaler is fit on early years and applied to later ones.

| Group | Feature | Description |
|---|---|---|
| Momentum | `returns` | Daily % change |
| | `rsi` | 14-day Relative Strength Index (0–100) |
| Trend | `macd_pct` | MACD line (12/26) ÷ Close |
| | `adx` | 14-day Average Directional Index (trend strength, 0–100) |
| Volatility | `volatility_20` | 20-day rolling std of returns |
| | `atr_pct` | 14-day Average True Range ÷ Close |
| Volume / macro | `obv_flow_20` | 20-day signed volume ÷ 20-day total volume, in [−1, 1] |
| | `vix` | CBOE VIX close, forward-filled onto each asset's calendar |

**Why the change (Step 2):** the earlier `atr` and `macd` were in price points. Over the last 2 years the S&P 500's ATR averaged ~6.7 standard deviations above its 2000–09 training mean, simply because the index is several times higher, so the model saw "extreme volatility" every day. The earlier `obv_ratio` (OBV ÷ its EMA) reached values like +1,016 (NASDAQ) and −419 (Gold) when OBV crossed zero. `obv_flow_20` keeps the same idea (is volume backing the up-days or the down-days?) but stays between −1 and 1.

---

## 6. Backtest Realism

- **Execution lag:** a weight decided at day *t*'s close is traded at day *t+1*'s close and earns returns from day *t+2*. Tested in `tests/test_backtest.py`.
- **Transaction costs:** spread + commission per unit of turnover: 1.5 bps for S&P 500 and NASDAQ, 2.5 bps for Gold, 5.5 bps for Bitcoin.
- **Probability-weighted sizing:** HMM exposure = trailing 21-day mean of P(Bull), between 0 and 1 (`src/models/signal.py`). Positions under 0.1% of the portfolio are set to 0: nobody would trade them, and floating-point noise around such tiny weights made some metrics change between runs.
- **Risk reporting:** historical and Cornish-Fisher VaR, historical CVaR, drawdown depth and duration.

---

## 7. Limitations

| Limitation | Explanation |
|---|---|
| **Volatility-based labeling** | The calmest state is always called Bull. When a training window is very volatile overall (Bitcoin since 2014), the calmest state can still include crashes. |
| **Gaussian emissions** | `hmmlearn`'s Gaussian HMM assumes normal returns within each state. Real returns have fat tails. |
| **Detection lag** | The 21-day trailing smoothing delays reactions to regime changes by up to ~3 weeks. That's the price of zero look-ahead. |
| **No Transition state for Gold and Bitcoin** | With the current labeling rule, every Gold and Bitcoin fold maps its 4 states to Bull or Crisis only. The rule is worth revisiting. |
| **Training time** | 5 EM starts per fold make training ~5 minutes, and `backfill_db.py` currently retrains instead of loading the saved models. (Step 3 merges the two.) |
| **Index-level backtest** | `^GSPC` / `^IXIC` are traded as if they were investable, which ignores ETF tracking error and fees. |
| **Private `hmmlearn` API** | Filtered probabilities use `hmmlearn._hmmc.forward_log`, so `hmmlearn` is pinned to `>=0.3,<0.4`. |
| **Notebooks and older docs** | `notebooks/` and the phase docs in `docs/` predate Steps 1–2. They may show old numbers, and notebooks 01 and 03 import functions that were replaced by `src/models/signal.py`. They'll be refreshed in Step 5. |

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

python scripts/train.py          # walk-forward models → models/  (~5 min)
python scripts/backfill_db.py    # evaluation → data/regime_store.db  (~5 min)
pytest tests/                    # test suite
streamlit run app.py             # dashboard at http://localhost:8501
```

`python scripts/backfill_db.py --check-idempotency` runs the backfill twice and fails if any table's row count changes.

---

## 9. Roadmap

| Step | Scope | Status |
|---|---|---|
| 1 | Out-of-sample-only evaluation window, per-asset annualization, one Sharpe definition, baselines + significance stored in the database | ✅ Done |
| 2 | One signal definition shared by backtest, store, dashboard and paper trading; scale-free features; continuous forward pass; EM restarts | ✅ Done |
| 3 | A single pipeline script that trains once and writes everything (incl. per-cycle metrics and fold info) | ⏳ Next |
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
