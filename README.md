# Market Regime Detection System

[![Tests](https://github.com/Anshuman0716/Stock-Market-Regime-Prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/Anshuman0716/Stock-Market-Regime-Prediction/actions/workflows/tests.yml)

**A quantitative data science & risk project**

This project labels each trading day as **Bull, Transition or Crisis** using a 4-state Gaussian Hidden Markov Model (HMM) trained on 8 technical and macro features. It then tests whether moving in and out of the market based on those labels beats simple baselines after trading costs. It covers the S&P 500, NASDAQ, Gold and Bitcoin.

The project's priority is an **honest evaluation**: no look-ahead bias anywhere in the pipeline, strictly out-of-sample results, the same costs and execution lag for every strategy, and statistical tests that correct for multiple comparisons.

> **Status:** all 5 steps of the improvement roadmap are complete (see [Roadmap](#9-roadmap)). All numbers below come from `scripts/run_pipeline.py` run on data up to **2026-10-26** and are read straight from `data/regime_store.db`. Separate runs agree to within 10⁻¹¹ on every metric, and a refresh that reuses the saved models reproduces a full retrain exactly.

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

### During equity stress episodes

Each episode runs from an S&P 500 closing high to the closing low that followed (dates in `config.yaml`, checked against the data). Total return over the episode, S&P 500 only; the other assets are in the database and dashboard.

| Episode (S&P 500 peak → trough) | Buy & Hold | 200-day MA | HMM | HMM avg. exposure |
|---|---|---|---|---|
| Euro debt crisis 2011 (2011-04-29 → 2011-10-03) | -19.2% | -7.4% | -3.0% | 16% |
| China / oil sell-off 2015–16 (2015-05-21 → 2016-02-11) | -14.0% | -16.4% | -5.2% | 27% |
| Q4 sell-off 2018 (2018-09-20 → 2018-12-24) | -19.1% | -7.8% | -6.6% | 52% |
| COVID crash 2020 (2020-02-19 → 2020-03-23) | -33.6% | -10.3% | -19.0% | 63% |
| Bear market 2022 (2022-01-03 → 2022-10-12) | -24.9% | -14.1% | -5.5% | 11% |
| Tariff sell-off 2025 (2025-02-19 → 2025-04-08) | -18.7% | -10.4% | -8.3% | 65% |

The HMM lost less than buy-and-hold in all six episodes, and less than the 200-day MA in five (not the fast COVID crash, where it was still 63% invested on average).

### So why doesn't it win overall?

What each regime looked like on the S&P 500 out-of-sample (the asset's own return on days with that label, *not* a tradable result):

| S&P 500 regime | Share of days | Avg. spell (days) | Ann. return | Ann. vol |
|---|---|---|---|---|
| Bull | 60% | 94 | +11.6% | 12.6% |
| Transition | 22% | 44 | +5.7% | 20.6% |
| Crisis | 18% | 40 | +26.3% | 24.3% |

"Crisis" days are the most volatile, but on average they were also the *best* days. Big rebounds cluster with big sell-offs, so a strategy that steps aside in volatile periods avoids the crashes and also misses the rebounds. Net of both effects, the HMM still ends 2010–2026 with a lower Sharpe than buy-and-hold on the S&P 500 (0.55 vs 0.75).

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
[yfinance loader] → [8 features] → [walk-forward HMM] → [labeling] → [signal] → [evaluation: HMM + baselines, risk] → [regime analysis] → [significance] → [SQLite store + models/] → [Streamlit app / notebooks]
```

`src/pipeline.py::run_pipeline` runs all of it in one pass. Notebooks and `app.py` only **read** from `src/` and the store; they never recompute pipeline logic.

**One training run, two outputs.** Each run fits every walk-forward fold once and writes both the saved models and the database from that same fit. `--reuse-models` skips fitting and predicts with the saved folds instead: about 15 seconds instead of about 3.5 minutes, with identical results (tested in `tests/test_pipeline.py`). A retrain is required after any change to `config.yaml` or the features, and when a new calendar year starts; the pipeline refuses to reuse models in those cases.

**Crash-safe publishing.** Models are trained into a staging folder, and the database is built as a separate file. Only when every asset has succeeded are the new models, the manifest and the database swapped in, so the dashboard never sees a half-written result.

### Dashboard

`streamlit run app.py` opens an interactive Plotly dashboard with seven tabs. Everything on it is read from the SQLite store; the app never fits a model or recomputes a metric.

| Tab | What it shows |
|---|---|
| **Overview** | Today's regime, regime probabilities and target exposure for all four assets; a cross-asset scoreboard (Sharpe, CAGR, max drawdown vs both baselines); headline findings generated from the stored results |
| **Regimes** | Price on a log scale shaded by regime, above the stacked regime probabilities (smoothed, or raw with a toggle); recent regime spells; time spent in each regime |
| **Performance** | Full metrics table; growth of $1, drawdown and rolling 1-year Sharpe for all three strategies; calendar-year returns; equity stress episodes |
| **Risk** | Historical VaR / CVaR and Cornish-Fisher VaR; how much each strategy was invested over time; turnover and total costs paid |
| **Model insight** | What each regime looked like (returns, volatility, average VIX / RSI / ADX); day-to-day transition probabilities; spell lengths; every yearly fold's HMM states and the labels they received |
| **Statistical evidence** | Sharpe differences vs each baseline with 95% bootstrap intervals; bootstrap, HAC and Holm-adjusted p-values |
| **Paper trading** | Latest signal, data and model freshness checks, how to run `scripts/paper_trade.py`, and its trade log |

Design notes:
- **Read-only.** The store is opened with SQLite's read-only mode and closed after each query, so the dashboard can't change results or block the pipeline from publishing a new database. Cached queries reload automatically when a new database appears.
- **Pinned to one run.** Every query is filtered to the latest `run_id`, so rows from different runs are never mixed.
- **Honest headlines.** The overview's summary sentences (e.g. on how many assets the HMM beat buy-and-hold, and whether any difference is significant) are computed from the store each time the page loads, never typed by hand.
- **Readable for everyone.** Colours follow the entity everywhere (each strategy and each regime has one fixed colour) and were checked with a colour-vision-deficiency validator in light and dark themes. Red-vs-green for Bull/Crisis was rejected because red-green colour-blind readers can't tell them apart. Regime names are always shown as text and symbols (▲ Bull, ◆ Transition, ▼ Crisis), never as colour alone. Charts never use two y-axes on one plot.
- **Tested.** `tests/test_dashboard.py` builds a small store with the real pipeline and runs the real `app.py` headlessly through every tab and asset.

### Directory structure
```
├── config/config.yaml            # Tickers, dates, HMM settings, assets, annualization, costs, stress episodes
├── src/
│   ├── pipeline.py               # run_pipeline(): the whole flow, start to finish
│   ├── data/loader.py            # yfinance download, parquet cache, config / annualization / cost helpers
│   ├── features/engineering.py   # The 8 features + look-ahead property check
│   ├── models/hmm.py             # Walk-forward fit/predict, filtered probabilities, reuse of saved folds
│   ├── models/labeling.py        # Per-fold state → regime mapping, fold artifacts
│   ├── models/signal.py          # THE signal: smoothed probabilities, regime, target weight
│   ├── backtest/engine.py        # Backtest with 2-day execution lag and costs
│   ├── backtest/baselines.py     # Buy & hold and 200-day MA target weights
│   ├── backtest/evaluation.py    # Out-of-sample window; scores all strategies on it
│   ├── backtest/risk.py          # VaR / CVaR
│   ├── backtest/metrics.py       # CAGR, Sharpe, Sortino, Calmar, drawdowns, period metrics, chart series
│   ├── analysis/regimes.py       # Regime profiles, transitions, spells, per-year / per-episode breakdown
│   ├── stats/significance.py     # Block bootstrap, HAC test, Holm correction
│   ├── db/schema.sql, store.py   # SQLite schema (versioned) and idempotent writers
│   └── dashboard/                # Read-only queries (data.py), Plotly charts (charts.py),
│                                 #   colours and labels (theme.py), generated headlines (summary.py)
├── scripts/
│   ├── run_pipeline.py           # Run everything → models/ + data/regime_store.db
│   └── paper_trade.py            # Alpaca paper-trading scaffold (dry run by default)
├── tests/                        # pytest: look-ahead, lag, costs, labeling, risk, store, signal, pipeline, dashboard
├── notebooks/                    # Analysis notebooks; read the store only (see below)
├── docs/archive/                 # Historical project notes (numbers superseded; see its README)
├── models/                       # Persisted fold artifacts (model + scaler + state map)
├── data/regime_store.db          # SQLite store read by the dashboard and notebooks
├── app.py                        # Streamlit dashboard (layout only; reads the store)
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # + notebook execution and lint tools
└── ruff.toml                     # Lint settings
```

### SQLite tables

| Table | Contents |
|---|---|
| `assets`, `prices`, `features` | Raw inputs (features in long format) |
| `backtest_runs` | One row per run: config (and its hash), git SHA, cost model, data end date, whether models were retrained |
| `regimes` | The full signal per out-of-sample day: raw and smoothed probabilities and labels, target weight |
| `backtest_daily` | Per strategy (`hmm` / `buy_hold` / `dma200`) and day: weight, returns, turnover, cost, equity, drawdown, rolling 1-year Sharpe |
| `backtest_metrics` | Per strategy and cycle: `full`, `year:2020`, `episode:covid_crash_2020`, ... |
| `evaluation_windows` | Out-of-sample window, annualization and cost per asset |
| `significance_tests` | Sharpe-difference tests vs each baseline (bootstrap, HAC, Holm) |
| `model_folds`, `fold_states` | Per-fold training window and fit diagnostics; each HMM state's training behaviour and label |
| `regime_profiles`, `regime_transitions`, `regime_spells` | Out-of-sample description of the regimes: frequency, persistence, returns, average features |

The schema is versioned with `PRAGMA user_version`; an out-of-date database is rebuilt automatically.

### Notebooks

All four read only `data/regime_store.db` through `src/`, so they run offline and always describe the same run as the dashboard. They are committed with their outputs, and CI re-executes them on every push.

| Notebook | What it covers |
|---|---|
| `01_eda_and_features` | Price histories; the 8 features over 26 years; checks that stored features rebuild exactly from stored prices and contain no look-ahead; feature correlations on the first training window; features by regime |
| `02_model_development` | Model selection on the **first training window only**: parameter counts, BIC / AIC for 2–6 states (diagonal vs full covariance), sensitivity to the random EM start, how fold 0's states were labeled |
| `03_backtest_and_risk` | Headline metrics, equity and drawdown curves, calendar years, stress episodes, how quickly the signal reacted in each episode, tail risk and costs, significance tests |
| `04_sql_analysis` | The same questions in SQL (window functions, CTEs, gaps-and-islands), each **cross-checked** against the tables the pipeline stored |

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
- **Transaction costs:** spread + commission per unit of turnover, set in `config.yaml` (`backtest.costs`): 1.5 bps for S&P 500 and NASDAQ, 2.5 bps for Gold, 5.5 bps for Bitcoin. `run_backtest` requires the cost explicitly, so no backtest can silently run cost-free.
- **Probability-weighted sizing:** HMM exposure = trailing 21-day mean of P(Bull), between 0 and 1 (`src/models/signal.py`). Positions under 0.1% of the portfolio are set to 0: nobody would trade them, and floating-point noise around such tiny weights made some metrics change between runs.
- **Risk reporting:** historical and Cornish-Fisher VaR, historical CVaR, drawdown depth and duration.

---

## 7. Limitations

| Limitation | Explanation |
|---|---|
| **Volatility-based labeling** | The calmest state is always called Bull. When a training window is very volatile overall (Bitcoin since 2014), the calmest state can still include crashes. |
| **Gaussian emissions** | `hmmlearn`'s Gaussian HMM assumes normal returns within each state. Real returns have fat tails. |
| **Detection lag** | The 21-day trailing smoothing delays reactions to regime changes. In the S&P 500 stress episodes the smoothed regime left Bull 13–36 trading days after the peak (it was already Transition at the 2011 peak), when buy-and-hold had already lost between 1% and 19% (notebook 03). That's the price of zero look-ahead. |
| **No Transition state for Gold and Bitcoin** | With the current labeling rule, every Gold and Bitcoin fold maps its 4 states to Bull or Crisis only. The rule is worth revisiting. |
| **Stress-episode dates are equity-based** | The episodes are S&P 500 peak-to-trough windows applied to every asset, which shows how Gold and Bitcoin behaved during *equity* stress, not during their own crashes. |
| **Index-level backtest** | `^GSPC` / `^IXIC` are traded as if they were investable, which ignores ETF tracking error and fees. |
| **Private `hmmlearn` API** | Filtered probabilities use `hmmlearn._hmmc.forward_log`, so `hmmlearn` is pinned to `>=0.3,<0.4`. |
| **Number of states** | BIC on the first training window keeps falling up to 6 states (notebook 02). Four states are used because each can be mapped to an economic regime; this is a judgement call, not a statistical optimum. |

---

## 8. How to Run Locally

Requires **Python 3.10–3.12**. `requirements.txt` pins `numpy<2.0`, which has no wheels for 3.13.

```bash
git clone https://github.com/Anshuman0716/Stock-Market-Regime-Prediction.git
cd Stock-Market-Regime-Prediction

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt               # or requirements-dev.txt for notebooks + lint

python scripts/run_pipeline.py                 # train + evaluate + write the store (~3.5 min)
python scripts/run_pipeline.py --reuse-models  # daily refresh with saved models (~15 s)
pytest                                         # test suite
streamlit run app.py                           # dashboard at http://localhost:8501
```

Development checks (what CI runs on every push, on Python 3.10, 3.11 and 3.12):

```bash
ruff check .                                   # lint (settings in ruff.toml)
pytest                                         # tests
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # re-run the notebooks
```

On Windows, `run_all.bat` does all of this with [uv](https://docs.astral.sh/uv/) on Python 3.12.

---

## 9. Roadmap

| Step | Scope | Status |
|---|---|---|
| 1 | Out-of-sample-only evaluation window, per-asset annualization, one Sharpe definition, baselines + significance stored in the database | ✅ Done |
| 2 | One signal definition shared by backtest, store, dashboard and paper trading; scale-free features; continuous forward pass; EM restarts | ✅ Done |
| 3 | A single pipeline that trains once and writes everything (per-year and stress-episode metrics, fold info, regime profiles, transitions, chart series); fast refresh with saved models; crash-safe publishing; costs in config | ✅ Done |
| 4 | Dashboard rebuild in Plotly: overview, regime timeline, performance vs baselines, risk, model insight, statistical evidence, paper trading; read-only, colour-blind-safe, tested headlessly | ✅ Done |
| 5 | Notebooks rewritten to read the store (offline, cross-checked, executed in CI); old docs archived; tests for labeling, VaR/CVaR, baselines and the store; tests no longer write into the repo; lint; CI on Python 3.10–3.12 for every branch | ✅ Done |

Possible next steps (not started): revisit the labeling rule so Gold and Bitcoin can have a Transition regime; backtest tradable ETFs instead of index levels; compare against a volatility-targeting baseline.

---

## 10. Tech Stack

- **Data science:** Python, `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Modeling:** `hmmlearn` (Gaussian HMM)
- **Features:** `ta`
- **Data & persistence:** `yfinance`, SQLite, `joblib`, parquet
- **UI:** `streamlit`, `plotly`
- **Analysis:** Jupyter notebooks, `matplotlib`
- **Testing / CI:** `pytest`, `ruff`, GitHub Actions

---

## 11. Deploy on Streamlit Community Cloud

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) choose **New app** → this repo, branch `main`, main file `app.py`. Under **Advanced settings** pick Python 3.12 or lower (`numpy<2` has no wheels for 3.13).
3. The app reads only `data/regime_store.db`, so commit it after running `scripts/run_pipeline.py` locally.
4. Deploy.
