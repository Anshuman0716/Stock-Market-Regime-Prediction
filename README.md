# Market Regime Detection System

**Rebuild specification — Data Scientist / quant-fintech track**

This file is both the build brief for the coding agent doing the rebuild and the
living README for the finished project. Sections marked `[TODO — fill after
Phase X]` are placeholders: leave them empty (or lightly stubbed) until the
corresponding phase is actually done and measured. Never write a number,
Sharpe ratio, or claim into this file that hasn't been produced by code that
has run.

---

## 0. Instructions for the build agent

Read this whole document before writing any code. Then work through
**Section 8 (Build Phases) in order, one phase at a time.** Each phase has a
task list and a "definition of done." Do not start Phase *N+1* until Phase
*N*'s definition of done is satisfied — every later number in this project
(Sharpe ratios, drawdowns, regime stability) depends on Phase 1 (removing
look-ahead bias) being correct first. Building statistical rigor or risk
features on top of a leaky pipeline just produces more confidently wrong
numbers.

A few standing rules that apply across every phase:

- **One source of truth.** Feature engineering and HMM logic live in `src/`
  only. Notebooks and `app.py` import from `src/`; they never redefine or
  recompute the same logic locally. If you catch yourself pasting a function
  into a second file, stop and import it instead.
- **No future information, anywhere.** Not in scaling, not in HMM fitting,
  not in smoothing, not in feature windows. If a computation for day *t*
  touches data from day *t+1* or later, it's a bug.
- **Comment for a learner, not just a reviewer.** Anshuman is building this
  project to learn ML/quant concepts, not just to ship it. In `src/`,
  comments should explain *why* a choice was made (e.g. why walk-forward
  instead of a single fit, why trailing smoothing instead of centered, why
  full covariance instead of diagonal) — not just restate what the line
  does.
- **Stay in scope.** See Section 2 for what is explicitly out of bounds. The
  fastest way to weaken this project is to widen it.

---

## 1. Project summary

Financial markets move through distinct regimes — sustained stretches with
different risk and return characteristics. This project detects those
regimes (Bull / Transition / Crisis) directly from price and volatility
data, with no manual labeling, using a 4-state Gaussian Hidden Markov Model
trained on 18 engineered technical indicators.

What makes this a *Data Scientist* project rather than an ML/AI/Data
Engineering one is the emphasis: rigorous validation (walk-forward, no
leakage), statistical honesty (baselines, confidence intervals, ablations),
and financial realism (transaction costs, position sizing, VaR/CVaR,
performance broken out by market cycle) — not infrastructure.

---

## 2. Role target and scope guardrails

### Role fit

| Role | Verdict | Why |
|---|---|---|
| **Data Scientist** | Strong native fit | Feature engineering, statistical modeling, backtesting, and insight storytelling are already the core of the work. |
| ML Engineer | Partial — not pursued here | Needs serving infrastructure, CI/CD, containerization, and experiment tracking — a distinct specialization, not this project's focus. |
| AI Engineer | Not a fit | Needs LLM / RAG / agent tooling unrelated to the core time-series modeling problem. |
| Data Engineer | Not a fit | Needs orchestration, warehousing, and streaming infrastructure at a data scale this project doesn't have. |

This rebuild deliberately targets **Data Scientist, quant/fintech flavor**,
rather than stretching to touch every role at once. That's the single
biggest scope risk to manage.

### Keep / Cut / Fix / Add

| Decision | Item | Rationale |
|---|---|---|
| **KEEP** | 18-indicator feature set (momentum, trend, volatility, volume) | Already well-designed — spans four genuinely distinct indicator families. |
| **KEEP** | 4-state Gaussian HMM core | Correct model choice for unsupervised regime detection — validate it properly, don't replace it. |
| **KEEP** | Multi-asset backtest (S&P 500, NASDAQ, Gold, Bitcoin) | Demonstrates generalization and surfaces a real insight (Gold's safe-haven behavior). |
| **CUT** | RAG / vector DB / LangChain / any LLM tooling | AI Engineer scope — would read as bolted-on, not depth. |
| **CUT** | Airflow / dbt / Snowflake / Spark / Kafka | Data Engineer scope — no data volume or streaming need exists here. |
| **CUT** | Docker / FastAPI / cloud deployment / MLflow | ML Engineer scope — a clean local pipeline plus Streamlit is the right footprint. |
| **FIX** | Duplicated feature/training code across notebooks + `app.py` | Replace with one shared `src/` module. Duplication is a correctness risk. |
| **FIX** | Stale 5-feature `01_data_pipeline.ipynb` | Rewrite to match the real 18-feature pipeline used everywhere else. |
| **ADD** | Walk-forward validation + trailing-only smoothing | Fixes the look-ahead bias currently inflating every backtest result. |
| **ADD** | Baseline strategy + statistical significance testing | Turns "it worked" into "it worked, and here's the confidence interval." |
| **ADD** | Transaction costs, slippage, position sizing | Removes the biggest fintech-credibility gap — costless, all-or-nothing trading. |
| **ADD** | VaR / CVaR reporting | Standard risk-desk vocabulary; cheap to add from data already on hand. |
| **ADD** | Out-of-sample testing across market cycles | Shows the edge isn't an artifact of one lucky historical period. |
| **ADD** | Lightweight SQL layer (SQLite) | Covers the Data Scientist role's SQL expectation honestly. |
| **ADD** | Model risk / limitations section in README | Institutional framing that fintech reviewers respond well to. |

**Do not add**, even if it seems like a nice-to-have mid-build: any LLM/agent
tooling, any workflow orchestrator, any container or cloud deployment layer,
any experiment-tracking platform. If a task on the roadmap seems to require
one of these, it's a sign the task has drifted out of scope — flag it
instead of building it.

---

## 3. Non-negotiable engineering principles

1. **Single direction data flow:**
   `loader → engineering → walk-forward HMM → labeling → backtest/risk → SQLite store`
   — notebooks and `app.py` *read* from `src/` and the store; they never
   recompute independently.
2. **No look-ahead, anywhere in the pipeline.** This is the reason Phase 1
   exists and gates everything else.
3. **No LLM tooling, no orchestration, no container/cloud layer.** See
   Section 2.
4. **Every notebook and `app.py` call into the same shared package** —
   nothing is computed twice.

---

## 4. Final repository structure

```
market-regime-detection/
├── README.md                    # methodology, results, limitations
├── requirements.txt
├── config/
│   └── config.yaml               # tickers, dates, features, n_states
├── src/
│   ├── data/
│   │   └── loader.py              # yfinance download + local caching
│   ├── features/
│   │   └── engineering.py         # 18 indicators, single source of truth
│   ├── models/
│   │   ├── hmm.py                 # walk-forward fit/predict, save/load
│   │   └── labeling.py            # rank-based regime labeling
│   ├── backtest/
│   │   ├── engine.py              # walk-forward backtest loop
│   │   ├── baselines.py           # buy-hold, 200-day MA filter
│   │   ├── risk.py                # costs, position sizing, VaR/CVaR
│   │   └── metrics.py             # Sharpe, Sortino, Calmar, drawdown
│   ├── stats/
│   │   └── significance.py        # bootstrap CIs, hypothesis tests
│   └── db/
│       └── store.py               # SQLite persistence + SQL helpers
├── notebooks/
│   ├── 01_eda_and_features.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_backtest_and_risk.ipynb
│   └── 04_sql_analysis.ipynb
├── tests/
│   ├── test_features.py
│   ├── test_hmm.py
│   └── test_backtest.py
├── models/                       # persisted .joblib artifacts
├── data/
│   └── regime_store.db           # SQLite database
├── app.py                        # Streamlit dashboard, imports src/ only
└── .github/workflows/tests.yml   # CI: run pytest on every push
```

---

## 5. Data specification

- **Source:** `yfinance`.
- **Tickers:** `^GSPC` (S&P 500), `^IXIC` (NASDAQ), `GC=F` (Gold),
  `BTC-USD` (Bitcoin), `^VIX` (volatility index, used as a feature for
  every asset).
- **Date range:** config-driven via `config/config.yaml`, not hardcoded.
  Default start `2000-01-01`, end defaults to the most recent available
  trading day. (The original build hardcoded `end="2024-12-31"` — this was
  a real limitation; make it dynamic.)
- **Caching:** raw downloads cached locally (e.g. parquet under `data/`) so
  repeated runs and CI don't re-hit the network every time.

---

## 6. Feature specification (18 indicators)

Keep this set as-is — it's a genuine strength of the project. All 18 live
in `src/features/engineering.py` as the single source of truth.

**Momentum (5)**
- `returns` — daily percent change, the base signal
- `rsi` (14-day) and `rsi_fast` (7-day) — standard and fast-window relative strength
- `stoch_k` — Stochastic %K, momentum relative to recent range
- `williams_r` — similar to Stochastic, different normalization

**Trend (5)**
- `macd`, `macd_signal`, `macd_hist` — trend direction and momentum crossovers
- `ema_ratio` (EMA20/EMA50) — short vs. long trend
- `adx` — trend strength, independent of direction

**Volatility (5)**
- `volatility_20`, `volatility_5` — rolling return standard deviation, two windows
- `atr` — Average True Range, volatility in price terms
- `bb_width`, `bb_percent` — Bollinger Band width and position within the bands

**Volume / macro (3)**
- `obv_ratio` — On-Balance Volume vs. its EMA, buying/selling pressure
- `vix`, `vix_ma` — the market's own fear gauge, raw and smoothed

---

## 7. Modeling specification

- **Model:** 4-state Gaussian HMM, full covariance (justify `n_states=4`
  with the BIC/AIC sweep in Phase 2 rather than asserting it).
- **Scaling:** `StandardScaler` fit *only* on the training window at each
  walk-forward step — never on the full dataset. Fitting the scaler on
  everything leaks future distributional information into "past"
  predictions, same as fitting the HMM on everything does.
- **Walk-forward fitting:** expanding or rolling window, refit on a defined
  cadence (e.g. annually) — the model at time *t* must never have seen data
  from after *t*.
- **Smoothing:** trailing-only rolling mode window (21 days), replacing the
  original centered window. No future days are allowed into today's label.
- **Labeling:** rank the learned states by volatility and return; merge
  into Bull / Transition / Crisis.

---

## 8. Build phases

Work through these in order. Each phase lists its tasks and its definition
of done — don't move to the next phase until the current one's boxes are
actually checked, not just attempted.

### Phase 1 — Fix the foundation


Tasks:
- [ ] Extract feature engineering and HMM training into `src/`; delete the
      duplicated logic in the notebooks and `app.py`.
- [ ] Refit the HMM walk-forward (expanding or rolling window) — never fit
      on data the model wouldn't have had yet.
- [ ] Change the smoothing window from centered to trailing-only.
- [ ] Rewrite `01_eda_and_features.ipynb` to build the real 18-feature set
      (retiring the stale 5-feature version).

Definition of done: there is exactly one implementation of feature
engineering and exactly one implementation of HMM fit/predict in the whole
repo; every regime label for day *t* is produced using only data through
day *t*.

### Phase 2 — Statistical rigor


Tasks:
- [ ] Run a BIC/AIC sweep over 2–6 states; justify `n_states=4` with
      numbers.
- [ ] Check model stability across random seeds and bootstrap resamples of
      the training data.
- [ ] Add a naive baseline — a 200-day moving-average regime filter —
      alongside buy-and-hold.
- [ ] Bootstrap a confidence interval on the Sharpe ratio difference
      between the HMM strategy and each baseline; report a p-value, not
      just a point estimate.
- [ ] Run a feature-category ablation: retrain with momentum-only,
      trend-only, volatility-only, and volume-only feature sets; compare
      regime stability and backtest performance.

Definition of done: `n_states=4` is backed by a BIC/AIC chart, every
reported Sharpe improvement has a bootstrap p-value next to it, and the
ablation results are written up somewhere in the repo (notebook or README
section).

### Phase 3 — Fintech risk realism


Tasks:
- [ ] Model transaction costs as spread + commission, not a flat
      percentage.
- [ ] Replace binary in/cash exposure with probability-weighted position
      sizing using the HMM's `predict_proba` output.
- [ ] Report 95%/99% VaR and CVaR alongside Sharpe and max drawdown.
- [ ] Break results out explicitly by market cycle: dot-com (2000–03),
      pre-GFC bull (2004–07), GFC (2008–09), 2010s bull (2013–19), COVID
      crash (2020), 2022 rate-hike bear.
- [ ] *Optional stretch, highest fintech impact if time allows:* connect
      the walk-forward model to a broker sandbox (e.g. Alpaca) and run it
      as a daily paper-trading loop, logging real signals against live
      data.

Definition of done: no backtest result anywhere in the project assumes
zero-cost, all-or-nothing trades; every headline result has a VaR/CVaR
figure and a per-cycle breakdown next to it.

### Phase 4 — Lightweight SQL layer


Tasks:
- [ ] Persist computed features, regime labels, and backtest runs into
      `data/regime_store.db` (SQLite).
- [ ] Write and showcase real SQL queries in
      `04_sql_analysis.ipynb` — e.g. average regime duration by decade,
      regime frequency by asset, rolling win rate by regime.

Definition of done: the SQL notebook runs real queries against the SQLite
store (not against an in-memory DataFrame relabeled as "SQL").

### Phase 5 — Engineering polish


Tasks:
- [ ] Persist the trained model with `joblib` instead of retraining on
      every Streamlit session.
- [ ] Add `pytest` coverage for the feature functions, the HMM wrapper, and
      the backtest math.
- [ ] Add a GitHub Actions workflow that runs the test suite on every push
      — right-sized CI, no Docker or cloud required.
- [ ] Rewrite this README's results and limitations sections (Sections 12
      and 13) with the real, walk-forward-corrected numbers, and drop any
      remaining placeholder content.

Definition of done: `streamlit run app.py` loads a persisted model instead
of retraining, `pytest` passes locally and in CI, and every number in this
README is one that actually came out of the code.



---

## 9. Config file spec

`config/config.yaml` should look roughly like this — adjust as the build
progresses, but keep tickers/dates/features/n_states out of code:

```yaml
tickers:
  spx: "^GSPC"
  nasdaq: "^IXIC"
  gold: "GC=F"
  bitcoin: "BTC-USD"
  vix: "^VIX"

date_range:
  start: "2000-01-01"
  end: null            # null = most recent available trading day

model:
  n_states: 4
  covariance_type: "full"
  n_iter: 2000
  smoothing_window: 21  # trailing, not centered
  refit_cadence: "annual"

backtest:
  assets: ["spx", "nasdaq", "gold", "bitcoin"]
  transaction_cost_bps: null   # set from spread + commission model, Phase 3
```

---

## 10. Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | yfinance |
| Feature engineering | pandas-ta, pandas, NumPy |
| Modeling | hmmlearn (GaussianHMM) |
| Preprocessing | scikit-learn (StandardScaler) |
| Stats | SciPy, bootstrap resampling |
| Persistence | SQLite, joblib |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Testing / CI | pytest, GitHub Actions |

Explicitly excluded: Docker, FastAPI, MLflow, any cloud deployment target,
any LLM/agent framework, any data-orchestration tool (Airflow/dbt/Spark/Kafka).

---

## 11. Running locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/market-regime-detection.git
cd market-regime-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the tests
pytest

# 5. Run the dashboard
streamlit run app.py
```

---

## 12. Results

`[TODO — fill after Phase 2/3, with real walk-forward numbers only]`

This section should end up containing, per asset (S&P 500, NASDAQ, Gold,
Bitcoin): CAGR, annualized volatility, Sharpe ratio, max drawdown, 95%/99%
VaR and CVaR, the bootstrap p-value on the Sharpe improvement over each
baseline, and the per-market-cycle breakdown. Do not backfill this with the
old (leaky) numbers from the original build — they're invalid once
walk-forward validation is in place, and may move in either direction.

---

## 13. Model risk and limitations

`[TODO — fill after rebuild is complete]`

At minimum, address: the assumption that historical regime dynamics are
stationary going forward; the relatively short and unusually volatile
history available for Bitcoin compared to the other assets; the sensitivity
of regime labels to the smoothing window and refit cadence; and any
features that turned out to be highly correlated with each other in the
ablation study.

---

## 14. Resume bullets

`[Do not publish until the corresponding numbers are actually measured]`

**Core model bullet:**
> "Built and validated a market regime detection system (4-state Gaussian
> HMM, 18 engineered technical indicators) using walk-forward validation to
> eliminate look-ahead bias; backtested across four asset classes with
> realistic transaction costs and bootstrap-tested statistical
> significance, improving [asset] Sharpe ratio from [X] to [Y] (p < [value])
> versus buy-and-hold."

**Risk & tooling bullet:**
> "Added a risk-management layer (VaR/CVaR, probability-weighted position
> sizing) and SQL-backed experiment storage to a market regime detection
> pipeline, and validated robustness across five distinct historical market
> cycles including the 2008 and 2020 crises."

---

## 15. Build order summary

| Phase | Deliverable | Why it matters |
|---|---|---|
| 1 | Foundation fix: walk-forward validation, deduplicated code | Every later number depends on this being correct |
| 2 | Statistical rigor: baselines, significance, ablation | Proves the model, doesn't just assert it |
| 3 | Fintech risk realism: costs, sizing, VaR, cycle testing | Main fintech-credibility lever |
| 4 | Lightweight SQL layer | Honest coverage of the SQL requirement |
| 5 | Engineering polish: tests, CI, README | Signals production-ready habits |
