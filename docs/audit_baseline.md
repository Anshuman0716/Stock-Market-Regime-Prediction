# Baseline Audit Report

**Date:** 2026-08-16
**Scope:** All code files in the `stock-regime-project` repo prior to rebuild.
**Files audited:**
- [`app.py`](file:///c:/Users/anshu/OneDrive/Desktop/stock-regime-project/app.py)
- [`01_data_pipeline.ipynb`](file:///c:/Users/anshu/OneDrive/Desktop/stock-regime-project/01_data_pipeline.ipynb)
- [`02_hmm_model.ipynb`](file:///c:/Users/anshu/OneDrive/Desktop/stock-regime-project/02_hmm_model.ipynb)
- [`03_backtesting.ipynb`](file:///c:/Users/anshu/OneDrive/Desktop/stock-regime-project/03_backtesting.ipynb)

> [!IMPORTANT]
> This is a **read-only** audit. No files have been modified.

---

## 1. Duplication Map

The same logic is implemented independently in multiple files. This is the primary structural problem: there is **no shared `src/` module** — each file rebuilds the pipeline from scratch.

### 1.1 Data Download

| Logic | File | Lines / Cell | Identical? | Notes |
|---|---|---|---|---|
| Download S&P 500 (`^GSPC`) | `app.py` | L44 | — | Hardcoded `end="2024-12-31"` |
| Download S&P 500 (`^GSPC`) | `01_data_pipeline.ipynb` | Cell 2 | ✅ Identical dates | `start="2000-01-01"`, `end="2024-12-31"` |
| Download S&P 500 (`^GSPC`) | `02_hmm_model.ipynb` | Cell 2 | ✅ Identical dates | Same as above |
| Download S&P 500 (`^GSPC`) | `03_backtesting.ipynb` | Cell 2 | ✅ Identical dates | Same as above |
| Download VIX (`^VIX`) | `app.py` | L45 | — | Same hardcoded end |
| Download VIX (`^VIX`) | `01_data_pipeline.ipynb` | Cell 3 | ✅ Identical | — |
| Download VIX (`^VIX`) | `02_hmm_model.ipynb` | Cell 2 | ✅ Identical | — |
| Download VIX (`^VIX`) | `03_backtesting.ipynb` | Cell 2 | ✅ Identical | — |

> [!WARNING]
> **All four files hardcode `end="2024-12-31"`.** The README spec explicitly calls this out as a limitation to fix.

### 1.2 Feature Engineering

This is the highest-risk duplication area. Three different feature sets exist across the repo:

| Logic | File | Location | Feature Count | Features |
|---|---|---|---|---|
| Feature engineering | `app.py` | L54–L86 (`load_data`) | **18** | Full set: returns, rsi, rsi_fast, stoch_k, williams_r, macd, macd_signal, macd_hist, ema_ratio, adx, volatility_20, volatility_5, atr, bb_width, bb_percent, obv_ratio, vix, vix_ma |
| Feature engineering | `01_data_pipeline.ipynb` | Cells 4–8 | **5** | returns, volatility (20d), rsi, macd, vix |
| Feature engineering | `02_hmm_model.ipynb` | Cell 2 | **5** | returns, volatility (20d), rsi, macd, vix |
| Feature engineering | `03_backtesting.ipynb` | Cell 2 | **18** | Full set (same as `app.py`) |

> [!CAUTION]
> **DRIFT DETECTED — CRITICAL.** Notebooks 01 and 02 use a **5-feature** pipeline while `app.py` and notebook 03 use the **18-feature** pipeline. This means:
> - Notebook 01 (`01_data_pipeline.ipynb`) is **completely stale** — it doesn't match the pipeline actually in use.
> - Notebook 02 (`02_hmm_model.ipynb`) trains the HMM on a **different feature set** than what `app.py` and notebook 03 use. Any model development insights from notebook 02 are **not comparable** to the model running in the dashboard or the backtest.
> - Results across files are **fundamentally inconsistent** — they're not even training on the same inputs.

### 1.3 HMM Training

| Logic | File | Location | n_states | n_iter | Scaler | Smoothing |
|---|---|---|---|---|---|---|
| HMM fit + predict | `app.py` | L96–L115 (`train_model`) | User-selected (default 4) | 2000 | Full dataset | Centered, window ±10 (21 total) |
| HMM fit + predict | `02_hmm_model.ipynb` | Cell 3 | **3** | **1000** | Full dataset | **None** |
| HMM fit + predict | `03_backtesting.ipynb` | Cell 4 (corrected: Cell 3 errored, Cell 4 succeeded) | **4** | 2000 | Full dataset | Centered, window ±10 (21 total) |

> [!CAUTION]
> **DRIFT DETECTED — CRITICAL.** Notebook 02 uses `n_components=3` and `n_iter=1000`, while `app.py` and notebook 03 use `n_components=4` and `n_iter=2000`. Notebook 02 also applies **no smoothing at all**, while the other two use centered smoothing. This means the model development notebook is training a fundamentally different model than what's deployed.

### 1.4 Regime Labeling

| Logic | File | Location | Method |
|---|---|---|---|
| Regime labeling | `app.py` | L117–L134 | Volatility + return rank-based, dynamic mapping |
| Regime labeling | `02_hmm_model.ipynb` | Cell 5 | **Hardcoded**: `{0: 'Bull', 1: 'Transition', 2: 'Crisis'}` |
| Regime labeling | `03_backtesting.ipynb` | Cell 4 | **Hardcoded**: `{0: 'Bull', 1: 'Bull', 2: 'Crisis', 3: 'Transition'}` |

> [!WARNING]
> **DRIFT DETECTED.** Three completely different labeling strategies:
> - `app.py` uses a **dynamic rank-based** approach (ranking states by volatility and returns, then assigning labels).
> - Notebook 02 uses a **hardcoded 3-state map** (no ranking logic).
> - Notebook 03 uses a **hardcoded 4-state map** that collapses two states into "Bull".
>
> Since HMM state numbering is arbitrary across runs, the hardcoded maps in the notebooks are only valid for the specific run that produced them. Re-running the notebooks could produce entirely different (incorrect) labels.

### 1.5 Smoothing

| Logic | File | Location | Window | Type |
|---|---|---|---|---|
| Regime smoothing | `app.py` | L110–L115 | 21 (`i - 10` to `i + 11`) | **Centered** (future-leaking) |
| Regime smoothing | `02_hmm_model.ipynb` | — | — | **None** |
| Regime smoothing | `03_backtesting.ipynb` | Cell 4 | 21 (`i - window//2` to `i + window//2 + 1`) | **Centered** (future-leaking) |

### 1.6 Backtest Logic / Metrics

| Logic | File | Location | Notes |
|---|---|---|---|
| Backtest (binary Bull/cash) | `app.py` | L185–L206 | Single-asset, binary exposure, zero transaction costs |
| Backtest (binary Bull/cash) | `03_backtesting.ipynb` | Cells 5–7 | Multi-asset, binary exposure, zero transaction costs |
| Performance metrics (CAGR, vol, Sharpe, max DD) | `app.py` | L211–L224 (`get_metrics`) | Sharpe uses `(cagr - 0.02) / annual_vol` |
| Performance metrics (CAGR, vol, Sharpe, max DD) | `03_backtesting.ipynb` | Cell 7 (`get_metrics` inside `backtest_asset`) | Identical formula to `app.py` |

> [!NOTE]
> The metrics functions are **identical** between `app.py` and notebook 03 — they've been copy-pasted. This is the one case where drift hasn't occurred, but the duplication remains a maintenance risk.

---

## 2. Look-Ahead Inventory

Every item below means the model for day *t* uses information from day *t+1* or later. This **invalidates all reported backtest results**.

### 2.1 StandardScaler Fit on Full Dataset

| # | File | Location | Code | Severity |
|---|---|---|---|---|
| LA-1 | `app.py` | L97–L98 | `scaler = StandardScaler()` then `X = scaler.fit_transform(df[feature_cols])` | 🔴 **CRITICAL** |
| LA-2 | `02_hmm_model.ipynb` | Cell 3, lines 1–2 | `scaler = StandardScaler()` then `X = scaler.fit_transform(df)` | 🔴 **CRITICAL** |
| LA-3 | `03_backtesting.ipynb` | Cell 4, lines 2–3 | `scaler = StandardScaler()` then `X = scaler.fit_transform(df[feature_columns])` | 🔴 **CRITICAL** |

**Why it leaks:** `fit_transform()` computes the mean and standard deviation across the **entire** dataset (2000–2024). This means the scaler for day *t* = 2005-01-03 "knows" the distributional statistics of 2006–2024 data. In a walk-forward setup, the scaler must be fit only on data through day *t*.

**Impact:** Every scaled feature value is shifted by future information. This contaminates all downstream HMM training and prediction. The bias direction is hard to predict — it could inflate or deflate performance — but the results are definitionally invalid.

### 2.2 HMM Fit on Full Dataset (No Walk-Forward)

| # | File | Location | Code | Severity |
|---|---|---|---|---|
| LA-4 | `app.py` | L106–L107 | `model.fit(X)` then `regimes = model.predict(X)` | 🔴 **CRITICAL** |
| LA-5 | `02_hmm_model.ipynb` | Cell 3, lines 8–11 | `model.fit(X)` then `regimes = model.predict(X)` | 🔴 **CRITICAL** |
| LA-6 | `03_backtesting.ipynb` | Cell 4, lines 10–11 | `model.fit(X)` then `regimes = model.predict(X)` | 🔴 **CRITICAL** |

**Why it leaks:** The HMM is trained on the entire 2000–2024 dataset, then used to "predict" regimes for that same dataset. The model has already seen every data point it's labeling. This is equivalent to fitting a classifier on both train and test sets — it maximally overfits to the historical data.

**Impact:** This is the **single most severe** look-ahead bias. The HMM's transition matrix, means, and covariances are optimized to explain the full history, including future crises and recoveries. Regime labels will appear far more accurate than any real-time model could achieve. Every backtest metric (Sharpe, drawdown, CAGR) is **inflated by an unknowable amount**.

### 2.3 Centered Smoothing Window (Future Days in Labels)

| # | File | Location | Code | Severity |
|---|---|---|---|---|
| LA-7 | `app.py` | L111–L114 | `start = max(0, i - 10)`, `end = min(len(regimes), i + 11)` | 🟠 **HIGH** |
| LA-8 | `03_backtesting.ipynb` | Cell 4, smoothing function | `start = max(0, i - window // 2)`, `end = min(len(regimes), i + window // 2 + 1)` | 🟠 **HIGH** |

**Why it leaks:** For day *t*, the smoothing window includes up to 10 **future** days (`i + 1` through `i + 10`). The mode of this centered window determines today's regime label using tomorrow's (and the next 9 days') raw labels.

**Impact:** This allows regime transitions to be "detected" up to 10 days before they actually occur in the raw HMM output. In a backtest, this is the equivalent of knowing a crash is coming ~2 weeks early and exiting. This alone could add significant phantom alpha to the strategy.

**Fix required:** Use trailing-only smoothing: `start = max(0, i - 20)`, `end = i + 1` (21-day window, all past + current).

### 2.4 Full-Sequence Viterbi Decode

| # | File | Location | Code | Severity |
|---|---|---|---|---|
| LA-9 | All files using `model.predict(X)` | See LA-4, LA-5, LA-6 | `model.predict(X)` on the full dataset | 🟠 **HIGH** |

**Why it leaks:** `hmmlearn`'s `predict()` method uses the Viterbi algorithm, which finds the **globally optimal** state sequence for the entire observation sequence. This means the label for day *t* is influenced by observations from day *t+1* through the last day. In contrast, a causal (online) decoder would label day *t* using only observations through day *t*.

**Impact:** Combined with the full-dataset fit (LA-4/5/6), this is a double leak — the model is optimized on all data AND the decoding uses all data. Even if the model were properly walk-forward trained, full-sequence Viterbi on the entire history would still leak future information into past labels.

### 2.5 Trade Signal / Return Alignment

| # | File | Location | Code | Severity |
|---|---|---|---|---|
| LA-10 | `app.py` | L185–L193 | `asset_returns = asset_close.pct_change().reindex(df.index)` then strategy uses same-day regime | 🟡 **MODERATE** |
| LA-11 | `03_backtesting.ipynb` | Cell 5 | `df['strategy_returns'] = np.where(df['regime_label'] == 'Bull', df['returns'], 0)` | 🟡 **MODERATE** |

**Why it leaks:** The regime label for day *t* is computed from day *t*'s features (which use day *t*'s close price). The strategy then captures day *t*'s return, which is also computed from day *t*'s close. In practice, a regime signal computed from today's close cannot be acted on until tomorrow's open at the earliest.

**Impact:** This is a standard off-by-one alignment issue. It grants the strategy a 1-day informational advantage on every trade entry and exit. The impact is moderate compared to the scaler/HMM/smoothing leaks, but still inflates results.

### 2.6 Summary of Look-Ahead Bias Severity

| Finding | Description | Severity |
|---|---|---|
| **LA-4/5/6** | HMM fit on full dataset | 🔴 **CRITICAL** — invalidates all results |
| **LA-1/2/3** | Scaler fit on full dataset | 🔴 **CRITICAL** — contaminates all feature values |
| **LA-7/8** | Centered smoothing window | 🟠 **HIGH** — adds ~10 days of future knowledge |
| **LA-9** | Full-sequence Viterbi decode | 🟠 **HIGH** — decoder uses future observations |
| **LA-10/11** | Same-day signal / return alignment | 🟡 **MODERATE** — 1-day informational advantage |

> [!CAUTION]
> **Every single backtest number in the current repo is invalid.** The combination of LA-1 through LA-9 means the model has seen the future at every stage of the pipeline. Walk-forward validation (Phase 1 of the rebuild) is the mandatory first fix — nothing else can be meaningfully evaluated until these leaks are eliminated.

---

## 3. Feature Inventory

Mapping the 18 indicators specified in README Section 6 to their presence in the current codebase:

### 3.1 Momentum (5)

| Indicator | README Spec | `app.py` | NB 01 | NB 02 | NB 03 |
|---|---|---|---|---|---|
| `returns` | daily pct_change | ✅ L54 | ✅ Cell 4 | ✅ Cell 2 | ✅ Cell 2 |
| `rsi` (14-day) | RSI length=14 | ✅ L55 | ✅ Cell 6 | ✅ Cell 2 | ✅ Cell 2 |
| `rsi_fast` (7-day) | RSI length=7 | ✅ L56 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `stoch_k` | Stochastic %K | ✅ L57 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `williams_r` | Williams %R | ✅ L58 | ❌ Missing | ❌ Missing | ✅ Cell 2 |

### 3.2 Trend (5)

| Indicator | README Spec | `app.py` | NB 01 | NB 02 | NB 03 |
|---|---|---|---|---|---|
| `macd` | MACD 12/26/9 | ✅ L60 | ✅ Cell 7 | ✅ Cell 2 | ✅ Cell 2 |
| `macd_signal` | MACD signal line | ✅ L61 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `macd_hist` | MACD histogram | ✅ L62 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `ema_ratio` | EMA20/EMA50 | ✅ L63–L65 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `adx` | ADX 14 | ✅ L66 | ❌ Missing | ❌ Missing | ✅ Cell 2 |

### 3.3 Volatility (5)

| Indicator | README Spec | `app.py` | NB 01 | NB 02 | NB 03 |
|---|---|---|---|---|---|
| `volatility_20` | rolling(20).std() | ✅ L67 | ✅ Cell 5 (as `volatility`) | ✅ Cell 2 (as `volatility`) | ✅ Cell 2 |
| `volatility_5` | rolling(5).std() | ✅ L68 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `atr` | ATR length=14 | ✅ L69 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `bb_width` | BBands width | ✅ L70–L71 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `bb_percent` | BBands %B | ✅ L72 | ❌ Missing | ❌ Missing | ✅ Cell 2 |

### 3.4 Volume / Macro (3)

| Indicator | README Spec | `app.py` | NB 01 | NB 02 | NB 03 |
|---|---|---|---|---|---|
| `obv_ratio` | OBV / EMA(OBV, 20) | ✅ L73–L75 | ❌ Missing | ❌ Missing | ✅ Cell 2 |
| `vix` | VIX close | ✅ L76 (as `vix_close`) | ✅ Cell 3 | ✅ Cell 2 | ✅ Cell 2 |
| `vix_ma` | EMA(VIX, 10) | ✅ L76 | ❌ Missing | ❌ Missing | ✅ Cell 2 |

### 3.5 Summary

| Location | Features Present | Features Missing (vs README spec) |
|---|---|---|
| **`app.py`** | **18/18** ✅ | None |
| **`01_data_pipeline.ipynb`** | **5/18** ❌ | 13 missing (rsi_fast, stoch_k, williams_r, macd_signal, macd_hist, ema_ratio, adx, volatility_5, atr, bb_width, bb_percent, obv_ratio, vix_ma) |
| **`02_hmm_model.ipynb`** | **5/18** ❌ | Same 13 as above |
| **`03_backtesting.ipynb`** | **18/18** ✅ | None |

> [!NOTE]
> The 18-feature set exists in `app.py` and `03_backtesting.ipynb`. Notebooks 01 and 02 are stuck on the original 5-feature pipeline. The feature engineering code is copy-pasted between `app.py` and notebook 03 — there is no shared source of truth.

---

## 4. Dead / Stale Code

### 4.1 `01_data_pipeline.ipynb` — **STALE**

This notebook is explicitly called out in the README as stale. Confirmed findings:

- Uses a **5-feature** pipeline (returns, volatility, rsi, macd, vix) — missing 13 of the 18 indicators used in the actual pipeline.
- Does **not** import `pandas_ta` until Cell 6 (RSI computation); the earlier cells use manual rolling std for volatility.
- Does **not** train an HMM — stops at feature visualization. Its stated purpose ("data pipeline") is misleading since it doesn't produce the pipeline actually used.
- The DataFrame it produces (`df` with 5 features, shape `(6263, 5)`) is incompatible with the downstream notebooks/app that expect 18 features.

**Verdict:** This notebook is completely non-functional as part of the current pipeline. It must be rewritten to use the 18-feature set and import from `src/`.

### 4.2 `02_hmm_model.ipynb` — **PARTIALLY STALE**

- Uses the **5-feature** pipeline, not the 18-feature pipeline.
- Uses `n_components=3` and `n_iter=1000`, while the actual model uses `n_components=4` and `n_iter=2000`.
- Applies **no smoothing** to regime labels.
- Uses **hardcoded** regime name mapping (`{0: 'Bull', 1: 'Transition', 2: 'Crisis'}`) instead of the dynamic rank-based labeling in `app.py`.
- Contains visualization and analysis code that is valid in concept but produces results from a **different model** than what's actually deployed.

**Verdict:** The model development notebook is training and analyzing a model that is fundamentally different from the one in production. Any insights about regime characteristics, state distributions, or model quality from this notebook are **not transferable** to the actual 18-feature, 4-state model.

### 4.3 `03_backtesting.ipynb` — **PARTIALLY FUNCTIONAL but DUPLICATED**

- Contains the correct 18-feature pipeline.
- Trains the correct 4-state HMM with `n_iter=2000`.
- Cell 3 contains a **failed execution** (tried to `scaler.fit_transform(df)` but `df` already contained string regime labels from a prior cell — the cell errored with `ValueError: could not convert string to float: 'Bull'`). Cell 4 corrects this by using `df[feature_columns]`.
- Contains the centered smoothing (look-ahead bias).
- Uses hardcoded regime names (`{0: 'Bull', 1: 'Bull', 2: 'Crisis', 3: 'Transition'}`) that merge states 0 and 1 into "Bull" — this is fragile and specific to one HMM run.
- All logic is duplicated from `app.py` rather than imported from a shared module.

**Verdict:** Functionally the closest to the actual pipeline, but all logic is copy-pasted. The hardcoded regime map is a correctness risk on re-runs.

### 4.4 Image Artifacts — **STALE**

Six `.png` files exist in the repo root:
- `asset_backtest_comparison.png`
- `features_plot.png`
- `final_regime_chart.png`
- `regime_chart.png`
- `smoothed_regime_chart.png`
- `smoothed_regime_chart_21.png`

These were generated from the **old (leaky) pipeline** and will need to be regenerated after the walk-forward fix. Their presence in the repo root (rather than a `figures/` directory) also suggests a lack of organizational structure.

---

## 5. Additional Observations

### 5.1 No `src/` module exists
There is no shared Python package. Every file independently downloads data, computes features, fits the model, and runs predictions. This violates the "one source of truth" principle.

### 5.2 No `config/` or `config.yaml`
All parameters (tickers, dates, n_states, n_iter, smoothing window) are hardcoded in each file individually. Changes must be manually synchronized across 4 files.

### 5.3 No tests
No `tests/` directory, no `pytest` configuration, no CI. There is no automated way to verify correctness of any component.

### 5.4 No `requirements.txt`
Dependencies are not formally declared.

### 5.5 No model persistence
The HMM is retrained from scratch on every Streamlit session (`app.py` L92–L106 inside `@st.cache_data`). While `st.cache_data` provides session-level caching, there is no `joblib` persistence.

### 5.6 `app.py` always computes features on S&P 500
In `app.py`, the `load_data` function (L43–L89) computes all 18 features from S&P 500 data (`sp500` high/low/close/volume), regardless of which asset the user selects. The selected asset's close price is only used for the backtest return calculation (L88). This is intentional design (regimes are detected from S&P 500 and applied to other assets), but it's not documented or commented.

---

## Appendix: File Cross-Reference

| File | Features | n_states | n_iter | Scaler Scope | Smoothing | Labeling | Backtest |
|---|---|---|---|---|---|---|---|
| `app.py` | 18 | 4 (user-selectable) | 2000 | Full dataset | Centered ±10 | Dynamic rank | Binary, 0-cost |
| `01_data_pipeline.ipynb` | 5 | — | — | — | — | — | — |
| `02_hmm_model.ipynb` | 5 | 3 | 1000 | Full dataset | None | Hardcoded 3-state | — |
| `03_backtesting.ipynb` | 18 | 4 | 2000 | Full dataset | Centered ±10 | Hardcoded 4-state | Binary, 0-cost, multi-asset |
