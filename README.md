# Market Regime Detection System

**A Quantitative Data Science & Risk Management Project**

This project detects financial market regimes (Bull / Transition / Crisis) directly from price and volatility data using a 4-state Gaussian Hidden Markov Model (HMM) trained on 18 engineered technical indicators. 

Unlike many academic or tutorial-level ML finance projects, this repository prioritizes **strict elimination of look-ahead bias, robust out-of-sample statistical validation, and realistic fintech risk modeling** (including dynamic transaction costs, probability-weighted position sizing, and VaR/CVaR risk reporting).

---

## 1. Executive Summary & Results

The fundamental question of this project was: *Can an unsupervised Hidden Markov Model reliably detect market crashes in time to protect a portfolio, and does that edge survive real-world trading costs?*

**The Conclusion:** The HMM acts as an exceptionally powerful risk-off filter that severely truncates tail-risk drawdowns—slashing the S&P 500 Max Drawdown from ~56% to ~22%, and volatility from 19.2% to 6.1%. However, the mathematical realities of detection lag (trailing smoothing), false-positive transitions, and transaction costs present a severe drag. The strategy systematically underperforms a passive Buy & Hold baseline on an absolute and risk-adjusted (Sharpe) basis. **It is a powerful capital preservation overlay, but not an absolute alpha generator.**

### Headline Metrics (Strict Out-of-Sample, Post-Cost)

| Asset | Strategy | CAGR | Ann. Vol | Sharpe | Max DD | 95% VaR (Hist) | 99% CVaR (Hist) |
|---|---|---|---|---|---|---|---|
| **S&P 500** | Buy & Hold | 6.54% | 19.23% | 0.340 | -56.78% | - | - |
| | **HMM Regime** | **1.50%** | **6.14%** | **0.243** | **-22.36%** | 0.41% | 2.18% |
| **NASDAQ** | Buy & Hold | 8.12% | 24.15% | 0.315 | -75.12% | - | - |
| | **HMM Regime** | **3.33%** | **9.05%** | **0.367** | **-25.18%** | 0.76% | 2.83% |
| **Gold** | Buy & Hold | 8.81% | 15.33% | 0.590 | -45.13% | - | - |
| | **HMM Regime** | **6.15%** | **11.89%** | **0.517** | **-31.98%** | 1.07% | 3.39% |
| **Bitcoin** | Buy & Hold | 45.10% | 61.22% | 0.791 | -83.15% | - | - |
| | **HMM Regime** | **1.36%** | **30.52%** | **0.045** | **-76.63%** | 2.67% | 8.65% |

*(Note: Bootstrap significance tests against Buy & Hold yielded p > 0.70 across all assets, confirming that the HMM strategy does not statistically significantly improve risk-adjusted returns).*

---

## 2. Core Concepts & Statistical Integrity

This project is built around overcoming the most common flaws in time-series machine learning.

### A. Walk-Forward Validation
To completely eliminate look-ahead bias, the model is never fit on the full dataset. Instead, it uses an **expanding window walk-forward approach**. The model evaluating day `t` has *never* seen data from day `t+1`. The `StandardScaler` used to normalize features is also strictly fit *only* on the training window at each step, preventing future distributional data from leaking into past predictions.

### B. Filtered vs. Smoothed Probabilities
Standard HMM implementations (like `hmmlearn`'s default `.predict()` and `.predict_proba()`) utilize the Viterbi or Forward-Backward algorithms. These algorithms "smooth" probabilities by conditioning on the *entire* sequence passed to them—meaning day 1's probability is influenced by day 250's data. 

To solve this, this pipeline explicitly isolates the **forward-pass lattice (alpha)** to generate strictly *filtered* probabilities, mathematically ensuring that today's state prediction is conditioned exclusively on data available up to today.

### C. Trailing Smoothing 
Raw HMM states can be erratic. We apply a 21-day rolling mode to smooth the signal. Crucially, this is a **trailing** window. Centered windows (often used in academic literature to produce beautiful, clean charts) leak future data into today's label. We explicitly sacrifice responsiveness at regime boundaries to mathematically guarantee zero look-ahead bias.

---

## 3. Architecture & Data Flow

The project strictly enforces a single-direction data flow. Notebooks and dashboards are strictly read-only consumers of the core logic to prevent duplicated, diverging code.

```text
[yfinance Loader] → [18-Feature Engineering] → [Walk-Forward HMM] → [Labeling & Smoothing] → [Backtest Engine & Risk] → [SQLite Store]
```

### Directory Structure
```
market-regime-detection/
├── config/
│   └── config.yaml               # Global configuration (tickers, dates, n_states)
├── src/
│   ├── data/loader.py            # yfinance retrieval & local parquet caching
│   ├── features/engineering.py   # Technical indicators & signal generation
│   ├── models/hmm.py             # Walk-forward fit/predict & filtered probabilities
│   ├── models/labeling.py        # Rank-based labeling & state smoothing
│   ├── backtest/engine.py        # Vectorized backtest with execution lag
│   ├── backtest/risk.py          # Dynamic transaction costs, position sizing, VaR
│   ├── backtest/metrics.py       # Sharpe, Sortino, Calmar, Max DD
│   └── db/store.py               # SQLite schema, upserts, & experiment tracking
├── notebooks/                    # Analytical sandbox (reads from DB & src/)
├── tests/                        # Right-sized pytest suite (look-ahead assertions)
├── scripts/
│   ├── train.py                  # Entrypoint: train models & generate joblib artifacts
│   ├── backfill_db.py            # Entrypoint: populate SQLite with backtest runs
│   └── paper_trade.py            # Alpaca API sandbox for live forward-testing
├── models/                       # Persisted .joblib artifacts & manifests
├── data/                         # regime_store.db SQLite database
└── app.py                        # Streamlit dashboard
```

---

## 4. Feature Engineering (The 18 Indicators)

The HMM is fed an 18-dimension feature vector spanning 4 orthogonal market factors, computed via `pandas-ta`.

**1. Momentum (5)**
- `returns`: Daily percent change (the base signal).
- `rsi` (14-day) & `rsi_fast` (7-day): Standard and fast-window relative strength.
- `stoch_k`: Stochastic %K (momentum relative to the recent high/low range).
- `williams_r`: Williams %R (momentum normalized differently to Stochastic).

**2. Trend (5)**
- `macd`, `macd_signal`, `macd_hist`: Moving Average Convergence Divergence capturing trend direction and acceleration.
- `ema_ratio`: Ratio of 20-day EMA to 50-day EMA.
- `adx`: Average Directional Index (trend strength, independent of direction).

**3. Volatility (5)**
- `volatility_20`, `volatility_5`: Rolling return standard deviation (fast and slow).
- `atr`: Average True Range (volatility in absolute price terms).
- `bb_width`, `bb_percent`: Bollinger Band width and price position relative to the bands.

**4. Volume / Macro (3)**
- `obv_ratio`: On-Balance Volume divided by its 20-day EMA (buying vs. selling pressure).
- `vix`, `vix_ma`: The CBOE Volatility Index (raw and smoothed), appended cross-asset as a universal macro fear gauge.

---

## 5. Risk Management & Backtest Realism

- **Execution Lag:** Target weights generated at the close of day $t$ are not executed flawlessly at the same close. The engine explicitly models execution lag, shifting positions to prevent impossible intra-day look-ahead.
- **Dynamic Transaction Costs:** Modeled as spread + commission (e.g., 1.5 bps for high-liquidity equities, 5.5 bps for crypto) applied strictly to portfolio turnover.
- **Probability-Weighted Sizing:** Instead of binary (100% in or 100% out) trading, the portfolio scales its exposure continuously based on the HMM's exact posterior probability of being in a Bull regime.
- **Tail Risk Reporting:** Generates historical 95% and 99% Value-at-Risk (VaR) and Conditional VaR (CVaR) to satisfy institutional risk-desk requirements.

---

## 6. Model Limitations & Quant Risk Audit

While the pipeline is statistically rigorous, it has real limitations that would block a hedge fund production deployment:

1. **Detection Lag:** The mathematically mandatory trailing smoothing window (21 days) forces severe lag. In the 2008 GFC, the model did not confidently output a `Crisis` label until the market had already drawn down significantly. In the 2020 COVID crash, the V-shape recovery occurred before the model could re-enter the `Bull` regime, missing the rally entirely.
2. **Transaction Cost Sensitivity:** The strategy trades frequently between Transition and Bull states. Applying realistic bid-ask spread and commission costs drastically drags the CAGR. 
3. **Gaussian HMM on Fat-Tailed Returns:** The `hmmlearn` Gaussian HMM assumes normally distributed emissions. Financial returns are non-Gaussian (excess kurtosis/fat tails). Forcing a Gaussian emission means the model mathematically underestimates extreme events, forcing it to violently flip states to accommodate outliers.
4. **Feature Correlation:** Trend features (MACD) and Momentum features (RSI) are highly collinear, meaning the 18-dimension vector is effectively over-weighting standard price momentum rather than orthogonal factors.
5. **Index Survivorship:** Modeling the `^GSPC` as a tradeable asset assumes zero tracking error and frictionless execution, ignoring the survivorship bias and rebalancing drag inherent in the actual index composition.

---

## 7. How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Anshuman0716/Stock-Market-Regime-Prediction.git
cd Stock-Market-Regime-Prediction

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the walk-forward models (outputs to models/ directory)
python scripts/train.py

# 5. Execute backtests and populate SQLite database
python scripts/backfill_db.py

# 6. Run the rigorous look-ahead test suite
pytest tests/

# 7. Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## 8. Tech Stack

- **Data Science:** Python 3.10+, `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Modeling:** `hmmlearn` (4-State Gaussian HMM)
- **Features:** `pandas-ta` (Technical Analysis)
- **Data & Persistence:** `yfinance`, SQLite (`sqlite3`), `joblib`
- **Visualization & UI:** `matplotlib`, `streamlit`
- **CI/CD & Testing:** `pytest`, GitHub Actions
