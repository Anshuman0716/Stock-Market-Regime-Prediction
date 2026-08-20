# Feature Notes — 18 Indicators and Their Regime Signals

This document maps each feature to the regime behaviour it is expected to
capture, and flags pairs that are likely to be highly correlated (feeding
the Phase 2 ablation study).

---

## Momentum (5 features)

| Feature | What it measures | Expected regime signal |
|---|---|---|
| `returns` | Daily percent change in close price | Negative in Crisis, positive in Bull, near-zero mean in Transition |
| `rsi` (14d) | Ratio of average gains to losses, smoothed | Persistently below 30 in Crisis, above 50 in Bull, oscillating in Transition |
| `rsi_fast` (7d) | Same as RSI but faster-reacting | Leads `rsi` into oversold/overbought; divergence between the two signals regime edges |
| `stoch_k` | Where close sits in the recent high-low range (0–100) | Locked near 0 in Crisis, near 100 in strong Bull; whipsawing in Transition |
| `williams_r` | Same concept as Stochastic, inverted scale (-100–0) | Mirror of `stoch_k`; -100 = oversold (Crisis), 0 = overbought (Bull) |

### Expected high correlations
- **`rsi` ↔ `rsi_fast`**: Same indicator, different lookback. Pearson ρ ≈ 0.90+.
- **`stoch_k` ↔ `williams_r`**: Nearly identical math, opposite sign convention. |ρ| ≈ 0.95+.
- **`rsi` ↔ `stoch_k`**: Both momentum oscillators, moderately correlated. ρ ≈ 0.65–0.80.

> **Ablation note:** The stoch_k / williams_r pair is the strongest candidate
> for redundancy. If the Phase 2 ablation shows removing one doesn't hurt
> regime stability, consider dropping it.

---

## Trend (5 features)

| Feature | What it measures | Expected regime signal |
|---|---|---|
| `macd` | Difference between 12-day and 26-day EMAs | Strongly positive in Bull, strongly negative in Crisis |
| `macd_signal` | 9-day EMA of the MACD line | Lags MACD; crossovers mark regime transitions |
| `macd_hist` | MACD minus signal line | Positive and expanding = strengthening Bull; negative and expanding = deepening Crisis |
| `ema_ratio` | EMA(20) / EMA(50), scale-independent trend | > 1 in Bull (short trend above long), < 1 in Crisis; ~1 in Transition |
| `adx` | Average Directional Index (14d), direction-independent | High (>25) in BOTH strong Bull and strong Crisis; low (<20) in Transition |

### Expected high correlations
- **`macd` ↔ `macd_signal`**: Signal is a smoothed version of MACD. ρ ≈ 0.95+.
- **`macd` ↔ `macd_hist`**: Histogram is MACD minus signal, so highly correlated with MACD. ρ ≈ 0.70–0.85.
- **`macd` ↔ `ema_ratio`**: Both measure EMA-based trend; different normalizations. ρ ≈ 0.70–0.85.

> **Key insight:** `adx` is the only feature in this family that is high in
> *both* Bull and Crisis regimes. This makes it uniquely useful for
> distinguishing "strong trend" from "no trend" (Transition), a distinction
> the other trend features cannot make on their own.

---

## Volatility (5 features)

| Feature | What it measures | Expected regime signal |
|---|---|---|
| `volatility_20` | 20-day rolling std of returns | Elevated and sustained in Crisis; low and flat in Bull |
| `volatility_5` | 5-day rolling std of returns | Captures sudden spikes at crisis onset; noisier than 20-day |
| `atr` | 14-day Average True Range (price terms) | Wide daily ranges in Crisis; narrow in calm Bull |
| `bb_width` | Bollinger Band width (proportional to vol) | Expands in Crisis (volatility expansion), contracts in Bull |
| `bb_percent` | Where close sits within Bollinger Bands (0–1) | Near 0 in selloffs (Crisis), near 1 in rallies (Bull) |

### Expected high correlations
- **`volatility_20` ↔ `bb_width`**: BB width is directly derived from rolling std. ρ ≈ 0.90+.
- **`volatility_20` ↔ `volatility_5`**: Same measure, different window. ρ ≈ 0.70–0.85.
- **`volatility_20` ↔ `atr`**: Both measure volatility, different units. ρ ≈ 0.70–0.85.

> **Ablation note:** `volatility_20` and `bb_width` will be very highly
> correlated because BB width is fundamentally a scaled version of rolling
> std. The ablation should test whether keeping both hurts or helps — the
> slight difference (BB width incorporates the moving average level) may
> or may not add information.

---

## Volume / Macro (3 features)

| Feature | What it measures | Expected regime signal |
|---|---|---|
| `obv_ratio` | On-Balance Volume vs. its 20d EMA | > 1 = above-average buying pressure (Bull); < 1 = distribution (Crisis) |
| `vix` | CBOE Volatility Index (implied vol) | Spikes in Crisis (>30), low in Bull (<15), moderate in Transition |
| `vix_ma` | 10-day EMA of VIX | Smoothed fear gauge; sustained high readings confirm Crisis regime |

### Expected high correlations
- **`vix` ↔ `vix_ma`**: Smoothed version of same series. ρ ≈ 0.95+.
- **`vix` ↔ `volatility_20`**: Implied vs. realized vol; moderately correlated. ρ ≈ 0.65–0.80.

> **Key insight:** VIX adds *forward-looking* information from the options
> market. Our volatility indicators are all *backward-looking* (computed from
> realized returns). VIX often rises BEFORE realized volatility spikes,
> giving the HMM an early warning signal that the other features miss.

---

## Summary: Correlation Clusters to Watch in Phase 2 Ablation

| Cluster | Features | Why they're correlated | Ablation question |
|---|---|---|---|
| Fast/slow momentum | `rsi`, `rsi_fast` | Same indicator, different window | Does the fast RSI add information beyond the standard? |
| Range oscillators | `stoch_k`, `williams_r` | Nearly identical math | Can one be dropped without losing regime accuracy? |
| MACD family | `macd`, `macd_signal`, `macd_hist` | Algebraically related | Does keeping all three vs. just `macd` help? |
| Realized vol | `volatility_20`, `bb_width` | BB width ≈ scaled rolling std | Is BB width redundant with volatility_20? |
| Fear gauge | `vix`, `vix_ma` | Smoothed version | Does smoothing add value or just lag? |
