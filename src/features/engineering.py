"""
Feature engineering: 8 core technical indicators to prevent overparameterization.

Why every feature is SCALE-FREE
-------------------------------
The HMM's StandardScaler is fit on the training window (e.g. 2000-2009) and
then applied to later years. A feature measured in price units drifts as
prices rise: by 2025 the S&P 500's ATR in index points sat ~7 training
standard deviations above its 2000-09 average, simply because the index is
~5x higher, not because markets were calmer or wilder. The model then sees
"extreme volatility" every day. So:

- ATR and MACD are divided by Close (percent of price, comparable across
  decades and across assets).
- OBV is a running total whose level depends on the start date and can
  cross zero; dividing it by its own EMA blew up to values like +1,000 and
  -400 near zero-crossings. It is replaced by a bounded 20-day volume-flow
  ratio in [-1, 1].
"""

import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "returns",
    "rsi",
    "macd_pct",
    "adx",
    "volatility_20",
    "atr_pct",
    "obv_flow_20",
    "vix"
]

# Window for the volume-flow ratio (about one trading month).
FLOW_WINDOW = 20

# ADX smooths twice over its 14-day window, so it needs at least 2 * 14 rows.
# The `ta` library crashes (instead of returning NaN) on anything shorter.
ADX_WINDOW = 14
MIN_INPUT_ROWS = 2 * ADX_WINDOW


def build_features(df, vix_close):
    """
    Build the 8 HMM features from one asset's OHLCV frame plus VIX.

    Every indicator is a trailing calculation (rolling / ewm / recursive
    smoothing over past rows only), so the value on day t depends only on
    data up to t. Leading warm-up rows, where an indicator is not yet
    defined, are dropped rather than filled.

    Returns an empty frame with the standard columns when the input is too
    short to compute every indicator.
    """
    if len(df) < MIN_INPUT_ROWS:
        logger.info(f"Input has {len(df)} rows (< {MIN_INPUT_ROWS}); returning no features.")
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=df.index[:0], dtype=float)

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # 1. Momentum & Returns
    returns = close.pct_change()
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

    # 2. Trend
    # MACD as a percent of price, so a 50-point MACD at S&P 1,000 and a
    # 300-point MACD at S&P 6,000 read as the same trend strength.
    macd_pct = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9).macd() / close

    adx = ta.trend.ADXIndicator(high, low, close, window=ADX_WINDOW).adx()

    # 3. Volatility
    volatility_20 = returns.rolling(20).std()
    # ATR as a percent of price (same reasoning as MACD).
    atr_pct = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close

    # 4. Volume/Macro
    # Volume flow: share of the last 20 days' volume traded on up-days minus
    # the share traded on down-days. It equals the 20-day change in OBV
    # divided by 20-day total volume, so it keeps OBV's idea (is volume
    # backing the up-moves or the down-moves?) but is bounded in [-1, 1]:
    # +1 = all volume on up-days, -1 = all on down-days.
    signed_volume = np.sign(close.diff()) * volume
    total_volume = volume.rolling(FLOW_WINDOW).sum()
    obv_flow_20 = signed_volume.rolling(FLOW_WINDOW).sum() / total_volume.where(total_volume > 0)

    features = pd.DataFrame({
        "returns": returns,
        "rsi": rsi,
        "macd_pct": macd_pct,
        "adx": adx,
        "volatility_20": volatility_20,
        "atr_pct": atr_pct,
        "obv_flow_20": obv_flow_20,
        "vix": vix_close
    })

    # Ensure column order
    features = features[FEATURE_COLUMNS]

    rows_before = len(features)
    features = features.dropna()
    rows_dropped = rows_before - len(features)

    if rows_dropped > 0:
        start_date_str = str(features.index[0].date()) if len(features) > 0 else "N/A"
        logger.info(
            f"Warmup trimming: dropped {rows_dropped} of {rows_before} "
            f"rows ({rows_dropped / rows_before * 100:.1f}%). "
            f"Features start at {start_date_str}."
        )

    nan_counts = features.isna().sum()
    if nan_counts.any():
        bad_cols = nan_counts[nan_counts > 0]
        raise ValueError(f"NaNs remain after warmup trimming:\n{bad_cols}")

    return features

def assert_no_lookahead(build_fn, df, vix_close, n_checks=20, seed=42):
    full_features = build_fn(df, vix_close)
    if full_features.empty:
        raise ValueError("build_fn returned an empty DataFrame on full data.")

    rng = np.random.default_rng(seed)
    min_pos = 100
    max_pos = len(df) - 1
    if max_pos <= min_pos:
        raise ValueError(f"DataFrame too short for look-ahead test (need > {min_pos} rows, got {len(df)}).")

    cut_positions = sorted(rng.integers(min_pos, max_pos, size=n_checks))
    passed = 0

    for t in cut_positions:
        trunc_df = df.iloc[: t + 1]
        trunc_vix = vix_close.iloc[: t + 1]
        partial_features = build_fn(trunc_df, trunc_vix)

        if partial_features.empty:
            continue

        last_date = partial_features.index[-1]
        if last_date not in full_features.index:
            continue

        full_row = full_features.loc[last_date]
        partial_row = partial_features.loc[last_date]

        for col in FEATURE_COLUMNS:
            full_val = full_row[col]
            partial_val = partial_row[col]
            if not np.isclose(full_val, partial_val, rtol=1e-10, atol=1e-14, equal_nan=True):
                raise AssertionError(f"LOOK-AHEAD DETECTED in '{col}' at t={t}")
        passed += 1

    return passed

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.loader import load_config, download_prices

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    data = download_prices(config=cfg)
    spx = data["spx"]
    features = build_features(spx, spx["VIX_Close"])
    assert len(features.columns) == 8, f"Expected 8 cols, got {len(features.columns)}"
    print("ALL CHECKS PASSED")
