"""
Feature engineering: 8 core technical indicators to prevent overparameterization.
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "returns",
    "rsi",
    "macd",
    "adx",
    "volatility_20",
    "atr",
    "obv_ratio",
    "vix"
]

def build_features(df, vix_close):
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # 1. Momentum & Returns
    returns = close.pct_change()
    rsi = ta.rsi(close, length=14)

    # 2. Trend
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and len(macd_df.columns) >= 1:
        macd = macd_df.iloc[:, 0]
    else:
        macd = pd.Series(np.nan, index=close.index)

    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and len(adx_df.columns) >= 1:
        adx = adx_df.iloc[:, 0]
    else:
        adx = pd.Series(np.nan, index=close.index)

    # 3. Volatility
    volatility_20 = returns.rolling(20).std()
    atr = ta.atr(high, low, close, length=14)

    # 4. Volume/Macro
    obv = ta.obv(close, volume)
    obv_ema = ta.ema(obv, length=20)
    if obv_ema is not None and obv is not None:
        obv_ratio = obv / obv_ema
    else:
        obv_ratio = pd.Series(np.nan, index=close.index)

    features = pd.DataFrame({
        "returns": returns,
        "rsi": rsi,
        "macd": macd,
        "adx": adx,
        "volatility_20": volatility_20,
        "atr": atr,
        "obv_ratio": obv_ratio,
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
    import time
    sys.path.insert(0, ".")
    from src.data.loader import load_config, download_prices

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    data = download_prices(config=cfg)
    spx = data["spx"]
    features = build_features(spx, spx["VIX_Close"])
    assert len(features.columns) == 8, f"Expected 8 cols, got {len(features.columns)}"
    print("ALL CHECKS PASSED")
