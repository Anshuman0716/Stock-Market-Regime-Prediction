"""
Data loader module: yfinance download + local parquet caching.

Design decisions documented for learning:

1. PARQUET CACHING — yfinance rate-limits aggressively and network calls are
   slow (~2-5 s per ticker). Caching raw downloads as parquet files under
   data/raw/ means repeated runs (during development, in notebooks, in CI)
   hit the filesystem instead of the network. Parquet preserves dtypes and
   DatetimeIndex exactly, unlike CSV which would require re-parsing.

2. VIX AS A FEATURE, NOT AN ASSET — VIX (^VIX) measures implied volatility
   of S&P 500 options. It's the market's own fear gauge. We load it once and
   join it onto every tradeable asset as a feature column, but we never
   "trade" VIX itself (you can't buy the VIX index directly).

3. CALENDAR ALIGNMENT — BTC-USD trades 7 days/week while US equities
   (^GSPC, ^IXIC) and Gold futures (GC=F) follow the US market calendar.
   VIX also follows the US calendar. When joining VIX onto an asset that
   trades on weekends (BTC), we forward-fill VIX: on Saturday, the most
   recent known VIX value is Friday's close. This is the standard approach
   for joining lower-frequency onto higher-frequency calendars. We NEVER
   forward-fill an asset's own OHLCV data — if it didn't trade, that day
   simply doesn't exist.

4. FAIL LOUDLY — A silent partial download (e.g., only 5 rows for a ticker
   that should have thousands) would propagate bad data through the entire
   pipeline and produce wrong features, wrong labels, and wrong backtest
   results. We raise immediately on empty or suspiciously short downloads.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config/config.yaml"):
    """
    Load project configuration from a YAML file.

    Why YAML instead of hardcoded constants: keeping tickers, dates, model
    hyperparameters, and backtest settings in a single config file means
    (a) notebooks, src/ modules, and app.py all read the same values, and
    (b) re-running the pipeline with different parameters is a one-line
    edit, not a multi-file search-and-replace.

    Parameters
    ----------
    path : str
        Path to the YAML config file. Defaults to "config/config.yaml".

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path.resolve()}'. "
            f"Make sure you're running from the project root."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker, start, end, cache_dir):
    """
    Build a deterministic cache filename for a ticker + date range.

    The ticker is sanitized for the filesystem (^ and / are replaced)
    so that tickers like ^GSPC and GC=F produce valid filenames.
    """
    safe_ticker = (
        ticker
        .replace("^", "_caret_")
        .replace("=", "_eq_")
        .replace("/", "_slash_")
    )
    filename = f"{safe_ticker}_{start}_{end}.parquet"
    return Path(cache_dir) / filename


# ---------------------------------------------------------------------------
# Single-ticker download
# ---------------------------------------------------------------------------

def _download_single_ticker(ticker, start, end, force_refresh=False,
                            cache_dir="data/raw"):
    """
    Download OHLCV data for one ticker, with local parquet caching.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker (e.g., "^GSPC", "BTC-USD").
    start : str
        Start date "YYYY-MM-DD".
    end : str or None
        End date "YYYY-MM-DD", or None for the most recent available day.
    force_refresh : bool
        If True, ignore cached data and re-download from yfinance.
    cache_dir : str
        Directory under which to store/read parquet cache files.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with a tz-naive DatetimeIndex named "Date".

    Raises
    ------
    RuntimeError
        If the download is empty or returns fewer than MIN_ROWS rows.
    """
    # Resolve end=None to today's date for cache keying.
    # Why: end=None means "most recent available," which changes daily.
    # Keying the cache to today's date means same-day re-runs hit cache,
    # but tomorrow's run will re-download (which is what we want for a
    # pipeline that should see the latest data).
    resolved_end = end if end is not None else str(date.today())

    cache_file = _cache_path(ticker, start, resolved_end, cache_dir)

    # ── Serve from cache if available ────────────────────────────────
    if cache_file.exists() and not force_refresh:
        df = pd.read_parquet(cache_file)
        if not df.empty:
            return df

    # ── Download from yfinance ───────────────────────────────────────
    raw = yf.download(
        ticker,
        start=start,
        end=resolved_end,
        progress=False,
    )

    # ── Validate download quality ────────────────────────────────────
    # Raise loudly rather than returning a short frame silently.
    if raw is None or raw.empty:
        raise RuntimeError(
            f"Download for '{ticker}' returned no data. "
            f"Check the ticker symbol and date range "
            f"({start} to {resolved_end})."
        )

    # yfinance sometimes returns MultiIndex columns like
    # ('Close', '^GSPC') when downloading a single ticker.
    # Flatten to simple column names so downstream code doesn't break.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Minimum row threshold: ~1.5 months of trading days. Anything
    # shorter almost certainly means a wrong ticker, too-narrow date
    # range, or a data outage.
    MIN_ROWS = 30
    if len(raw) < MIN_ROWS:
        raise RuntimeError(
            f"Download for '{ticker}' returned only {len(raw)} rows "
            f"(minimum {MIN_ROWS} expected). "
            f"Date range: {start} to {resolved_end}."
        )

    # Ensure a clean, tz-naive DatetimeIndex.
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "Date"

    # ── Cache to parquet ─────────────────────────────────────────────
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_file)

    return raw


# ---------------------------------------------------------------------------
# Multi-asset download with VIX join
# ---------------------------------------------------------------------------

def download_prices(config=None, config_path="config/config.yaml",
                    force_refresh=False, cache_dir="data/raw"):
    """
    Download price data for all configured tickers, with VIX joined as a
    market-wide feature column onto each tradeable asset.

    Design decisions (documented for learning):

    1. Config-driven tickers and dates — no ticker symbol or date string
       is hardcoded in this function. Everything comes from config.yaml so
       that a single file controls the entire pipeline.

    2. VIX handling — VIX is downloaded once and its Close column is
       joined onto every tradeable asset via a left join + forward-fill.

    3. Calendar alignment rule —
       - Each asset keeps its OWN trading calendar (its DatetimeIndex).
       - VIX is joined with ``how="left"`` (asset calendar is authority).
       - VIX_Close is then forward-filled: on days the asset trades but
         VIX doesn't (e.g., BTC on a Saturday), the most recently known
         VIX close is used. This is standard for joining lower-frequency
         market data onto higher-frequency calendars.
       - Asset OHLCV is NEVER forward-filled across gaps. If Gold didn't
         trade on Saturday, Gold has no row for Saturday. Manufacturing
         fake OHLCV data would distort returns, volatility, and every
         indicator computed downstream.

    Parameters
    ----------
    config : dict or None
        Pre-loaded config dict. If None, loads from config_path.
    config_path : str
        Path to config YAML (used only if config is None).
    force_refresh : bool
        If True, bypass cache and re-download all tickers from yfinance.
    cache_dir : str
        Directory for parquet cache files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps asset keys (e.g., "spx", "bitcoin") to DataFrames. Each
        DataFrame has the asset's OHLCV columns plus a ``VIX_Close``
        column from the VIX series, forward-filled onto the asset's
        own trading calendar.
    """
    if config is None:
        config = load_config(config_path)

    tickers = config["tickers"]
    date_range = config["date_range"]
    start = date_range["start"]
    end = date_range.get("end")  # None → most recent available

    # ── Step 1: Download VIX (a feature, not a tradeable asset) ──────
    vix_ticker = tickers["vix"]
    vix_data = _download_single_ticker(
        vix_ticker, start, end,
        force_refresh=force_refresh, cache_dir=cache_dir,
    )
    # Keep only VIX Close, renamed to avoid collision with asset columns.
    vix_close = vix_data[["Close"]].rename(columns={"Close": "VIX_Close"})

    # ── Step 2: Download each tradeable asset and join VIX ───────────
    results = {}
    for key, ticker in tickers.items():
        if key == "vix":
            # VIX is a feature, not an asset — skip it in the output dict
            continue

        asset_data = _download_single_ticker(
            ticker, start, end,
            force_refresh=force_refresh, cache_dir=cache_dir,
        )

        # Join VIX onto this asset's trading calendar.
        # LEFT join: asset's calendar is the authority.
        # Forward-fill VIX: on weekend BTC rows, carry Friday's VIX.
        # NEVER forward-fill asset OHLCV.
        asset_with_vix = asset_data.join(vix_close, how="left")
        asset_with_vix["VIX_Close"] = asset_with_vix["VIX_Close"].ffill()

        # Drop leading rows where VIX is still NaN (asset started trading
        # before VIX data begins).
        asset_with_vix = asset_with_vix.dropna(subset=["VIX_Close"])

        results[key] = asset_with_vix

    return results


# ---------------------------------------------------------------------------
# Quick CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  Data Loader — Smoke Test")
    print("=" * 60)

    cfg = load_config()
    print(f"\nConfig loaded. Tickers: {list(cfg['tickers'].keys())}")
    print(f"Date range: {cfg['date_range']['start']} → "
          f"{cfg['date_range']['end'] or 'latest'}")

    # Cold cache
    print("\n── Cold-cache download (network) ──")
    t0 = time.perf_counter()
    data = download_prices(config=cfg, force_refresh=True)
    t1 = time.perf_counter()
    print(f"   Elapsed: {t1 - t0:.2f} s")
    for key, df in data.items():
        print(f"   {key:>10s}: {df.shape[0]:>6d} rows, "
              f"{df.shape[1]} cols, "
              f"{df.index.min().date()} → {df.index.max().date()}")

    # Warm cache
    print("\n── Warm-cache download (parquet) ──")
    t0 = time.perf_counter()
    data2 = download_prices(config=cfg)
    t1 = time.perf_counter()
    print(f"   Elapsed: {t1 - t0:.2f} s")
    for key, df in data2.items():
        print(f"   {key:>10s}: {df.shape[0]:>6d} rows, "
              f"{df.shape[1]} cols, "
              f"{df.index.min().date()} → {df.index.max().date()}")

    print("\n✅ Loader smoke test passed.")
