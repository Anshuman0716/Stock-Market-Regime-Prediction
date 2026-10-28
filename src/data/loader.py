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

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

# Network downloads are retried this many times, waiting 2 s, 4 s, ... between
# attempts. Yahoo occasionally drops a request or rate-limits a burst; one
# transient failure should not kill a multi-minute pipeline run.
DOWNLOAD_ATTEMPTS = 3
RETRY_BASE_SECONDS = 2.0

# Minimum rows for a download to be believable (~1.5 months of trading days).
# Anything shorter almost certainly means a wrong ticker, a too-narrow date
# range or a data outage.
MIN_ROWS = 30

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


def get_ann_factor(config, asset_name):
    """
    Number of rows per year for an asset, used to annualize statistics.

    Why per-asset: annualizing multiplies a daily mean by rows-per-year and a
    daily std by sqrt(rows-per-year). Bitcoin has ~365 rows a year, equities
    ~252. A single hardcoded 252 silently mis-states every crypto metric
    (CAGR, volatility, Sharpe) and even shifts which HMM state gets labelled
    Bull, because the labelling thresholds are on annualized Sharpe.

    Falls back to 252 when the config has no annualization block (e.g. the
    small mock configs used in tests).
    """
    table = (config or {}).get("backtest", {}).get("annualization", {})
    return int(table.get(asset_name, table.get("default", 252)))


def get_cost_bps(config, asset_name):
    """
    Trading cost in basis points per unit of turnover for an asset:
    its spread (from config backtest.costs.spread_bps, falling back to
    'default') plus the flat commission.

    Why spread + commission rather than one flat number: liquidity differs
    a lot between assets. Crossing the spread on S&P 500 products costs
    about 1 bp, on Bitcoin several. One flat figure would flatter the
    expensive assets and punish the cheap ones.
    """
    costs = (config or {}).get("backtest", {}).get("costs")
    if costs is None:
        raise KeyError("config is missing backtest.costs; refusing to guess trading costs.")
    spreads = costs.get("spread_bps", {})
    spread = spreads.get(asset_name, spreads.get("default"))
    if spread is None:
        raise KeyError(f"No spread configured for '{asset_name}' and no default.")
    return float(spread) + float(costs.get("commission_bps", 0.0))


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
# Download quality
# ---------------------------------------------------------------------------

def clean_price_frame(raw, ticker, is_rate=False):
    """
    Normalise and validate one ticker's download.

    - Flatten yfinance's ('Close', '^GSPC') MultiIndex columns.
    - tz-naive, sorted DatetimeIndex named "Date"; duplicate dates (Yahoo
      sometimes repeats the latest bar) keep the last copy.
    - Rows without a Close are dropped: a missing close is a missing day,
      never something to fill in.
    - Fail loudly on non-positive prices or a suspiciously short history.
      `is_rate=True` (an interest rate such as ^IRX) skips the positivity
      check: T-bill yields really were 0 or slightly negative in 2020.

    Why here: every downstream step (returns, indicators, the HMM) assumes
    one clean row per trading day. Fixing that once at the source is safer
    than every consumer guessing.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise RuntimeError(f"Download for '{ticker}' has no Close column (got {list(df.columns)}).")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df.sort_index()

    n_dupes = int(df.index.duplicated(keep="last").sum())
    if n_dupes:
        logger.warning(f"{ticker}: dropped {n_dupes} duplicate date(s).")
        df = df[~df.index.duplicated(keep="last")]

    n_missing = int(df["Close"].isna().sum())
    if n_missing:
        logger.warning(f"{ticker}: dropped {n_missing} row(s) without a Close.")
        df = df.dropna(subset=["Close"])

    bad = df["Close"] <= 0
    if bad.any() and not is_rate:
        raise RuntimeError(f"Download for '{ticker}' has {int(bad.sum())} non-positive Close value(s), "
                           f"first on {df.index[bad][0].date()}.")
    if len(df) < MIN_ROWS:
        raise RuntimeError(f"Download for '{ticker}' has only {len(df)} usable rows "
                           f"(minimum {MIN_ROWS} expected).")
    return df


def _fetch_with_retry(ticker, start, end):
    """yf.download with a few retries; raises after the last failed attempt."""
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False,
                              auto_adjust=True)  # adjusted prices: dividends and splits included
            if raw is not None and not raw.empty:
                return raw
            last_error = RuntimeError("empty response")
        except Exception as e:  # network errors come in many types; retry them all
            last_error = e
        if attempt < DOWNLOAD_ATTEMPTS:
            wait = RETRY_BASE_SECONDS * 2 ** (attempt - 1)
            logger.warning(f"{ticker}: download attempt {attempt} failed ({last_error}); retrying in {wait:.0f}s.")
            time.sleep(wait)
    raise RuntimeError(f"Download for '{ticker}' failed after {DOWNLOAD_ATTEMPTS} attempts "
                       f"({start} to {end}): {last_error}")


# ---------------------------------------------------------------------------
# Single-ticker download
# ---------------------------------------------------------------------------

def _download_single_ticker(ticker, start, end, force_refresh=False,
                            cache_dir="data/raw", is_rate=False):
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
        If every download attempt fails or the data fails validation
        (see clean_price_frame).
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

    # ── Download from yfinance, then validate ────────────────────────
    raw = clean_price_frame(_fetch_with_retry(ticker, start, resolved_end), ticker, is_rate=is_rate)

    # ── Cache to parquet ─────────────────────────────────────────────
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_file)

    return raw


# ---------------------------------------------------------------------------
# Multi-asset download with VIX join
# ---------------------------------------------------------------------------

def as_of(series, index):
    """
    The latest value of `series` at or before each date in `index`.

    Used to put another instrument's data (VIX, the traded fund, the T-bill
    yield) onto an asset's own trading calendar. Only past values are
    carried forward, never later ones, so a date never sees the future.
    """
    combined = series.reindex(series.index.union(index)).ffill()
    return combined.reindex(index)


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
        own trading calendar. When the config lists them, it also has
        ``Trade_Close`` (the traded instrument's adjusted close) and
        ``Cash_Yield`` (T-bill yield, % a year), aligned the same way.
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

    # ── Step 1b: Traded instruments and the cash rate (optional) ─────
    backtest_cfg = config.get("backtest", {}) or {}
    instruments = backtest_cfg.get("instruments") or {}
    cash_ticker = backtest_cfg.get("cash_ticker")
    cash_yield = None
    if cash_ticker:
        cash_yield = _download_single_ticker(
            cash_ticker, start, end, force_refresh=force_refresh, cache_dir=cache_dir, is_rate=True)["Close"]

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

        # Trade_Close: price of what is actually traded (e.g. SPY for the
        # S&P 500), on the asset's own calendar. NaN before the fund existed;
        # the evaluation checks it covers the whole out-of-sample window.
        instrument = instruments.get(key)
        if instrument:
            trade_close = (asset_data["Close"] if instrument == ticker else _download_single_ticker(
                instrument, start, end, force_refresh=force_refresh, cache_dir=cache_dir)["Close"])
            asset_with_vix["Trade_Close"] = as_of(trade_close, asset_with_vix.index)
        # Cash_Yield: annual T-bill yield in percent, as known on each date.
        if cash_yield is not None:
            asset_with_vix["Cash_Yield"] = as_of(cash_yield, asset_with_vix.index)

        results[key] = asset_with_vix

    return results


