"""
Tests for the data loader: download validation, retries and calendar alignment.
No network: yfinance is replaced by fakes.
"""
import numpy as np
import pandas as pd
import pytest

import src.data.loader as loader


def _ohlcv(dates, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": 1000.0}, index=pd.DatetimeIndex(dates))


def test_clean_price_frame_sorts_dedupes_and_drops_missing_closes():
    dates = pd.bdate_range("2020-01-01", periods=40)
    raw = _ohlcv(dates, np.arange(100, 140))
    raw.iloc[5, raw.columns.get_loc("Close")] = np.nan                   # a missing close
    raw = pd.concat([raw, raw.iloc[[-1]]]).iloc[::-1]                      # duplicate last bar, reversed order
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["^TEST"]])     # yfinance-style columns

    clean = loader.clean_price_frame(raw, "^TEST")
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert clean["Close"].notna().all()
    assert len(clean) == 39 and list(clean.columns)[:4] == ["Open", "High", "Low", "Close"]


def test_clean_price_frame_rejects_bad_data():
    dates = pd.bdate_range("2020-01-01", periods=40)
    bad = _ohlcv(dates, np.arange(100, 140))
    bad.iloc[10, bad.columns.get_loc("Close")] = 0.0
    with pytest.raises(RuntimeError, match="non-positive"):
        loader.clean_price_frame(bad, "X")
    # An interest rate may be zero or negative (T-bill yields in 2020); a price may not.
    assert len(loader.clean_price_frame(bad, "^IRX", is_rate=True)) == 40
    with pytest.raises(RuntimeError, match="only 5 usable rows"):
        loader.clean_price_frame(_ohlcv(dates[:5], [1, 2, 3, 4, 5]), "X")


def test_download_retries_then_succeeds(monkeypatch, tmp_path):
    calls = {"n": 0}
    good = _ohlcv(pd.bdate_range("2020-01-01", periods=40), np.arange(100, 140))

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temporary outage")
        return good

    monkeypatch.setattr(loader.yf, "download", flaky)
    monkeypatch.setattr(loader.time, "sleep", lambda s: None)
    df = loader._download_single_ticker("^TEST", "2020-01-01", "2020-03-01", cache_dir=str(tmp_path))
    assert calls["n"] == 3 and len(df) == 40


def test_download_gives_up_with_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.yf, "download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(loader.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        loader._download_single_ticker("^NOPE", "2020-01-01", "2020-03-01", cache_dir=str(tmp_path))


def test_as_of_only_carries_past_values_forward():
    fund = pd.Series([10.0, 11.0, 12.0], index=pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-08"]))
    calendar = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-07", "2020-01-08"])
    out = loader.as_of(fund, calendar)
    assert np.isnan(out.iloc[0])                    # before the fund's first price: unknown, not back-filled
    assert list(out.iloc[1:]) == [10.0, 10.0, 11.0, 12.0]


def test_download_prices_adds_instrument_and_cash_columns(monkeypatch, tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=60)
    frames = {
        "^GSPC": _ohlcv(dates, np.linspace(3000, 3300, 60)),
        "^VIX": _ohlcv(dates, np.full(60, 20.0)),
        "SPY": _ohlcv(dates[3:], np.linspace(300, 330, 57)),   # fund starts later
        "^IRX": _ohlcv(dates, np.full(60, 5.0)),
    }
    monkeypatch.setattr(loader.yf, "download", lambda ticker, **k: frames[ticker])
    cfg = {"tickers": {"spx": "^GSPC", "vix": "^VIX"}, "date_range": {"start": "2020-01-01", "end": "2020-04-01"},
           "backtest": {"instruments": {"spx": "SPY"}, "cash_ticker": "^IRX"}}
    spx = loader.download_prices(cfg, cache_dir=str(tmp_path))["spx"]
    assert {"Close", "VIX_Close", "Trade_Close", "Cash_Yield"} <= set(spx.columns)
    assert spx["Trade_Close"].iloc[:3].isna().all() and spx["Trade_Close"].iloc[3:].notna().all()
    assert (spx["Cash_Yield"] == 5.0).all()
