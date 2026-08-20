import pytest
import numpy as np
import pandas as pd
from src.features.engineering import build_features, FEATURE_COLUMNS, assert_no_lookahead

def test_hand_computed_indicators():
    # 1. each indicator against a hand-computed expected value on a small fixed input frame
    dates = pd.date_range('2020-01-01', periods=50)
    
    # We will engineer prices to produce a known RSI and simple returns
    close = np.arange(100, 150, dtype=float)
    high = close + 1
    low = close - 1
    open_p = close
    volume = np.ones(50) * 1000
    
    df = pd.DataFrame({'Close': close, 'High': high, 'Low': low, 'Open': open_p, 'Volume': volume}, index=dates)
    vix = pd.Series(np.ones(50) * 20, index=dates)
    
    features = build_features(df, vix)
    
    # Check returns manually for the last day
    # Close today = 149, yesterday = 148 -> return = 1/148
    assert np.isclose(features['returns'].iloc[-1], 1 / 148.0)
    
    # Since price strictly went up every single day, RSI must be exactly 100.0
    assert np.isclose(features['rsi'].iloc[-1], 100.0)
    
    # ATR with High = Close+1, Low = Close-1 => True Range is always exactly 2.0
    assert np.isclose(features['atr'].iloc[-1], 2.0)
    
    # VIX should map exactly
    assert features['vix'].iloc[-1] == 20.0

def test_no_lookahead(sample_spx, sample_vix):
    # 2. the no-look-ahead property test: df[:t] match full df at row t
    passed = assert_no_lookahead(build_features, sample_spx, sample_vix['Close'], n_checks=10, seed=42)
    assert passed == 10

def test_nan_and_short_frame():
    # 3. NaN/warmup handling, and behaviour on a short input frame
    dates = pd.date_range('2020-01-01', periods=5)
    df = pd.DataFrame({
        'Close': np.random.rand(5),
        'High': np.random.rand(5),
        'Low': np.random.rand(5),
        'Open': np.random.rand(5),
        'Volume': np.random.rand(5)
    }, index=dates)
    vix = pd.Series(np.random.rand(5), index=dates)
    
    features = build_features(df, vix)
    assert len(features) == 0
