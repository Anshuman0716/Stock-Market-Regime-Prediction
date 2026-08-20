import pytest
import numpy as np
import pandas as pd
from src.backtest.engine import run_backtest
from src.backtest.metrics import get_all_metrics

def test_hand_computed_metrics():
    # 1. metrics against hand-computed values on a tiny synthetic return series
    # A series with known Sharpe and a known 50% drawdown
    # To get 50% drawdown: 1.0 -> 0.5. Let's make returns: 0.0, -0.5, +1.0
    # Portfolio: 1.0 -> 1.0 -> 0.5 -> 1.0. Max DD is exactly 0.5 (50%).
    returns = pd.Series([0.0, -0.5, 1.0])
    weights = pd.Series([1.0, 1.0, 1.0])
    
    metrics = get_all_metrics(returns, weights, ann_factor=252, asset_name='test')
    assert np.isclose(metrics['Max DD'], -0.5)

def test_signal_lag():
    # 2. signal lag: construct a synthetic series where a lookahead bug would
    # produce an impossibly high Sharpe, and assert it doesn't.
    # If the market alternates +10%, -10%, +10%, -10%
    # And our target_weight perfectly predicts it on the SAME day:
    dates = pd.date_range('2020-01-01', periods=10)
    returns = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], index=dates)
    
    # Perfect oracle targets
    target_weights = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=dates)
    
    df = pd.DataFrame({'returns': returns})
    
    res = run_backtest(df, target_weights, asset_name="spx", use_dynamic_costs=False, cost_bps=0.0)
    
    # Due to lag, the target_weight for day t is executed at t+1 and earns return at t+2.
    # So the weights earned will NOT match the perfect oracle, meaning the strategy shouldn't just win perfectly.
    # A lookahead bug would give us only the +0.1 returns.
    # Because of lag, we will actually hold the asset on the -0.1 days instead!
    # target at t=0 (weight=1) -> executed entering t=2. t=2 return is 0.1, so we win.
    # target at t=1 (weight=0) -> executed entering t=3. t=3 return is -0.1, we hold 0 weight.
    # Wait, t=0 target is 1. executed_weight at t=2 is 1. 
    # With this exact pattern, lag 2 actually perfectly aligns with the 2-day cycle!
    # Let's use a non-repeating pattern to prove it.
    
    returns2 = pd.Series([0.1, 0.1, -0.5, 0.1, 0.1], index=dates[:5])
    target_weights2 = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates[:5])
    # The oracle says 0 everywhere because of the -0.5. If we set target to 1 right BEFORE the drop:
    # Say we know day 2 is -0.5. 
    target_weights_buggy = pd.Series([1.0, 0.0, 1.0, 1.0, 1.0], index=dates[:5])
    # If there was no lag, we would have 0 weight on day 2 and miss the -0.5.
    df2 = pd.DataFrame({'returns': returns2})
    res2 = run_backtest(df2, target_weights_buggy, use_dynamic_costs=False)
    
    # Because of lag, the weight held at day 2 is the target from day 0, which is 1.0.
    # So we DO suffer the -0.5 drop!
    assert res2['net_return'].iloc[2] == -0.5

def test_cost_model_arithmetic():
    # 3. cost model arithmetic: turnover x cost equals the expected drag
    dates = pd.date_range('2020-01-01', periods=4)
    returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates)
    # Target weights cause turnover
    target_weights = pd.Series([1.0, 0.0, 1.0, 0.0], index=dates)
    df = pd.DataFrame({'returns': returns})
    
    # Cost = 100 bps = 0.01 per unit of turnover
    res = run_backtest(df, target_weights, use_dynamic_costs=False, cost_bps=100.0)
    
    # turnover is diff of executed_weights (shifted 2).
    # target: [1.0, 0.0, 1.0, 0.0]
    # executed: [0.0, 0.0, 1.0, 0.0]
    # turnover: [0.0, 0.0, 1.0, 1.0]
    # expected cost: [0, 0, 0.01, 0.01]
    
    np.testing.assert_allclose(res['cost'].values, [0.0, 0.0, 0.01, 0.01])
    
    # 4. zero-cost vs full-cost runs differ
    res_zero = run_backtest(df, target_weights, use_dynamic_costs=False, cost_bps=0.0)
    assert res_zero['cost'].sum() == 0.0
    assert res['cost'].sum() > 0.0
    assert res_zero['net_return'].sum() > res['net_return'].sum()
