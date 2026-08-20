import pandas as pd
import numpy as np
from src.backtest.engine import run_backtest

def get_buy_and_hold(df, asset_name="spx"):
    """
    Buy and hold baseline.
    Always 100% exposed.
    """
    target_weights = pd.Series(1.0, index=df.index)
    # B&H has no lag or turnover costs after initial entry
    return run_backtest(df, target_weights, asset_name=asset_name, use_dynamic_costs=True)

def get_200dma_filter(df, asset_name="spx"):
    """
    200-Day Moving Average Filter baseline.
    Long (100%) when Close > 200DMA, flat (0%) otherwise.
    
    Must undergo the exact same strict lag and cost mechanics as the HMM.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' to compute 200DMA.")
        
    dma_200 = df['Close'].rolling(window=200).mean()
    
    # Signal computed at day t close
    target_weights = np.where(df['Close'] > dma_200, 1.0, 0.0)
    
    # We must preserve NaNs for the 200-day warmup
    target_weights = pd.Series(target_weights, index=df.index)
    target_weights[dma_200.isna()] = 0.0
    
    return run_backtest(df, target_weights, asset_name=asset_name, use_dynamic_costs=True)
