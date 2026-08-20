import sys
sys.path.insert(0, ".")

import pandas as pd
import logging
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels
from src.backtest.engine import compute_target_weight, run_backtest
from src.backtest.baselines import get_buy_and_hold, get_200dma_filter
from src.backtest.metrics import get_all_metrics

logging.basicConfig(level=logging.WARNING)

def run_all_assets():
    cfg = load_config('config/config.yaml')
    data = download_prices(cfg)
    
    # We will compute results for each asset and build a list of dicts for the table
    table_rows = []
    
    assets = ['spx', 'nasdaq', 'gold', 'bitcoin']
    
    print(f"{'='*100}")
    print(f"{'Asset':<10} | {'Strategy':<15} | {'CAGR (%)':<10} | {'Vol (%)':<10} | {'Sharpe':<8} | {'Max DD (%)':<12} | {'Time in Mkt':<12}")
    print(f"{'-'*100}")
    
    for asset_name in assets:
        df = data[asset_name].copy()
        
        # Determine annualization factor
        # Bitcoin trades 365 days a year, others 252.
        ann_factor = 365 if asset_name == 'bitcoin' else 252
        
        # 1. Feature Engineering
        # Make sure df contains VIX_Close
        if 'VIX_Close' not in df.columns:
            # Re-align VIX if not there, but loader should have added it.
            pass
            
        features = build_features(df, df['VIX_Close'])
        features['returns'] = df['Close'].pct_change()
        features = features.dropna()
        
        # 2. HMM Pipeline
        raw_labels, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name=asset_name)
        smoothed_labels = smooth_regime_labels(raw_labels, window_size=cfg['model']['smoothing_window'])
        
        # Align features with the valid dates where labels exist (out-of-sample)
        # walk_forward_predict produces labels starting from the first out-of-sample year
        oos_mask = smoothed_labels.notna()
        oos_features = features[oos_mask].copy()
        oos_labels = smoothed_labels[oos_mask]
        
        # Align original df with returns
        full_df = df.copy()
        full_df['returns'] = full_df['Close'].pct_change()
        
        # 3. Strategy Engines
        # HMM Strategy
        hmm_targets = compute_target_weight(oos_labels)
        hmm_bt = run_backtest(oos_features, hmm_targets, cost_bps=0.0) # WARNING prints automatically
        
        # Baselines (computed on full df to preserve 200DMA history, then sliced)
        bh_bt = get_buy_and_hold(full_df).loc[oos_features.index]
        dma_bt = get_200dma_filter(full_df, cost_bps=0.0).loc[oos_features.index]
        
        # 4. Metrics Calculation
        metrics_hmm = get_all_metrics(hmm_bt['net_return'], hmm_bt['executed_weight'], ann_factor, asset_name)
        metrics_bh = get_all_metrics(bh_bt['net_return'], bh_bt['executed_weight'], ann_factor, asset_name)
        metrics_dma = get_all_metrics(dma_bt['net_return'], dma_bt['executed_weight'], ann_factor, asset_name)
        
        # Append to table
        for m, s_name in zip([metrics_hmm, metrics_bh, metrics_dma], ['HMM Regime', 'Buy & Hold', '200DMA Filter']):
            m['Strategy'] = s_name
            m['Ann_Factor'] = ann_factor
            table_rows.append(m)
            
            print(f"{m['Asset'].upper():<10} | {s_name:<15} | {m['CAGR']*100:>8.2f}% | {m['Ann. Vol']*100:>8.2f}% | {m['Sharpe']:>8.2f} | {m['Max DD']*100:>10.2f}% | {m['Time in Market']*100:>10.2f}%")
            
        print(f"{'-'*100}")

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv('scratch/backtest_results.csv', index=False)
    print("Results saved to scratch/backtest_results.csv")

if __name__ == "__main__":
    run_all_assets()
