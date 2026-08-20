import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import adjusted_rand_score
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels
from src.backtest.engine import compute_target_weight, run_backtest
from src.backtest.baselines import get_buy_and_hold, get_200dma_filter
from src.stats.significance import block_bootstrap_sharpe_diff, ledoit_wolf_sharpe_diff, adjust_pvalues

logging.basicConfig(level=logging.WARNING)

# Current 8-feature categories
CATEGORIES = {
    "Momentum": ["returns", "rsi"],
    "Trend": ["macd", "adx"],
    "Volatility": ["volatility_20", "atr"],
    "Volume": ["obv_ratio", "vix"]
}

def analyze_significance(cfg, data):
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (HMM vs Baselines)")
    print("="*80)
    
    assets = ['spx', 'nasdaq', 'gold', 'bitcoin']
    tests = []
    
    for asset_name in assets:
        df = data[asset_name].copy()
        ann_factor = 365 if asset_name == 'bitcoin' else 252
        
        features = build_features(df, df['VIX_Close'])
        features['returns'] = df['Close'].pct_change()
        features = features.dropna()
        
        raw_labels, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name=asset_name)
        smoothed_labels = smooth_regime_labels(raw_labels, window_size=cfg['model']['smoothing_window'])
        
        oos_mask = smoothed_labels.notna()
        oos_features = features[oos_mask].copy()
        oos_labels = smoothed_labels[oos_mask]
        
        full_df = df.copy()
        full_df['returns'] = full_df['Close'].pct_change()
        
        hmm_targets = compute_target_weight(oos_labels)
        hmm_bt = run_backtest(oos_features, hmm_targets, cost_bps=0.0)
        
        bh_bt = get_buy_and_hold(full_df).loc[oos_features.index]
        dma_bt = get_200dma_filter(full_df, cost_bps=0.0).loc[oos_features.index]
        
        # Test HMM vs B&H
        boot_bh = block_bootstrap_sharpe_diff(hmm_bt['net_return'], bh_bt['net_return'], ann_factor)
        lw_bh = ledoit_wolf_sharpe_diff(hmm_bt['net_return'], bh_bt['net_return'], ann_factor)
        
        tests.append({
            "Asset": asset_name.upper(),
            "Baseline": "Buy & Hold",
            "Diff": boot_bh['diff'],
            "CI": (boot_bh['ci_lower'], boot_bh['ci_upper']),
            "p_boot": boot_bh['p_value'],
            "p_lw": lw_bh['p_value']
        })
        
        # Test HMM vs 200DMA
        boot_dma = block_bootstrap_sharpe_diff(hmm_bt['net_return'], dma_bt['net_return'], ann_factor)
        lw_dma = ledoit_wolf_sharpe_diff(hmm_bt['net_return'], dma_bt['net_return'], ann_factor)
        
        tests.append({
            "Asset": asset_name.upper(),
            "Baseline": "200DMA",
            "Diff": boot_dma['diff'],
            "CI": (boot_dma['ci_lower'], boot_dma['ci_upper']),
            "p_boot": boot_dma['p_value'],
            "p_lw": lw_dma['p_value']
        })

    # Adjust p-values (Holm-Bonferroni is more conservative, BH is False Discovery Rate)
    # The prompt says "Report both raw and adjusted p-values (Holm or BH) and say which you consider the honest headline"
    pvals = [t["p_boot"] for t in tests]
    holm_pvals = adjust_pvalues(pvals, method='holm')
    
    print(f"{'Asset':<10} | {'Baseline':<12} | {'SR Diff':<8} | {'95% CI (Boot)':<25} | {'Raw p':<8} | {'Holm p':<8} | {'LW p':<8}")
    print("-" * 100)
    
    for i, t in enumerate(tests):
        ci_str = f"[{t['CI'][0]:.2f}, {t['CI'][1]:.2f}]"
        print(f"{t['Asset']:<10} | {t['Baseline']:<12} | {t['Diff']:>8.2f} | {ci_str:<25} | {t['p_boot']:>8.4f} | {holm_pvals[i]:>8.4f} | {t['p_lw']:>8.4f}")

    return tests

def run_ablation(cfg, data):
    print("\n" + "="*80)
    print("FEATURE-CATEGORY ABLATION (SPX Only)")
    print("="*80)
    
    asset_name = 'spx'
    df = data[asset_name].copy()
    ann_factor = 252
    
    features = build_features(df, df['VIX_Close'])
    features['returns'] = df['Close'].pct_change()
    features = features.dropna()
    
    # Get Full Baseline Labels
    raw_full, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name=asset_name)
    smooth_full = smooth_regime_labels(raw_full, window_size=cfg['model']['smoothing_window'])
    oos_mask = smooth_full.notna()
    
    print(f"{'Category':<15} | {'Features':<5} | {'ARI vs Full':<12} | {'Switches':<10} | {'Mean Dur (d)':<12} | {'CAGR (%)':<8} | {'Sharpe':<6}")
    print("-" * 90)
    
    # Calculate for Full Features first
    oos_labels_full = smooth_full[oos_mask]
    switches_full = (oos_labels_full != oos_labels_full.shift(1)).sum()
    mean_dur_full = len(oos_labels_full) / switches_full if switches_full > 0 else len(oos_labels_full)
    
    hmm_targets = compute_target_weight(oos_labels_full)
    bt_full = run_backtest(features[oos_mask], hmm_targets, cost_bps=0.0)
    cagr_full = (1 + bt_full['net_return']).prod() ** (ann_factor / len(bt_full)) - 1
    sr_full = (cagr_full / (bt_full['net_return'].std() * np.sqrt(ann_factor))) if bt_full['net_return'].std() > 0 else 0
    
    print(f"{'Full':<15} | {8:<5} | {'1.0000':<12} | {switches_full:<10} | {mean_dur_full:<12.1f} | {cagr_full*100:>8.2f} | {sr_full:>6.2f}")
    
    for category, cols in CATEGORIES.items():
        # Abalation: Fit only on these columns
        raw_ab, _ = walk_forward_predict(features, cols, cfg, asset_name=asset_name)
        smooth_ab = smooth_regime_labels(raw_ab, window_size=cfg['model']['smoothing_window'])
        
        oos_labels_ab = smooth_ab[oos_mask]
        
        # Stability Metrics
        ari = adjusted_rand_score(oos_labels_full, oos_labels_ab)
        switches = (oos_labels_ab != oos_labels_ab.shift(1)).sum()
        mean_dur = len(oos_labels_ab) / switches if switches > 0 else len(oos_labels_ab)
        
        # Performance Metrics
        targets_ab = compute_target_weight(oos_labels_ab)
        bt_ab = run_backtest(features[oos_mask], targets_ab, cost_bps=0.0)
        cagr_ab = (1 + bt_ab['net_return']).prod() ** (ann_factor / len(bt_ab)) - 1
        vol = bt_ab['net_return'].std() * np.sqrt(ann_factor)
        sr_ab = cagr_ab / vol if vol > 0 else 0
        
        print(f"{category:<15} | {len(cols):<5} | {ari:<12.4f} | {switches:<10} | {mean_dur:<12.1f} | {cagr_ab*100:>8.2f} | {sr_ab:>6.2f}")


if __name__ == "__main__":
    cfg = load_config('config/config.yaml')
    data = download_prices(cfg)
    
    tests = analyze_significance(cfg, data)
    run_ablation(cfg, data)
