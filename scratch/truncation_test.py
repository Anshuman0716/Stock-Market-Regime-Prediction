import sys
sys.path.insert(0, '.')

import pandas as pd
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels

cfg = load_config('config/config.yaml')
data = download_prices(cfg)
spx = data['spx']
features = build_features(spx, spx['VIX_Close'])
features['returns'] = spx['Close'].pct_change()
features = features.dropna()

print("Generating full pipeline labels...")
full_raw, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name='spx')
full_smoothed = smooth_regime_labels(full_raw, window_size=21)

cut_dates = ['2019-06-14', '2012-03-10', '2023-11-01']

for cut_date in cut_dates:
    print(f"\n--- Truncation test for {cut_date} ---")
    truncated_features = features.loc[:cut_date]
    trunc_raw, _ = walk_forward_predict(truncated_features, FEATURE_COLUMNS, cfg, asset_name='spx')
    trunc_smoothed = smooth_regime_labels(trunc_raw, window_size=21)
    
    # Compare
    common_idx = trunc_raw.dropna().index
    raw_diffs = (full_raw.loc[common_idx] != trunc_raw.loc[common_idx]).sum()
    smoothed_diffs = (full_smoothed.loc[common_idx] != trunc_smoothed.loc[common_idx]).sum()
    
    print(f"Comparing on {len(common_idx)} days...")
    print(f"RAW differences: {raw_diffs}")
    print(f"SMOOTHED differences: {smoothed_diffs}")
