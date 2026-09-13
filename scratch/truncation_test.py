import sys
sys.path.insert(0, ".")
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels

cfg = load_config()
data = download_prices(cfg)
spx = data["spx"]
features = build_features(spx, spx["VIX_Close"])
features['returns'] = spx['Close'].pct_change()
features = features.dropna()

print("Generating full pipeline labels...")
full_raw, full_probs, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name='spx')

cut_dates = ['2019-06-14', '2012-03-10', '2023-11-01']

for cut_date in cut_dates:
    print(f"\n--- Truncation test for {cut_date} ---")
    truncated_features = features.loc[:cut_date]
    if len(truncated_features) < 1000:
        print("Not enough data to run walk-forward.")
        continue
    trunc_raw, trunc_probs, _ = walk_forward_predict(truncated_features, FEATURE_COLUMNS, cfg, asset_name='spx')
    
    common_idx = trunc_raw.dropna().index
    raw_diffs = (full_raw.loc[common_idx] != trunc_raw.loc[common_idx]).sum()
    
    print(f"Comparing on {len(common_idx)} days...")
    print(f"RAW differences: {raw_diffs}")
    if raw_diffs > 0:
        print("FAIL: Look-ahead bias detected in raw labels.")
    else:
        print("PASS: No look-ahead bias in raw labels.")
