import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels

print("=" * 65)
print("  Modeling Layer - Verification & Look-Ahead Proof")
print("=" * 65)

# 1. Load Data & Build Features
cfg = load_config()
data = download_prices(config=cfg)
spx = data["spx"]

print("\nBuilding features for S&P 500...")
t0 = time.perf_counter()
features = build_features(spx, spx["VIX_Close"])
# Reattach target variables (Close) needed for plotting, and 'returns' for ranking
features["Close"] = spx.loc[features.index, "Close"]
t1 = time.perf_counter()
print(f"Features built in {t1 - t0:.2f}s")

# 2. Full Walk-Forward Run
print("\n-- Test 1: Full Walk-Forward Predict --")
t0 = time.perf_counter()
raw_labels, folds_info = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name="spx")
t1 = time.perf_counter()

print(f"\nCompleted in {t1 - t0:.2f} s")
print(f"Total Folds: {len(folds_info)}")
print("\nFold Boundaries & State Mappings:")
for f in folds_info:
    mapping_str = ", ".join([f"State {k}->{v}" for k, v in sorted(f['state_map'].items())])
    print(f"Fold {f['fold_idx']:02d} | Test: {f['test_start']} to {f['test_end']} | {mapping_str}")

# Apply trailing smoothing
smoothed_labels = smooth_regime_labels(raw_labels, window_size=cfg["model"]["smoothing_window"])
features["Regime_Raw"] = raw_labels
features["Regime_Smoothed"] = smoothed_labels

# Drop the NA portion (initial train window where out-of-sample hasn't started)
labeled_features = features.dropna(subset=["Regime_Smoothed"]).copy()
print(f"\nFirst labeled date (after initial training): {labeled_features.index[0].date()}")


# 3. Prove No Look-Ahead
print("\n-- Test 2: Prove No Look-Ahead (Truncation Test) --")
trunc_date = pd.to_datetime("2019-06-14")  # Arbitrary date mid-year
print(f"Truncating full dataset at {trunc_date.date()}...")
trunc_features = features.loc[:trunc_date].copy()

# Note: to prove no look-ahead in the HMM, we pass exactly the truncated dataset 
# through the pipeline.
raw_labels_trunc, _ = walk_forward_predict(trunc_features, FEATURE_COLUMNS, cfg, asset_name="spx_trunc")
smoothed_labels_trunc = smooth_regime_labels(raw_labels_trunc, window_size=cfg["model"]["smoothing_window"])

# Compare labels
# Get intersection of dates where both have labels
common_dates = raw_labels.dropna().index.intersection(raw_labels_trunc.dropna().index)

print(f"Comparing labels on {len(common_dates)} days...")
diff_raw = (raw_labels.loc[common_dates] != raw_labels_trunc.loc[common_dates]).sum()
diff_smoothed = (smoothed_labels.loc[common_dates] != smoothed_labels_trunc.loc[common_dates]).sum()

print(f"Differences in RAW labels:      {diff_raw}")
print(f"Differences in SMOOTHED labels: {diff_smoothed}")

if diff_raw == 0 and diff_smoothed == 0:
    print("[OK] ZERO look-ahead detected. Labels are perfectly stable.")
else:
    print("[FAIL] LOOK-AHEAD DETECTED!")
    
# 4. Plotting
print("\n-- Test 3: Plotting Regimes over Price --")
plot_df = labeled_features.copy()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(plot_df.index, plot_df["Close"], color='black', linewidth=1, alpha=0.7)

# Shade regimes
colors = {"Bull": "green", "Transition": "orange", "Crisis": "red"}
for i in range(1, len(plot_df)):
    start = plot_df.index[i-1]
    end = plot_df.index[i]
    regime = plot_df["Regime_Smoothed"].iloc[i]
    if pd.notna(regime):
        ax.axvspan(start, end, color=colors[regime], alpha=0.3, lw=0)

plt.title("S&P 500 Regimes (Walk-Forward, No Look-Ahead)")
plt.yscale("log")
plt.ylabel("Log Price")
plt.tight_layout()
plt.savefig("artifacts/regime_chart.png")
print("Plot saved to artifacts/regime_chart.png")

print("=" * 65)
print("  ALL TESTS COMPLETE")
print("=" * 65)
