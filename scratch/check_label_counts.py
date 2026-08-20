import sys
sys.path.insert(0, ".")

import pandas as pd
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels

cfg = load_config()
data = download_prices(config=cfg)
spx = data['spx']
features = build_features(spx, spx['VIX_Close'])
features['returns'] = spx.loc[features.index, 'Close'].pct_change()
features = features.dropna()
cfg["model"]["refit_cadence"] = "annual"
raw_labels, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name="spx")
smoothed_labels = smooth_regime_labels(raw_labels, window_size=21)

print("2008 Q4 RAW counts:")
print(raw_labels.loc["2008-09-01":"2008-12-31"].value_counts())
print("\n2008 Q4 SMOOTHED counts:")
print(smoothed_labels.loc["2008-09-01":"2008-12-31"].value_counts())

print("\n2022 RAW counts:")
print(raw_labels.loc["2022-01-01":"2022-12-31"].value_counts())
print("\n2022 SMOOTHED counts:")
print(smoothed_labels.loc["2022-01-01":"2022-12-31"].value_counts())

print("\n2020 Q1 RAW counts:")
print(raw_labels.loc["2020-02-01":"2020-04-30"].value_counts())
print("\n2020 Q1 SMOOTHED counts:")
print(smoothed_labels.loc["2020-02-01":"2020-04-30"].value_counts())
