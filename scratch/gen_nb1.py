import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# EDA and Feature Engineering

This notebook demonstrates the new single source of truth for data loading and feature engineering using the `src/` package.
"""))

cells.append(nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels

cfg = load_config('../config/config.yaml')
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Raw Data EDA"))

cells.append(nbf.v4.new_code_cell("""data = download_prices(cfg)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

assets = ['spx', 'nasdaq', 'gold', 'bitcoin']
for i, asset in enumerate(assets):
    df = data[asset]
    axes[i].plot(df.index, df['Close'])
    axes[i].set_title(f"{asset.upper()} Price")
    axes[i].set_yscale('log')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Feature Engineering & NaN Warmup"))

cells.append(nbf.v4.new_code_cell("""spx = data['spx']

# The single source of truth for features
features = build_features(spx, spx['VIX_Close'])
features['returns'] = spx['Close'].pct_change()

print("Shape with NaNs:", features.shape)
features = features.dropna()
print("Shape after warmup dropna:", features.shape)
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Distributions and Correlations"))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 10))
corr = features[FEATURE_COLUMNS].corr()
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Feature Correlations")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Regime-Conditional Feature Behavior"))

cells.append(nbf.v4.new_code_cell("""# Walk-forward prediction to avoid look-ahead
raw_labels, folds = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name='spx')
features['regime'] = smooth_regime_labels(raw_labels, window_size=cfg['model']['smoothing_window'])

regime_stats = features.groupby('regime')[FEATURE_COLUMNS].mean().T
display(regime_stats)
"""))

nb['cells'] = cells

with open('notebooks/01_eda_and_features.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
