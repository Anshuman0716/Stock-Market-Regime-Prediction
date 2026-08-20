import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Model Development

This notebook evaluates the walk-forward HMM pipeline imported from `src/`.
Local reimplementations of the Gaussian model have been removed to preserve the single source of truth.
"""))

cells.append(nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

import pandas as pd
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.models.labeling import smooth_regime_labels

cfg = load_config('../config/config.yaml')
data = download_prices(cfg)
spx = data['spx']

features = build_features(spx, spx['VIX_Close'])
features['returns'] = spx['Close'].pct_change()
features = features.dropna()

raw_labels, folds_info = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name='spx')
smoothed_labels = smooth_regime_labels(raw_labels, window_size=cfg['model']['smoothing_window'])
"""))

nb['cells'] = cells
with open('notebooks/02_model_development.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)


nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Backtest and Risk

This notebook simulates the strategy using regimes obtained from the `src/` modules.
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
data = download_prices(cfg)
asset = data['spx']

features = build_features(asset, asset['VIX_Close'])
features['returns'] = asset['Close'].pct_change()
features = features.dropna()

raw_labels, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name='spx')
features['regime'] = smooth_regime_labels(raw_labels, window_size=cfg['model']['smoothing_window'])

features['strategy_returns'] = np.where(features['regime'] == 'Bull', features['returns'], 0)

plt.plot((1 + features['returns']).cumprod(), label='Buy & Hold')
plt.plot((1 + features['strategy_returns']).cumprod(), label='Regime Strategy')
plt.legend()
plt.show()
"""))

nb['cells'] = cells
with open('notebooks/03_backtest_and_risk.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
