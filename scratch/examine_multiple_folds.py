import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

cfg = load_config()
data = download_prices(config=cfg)
spx = data['spx']
features = build_features(spx, spx['VIX_Close'])
# Need returns for stats
features['returns'] = spx.loc[features.index, 'Close'].pct_change()
features = features.dropna()

for test_year in [2008, 2010, 2018, 2020, 2024]:
    train_df = features[features.index.year < test_year]
    X_train = StandardScaler().fit_transform(train_df[FEATURE_COLUMNS].values)
    model = GaussianHMM(n_components=4, covariance_type="full", n_iter=2000, random_state=42 + (test_year - 2005))
    model.fit(X_train)
    states = model.predict(X_train)

    print(f"\n--- Train ending before {test_year} ---")
    returns = train_df["returns"]
    for i in range(4):
        mask = (states == i)
        r = returns[mask]
        vol = r.std() * np.sqrt(252) if len(r) > 1 else 0
        mu = r.mean() * 252 if len(r) > 0 else 0
        sharpe = mu / vol if vol > 0 else 0
        print(f"State {i}: Vol={vol:.4f}, Mu={mu:.4f}, Sharpe={sharpe:.4f}, count={len(r)}")
