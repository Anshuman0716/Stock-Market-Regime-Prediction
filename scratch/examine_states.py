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

train_df = features[features.index.year < 2024]
X_train = StandardScaler().fit_transform(train_df[FEATURE_COLUMNS].values)

model = GaussianHMM(n_components=4, covariance_type="full", n_iter=2000, random_state=42 + 19)
model.fit(X_train)
states = model.predict(X_train)

returns = train_df["returns"]
for i in range(4):
    mask = (states == i)
    r = returns[mask]
    vol = r.std() * np.sqrt(252)
    mu = r.mean() * 252
    sharpe = mu / vol if vol > 0 else 0
    print(f"State {i}: Vol={vol:.4f}, Mu={mu:.4f}, Sharpe={sharpe:.4f}, count={len(r)}")
