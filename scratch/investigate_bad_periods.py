import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from src.models.hmm import predict_proba_filtered

cfg = load_config()
data = download_prices(config=cfg)
spx = data['spx']
features = build_features(spx, spx['VIX_Close'])
features['returns'] = spx.loc[features.index, 'Close'].pct_change()
features = features.dropna()

periods = {
    "2008 Crash": ("2008-09-01", "2008-12-31", 2008),
    "2018 Q4": ("2018-10-01", "2018-12-31", 2018),
    "2020 COVID": ("2020-02-20", "2020-04-30", 2020),
    "2022 Bear": ("2022-01-01", "2022-10-31", 2022),
    "2024 Bull": ("2024-01-01", "2024-08-01", 2024)
}

for name, (start, end, test_year) in periods.items():
    train_df = features[features.index.year < test_year]
    test_df = features[(features.index >= start) & (features.index <= end)]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)
    
    model = GaussianHMM(n_components=4, covariance_type="full", n_iter=2000, random_state=42 + (test_year - 2005))
    model.fit(X_train)
    
    train_states = model.predict(X_train)
    
    print(f"\n=== {name} ===")
    stats = []
    returns = train_df["returns"]
    for i in range(4):
        mask = (train_states == i)
        r = returns[mask]
        vol = r.std() * np.sqrt(252) if len(r) > 1 else 0
        mu = r.mean() * 252 if len(r) > 0 else 0
        sharpe = mu / vol if vol > 0 else -np.inf
        stats.append({"state": i, "vol": vol, "mu": mu, "sharpe": sharpe})
        print(f"  State {i}: Vol={vol:.4f}, Mu={mu:.4f}, Sharpe={sharpe:.4f}")
        
    test_probs = predict_proba_filtered(model, X_test)
    test_states = np.argmax(test_probs, axis=1)
    
    # Mode of the states during this period
    active_state = pd.Series(test_states).mode()[0]
    active_stat = next(s for s in stats if s["state"] == active_state)
    print(f"-> Active State during {name}: {active_state}")
    print(f"   (Vol={active_stat['vol']:.4f}, Mu={active_stat['mu']:.4f}, Sharpe={active_stat['sharpe']:.4f})")
