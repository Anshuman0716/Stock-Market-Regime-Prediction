import sys
sys.path.insert(0, ".")
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
import warnings
warnings.filterwarnings('ignore')

cfg = load_config('config/config.yaml')
data = download_prices(cfg)
spx = data['spx']
features = build_features(spx, spx['VIX_Close'])
train_df = features[features.index.year < 2005]
X_train = StandardScaler().fit_transform(train_df[FEATURE_COLUMNS].values)

for k in [2, 3]:
    print(f"\nEvaluating stability for k={k} (diagonal)...")
    base = GaussianHMM(n_components=k, covariance_type="diag", n_iter=2000, random_state=42).fit(X_train).predict(X_train)
    aris = []
    for s in range(100, 120):
        preds = GaussianHMM(n_components=k, covariance_type="diag", n_iter=2000, random_state=s).fit(X_train).predict(X_train)
        aris.append(adjusted_rand_score(base, preds))
    print(f"Mean ARI: {np.mean(aris):.4f}, Min: {np.min(aris):.4f}, Max: {np.max(aris):.4f}")
