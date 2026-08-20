import pytest
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.models.hmm import walk_forward_predict, predict_proba_filtered
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.labeling import rank_and_map_states

@pytest.fixture
def mock_features(sample_spx, sample_vix):
    # build a small feature set spanning a few years for walk forward
    feat = build_features(sample_spx, sample_vix['Close'])
    feat['returns'] = sample_spx['Close'].pct_change()
    return feat.dropna()

def test_walk_forward_no_future_data_and_scaler_differs(mock_features):
    # Mock config
    cfg = {
        'model': {
            'n_states': 3,
            'covariance_type': 'diag',
            'n_iter': 10,
            'min_train_days': 200,
            'smoothing_window': 5
        }
    }
    
    # We will modify hmm.py slightly in our minds to assert on the artifacts if needed, 
    # or we can test walk_forward_predict and check folds_info
    raw_labels, out_probs, folds_info = walk_forward_predict(mock_features, FEATURE_COLUMNS, cfg, asset_name="test")
    
    # Check folds_info
    assert len(folds_info) >= 1
    
    # 1. Walk-forward folds never include future data in their training window
    for fold in folds_info:
        # train_end must be strictly less than test_start
        assert fold['train_end'] < fold['test_start']
        
    # The scaler is fit per fold (assert scaler statistics differ across folds)
    # We can load the artifacts saved by walk_forward_predict to verify this.
    import joblib
    import os
    if len(folds_info) > 1:
        scaler0 = joblib.load(os.path.join("models", "test", "hmm_fold_0.joblib"))['scaler']
        scaler1 = joblib.load(os.path.join("models", "test", "hmm_fold_1.joblib"))['scaler']
        
        # Check that the means are not exactly identical, proving they were fit independently on different data
        assert not np.array_equal(scaler0.mean_, scaler1.mean_)

def test_filtered_vs_smoothed():
    # Filtered probabilities differ from full-sequence smoothed probabilities
    # (if they are identical, the forward-only implementation silently regressed)
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=10, random_state=42)
    model.fit(X)
    
    # hmmlearn's predict_proba uses the forward-backward algorithm (smoothed)
    smoothed_probs = model.predict_proba(X)
    
    # Our explicitly filtered probabilities (forward-only)
    filtered_probs = predict_proba_filtered(model, X)
    
    # They should sum to 1
    assert np.allclose(filtered_probs.sum(axis=1), 1.0)
    assert np.allclose(smoothed_probs.sum(axis=1), 1.0)
    
    # They should NOT be identical, because smoothed uses future data within the sequence
    # and filtered does not.
    assert not np.allclose(smoothed_probs, filtered_probs)
    
def test_state_mapping_deterministic():
    # state->label mapping is deterministic given a seed
    df = pd.DataFrame({
        'returns': np.random.randn(1000)
    })
    states = np.random.randint(0, 3, size=1000)
    
    map1 = rank_and_map_states(df, states, n_states=3)
    map2 = rank_and_map_states(df, states, n_states=3)
    
    assert map1 == map2
