"""
Tests for the walk-forward HMM: training windows, per-fold scaling and
filtered (forward-only) probabilities.
"""
import os

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.features.engineering import FEATURE_COLUMNS, build_features
from src.models.hmm import predict_proba_filtered, walk_forward_predict

CFG = {"model": {"n_states": 3, "covariance_type": "diag", "n_iter": 10, "n_init": 1,
                 "min_train_days": 200, "smoothing_window": 5}}


def test_walk_forward_trains_on_the_past_and_scales_per_fold(sample_spx, sample_vix, tmp_path):
    features = build_features(sample_spx, sample_vix["Close"])
    _, _, folds = walk_forward_predict(features, FEATURE_COLUMNS, CFG, asset_name="test",
                                       model_dir=str(tmp_path))
    assert len(folds) >= 2, "sample data should give at least two yearly folds"

    for fold in folds:
        # Training always ends before the year it predicts.
        assert fold["train_end"] < fold["test_start"]

    # Each fold fits its own scaler on its own (longer) training window, so
    # the fitted means differ. Identical means would suggest one scaler was
    # fit once and reused, e.g. on the full sample.
    s0 = joblib.load(os.path.join(tmp_path, "hmm_fold_0.joblib"))["scaler"]
    s1 = joblib.load(os.path.join(tmp_path, "hmm_fold_1.joblib"))["scaler"]
    assert not np.array_equal(s0.mean_, s1.mean_)


def test_filtered_probabilities_differ_from_smoothed():
    """Forward-only (filtered) probabilities must not equal forward-backward
    (smoothed) ones; if they did, the filter would be silently using future rows."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=10, random_state=42).fit(X)

    smoothed = model.predict_proba(X)
    filtered = predict_proba_filtered(model, X)

    assert np.allclose(filtered.sum(axis=1), 1.0)
    assert not np.allclose(smoothed, filtered)
    # The last row has no future to look at, so there the two must agree.
    np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-8)


def test_filtered_probability_at_t_ignores_later_rows():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 2))
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=10, random_state=0).fit(X)
    full = predict_proba_filtered(model, X)
    prefix = predict_proba_filtered(model, X[:50])
    np.testing.assert_allclose(full[:50], prefix)


def test_number_of_states_can_be_set_per_asset(sample_spx, sample_vix, tmp_path):
    features = build_features(sample_spx, sample_vix["Close"])
    cfg = {"model": dict(CFG["model"], n_states_by_asset={"btc": 5})}
    _, _, folds = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name="btc", model_dir=str(tmp_path))
    assert all(len(f["state_map"]) == 5 for f in folds)
    _, _, folds = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name="spx",
                                       model_dir=str(tmp_path / "spx"))
    assert all(len(f["state_map"]) == CFG["model"]["n_states"] for f in folds)
