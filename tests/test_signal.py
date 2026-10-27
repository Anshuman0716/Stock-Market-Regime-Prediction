"""
Tests for the shared signal definition and the walk-forward HMM's causality.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import (
    walk_forward_predict, predict_latest_probs, map_to_regimes, fit_best_hmm,
)
from src.models.labeling import load_fold_artifact
from src.models.signal import compute_signal, REGIME_LABELS

CFG = {"model": {"n_states": 3, "covariance_type": "diag", "n_iter": 20,
                 "n_init": 2, "min_train_days": 400, "smoothing_window": 5}}


@pytest.fixture
def features(sample_spx, sample_vix):
    return build_features(sample_spx, sample_vix["Close"])


def _probs(n=60, start_pred=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    raw = rng.dirichlet([1, 1, 1], size=n)
    probs = pd.DataFrame(raw, index=idx, columns=REGIME_LABELS)
    probs.iloc[:start_pred] = np.nan
    return probs


def test_signal_columns_and_invariants():
    probs = _probs()
    sig = compute_signal(probs, window=5)
    pred = sig.iloc[10:]
    # smoothed probabilities still sum to 1
    np.testing.assert_allclose(pred[["sp_bull", "sp_transition", "sp_crisis"]].sum(axis=1), 1.0)
    # target weight IS smoothed P(Bull) (Dirichlet draws here are all >= 0.1%)
    np.testing.assert_allclose(pred["target_weight"], pred["sp_bull"])
    # smoothed label = most likely smoothed regime
    expected = pred[["sp_bull", "sp_transition", "sp_crisis"]].idxmax(axis=1).str[3:].str.capitalize()
    assert (pred["label"] == expected).all()
    # no prediction before the first out-of-sample row; weight 0 there
    assert sig["label"].iloc[:10].isna().all()
    assert (sig["target_weight"].iloc[:10] == 0).all()


def test_signal_is_trailing_only():
    probs = _probs()
    future_changed = probs.copy()
    future_changed.iloc[40:] = [1.0, 0.0, 0.0]
    a = compute_signal(probs, 5).iloc[:40]
    b = compute_signal(future_changed, 5).iloc[:40]
    pd.testing.assert_frame_equal(a, b)


def test_map_to_regimes_sums_shared_states():
    state_probs = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    out = map_to_regimes(state_probs, {0: "Bull", 1: "Bull", 2: "Crisis"})
    np.testing.assert_allclose(out, [[0.5, 0.0, 0.5], [0.7, 0.0, 0.3]])


def test_fit_best_hmm_keeps_highest_likelihood():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (150, 2)), rng.normal(3, 0.5, (150, 2))])
    model, diag = fit_best_hmm(X, 2, "diag", 20, n_init=3, seed=0)
    lls = [s["log_likelihood"] for s in diag["starts"]]
    assert len(lls) == 3
    assert diag["best_log_likelihood"] == pytest.approx(max(lls))
    assert model.score(X) == pytest.approx(max(lls))


def test_walk_forward_is_causal(features):
    """Deleting the last year of data must not change any earlier prediction.

    If any step (scaling, fitting, labeling, forward pass) peeked at later
    data, the predictions for the earlier years would move.
    """
    _, full, _ = walk_forward_predict(features, FEATURE_COLUMNS, CFG, asset_name="test")
    cut_year = features.index.year.max()
    truncated = features[features.index.year < cut_year]
    _, part, _ = walk_forward_predict(truncated, FEATURE_COLUMNS, CFG, asset_name="test")
    pd.testing.assert_frame_equal(full.loc[part.index], part)


def test_live_signal_matches_backtest(features):
    """The paper-trading path reproduces the backtest's probabilities exactly.

    For the latest fold, running predict_latest_probs (what paper_trade.py
    uses) over the full history must give the same test-year probabilities
    as walk_forward_predict produced.
    """
    _, probs, folds = walk_forward_predict(features, FEATURE_COLUMNS, CFG, asset_name="test")
    last = folds[-1]
    artifact = load_fold_artifact(last["fold_idx"], directory="models/test")
    live = predict_latest_probs(features, artifact, FEATURE_COLUMNS)
    test_rows = probs.index[probs.index.date >= last["test_start"]]
    pd.testing.assert_frame_equal(live.loc[test_rows], probs.loc[test_rows], check_dtype=False)


def test_probabilities_continue_across_year_boundary(features):
    """No reset on 1 January: the first test day of a fold is not the
    model's generic start distribution. We check the walk-forward output
    equals a single continuous forward pass for that fold."""
    _, probs, folds = walk_forward_predict(features, FEATURE_COLUMNS, CFG, asset_name="test")
    assert len(folds) >= 2
    fold = folds[1]
    artifact = load_fold_artifact(fold["fold_idx"], directory="models/test")
    history = features[features.index.date <= fold["test_end"]]
    continuous = predict_latest_probs(history, artifact, FEATURE_COLUMNS)
    first_day = probs.index[probs.index.date >= fold["test_start"]][0]
    np.testing.assert_allclose(probs.loc[first_day].values, continuous.loc[first_day].values)


def test_tiny_weights_snap_to_zero():
    """Positions under MIN_WEIGHT are 0, so near-zero noise can't flip
    'in the market' on and off between runs."""
    from src.models.signal import target_weight, MIN_WEIGHT
    sp = pd.DataFrame({"Bull": [1e-9, MIN_WEIGHT / 2, MIN_WEIGHT, 0.5, 1.2, np.nan]})
    np.testing.assert_allclose(target_weight(sp).values, [0, 0, MIN_WEIGHT, 0.5, 1.0, 0])
