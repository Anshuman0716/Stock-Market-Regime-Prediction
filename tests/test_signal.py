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


def test_walk_forward_is_causal(features, tmp_path):
    """Deleting the last year of data must not change any earlier prediction.

    If any step (scaling, fitting, labeling, forward pass) peeked at later
    data, the predictions for the earlier years would move.
    """
    _, full, _ = walk_forward_predict(features, FEATURE_COLUMNS, CFG, model_dir=str(tmp_path / "full"))
    cut_year = features.index.year.max()
    truncated = features[features.index.year < cut_year]
    _, part, _ = walk_forward_predict(truncated, FEATURE_COLUMNS, CFG, model_dir=str(tmp_path / "part"))
    pd.testing.assert_frame_equal(full.loc[part.index], part)


def test_live_signal_matches_backtest(features, tmp_path):
    """The paper-trading path reproduces the backtest's probabilities exactly.

    For the latest fold, running predict_latest_probs (what paper_trade.py
    uses) over the full history must give the same test-year probabilities
    as walk_forward_predict produced.
    """
    _, probs, folds = walk_forward_predict(features, FEATURE_COLUMNS, CFG, model_dir=str(tmp_path))
    last = folds[-1]
    artifact = load_fold_artifact(last["fold_idx"], directory=str(tmp_path))
    live = predict_latest_probs(features, artifact, FEATURE_COLUMNS)
    test_rows = probs.index[probs.index.date >= last["test_start"]]
    pd.testing.assert_frame_equal(live.loc[test_rows], probs.loc[test_rows], check_dtype=False)


def test_probabilities_continue_across_year_boundary(features, tmp_path):
    """No reset on 1 January: the first test day of a fold is not the
    model's generic start distribution. We check the walk-forward output
    equals a single continuous forward pass for that fold."""
    _, probs, folds = walk_forward_predict(features, FEATURE_COLUMNS, CFG, model_dir=str(tmp_path))
    assert len(folds) >= 2
    fold = folds[1]
    artifact = load_fold_artifact(fold["fold_idx"], directory=str(tmp_path))
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


# ── Sizing with regime weights (Step 8) ──────────────────────────────────────

def test_target_weight_is_probability_weighted_regime_exposure():
    from src.models.signal import target_weight
    sp = pd.DataFrame({"Bull": [0.6, 0.1, 0.0], "Transition": [0.3, 0.2, 0.0005], "Crisis": [0.1, 0.7, 0.9995]})
    w = target_weight(sp, {"Bull": 1.0, "Transition": 1.0, "Crisis": 0.0})
    np.testing.assert_allclose(w.values, [0.9, 0.3, 0.0])   # last row 0.05% < MIN_WEIGHT -> 0
    np.testing.assert_allclose(target_weight(sp).values, [0.6, 0.1, 0.0])  # default: P(Bull)


def test_regime_weights_can_change_per_row():
    from src.models.signal import target_weight
    sp = pd.DataFrame({"Bull": [0.5, 0.5], "Transition": [0.5, 0.5], "Crisis": [0.0, 0.0]})
    weights = pd.DataFrame({"Bull": [1.0, 1.0], "Transition": [0.0, 1.0], "Crisis": [0.0, 0.0]})
    np.testing.assert_allclose(target_weight(sp, weights).values, [0.5, 1.0])


def test_regime_weight_frame_follows_each_folds_year():
    from src.models.hmm import regime_weight_frame
    idx = pd.to_datetime(["2009-12-31", "2010-06-01", "2011-03-01"])
    folds = [{"test_start": "2010-01-04", "regime_weights": {"Bull": 1.0, "Transition": 1.0, "Crisis": 0.0}},
             {"test_start": "2011-01-03", "regime_weights": {"Bull": 1.0, "Transition": 0.0, "Crisis": 0.0}}]
    frame = regime_weight_frame(idx, folds)
    assert frame["Transition"].tolist() == [0.0, 1.0, 0.0]  # before any fold: P(Bull) default


def test_volatility_scaling_only_shrinks_and_stays_causal():
    probs = _probs()
    scale = pd.Series(np.linspace(1.0, 0.2, len(probs)), index=probs.index)
    plain = compute_signal(probs, 5)["target_weight"]
    scaled = compute_signal(probs, 5, scale=scale)["target_weight"]
    assert (scaled <= plain + 1e-12).all()
    changed = scale.copy()
    changed.iloc[40:] = 0.0
    pd.testing.assert_series_equal(compute_signal(probs, 5, scale=changed)["target_weight"].iloc[:40],
                                   scaled.iloc[:40])
