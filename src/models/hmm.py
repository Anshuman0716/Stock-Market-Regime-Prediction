"""
Modeling layer: HMM walk-forward fitting and strictly filtered predictions.

This module implements the core walk-forward loop. By refitting the HMM on an
expanding window, we guarantee that the model evaluating day `t` has never
seen data from day `t+1`.

It handles the subtleties of out-of-sample HMM evaluation:
1. It uses FILTERED probabilities (forward pass only), not smoothed ones.
2. It re-ranks and maps the arbitrary state identities on every refit.
3. It fits the StandardScaler ONLY on the training window.
4. It runs the forward pass continuously from the start of the history, so
   the state belief carries over from 31 December to 2 January instead of
   being reset every year.
5. It fits each fold from several random starts and keeps the best one,
   because EM only finds a local optimum.
"""

import logging

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from hmmlearn import _hmmc

from src.data.loader import get_ann_factor
from src.models.labeling import rank_and_map_states, save_fold_artifact, state_statistics
from src.models.signal import REGIME_LABELS, regime_labels

logger = logging.getLogger(__name__)


def predict_proba_filtered(model, X):
    """
    Computes FILTERED probabilities (forward pass only).

    Why this matters:
    hmmlearn's standard `predict()` (Viterbi) and `predict_proba()` (smoothed
    posterior gamma) condition on the ENTIRE sequence passed in. If you pass an
    out-of-sample window of 252 days, day 1's label will be informed by day
    252's data.

    This function explicitly computes the forward variable (alpha), normalizing
    it at each step to yield P(z_t | X_{1:t}). This strictly prevents any future
    data within the sequence from influencing the current day's probability.

    Parameters
    ----------
    model : GaussianHMM
        Fitted hmmlearn model.
    X : np.ndarray
        Feature matrix for the sequence to decode.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_components) containing filtered state probabilities.
    """
    # 1. Emission log probabilities for all observations
    log_frameprob = model._compute_log_likelihood(X)

    # 2. Forward pass -> forward lattice (log alpha)
    _, fwdlattice = _hmmc.forward_log(model.startprob_, model.transmat_, log_frameprob)

    # 3. Normalize each row to get P(z_t | X_{1:t})
    filtered_probs = np.exp(fwdlattice - logsumexp(fwdlattice, axis=1, keepdims=True))
    assert np.allclose(filtered_probs.sum(axis=1), 1.0), "Filtered probs do not sum to 1"
    return filtered_probs


def map_to_regimes(state_probs, state_map):
    """
    Convert per-state probabilities into Bull / Transition / Crisis columns.

    Several raw states can share a label (e.g. two "Bull" states), so
    P(Bull) is the SUM of their probabilities. A label no state maps to in
    this model gets probability 0, so every row still sums to 1.

    Used by both the walk-forward backtest and live paper trading, so the
    two always aggregate states the same way.
    """
    out = np.zeros((state_probs.shape[0], len(REGIME_LABELS)))
    for s, label in state_map.items():
        out[:, REGIME_LABELS.index(label)] += state_probs[:, s]
    return out


def fit_best_hmm(X_train, n_states, covariance_type, n_iter, n_init, seed):
    """
    Fit `n_init` HMMs from different random starts; keep the highest
    training log-likelihood.

    Why restarts: EM (Baum-Welch) climbs to the nearest local optimum of the
    likelihood, and which one it reaches depends on the random start. A
    single start can land in a poor solution (e.g. two states describing
    the same calm market). Keeping the best of several starts makes the fit
    more stable from fold to fold. Only TRAINING data is used to choose, so
    this adds no look-ahead.

    Returns
    -------
    (model, diagnostics) where diagnostics records every start's
    log-likelihood and whether it converged.
    """
    best_model, best_ll, starts = None, -np.inf, []
    # hmmlearn logs a warning whenever one EM step lowers the likelihood
    # slightly (floating-point noise near convergence). We record convergence
    # per start in `starts` instead of printing hundreds of these lines.
    hmm_logger = logging.getLogger("hmmlearn.base")
    previous_level = hmm_logger.level
    hmm_logger.setLevel(logging.ERROR)
    try:
        for k in range(n_init):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=seed + k,
                tol=1e-4,
            )
            model.fit(X_train)
            ll = model.score(X_train)
            starts.append({"seed": seed + k, "log_likelihood": float(ll),
                           "converged": bool(model.monitor_.converged),
                           "iterations": int(model.monitor_.iter)})
            if ll > best_ll:
                best_model, best_ll = model, ll
    finally:
        hmm_logger.setLevel(previous_level)
    return best_model, {"best_log_likelihood": float(best_ll), "starts": starts}


def walk_forward_predict(df, feature_cols, config, asset_name="spx", model_dir=None):
    """
    Perform an expanding-window walk-forward fit and predict loop.

    Parameters
    ----------
    df : pd.DataFrame
        Asset data containing features and at least a 'returns' column.
        Index must be a DatetimeIndex.
    feature_cols : list of str
        The columns to use as features for the HMM.
    config : dict
        Configuration dict containing 'model' parameters.
    asset_name : str
        Name of the asset (used for annualization, logging and saving models).
    model_dir : str or None
        Where to save the fold artifacts. Defaults to models/<asset_name>.

    Returns
    -------
    tuple
        (raw_labels, probs, folds_info)
        raw_labels: pd.Series of the most likely regime each day (unsmoothed).
        probs: pd.DataFrame with columns Bull / Transition / Crisis holding
            filtered probabilities. Rows inside the initial training window
            (before the first out-of-sample year) stay NaN on purpose: no
            model existed yet, so there is no honest prediction for them.
            Downstream evaluation uses this NaN boundary to find where the
            out-of-sample period starts.
        folds_info: list of dicts describing fold boundaries, mappings and
            fit diagnostics.
    """
    model_cfg = config["model"]
    n_states = model_cfg.get("n_states", 4)
    covariance_type = model_cfg.get("covariance_type", "diag")
    n_iter = model_cfg.get("n_iter", 2000)
    n_init = model_cfg.get("n_init", 1)
    min_train_days = model_cfg.get("min_train_days", 1008)
    # Labeling rule and thresholds (src/models/labeling.py). Configs without a
    # labeling block get the original "absolute" rule.
    labeling = {"rule": "absolute", **model_cfg.get("labeling", {})}
    ann_factor = get_ann_factor(config, asset_name)
    model_dir = model_dir or f"models/{asset_name}"

    years = df.index.year.unique()
    out_probs = pd.DataFrame(index=df.index, columns=REGIME_LABELS, dtype=float)
    folds_info = []

    # First test year = first year with at least min_train_days rows before it.
    first_test_year = None
    for y in years:
        if (df.index.year < y).sum() >= min_train_days:
            first_test_year = y
            break
    if first_test_year is None:
        raise ValueError(
            f"Dataset too short for min_train_days={min_train_days}. "
            f"Total rows: {len(df)}"
        )

    logger.info(f"[{asset_name}] Walk-forward starting. First out-of-sample year: {first_test_year}")

    fold_idx = 0
    for test_year in range(first_test_year, years[-1] + 1):
        # 1. Windows. `history` = everything up to the end of the test year.
        train_mask = df.index.year < test_year
        test_mask = df.index.year == test_year
        history_mask = df.index.year <= test_year
        train_df = df[train_mask]
        test_df = df[test_mask]
        if test_df.empty:
            continue

        # 2. Scaler fit on the training window only.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)

        # 3. HMM: best of n_init random starts, chosen on training likelihood.
        # Seeds are spaced per fold so folds don't share starting points.
        model, fit_diag = fit_best_hmm(
            X_train, n_states, covariance_type, n_iter, n_init, seed=42 + 1000 * fold_idx)

        # 4. Rank and map states using TRAINING data only. Viterbi on the
        # training set is fine here: it only describes past behaviour.
        train_states = model.predict(X_train)
        state_map = rank_and_map_states(train_df, train_states, n_states, ann_factor=ann_factor, **labeling)
        state_stats = state_statistics(train_df, train_states, n_states, ann_factor)
        for st in state_stats:
            st["label"] = state_map[st["state"]]

        # 5. Persist model + scaler + mapping (one is useless without the others).
        save_fold_artifact(fold_idx, model, scaler, state_map, directory=model_dir,
                           train_end=train_df.index[-1].date())

        # 6. Filtered probabilities over the whole history up to the end of
        # the test year, keeping only the test-year rows. The forward pass
        # is causal (row t uses rows 1..t only), so feeding it the training
        # years first adds no look-ahead. It just means the state belief on
        # 2 January continues from 31 December instead of restarting from
        # the model's generic starting distribution.
        X_history = scaler.transform(df.loc[history_mask, feature_cols].values)
        history_probs = predict_proba_filtered(model, X_history)
        test_state_probs = history_probs[-len(test_df):]

        # 7. Store out-of-sample regime probabilities.
        out_probs.loc[test_df.index, REGIME_LABELS] = map_to_regimes(test_state_probs, state_map)

        folds_info.append({
            "fold_idx": fold_idx,
            "train_start": train_df.index[0].date(),
            "train_end": train_df.index[-1].date(),
            "test_start": test_df.index[0].date(),
            "test_end": test_df.index[-1].date(),
            "state_map": state_map,
            "states": state_stats,
            "fit": fit_diag,
        })
        fold_idx += 1

    return regime_labels(out_probs), out_probs, folds_info


def predict_latest_probs(features, artifact, feature_cols):
    """
    Regime probabilities for every row of `features` using one saved fold
    artifact. Used for live / paper trading with the most recent fold.

    It runs the same continuous forward pass as the backtest: over the full
    feature history, scaled with the fold's own training scaler.
    """
    X = artifact["scaler"].transform(features[feature_cols].values)
    state_probs = predict_proba_filtered(artifact["model"], X)
    return pd.DataFrame(map_to_regimes(state_probs, artifact["state_map"]),
                        index=features.index, columns=REGIME_LABELS)


class RetrainRequired(RuntimeError):
    """Saved models cannot cover the data (e.g. a new calendar year began)."""


def predict_from_artifacts(df, feature_cols, folds_info, model_dir):
    """
    Rebuild walk-forward probabilities from SAVED fold models, without refitting.

    Each fold's model predicts its own test year with the same continuous
    forward pass used in training, so for the same data and config the
    output matches walk_forward_predict exactly (tests/test_pipeline.py).
    New days inside the latest fold's year are predicted by that fold, which
    is exactly what would have happened live. A day in a year no fold
    covers needs a new fold, so we raise RetrainRequired instead of guessing.

    Returns (raw_labels, probs, folds_info) like walk_forward_predict.
    """
    from src.models.labeling import load_fold_artifact

    out_probs = pd.DataFrame(index=df.index, columns=REGIME_LABELS, dtype=float)
    fold_years = {}
    for fold in folds_info:
        year = pd.Timestamp(fold["test_start"]).year
        fold_years[year] = fold
    first_year = min(fold_years)
    uncovered = sorted(set(df.index.year[df.index.year >= first_year]) - set(fold_years))
    if uncovered:
        raise RetrainRequired(f"No saved fold covers year(s) {uncovered}; run the pipeline with training.")

    for year, fold in sorted(fold_years.items()):
        artifact = load_fold_artifact(fold["fold_idx"], directory=model_dir)
        history = df[df.index.year <= year]
        test_index = history.index[history.index.year == year]
        if len(test_index) == 0:
            continue
        X_history = artifact["scaler"].transform(history[feature_cols].values)
        state_probs = predict_proba_filtered(artifact["model"], X_history)[-len(test_index):]
        out_probs.loc[test_index, REGIME_LABELS] = map_to_regimes(state_probs, artifact["state_map"])
    return regime_labels(out_probs), out_probs, folds_info
