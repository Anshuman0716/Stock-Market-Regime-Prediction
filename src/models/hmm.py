"""
Modeling layer: HMM walk-forward fitting and strictly filtered predictions.

This module implements the core walk-forward loop. By refitting the HMM on an
expanding window, we guarantee that the model evaluating day `t` has never 
seen data from day `t+1`. 

Crucially, it handles the three subtleties of out-of-sample HMM evaluation:
1. It uses FILTERED probabilities (forward pass only), not smoothed ones.
2. It re-ranks and maps the arbitrary state identities on every refit.
3. It fits the StandardScaler ONLY on the training window.
"""

import numpy as np
import pandas as pd
import logging
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from hmmlearn import _hmmc

from src.models.labeling import rank_and_map_states, save_fold_artifact

logger = logging.getLogger(__name__)


def predict_proba_filtered(model, X):
    """
    Computes FILTERED probabilities (forward pass only).
    
    Why this matters:
    hmmlearn's standard `predict()` (Viterbi) and `predict_proba()` (smoothed posterior
    gamma) condition on the ENTIRE sequence passed in. If you pass an out-of-sample
    window of 252 days, day 1's label will be informed by day 252's data.
    
    This function explicitly computes the forward variable (alpha), normalizing it 
    at each step to yield P(z_t | X_{1:t}). This strictly prevents any future data
    within the test window from influencing the current day's state probability.

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
    # 1. Compute emission log probabilities for all observations
    log_frameprob = model._compute_log_likelihood(X)
    
    # 2. Run the forward pass to get the forward lattice (alpha)
    # _hmmc.forward_log returns (total_log_prob, fwdlattice)
    _, fwdlattice = _hmmc.forward_log(
        model.startprob_, model.transmat_, log_frameprob)
        
    # 3. Normalize at each time step to get P(z_t | X_{1:t})
    filtered_probs = np.exp(fwdlattice - logsumexp(fwdlattice, axis=1, keepdims=True))
    
    # Sanity check: probabilities should sum to 1 across states
    assert np.allclose(filtered_probs.sum(axis=1), 1.0), "Filtered probs do not sum to 1"
    
    return filtered_probs


def walk_forward_predict(df, feature_cols, config, asset_name="spx"):
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
        Name of the asset (used for logging and saving models).

    Returns
    -------
    tuple
        (raw_labels_series, folds_info)
        raw_labels_series: pd.Series of mapped string labels ('Bull', 'Crisis', etc.), indexed identically to df.
        folds_info: list of dicts describing fold boundaries and mappings.
    """
    model_cfg = config["model"]
    n_states = model_cfg.get("n_states", 4)
    covariance_type = model_cfg.get("covariance_type", "full")
    n_iter = model_cfg.get("n_iter", 2000)
    min_train_days = model_cfg.get("min_train_days", 1008)
    
    # Identify unique years to establish boundaries
    years = df.index.year.unique()
    
    # Output arrays
    out_labels = pd.Series(index=df.index, dtype="object")
    out_probs = pd.DataFrame(index=df.index, columns=["Bull", "High-Vol Bull", "Transition", "Crisis"], dtype=float)
    folds_info = []
    
    # We will expand the training window by moving the test_year forward
    # Find the first test year such that the preceding days >= min_train_days
    first_test_year = None
    for y in years:
        train_len = len(df[df.index.year < y])
        if train_len >= min_train_days:
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
        # 1. Define windows
        train_mask = df.index.year < test_year
        test_mask = df.index.year == test_year
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if test_df.empty:
            continue
            
        X_train_raw = train_df[feature_cols].values
        X_test_raw = test_df[feature_cols].values
        
        # 2. Fit StandardScaler purely on the training window
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # 3. Fit HMM on scaled training data
        # Fix the random state so results are deterministic within the fold
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42 + fold_idx,
            tol=1e-4
        )
        model.fit(X_train)
        
        # 4. Rank and map states using the TRAINING data
        # We decode the training set using standard predict (Viterbi) since it's 
        # just for evaluating empirical state behavior historically.
        train_states = model.predict(X_train)
        state_map = rank_and_map_states(train_df, train_states, n_states)
        
        # 5. Save fold artifacts (model, scaler, state_map)
        artifact_path = save_fold_artifact(fold_idx, model, scaler, state_map, directory=f"models/{asset_name}")
        
        # 6. Out-of-sample prediction using strictly FILTERED probabilities
        test_probs = predict_proba_filtered(model, X_test)
        test_states = np.argmax(test_probs, axis=1)
        
        # 7. Map out-of-sample states to semantic labels
        test_labels = [state_map[s] for s in test_states]
        out_labels.loc[test_df.index] = test_labels

        # 8. Store out-of-sample probabilities aligned to semantic states
        for i, row_date in enumerate(test_df.index):
            for s in range(n_states):
                semantic_state = state_map[s]
                if pd.isna(out_probs.loc[row_date, semantic_state]):
                    out_probs.loc[row_date, semantic_state] = test_probs[i, s]
                else:
                    out_probs.loc[row_date, semantic_state] += test_probs[i, s]
        
        folds_info.append({
            "fold_idx": fold_idx,
            "train_end": train_df.index[-1].date(),
            "test_start": test_df.index[0].date(),
            "test_end": test_df.index[-1].date(),
            "state_map": state_map
        })
        
        fold_idx += 1

    return out_labels, out_probs, folds_info
