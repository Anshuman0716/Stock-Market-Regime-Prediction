"""
Regime labeling, state mapping, and trailing smoothing.

HMM state identities are not stable across refits (EM converges to arbitrary
permutations). We must rank the states within each fold based on their economic
behavior (volatility and return) on the training set, and map them to fixed
economic labels: Bull, Transition, Crisis.

We also apply a trailing-only smoothing window to the predicted regimes. A centered
window (as used in the original build) leaks future data into today's label. By
using a trailing window, we sacrifice some responsiveness at regime turns to 
guarantee zero look-ahead bias.
"""

import os
import joblib
import pandas as pd
import numpy as np


def rank_and_map_states(train_df, train_hidden_states, n_states=4):
    """
    Rank states by volatility and mean return, merging them into Bull, Transition,
    and Crisis labels.

    Subtlety handled: THE RANKING MUST USE TRAINING DATA ONLY.
    We compute empirical stats directly from the training data's returns, masked
    by the training hidden states.

    Merge Rule:
    - Top 2 states by Sharpe Ratio (Mu/Vol) -> Bull (captures both quiet and volatile rallies)
    - 3rd state by Sharpe Ratio -> Transition (choppy/sideways)
    - Lowest state by Sharpe Ratio -> Crisis (steep drawdowns)
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing the "returns" column.
    train_hidden_states : np.ndarray
        Array of state predictions on the training set.
    n_states : int
        Number of states.

    Returns
    -------
    dict
        Mapping from state integer (0..n_states-1) to string label.
    """
    stats = []
    
    if "returns" not in train_df.columns:
        raise ValueError("train_df must contain 'returns' column for ranking.")
        
    returns = train_df["returns"]
    
    for i in range(n_states):
        mask = (train_hidden_states == i)
        state_returns = returns[mask]
        
        # In case a state has zero assignments, assign worst possible stats
        if len(state_returns) < 2:
            vol = np.inf
            mu = -np.inf
            sharpe = -np.inf
        else:
            vol = state_returns.std() * np.sqrt(252)
            mu = state_returns.mean() * 252
            sharpe = mu / vol if vol > 0 else -np.inf
            
        stats.append({"state": i, "vol": vol, "mu": mu, "sharpe": sharpe})
        
    # Sort states by volatility ascending
    stats.sort(key=lambda x: x["vol"])
    
    # Merge rule:
    # 1. Lowest volatility state -> Bull
    # 2. Highest volatility state -> Crisis
    # 3. Middle states:
    #    - If Sharpe > 0.2 -> Bull (captures volatile rallies like 2024)
    #    - If Sharpe < -0.1 -> Crisis (captures secondary selloffs)
    #    - Else -> Transition (choppy/sideways)
    
    state_map = {}
    if n_states == 4:
        state_map[stats[0]["state"]] = "Bull"
        state_map[stats[3]["state"]] = "Crisis"
        
        for i in [1, 2]:
            if stats[i]["sharpe"] > 0.2:
                state_map[stats[i]["state"]] = "Bull"
            elif stats[i]["sharpe"] < -0.1:
                state_map[stats[i]["state"]] = "Crisis"
            else:
                state_map[stats[i]["state"]] = "Transition"
    else:
        # Fallback if n_states varies
        state_map[stats[0]["state"]] = "Bull"
        state_map[stats[-1]["state"]] = "Crisis"
        for i in range(1, len(stats) - 1):
            state_map[stats[i]["state"]] = "Transition"
            
    return state_map


def smooth_regime_labels(labels_series, window_size=21):
    """
    Apply trailing-only rolling mode to regime labels.

    Why trailing-only? A centered window looks into the future (days t+1 to t+10) 
    to label day t. A trailing window (days t-20 to t) ensures that day t's label 
    is computed exclusively from data known on or before day t. This costs 
    responsiveness at regime boundaries but guarantees out-of-sample validity.

    Parameters
    ----------
    labels_series : pd.Series
        Series of categorical labels (e.g. 'Bull', 'Crisis').
    window_size : int
        Trailing window size for rolling mode.

    Returns
    -------
    pd.Series
        Smoothed regime labels.
    """
    # Map strings to integers for fast rolling mode calculation
    mapping = {"Bull": 0, "Transition": 1, "Crisis": 2}
    inv_mapping = {0: "Bull", 1: "Transition", 2: "Crisis"}
    
    numeric_series = labels_series.map(mapping)
    
    # Use pandas rolling with a custom mode function.
    # We must handle ties. scipy.stats.mode returns the smallest value in case of ties.
    # This means a tie between Bull (0) and Transition (1) will favor Bull.
    def get_mode(x):
        return pd.Series(x).mode(dropna=True).iloc[0]
        
    smoothed_numeric = numeric_series.rolling(window=window_size, min_periods=1).apply(get_mode, raw=True)
    smoothed_labels = smoothed_numeric.map(inv_mapping)
    
    return smoothed_labels


def save_fold_artifact(fold_idx, model, scaler, state_map, directory="models"):
    """
    Save the fitted model, scaler, and state_map per fold.
    A model without its scaler and mapping is unusable.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"hmm_fold_{fold_idx}.joblib")
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "state_map": state_map,
        "fold_idx": fold_idx
    }
    
    joblib.dump(artifact, filepath)
    return filepath


def load_fold_artifact(fold_idx, directory="models"):
    """Load a persisted model artifact for a specific fold."""
    filepath = os.path.join(directory, f"hmm_fold_{fold_idx}.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model artifact found at {filepath}")
    return joblib.load(filepath)
