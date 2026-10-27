"""
Regime labeling and state mapping (plus fold artifact persistence).

HMM state identities are not stable across refits (EM converges to arbitrary
permutations). We must rank the states within each fold based on their economic
behavior (volatility and return) on the training set, and map them to fixed
economic labels: Bull, Transition, Crisis.

Smoothing of the predicted regimes lives in src/models/signal.py, the single
definition shared by the backtest, store, dashboard and paper trading.
"""

import os
import joblib
import numpy as np


def rank_and_map_states(train_df, train_hidden_states, n_states=4, ann_factor=252):
    """
    Rank states by volatility and annualized Sharpe, merging them into Bull,
    Transition, and Crisis labels.

    Subtlety handled: THE RANKING MUST USE TRAINING DATA ONLY.
    We compute empirical stats directly from the training data's returns, masked
    by the training hidden states.

    Merge rule (4 states):
    - Lowest-volatility state -> Bull (calm, steadily rising markets)
    - Highest-volatility state -> Crisis (stress, steep drawdowns)
    - The two middle states are judged by annualized Sharpe:
        Sharpe > 0.2  -> Bull       (volatile rallies)
        Sharpe < -0.1 -> Crisis     (secondary sell-offs)
        otherwise     -> Transition (choppy / sideways)
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data containing the "returns" column.
    train_hidden_states : np.ndarray
        Array of state predictions on the training set.
    n_states : int
        Number of states.
    ann_factor : int
        Rows per year for this asset (252 equities, 365 crypto). The Sharpe
        thresholds above are annualized, so the wrong factor would move
        states across the Bull/Transition/Crisis boundaries.

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
            vol = state_returns.std() * np.sqrt(ann_factor)
            mu = state_returns.mean() * ann_factor
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


def save_fold_artifact(fold_idx, model, scaler, state_map, directory="models", train_end=None):
    """
    Save the fitted model, scaler, and state_map per fold.
    A model without its scaler and mapping is unusable. `train_end` (last
    training date) lets live code warn when the latest model is stale.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"hmm_fold_{fold_idx}.joblib")
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "state_map": state_map,
        "fold_idx": fold_idx,
        "train_end": None if train_end is None else str(train_end),
    }
    
    joblib.dump(artifact, filepath)
    return filepath


def load_fold_artifact(fold_idx, directory="models"):
    """Load a persisted model artifact for a specific fold."""
    filepath = os.path.join(directory, f"hmm_fold_{fold_idx}.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model artifact found at {filepath}")
    return joblib.load(filepath)
