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


def state_statistics(train_df, train_hidden_states, n_states, ann_factor=252):
    """
    Empirical behaviour of each raw HMM state on the TRAINING window.

    For every state: how many training days it covers, and the annualized
    mean return, volatility and Sharpe of the asset on those days. These
    numbers drive the labeling rule below and are also stored so the
    dashboard can show what each fold's states looked like.

    A state with fewer than 2 training days gets vol=inf, mean=-inf,
    Sharpe=-inf so it can never be ranked as the calm Bull state.
    """
    if "returns" not in train_df.columns:
        raise ValueError("train_df must contain 'returns' column for ranking.")
    returns = train_df["returns"]
    stats = []
    for i in range(n_states):
        state_returns = returns[train_hidden_states == i]
        if len(state_returns) < 2:
            vol, mu, sharpe = np.inf, -np.inf, -np.inf
        else:
            vol = state_returns.std() * np.sqrt(ann_factor)
            mu = state_returns.mean() * ann_factor
            sharpe = mu / vol if vol > 0 else -np.inf
        stats.append({
            "state": i,
            "train_days": int(len(state_returns)),
            "train_share": float(len(state_returns) / max(len(returns), 1)),
            "vol": float(vol),
            "mu": float(mu),
            "sharpe": float(sharpe),
        })
    return stats


LABELING_RULES = ("relative", "absolute")


def asset_sharpe(train_df, ann_factor):
    """Annualized Sharpe of the asset over the whole training window."""
    r = train_df["returns"].dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann_factor))


def rank_and_map_states(train_df, train_hidden_states, n_states=4, ann_factor=252,
                        rule="absolute", bull_sharpe=0.2, crisis_sharpe=-0.1):
    """
    Map each raw HMM state to Bull, Transition or Crisis from its behaviour
    on the TRAINING window only.

    Both rules share the volatility anchors:
    - lowest-volatility state  -> Bull   (calm, steadily rising markets)
    - highest-volatility state -> Crisis (stress, steep drawdowns)
    and judge the states in between by annualized Sharpe:

    rule="absolute" (used up to Step 5):
        Sharpe > bull_sharpe (0.2)   -> Bull
        Sharpe < crisis_sharpe (-0.1)-> Crisis
        otherwise                    -> Transition

    rule="relative" (default from Step 6):
        Sharpe < crisis_sharpe            -> Crisis
        Sharpe > max(bull_sharpe, S_asset)-> Bull
        otherwise                         -> Transition
      where S_asset is the asset's own Sharpe over the training window.

    Why "relative": a fixed 0.2 bar means very different things for
    different assets. For an asset with a strong long-run trend, a middle
    state that is positive but clearly worse than simply holding the asset
    is not a "Bull" market for that asset. The relative rule asks "is this
    state better than holding the asset over the training years?"; a state
    that is positive but worse is Transition. For an asset whose training
    Sharpe is below bull_sharpe (the S&P 500 in 2000-09) the rules agree.
    The rule was fixed before its out-of-sample results were seen and uses
    training data only.

    What it does NOT do: give Gold and Bitcoin a Transition regime. For
    those assets the HMM's two middle states turned out to be a strong
    rally state and a sell-off state (training Sharpe far above any bar and
    far below zero) in every fold, so no threshold on Sharpe makes either
    of them "sideways". A Transition regime there would need the model to
    find a sideways state, not a different label for the states it finds.

    Parameters
    ----------
    train_df : pd.DataFrame with a "returns" column (training window).
    train_hidden_states : np.ndarray of each training day's state.
    n_states : int
    ann_factor : int
        Rows per year (252 equities, 365 crypto); the thresholds are annualized.
    rule : "relative" or "absolute"
    bull_sharpe, crisis_sharpe : float
        Annualized Sharpe thresholds (config: model.labeling).

    Returns
    -------
    dict mapping state integer -> label.
    """
    if rule not in LABELING_RULES:
        raise ValueError(f"Unknown labeling rule '{rule}'; choose from {LABELING_RULES}.")
    stats = state_statistics(train_df, train_hidden_states, n_states, ann_factor)
    stats.sort(key=lambda x: x["vol"])  # calmest first

    state_map = {stats[0]["state"]: "Bull", stats[-1]["state"]: "Crisis"}
    middle = stats[1:-1]

    if rule == "absolute":
        # Exactly the pre-Step-6 behaviour, kept for comparison: only the
        # 4-state model judged middle states by Sharpe; other sizes called
        # every middle state Transition.
        for st in middle:
            if n_states == 4 and st["sharpe"] > bull_sharpe:
                state_map[st["state"]] = "Bull"
            elif n_states == 4 and st["sharpe"] < crisis_sharpe:
                state_map[st["state"]] = "Crisis"
            else:
                state_map[st["state"]] = "Transition"
        return state_map

    bull_bar = max(bull_sharpe, asset_sharpe(train_df, ann_factor))
    for st in middle:
        if st["sharpe"] < crisis_sharpe:
            state_map[st["state"]] = "Crisis"
        elif st["sharpe"] > bull_bar:
            state_map[st["state"]] = "Bull"
        else:
            state_map[st["state"]] = "Transition"
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
