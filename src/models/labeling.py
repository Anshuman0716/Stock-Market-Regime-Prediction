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


LABELING_RULES = ("sharpe", "relative", "absolute")


def asset_sharpe(train_df, ann_factor):
    """Annualized Sharpe of the asset over the whole training window."""
    r = train_df["returns"].dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann_factor))


def _by_sharpe(st, bull_bar, crisis_sharpe):
    """Label one state from its training Sharpe."""
    if st["sharpe"] < crisis_sharpe:
        return "Crisis"
    if st["sharpe"] > bull_bar:
        return "Bull"
    return "Transition"


def rank_and_map_states(train_df, train_hidden_states, n_states=4, ann_factor=252,
                        rule="absolute", bull_sharpe=0.2, crisis_sharpe=-0.1):
    """
    Map each raw HMM state to Bull, Transition or Crisis from its behaviour
    on the TRAINING window only.

    Every rule first fixes two anchors so that Bull and Crisis always
    exist, then judges the remaining states by annualized Sharpe.

    rule="sharpe" (default from Step 7) - anchors by what a state EARNED:
        highest-Sharpe state -> Bull, lowest-Sharpe state -> Crisis;
        others: Sharpe < crisis_sharpe -> Crisis,
                Sharpe > max(bull_sharpe, S_asset) -> Bull,
                otherwise -> Transition.
    rule="relative" (Step 6) - anchors by volatility:
        calmest state -> Bull, most volatile -> Crisis; others as above.
    rule="absolute" (up to Step 5) - volatility anchors, and the middle
        states of a 4-state model against fixed bars only
        (> bull_sharpe Bull, < crisis_sharpe Crisis, else Transition).
    S_asset is the asset's own Sharpe over the training window.

    Why anchor by Sharpe, not volatility: "the calmest state is a bull
    market" is true for stocks, where prices fall fast and rise slowly. It
    is not true for Gold or Bitcoin, whose rallies are volatile. On Gold's
    training windows the calmest state is a quiet drift that earns LESS
    than simply holding gold, while its rallies sit in a more volatile
    state; the volatility anchor called that drift "Bull" just for being
    calm. Ranking by Sharpe labels a state by its returns, which is what
    the strategy trades on. For stocks the calmest state is usually also
    the best-earning one, so the labels barely change there.

    Why judge against S_asset: a fixed 0.2 bar means very different things
    for a flat asset and a strongly trending one. "Bull" should mean better
    than simply holding the asset; positive-but-worse is Transition.

    All rules use training data only. Each new rule was fixed from
    training-window diagnostics before its out-of-sample results were seen,
    and the older rules stay available so results can be compared.

    Parameters
    ----------
    train_df : pd.DataFrame with a "returns" column (training window).
    train_hidden_states : np.ndarray of each training day's state.
    n_states : int
    ann_factor : int
        Rows per year (252 equities, 365 crypto); the thresholds are annualized.
    rule : "sharpe", "relative" or "absolute"
    bull_sharpe, crisis_sharpe : float
        Annualized Sharpe thresholds (config: model.labeling).

    Returns
    -------
    dict mapping state integer -> label.
    """
    if rule not in LABELING_RULES:
        raise ValueError(f"Unknown labeling rule '{rule}'; choose from {LABELING_RULES}.")
    stats = state_statistics(train_df, train_hidden_states, n_states, ann_factor)

    if rule == "sharpe":
        stats.sort(key=lambda x: x["sharpe"])  # worst-earning first
        state_map = {stats[-1]["state"]: "Bull", stats[0]["state"]: "Crisis"}
    else:
        stats.sort(key=lambda x: x["vol"])  # calmest first
        state_map = {stats[0]["state"]: "Bull", stats[-1]["state"]: "Crisis"}
    others = stats[1:-1]

    if rule == "absolute":
        # Exactly the pre-Step-6 behaviour, kept for comparison: only the
        # 4-state model judged middle states by Sharpe; other sizes called
        # every middle state Transition.
        for st in others:
            if n_states == 4 and st["sharpe"] > bull_sharpe:
                state_map[st["state"]] = "Bull"
            elif n_states == 4 and st["sharpe"] < crisis_sharpe:
                state_map[st["state"]] = "Crisis"
            else:
                state_map[st["state"]] = "Transition"
        return state_map

    bull_bar = max(bull_sharpe, asset_sharpe(train_df, ann_factor))
    for st in others:
        state_map[st["state"]] = _by_sharpe(st, bull_bar, crisis_sharpe)
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
