"""
Tests for the per-fold labeling rule (src/models/labeling.py).

Each HMM state is labeled from its behaviour on the TRAINING window:
lowest volatility -> Bull, highest -> Crisis, the two middle states by
annualized Sharpe (> 0.2 Bull, < -0.1 Crisis, otherwise Transition).

The synthetic returns alternate mean + s, mean - s, so each state's mean and
volatility are known exactly and the expected labels follow from the rule.
"""
import numpy as np
import pandas as pd

from src.models.labeling import rank_and_map_states, state_statistics


def _states(spec, days=200):
    """spec: {state: (daily_mean, daily_spread)} -> (train_df, hidden_states)."""
    returns, states = [], []
    for state, (mean, spread) in spec.items():
        signs = np.where(np.arange(days) % 2 == 0, 1.0, -1.0)
        returns.append(mean + spread * signs)
        states.append(np.full(days, state))
    return pd.DataFrame({"returns": np.concatenate(returns)}), np.concatenate(states)


def test_volatility_extremes_and_sharpe_for_middle_states():
    df, hidden = _states({
        0: (0.0, 0.030),     # most volatile                -> Crisis
        1: (0.001, 0.010),   # middle, Sharpe ~ +1.6         -> Bull (volatile rally)
        2: (0.0005, 0.005),  # calmest                       -> Bull
        3: (0.0, 0.015),     # middle, Sharpe 0              -> Transition
    })
    assert rank_and_map_states(df, hidden, n_states=4) == {0: "Crisis", 1: "Bull", 2: "Bull", 3: "Transition"}


def test_middle_state_with_negative_sharpe_is_crisis():
    df, hidden = _states({
        0: (0.0, 0.030),
        1: (-0.002, 0.012),  # middle, Sharpe ~ -2.6 -> Crisis (secondary sell-off)
        2: (0.0005, 0.005),
        3: (0.0, 0.015),
    })
    assert rank_and_map_states(df, hidden, n_states=4)[1] == "Crisis"


def test_annualization_moves_a_borderline_state():
    """Daily mean/std of 0.0115 annualizes to Sharpe 0.18 with 252 days
    (Transition) but 0.22 with 365 days (Bull). This is why each asset's own
    annualization factor is passed to the labeling rule."""
    spread = 0.010
    df, hidden = _states({
        0: (0.0, 0.030),
        1: (0.0115 * spread, spread),  # the borderline middle state
        2: (0.0005, 0.005),
        3: (0.0, 0.015),
    })
    assert rank_and_map_states(df, hidden, n_states=4, ann_factor=252)[1] == "Transition"
    assert rank_and_map_states(df, hidden, n_states=4, ann_factor=365)[1] == "Bull"


def test_nearly_empty_state_can_never_be_bull():
    """A state seen on fewer than 2 training days has no measurable volatility.
    It must not be mistaken for the calm Bull state."""
    df, hidden = _states({0: (0.0, 0.030), 1: (0.001, 0.010), 2: (0.0005, 0.005)})
    df = pd.concat([df, pd.DataFrame({"returns": [0.0]})], ignore_index=True)
    hidden = np.append(hidden, 3)
    labels = rank_and_map_states(df, hidden, n_states=4)
    assert labels[3] == "Crisis"
    assert labels[2] == "Bull"


def test_state_statistics_are_exact():
    df, hidden = _states({0: (0.001, 0.01), 1: (0.0, 0.02)}, days=100)
    stats = {s["state"]: s for s in state_statistics(df, hidden, n_states=2, ann_factor=252)}
    assert stats[0]["train_days"] == 100 and stats[0]["train_share"] == 0.5
    assert np.isclose(stats[0]["mu"], 0.001 * 252)
    # Sample std of an alternating +/- s series with n=100 is s * sqrt(n / (n - 1)).
    assert np.isclose(stats[0]["vol"], 0.01 * np.sqrt(100 / 99) * np.sqrt(252))
