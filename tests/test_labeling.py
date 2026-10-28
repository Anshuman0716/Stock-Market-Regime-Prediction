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
import pytest

from src.models.labeling import asset_sharpe, rank_and_map_states, regime_weights, state_statistics


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


# ── The "relative" rule (default from Step 6) ──────────────────────────────

def _strong_uptrend_states():
    """A strongly rising asset: its own training Sharpe is high. Middle
    states: Sharpe ~1.6 and ~0.6 (both above the fixed 0.2 bar)."""
    return _states({
        0: (0.001, 0.030),    # most volatile
        1: (0.001, 0.010),    # middle, Sharpe ~ 1.6
        2: (0.0012, 0.005),   # calmest
        3: (0.0006, 0.016),   # middle, Sharpe ~ 0.6
    })


def test_absolute_rule_never_gives_transition_in_a_strong_uptrend():
    df, hidden = _strong_uptrend_states()
    labels = rank_and_map_states(df, hidden, n_states=4, rule="absolute")
    assert "Transition" not in labels.values()  # a weak-but-positive state still counts as Bull


def test_relative_rule_judges_middle_states_against_the_asset_itself():
    df, hidden = _strong_uptrend_states()
    s_asset = asset_sharpe(df, 252)
    assert 0.6 < s_asset < 1.6  # the asset's own bar sits between the two middle states
    labels = rank_and_map_states(df, hidden, n_states=4, rule="relative")
    assert labels == {0: "Crisis", 1: "Bull", 2: "Bull", 3: "Transition"}


def test_rules_agree_when_the_asset_itself_is_weak():
    """With a training Sharpe below bull_sharpe (the S&P 500 in 2000-09) the
    bar stays at 0.2, so both rules give the same labels."""
    df, hidden = _states({
        0: (-0.002, 0.030),
        1: (0.001, 0.010),
        2: (0.0005, 0.005),
        3: (0.0, 0.015),
    })
    assert asset_sharpe(df, 252) < 0.2
    assert (rank_and_map_states(df, hidden, n_states=4, rule="relative")
            == rank_and_map_states(df, hidden, n_states=4, rule="absolute"))


def test_relative_rule_still_sends_losing_middle_states_to_crisis():
    df, hidden = _states({0: (0.001, 0.030), 1: (-0.002, 0.012), 2: (0.0012, 0.005), 3: (0.0006, 0.016)})
    assert rank_and_map_states(df, hidden, n_states=4, rule="relative")[1] == "Crisis"


def test_unknown_rule_is_rejected():
    df, hidden = _strong_uptrend_states()
    with pytest.raises(ValueError, match="labeling rule"):
        rank_and_map_states(df, hidden, n_states=4, rule="mode")


# ── The "sharpe" rule (default from Step 7): anchors by earnings, not calm ───

def _gold_like_states():
    """Calmest state is a quiet drift that earns less than holding the asset;
    the rally happens in a more volatile state."""
    return _states({
        0: (0.0001, 0.005),   # calmest: quiet drift, Sharpe ~ 0.3
        1: (0.005, 0.015),    # rally, Sharpe ~ 5.3
        2: (-0.002, 0.010),   # sell-off, Sharpe ~ -3.2
        3: (0.0001, 0.025),   # most volatile, but directionless: Sharpe ~ 0.06
    })


def test_sharpe_rule_does_not_call_a_quiet_drift_bull():
    df, hidden = _gold_like_states()
    assert 0.3 < asset_sharpe(df, 252)  # the drift earns less than holding the asset
    assert rank_and_map_states(df, hidden, n_states=4, rule="sharpe") == {
        0: "Transition", 1: "Bull", 2: "Crisis", 3: "Transition"}
    # The volatility anchor calls the drift Bull and the directionless state Crisis.
    assert rank_and_map_states(df, hidden, n_states=4, rule="relative") == {
        0: "Bull", 1: "Bull", 2: "Crisis", 3: "Crisis"}


def test_sharpe_rule_matches_volatility_anchors_for_stock_like_states():
    """Stocks: the calmest state earns the most and the most volatile the
    least, so both anchoring choices give the same labels."""
    df, hidden = _states({
        0: (0.0008, 0.006),    # calm rally
        1: (0.0001, 0.011),    # choppy
        2: (-0.0003, 0.016),   # weak
        3: (-0.003, 0.035),    # crash
    })
    assert (rank_and_map_states(df, hidden, n_states=4, rule="sharpe")
            == rank_and_map_states(df, hidden, n_states=4, rule="relative"))


def test_sharpe_rule_always_has_a_bull_and_a_crisis_state():
    rng = np.random.default_rng(3)
    for _ in range(20):
        spec = {k: (rng.normal(0, 0.002), rng.uniform(0.004, 0.03)) for k in range(4)}
        df, hidden = _states(spec)
        labels = set(rank_and_map_states(df, hidden, n_states=4, rule="sharpe").values())
        assert {"Bull", "Crisis"} <= labels


# ── Regime weights: how much to hold in each regime (Step 8) ─────────────────

def test_regime_weights_hold_regimes_that_earned_money_in_training():
    df, hidden = _gold_like_states()
    labels = rank_and_map_states(df, hidden, n_states=4, rule="sharpe")
    w = regime_weights(df, hidden, labels, sizing="regime_returns")
    # Bull (rally) earned money, Crisis (sell-off) lost it, Transition (quiet
    # drift + directionless volatile state) earned a little: hold it.
    assert w == {"Bull": 1.0, "Transition": 1.0, "Crisis": 0.0}


def test_regime_weights_skip_a_losing_transition_regime():
    df, hidden = _states({0: (0.0008, 0.006), 1: (-0.00005, 0.011), 2: (-0.0003, 0.016), 3: (-0.003, 0.035)})
    labels = {0: "Bull", 1: "Transition", 2: "Crisis", 3: "Crisis"}
    assert regime_weights(df, hidden, labels, sizing="regime_returns") == {"Bull": 1.0, "Transition": 0.0,
                                                                            "Crisis": 0.0}


def test_p_bull_sizing_and_unknown_rule():
    df, hidden = _gold_like_states()
    labels = rank_and_map_states(df, hidden, n_states=4, rule="sharpe")
    assert regime_weights(df, hidden, labels, sizing="p_bull") == {"Bull": 1.0, "Transition": 0.0, "Crisis": 0.0}
    with pytest.raises(ValueError, match="sizing rule"):
        regime_weights(df, hidden, labels, sizing="kelly")
