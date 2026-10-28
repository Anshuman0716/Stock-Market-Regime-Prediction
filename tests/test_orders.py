"""Tests for paper-trading order sizing (src/trading/orders.py)."""
import pytest

from src.trading.orders import plan_rebalance


def test_first_entry_is_capped_not_refused():
    """Going from 0% to 100% with a 50% cap buys half now (the old guardrail
    refused the trade outright, so a new account could never enter)."""
    plan = plan_rebalance(target_weight=1.0, current_qty=0, price=500.0, account_value=100_000,
                          max_weight_change=0.5)
    assert plan.capped and plan.new_weight == pytest.approx(0.5)
    assert plan.qty == 100 and plan.side == "buy"


def test_small_move_goes_all_the_way():
    plan = plan_rebalance(0.6, current_qty=100, price=500.0, account_value=100_000, max_weight_change=0.5)
    assert not plan.capped and plan.qty == 20 and plan.side == "buy"


def test_selling_and_no_trade():
    sell = plan_rebalance(0.0, current_qty=100, price=500.0, account_value=100_000, max_weight_change=1.0)
    assert sell.qty == -100 and sell.side == "sell"
    hold = plan_rebalance(0.5, current_qty=100, price=500.0, account_value=100_000)
    assert hold.qty == 0 and hold.side == "none"


def test_whole_shares_never_overshoot_the_target():
    plan = plan_rebalance(0.5, current_qty=0, price=333.0, account_value=100_000)
    assert plan.qty == 150  # 150.15 shares rounded down, never up
    assert plan.qty * 333.0 <= 0.5 * 100_000


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        plan_rebalance(0.5, 0, price=0.0, account_value=100_000)
    with pytest.raises(ValueError):
        plan_rebalance(1.5, 0, price=100.0, account_value=100_000)
