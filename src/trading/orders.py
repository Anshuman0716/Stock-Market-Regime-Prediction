"""
Turning a target weight into an order.

Kept separate from scripts/paper_trade.py (which talks to the broker) so the
arithmetic can be tested without a network or an account.
"""
from dataclasses import dataclass


@dataclass
class RebalancePlan:
    current_weight: float   # share of the account held in the asset now
    target_weight: float    # what the signal asks for
    new_weight: float       # what this order moves to (after the step cap)
    capped: bool            # True if the step cap limited the move
    qty: int                # shares to trade: + buy, - sell, 0 nothing
    side: str               # "buy", "sell" or "none"


def plan_rebalance(target_weight, current_qty, price, account_value, max_weight_change=1.0):
    """
    Size the order that moves the account towards `target_weight`.

    Why cap the step instead of refusing a big move: an earlier version
    refused any day where the weight had to change by more than the cap.
    That blocked the very first trade (0% -> 100%) forever, so a new account
    could never enter the market. Capping moves at most `max_weight_change`
    per run, so a large swing (often a data glitch) is spread over several
    days but still happens.

    Quantities are whole shares, rounded towards zero, so the account never
    buys more than the target or sells more than it holds.
    """
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}.")
    if account_value <= 0:
        raise ValueError(f"Account value must be positive, got {account_value}.")
    if not 0.0 <= target_weight <= 1.0:
        raise ValueError(f"Target weight must be between 0 and 1, got {target_weight}.")

    current_weight = current_qty * price / account_value
    wanted = target_weight - current_weight
    step = max(-max_weight_change, min(max_weight_change, wanted))
    new_weight = current_weight + step

    qty = int(account_value * new_weight / price - current_qty)  # int() rounds towards zero
    side = "buy" if qty > 0 else "sell" if qty < 0 else "none"
    return RebalancePlan(current_weight=current_weight, target_weight=target_weight,
                         new_weight=new_weight, capped=abs(wanted) > max_weight_change,
                         qty=qty, side=side)
