"""
The v2.0 pass/fail rule, applied to stored results.

Why this lives in src/: the README and the dashboard both show the verdict.
If each decided "pass" on its own, two slightly different readings of the
rule could disagree. Here the rule is written once, exactly as
pre-registered (docs/v2_preregistration.md, Section 8), and both read it.

The rule, and only the rule:
- Pass: the regime-aware portfolio's Sharpe ratio (above cash) is strictly
  higher than BOTH baselines', and its maximum drawdown is no deeper than
  EITHER baseline's. Full-precision numbers, no tolerance band: a Sharpe tie
  fails, an exactly equal drawdown is allowed.
- Significant pass: a pass in which both Sharpe differences also have a
  Holm-adjusted p-value below alpha (0.05).
- Fail: anything else.

The forward window has one more state, Interim: before forward_min_days
forward trading days there is no verdict at all (Section 10).
"""
from dataclasses import dataclass, field

import pandas as pd

from src.backtest.portfolio import PORTFOLIO_BASELINES, PORTFOLIO_STRATEGY

# Word for word from docs/v2_preregistration.md, Section 8 (checked by
# tests/test_verdict.py). Shown verbatim next to every verdict.
PASS_RULE = (
    "**Pass:** the regime-aware portfolio has a strictly higher Sharpe ratio above cash than both "
    "equal weight and plain risk parity, and a maximum drawdown no deeper than either of them. "
    "**Significant pass:** a pass in which both Sharpe differences also have a Holm-adjusted p-value "
    "below 0.05."
)

FAIL = "Fail"
PASS = "Pass"
SIGNIFICANT_PASS = "Significant pass"
INTERIM = "Interim"


@dataclass
class Comparison:
    """The regime-aware portfolio against one baseline."""
    baseline: str
    sharpe: float
    baseline_sharpe: float
    max_dd: float            # negative: -0.20 is a 20% drawdown
    baseline_max_dd: float
    p_holm: float = None     # None when no test was run

    @property
    def higher_sharpe(self):
        return self.sharpe > self.baseline_sharpe          # strictly: a tie is not a pass

    @property
    def drawdown_ok(self):
        return self.max_dd >= self.baseline_max_dd        # "no deeper": equal is allowed

    def significant(self, alpha):
        return self.p_holm is not None and self.p_holm < alpha


@dataclass
class Verdict:
    outcome: str                                  # Fail / Pass / Significant pass / Interim
    window: str                                   # "historical" or "forward"
    days: int
    comparisons: list = field(default_factory=list)
    min_days: int = None                          # forward only: days needed before a verdict

    def failed_conditions(self, names=None):
        """Plain-language list of the conditions that were not met (empty on a pass).
        `names` maps baseline ids to display names."""
        names = names or {}
        out = []
        for c in self.comparisons:
            name = names.get(c.baseline, c.baseline)
            if not c.higher_sharpe:
                out.append(f"Sharpe not above {name}")
            if not c.drawdown_ok:
                out.append(f"max drawdown deeper than {name}")
        return out


def judge(metrics, significance, alpha=0.05, window="historical", strategy=PORTFOLIO_STRATEGY,
          baselines=PORTFOLIO_BASELINES):
    """
    Apply the pre-registered rule to one window's stored results.

    Parameters
    ----------
    metrics : pd.DataFrame
        Index = portfolio strategy, columns include "Sharpe" and "Max DD"
        (and "Years" or "Days" for the window length), for ONE cycle.
    significance : pd.DataFrame
        That window's Holm family: rows with "baseline" and "p_value_holm".
        May be empty (no tests run), which rules out a significant pass.
    """
    p = {} if significance is None or significance.empty else \
        significance.set_index("baseline")["p_value_holm"].to_dict()
    comparisons = [
        Comparison(baseline=b,
                   sharpe=float(metrics.loc[strategy, "Sharpe"]), baseline_sharpe=float(metrics.loc[b, "Sharpe"]),
                   max_dd=float(metrics.loc[strategy, "Max DD"]), baseline_max_dd=float(metrics.loc[b, "Max DD"]),
                   p_holm=None if p.get(b) is None or pd.isna(p.get(b)) else float(p[b]))
        for b in baselines
    ]
    passed = all(c.higher_sharpe and c.drawdown_ok for c in comparisons)
    if not passed:
        outcome = FAIL
    elif all(c.significant(alpha) for c in comparisons):
        outcome = SIGNIFICANT_PASS
    else:
        outcome = PASS
    return Verdict(outcome=outcome, window=window, days=_days(metrics, strategy), comparisons=comparisons)


def judge_forward(metrics, significance, min_days, alpha=0.05, strategy=PORTFOLIO_STRATEGY,
                  baselines=PORTFOLIO_BASELINES):
    """The same rule on the forward window, or Interim before min_days forward days."""
    days = _days(metrics, strategy)
    if days < min_days:
        return Verdict(outcome=INTERIM, window="forward", days=days, min_days=min_days)
    v = judge(metrics, significance, alpha, window="forward", strategy=strategy, baselines=baselines)
    v.min_days = min_days
    return v


def _days(metrics, strategy):
    row = metrics.loc[strategy]
    if "Days" in row and pd.notna(row["Days"]):
        return int(row["Days"])
    return int(round(float(row["Years"]) * 252))
