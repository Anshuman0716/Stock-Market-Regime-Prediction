"""
Tests for the pre-registered v2.0 pass/fail rule (src/analysis/verdict.py).
"""
import re

import pandas as pd
import pytest

from src.analysis.verdict import (
    FAIL, INTERIM, PASS, PASS_RULE, SIGNIFICANT_PASS, judge, judge_forward,
)


def _metrics(ra=(0.80, -0.20), ew=(0.70, -0.25), rp=(0.75, -0.22), years=10.0):
    """(Sharpe, Max DD) per strategy, in the stored layout."""
    rows = {"regime_aware": ra, "equal_weight": ew, "risk_parity": rp}
    return pd.DataFrame({"Sharpe": [v[0] for v in rows.values()], "Max DD": [v[1] for v in rows.values()],
                         "Years": years}, index=list(rows))


def _sig(p_ew, p_rp):
    return pd.DataFrame({"baseline": ["equal_weight", "risk_parity"], "p_value_holm": [p_ew, p_rp]})


def test_rule_text_is_the_preregistered_sentence_word_for_word():
    """The README and dashboard quote PASS_RULE; it must be exactly the
    blockquote in Section 8 of the pre-registration."""
    doc = open("docs/v2_preregistration.md", encoding="utf-8").read()
    section = doc[doc.index("## 8. The pass/fail test"):doc.index("## 9.")]
    quote = " ".join(line[2:].strip() for line in section.splitlines() if line.startswith("> "))
    assert re.sub(r"\s+", " ", quote) == PASS_RULE


def test_pass_needs_higher_sharpe_and_no_deeper_drawdown_than_both():
    assert judge(_metrics(), _sig(0.5, 0.5)).outcome == PASS


def test_beating_only_one_baseline_fails():
    v = judge(_metrics(rp=(0.85, -0.22)), _sig(0.5, 0.5))
    assert v.outcome == FAIL
    assert v.failed_conditions({"risk_parity": "Risk parity"}) == ["Sharpe not above Risk parity"]


def test_a_deeper_drawdown_fails_even_with_a_higher_sharpe():
    v = judge(_metrics(ra=(0.90, -0.30)), _sig(0.01, 0.01))
    assert v.outcome == FAIL
    assert "max drawdown deeper than equal_weight" in v.failed_conditions()


def test_a_sharpe_tie_is_not_a_pass_but_an_equal_drawdown_is_allowed():
    assert judge(_metrics(ra=(0.75, -0.20)), _sig(0.5, 0.5)).outcome == FAIL       # ties risk parity's Sharpe
    assert judge(_metrics(ra=(0.80, -0.22)), _sig(0.5, 0.5)).outcome == PASS       # equals risk parity's drawdown


def test_full_precision_no_rounding():
    """0.7500001 beats 0.75 even though both print as 0.75."""
    assert judge(_metrics(ra=(0.7500001, -0.20)), _sig(0.5, 0.5)).outcome == PASS


def test_significant_pass_needs_both_adjusted_p_values_below_alpha():
    assert judge(_metrics(), _sig(0.01, 0.04)).outcome == SIGNIFICANT_PASS
    assert judge(_metrics(), _sig(0.01, 0.05)).outcome == PASS          # 0.05 is not below 0.05
    assert judge(_metrics(), _sig(0.01, None)).outcome == PASS          # a missing test is not significance
    assert judge(_metrics(), pd.DataFrame()).outcome == PASS
    assert judge(_metrics(ra=(0.6, -0.2)), _sig(0.001, 0.001)).outcome == FAIL   # significance can't rescue a fail


def test_forward_verdict_waits_for_the_full_window():
    early = _metrics().assign(Days=100)
    v = judge_forward(early, _sig(0.01, 0.01), min_days=252)
    assert v.outcome == INTERIM and v.days == 100 and v.comparisons == []
    full = _metrics().assign(Days=252)
    assert judge_forward(full, _sig(0.01, 0.01), min_days=252).outcome == SIGNIFICANT_PASS


def test_days_come_from_years_when_no_day_count_is_stored():
    assert judge(_metrics(years=2.0), _sig(0.5, 0.5)).days == 504


@pytest.mark.parametrize("missing", ["regime_aware", "risk_parity"])
def test_missing_strategy_is_an_error_not_a_silent_pass(missing):
    with pytest.raises(KeyError):
        judge(_metrics().drop(index=missing), _sig(0.5, 0.5))
