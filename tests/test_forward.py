"""
Tests for the forward test (src/analysis/forward.py): the freeze record, its
validity check, and scoring only the days after the freeze.
"""
import json

import numpy as np
import pandas as pd

from src.analysis.forward import forward_start, freeze_status, load_freeze
from src.analysis.regimes import period_breakdown

FREEZE = {"version": "v1.0", "frozen_on": "2026-10-28", "data_end_at_freeze": "2026-10-27", "config_hash": "abc"}


def test_freeze_status():
    assert freeze_status(None, "abc")[0] == "none"
    assert freeze_status(FREEZE, "abc")[0] == "valid"
    status, message = freeze_status(FREEZE, "changed")
    assert status == "invalid" and "no longer a forward test" in message


def test_forward_window_starts_after_the_last_data_the_design_saw():
    assert forward_start(FREEZE) == pd.Timestamp("2026-10-28")
    assert forward_start(None) is None


def test_load_freeze(tmp_path):
    assert load_freeze(str(tmp_path / "missing.json")) is None
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps(FREEZE))
    assert load_freeze(str(p)) == FREEZE


def test_only_days_after_the_freeze_are_scored():
    idx = pd.bdate_range("2026-10-20", periods=10)  # 5 days up to 10-26 ... then 10-27 onward
    bt = pd.DataFrame({"net_return": np.full(10, 0.01), "executed_weight": 1.0, "cash_return": 0.0}, index=idx)
    out = {(s, c): m for s, c, m in period_breakdown({"hmm": bt}, 252, forward_start=forward_start(FREEZE))}
    fwd = out[("hmm", "forward")]
    after = idx[idx >= pd.Timestamp("2026-10-28")]
    assert fwd["Days"] == len(after)
    assert np.isclose(fwd["Total Return"], 1.01 ** len(after) - 1)


def test_no_forward_cycle_before_any_new_day():
    idx = pd.bdate_range("2026-10-01", "2026-10-27")
    bt = pd.DataFrame({"net_return": 0.0, "executed_weight": 1.0, "cash_return": 0.0}, index=idx)
    cycles = {c for _, c, _ in period_breakdown({"hmm": bt}, 252, forward_start=forward_start(FREEZE))}
    assert "forward" not in cycles
