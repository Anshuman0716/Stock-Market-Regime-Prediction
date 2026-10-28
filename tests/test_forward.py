"""
Tests for the forward test (src/analysis/forward.py): the freeze record, its
validity check, and scoring only the days after the freeze.
"""
import json

import numpy as np
import pandas as pd

from src.analysis.forward import (
    config_fingerprint, forward_start, freeze_status, freeze_status_for, historical_end, load_freeze,
    load_freezes,
)
from src.db.store import get_config_hash, get_portfolio_hash
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
    # Records written before freeze.json became a list count as per-asset versions.
    assert load_freeze(str(p)) == {"scope": "assets", **FREEZE}


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



# --- several frozen versions -------------------------------------------------

CFG = {"model": {"n_states": 4}, "portfolio": {"assets": ["spx", "gold"]}}
V1 = {"version": "v1.0", "scope": "assets", "frozen_on": "2026-10-28", "data_end_at_freeze": "2026-10-27",
      "config_hash": get_config_hash(CFG)}
V2 = {"version": "v2.0", "scope": "portfolio", "frozen_on": "2026-10-28", "data_end_at_freeze": "2026-10-27",
      "config_hash": get_portfolio_hash(CFG)}


def test_load_freezes_reads_the_list_and_the_old_single_record(tmp_path):
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps({"versions": [V1, V2]}))
    assert [r["version"] for r in load_freezes(str(p))] == ["v1.0", "v2.0"]
    assert load_freeze(str(p))["version"] == "v1.0"                 # default scope: the per-asset version
    assert load_freeze(str(p), scope="portfolio")["version"] == "v2.0"

    p.write_text(json.dumps(FREEZE))                                  # v1.0's original single-record layout
    assert load_freezes(str(p)) == [{"scope": "assets", **FREEZE}]
    assert load_freeze(str(p), scope="portfolio") is None
    assert load_freezes(str(tmp_path / "missing.json")) == []


def test_portfolio_section_is_outside_the_per_asset_fingerprint():
    without = {k: v for k, v in CFG.items() if k != "portfolio"}
    assert get_config_hash(CFG) == get_config_hash(without)
    assert get_portfolio_hash(CFG) != get_portfolio_hash(without)
    assert config_fingerprint(CFG, "assets") == get_config_hash(CFG)
    assert config_fingerprint(CFG, "portfolio") == get_portfolio_hash(CFG)


def test_each_version_is_checked_against_its_own_fingerprint():
    assert freeze_status_for(V1, CFG)[0] == "valid" and freeze_status_for(V2, CFG)[0] == "valid"

    # Editing only the portfolio rules invalidates v2.0 but leaves v1.0 valid.
    edited = {**CFG, "portfolio": {"assets": ["spx"]}}
    assert freeze_status_for(V1, edited)[0] == "valid"
    assert freeze_status_for(V2, edited)[0] == "invalid"

    # Editing the models invalidates both: v2.0 is built on v1.0's signals.
    retrained = {**CFG, "model": {"n_states": 6}}
    assert freeze_status_for(V1, retrained)[0] == "invalid"
    assert freeze_status_for(V2, retrained)[0] == "invalid"
    assert freeze_status_for(None, CFG)[0] == "none"


def test_historical_window_ends_where_the_forward_window_starts():
    assert historical_end(V2) == pd.Timestamp("2026-10-27")
    assert forward_start(V2) == historical_end(V2) + pd.Timedelta(days=1)
    assert historical_end(None) is None
