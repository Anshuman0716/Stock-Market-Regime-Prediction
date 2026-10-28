"""
Tests for the SQLite store (src/db/store.py): value conversion, idempotent
upserts and the rebuild of an out-of-date schema.
"""
import datetime
import sqlite3

import numpy as np
import pandas as pd

from src.db.store import (
    SCHEMA_VERSION, _to_sql_value, get_db_connection, init_db, write_frame, write_portfolio_weights,
    write_significance,
)


def test_values_are_converted_to_plain_sqlite_types():
    assert _to_sql_value(np.nan) is None          # missing stays missing, never a fake number
    assert _to_sql_value(np.inf) is None
    assert _to_sql_value(np.float64(1.5)) == 1.5
    assert _to_sql_value(np.int64(7)) == 7 and isinstance(_to_sql_value(np.int64(7)), int)
    assert _to_sql_value(np.bool_(True)) == 1
    assert _to_sql_value(pd.Timestamp("2020-03-23 15:30")) == "2020-03-23"
    assert _to_sql_value(datetime.date(2020, 3, 23)) == "2020-03-23"
    assert _to_sql_value("Bull") == "Bull"


def test_fresh_database_gets_current_schema(tmp_path):
    db = str(tmp_path / "store.db")
    init_db(db)
    with sqlite3.connect(db) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"regimes", "backtest_daily", "backtest_metrics", "regime_spells"} <= tables


def test_outdated_database_is_rebuilt(tmp_path):
    """An old layout must be replaced, not silently kept by CREATE TABLE IF NOT EXISTS."""
    db = str(tmp_path / "old.db")
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE regimes (asset_id TEXT, date TEXT, label TEXT)")
        c.execute("INSERT INTO regimes VALUES ('spx', '2020-01-02', 'High-Vol Bull')")
        c.execute("PRAGMA user_version = 3")
    init_db(db)
    with sqlite3.connect(db) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        cols = {r[1] for r in c.execute("PRAGMA table_info(regimes)")}
        assert {"sp_bull", "target_weight", "run_id"} <= cols
        assert c.execute("SELECT COUNT(*) FROM regimes").fetchone()[0] == 0


def test_rewriting_rows_updates_instead_of_duplicating(tmp_path):
    db = str(tmp_path / "store.db")
    init_db(db)
    with get_db_connection(db) as conn:
        write_frame(conn, "assets", pd.DataFrame([{"asset_id": "spx", "ticker": "^GSPC", "name": "old"}]),
                    ["asset_id"])
        write_frame(conn, "assets", pd.DataFrame([{"asset_id": "spx", "ticker": "^GSPC", "name": "new"}]),
                    ["asset_id"])
        rows = conn.execute("SELECT asset_id, name FROM assets").fetchall()
    assert rows == [("spx", "new")]


def test_nan_metric_is_stored_as_null(tmp_path):
    db = str(tmp_path / "store.db")
    init_db(db)
    row = {"run_id": "r", "asset_id": "spx", "strategy": "hmm", "cycle": "full",
           "metric_name": "Calmar", "metric_value": np.nan}
    with get_db_connection(db) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        write_frame(conn, "backtest_metrics", pd.DataFrame([row]),
                    ["run_id", "asset_id", "strategy", "cycle", "metric_name"])
        value = conn.execute("SELECT metric_value FROM backtest_metrics").fetchone()[0]
    assert value is None


def test_portfolio_weights_round_trip_and_rewrite_without_duplicates(tmp_path):
    db = str(tmp_path / "store.db")
    init_db(db)
    dates = pd.bdate_range("2021-01-04", periods=3)
    held = pd.DataFrame({"spx": [0.0, 0.6, 0.5], "gold": [0.0, 0.4, 0.25]}, index=dates)
    with get_db_connection(db) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        write_portfolio_weights(conn, held, "r", "regime_aware")
        write_portfolio_weights(conn, held, "r", "regime_aware")   # same run again: update, not duplicate
        rows = pd.read_sql("SELECT date, asset_id, weight FROM portfolio_weights ORDER BY date, asset_id", conn)
    assert len(rows) == 6
    back = rows.pivot(index="date", columns="asset_id", values="weight")
    np.testing.assert_allclose(back[["spx", "gold"]].to_numpy(), held.to_numpy())


def test_significance_family_defaults_to_asset(tmp_path):
    """Results from code that predates the portfolio (no 'family' column)
    are stored as the per-asset family."""
    db = str(tmp_path / "store.db")
    init_db(db)
    sig = pd.DataFrame([{"asset": "spx", "strategy": "hmm", "baseline": "buy_hold", "sharpe_diff": 0.1,
                         "ci_lower": -0.1, "ci_upper": 0.3, "p_value": 0.4, "p_value_hac": 0.5,
                         "p_value_holm": 0.8}])
    with get_db_connection(db) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        write_significance(conn, sig, "r")
        family = conn.execute("SELECT family FROM significance_tests").fetchone()[0]
    assert family == "asset"
