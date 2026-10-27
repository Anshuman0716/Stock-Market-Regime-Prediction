"""
SQLite store: schema management and idempotent writers.

Every writer upserts on the table's primary key, so writing the same run
twice never duplicates rows. The pipeline builds a fresh database file and
swaps it in when complete (see src/pipeline.py), so readers never see a
half-written store.
"""
import hashlib
import json
import math
import os
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

# Must match "PRAGMA user_version" in schema.sql. Bump both together whenever
# a table's columns or keys change.
SCHEMA_VERSION = 4

DEFAULT_DB_PATH = "data/regime_store.db"


def get_git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"


def get_config_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode('utf-8')).hexdigest()


@contextmanager
def get_db_connection(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path=DEFAULT_DB_PATH):
    """
    Create the tables, rebuilding the database if its schema is out of date.

    Why rebuild instead of migrate: everything in the store is derived from
    the pipeline (prices are re-downloadable, the rest is recomputed), so the
    simplest correct upgrade is drop-and-recreate. CREATE TABLE IF NOT EXISTS
    alone would keep an old table with the wrong columns and fail later in a
    confusing way.
    """
    with get_db_connection(db_path) as conn:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] > 0
        if has_tables and current != SCHEMA_VERSION:
            print(f"Schema version {current} != {SCHEMA_VERSION}: rebuilding {db_path}")
            conn.execute("PRAGMA foreign_keys = OFF;")
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for t in tables:
                conn.execute(f"DROP TABLE IF EXISTS {t}")
            conn.commit()
            conn.execute("VACUUM")
            conn.execute("PRAGMA foreign_keys = ON;")
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            conn.executescript(f.read())
        conn.commit()


# --- Generic helpers ---------------------------------------------------------

def _to_sql_value(v):
    """Convert pandas/numpy values to plain Python types SQLite accepts.
    NaN / inf become NULL, so 'no value' is never stored as a fake number."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(pd.Timestamp(v).date())
    if hasattr(v, "isoformat") and not isinstance(v, str):  # datetime.date
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    return v


def _upsert_many(conn, table, columns, data, conflict_keys):
    """Insert rows, updating non-key columns when the key already exists."""
    if not data:
        return
    cols_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    conflict_str = ", ".join(conflict_keys)
    update_cols = [c for c in columns if c not in conflict_keys]
    if update_cols:
        action = "DO UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    else:
        action = "DO NOTHING"
    sql = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) ON CONFLICT({conflict_str}) {action}"
    conn.executemany(sql, data)


def write_frame(conn, table, df, key_cols):
    """Upsert every row of `df` (columns = table columns) into `table`."""
    if df is None or df.empty:
        return
    cols = list(df.columns)
    data = [tuple(_to_sql_value(v) for v in row) for row in df.itertuples(index=False, name=None)]
    _upsert_many(conn, table, cols, data, key_cols)
    conn.commit()


def _dated(df, asset_id, run_id=None, date_col="date"):
    """Move a DatetimeIndex into a 'date' column and add asset/run ids."""
    out = df.copy()
    out.insert(0, date_col, [str(d.date()) for d in out.index])
    out.insert(0, "asset_id", asset_id)
    if run_id is not None:
        out.insert(0, "run_id", run_id)
    return out.reset_index(drop=True)


# --- Writers -------------------------------------------------------------------

def write_asset(conn, asset_id, ticker, name="", calendar=None):
    write_frame(conn, "assets",
                pd.DataFrame([{"asset_id": asset_id, "ticker": ticker, "name": name, "calendar": calendar}]),
                ["asset_id"])


def write_prices(conn, df, asset_id):
    px = df.reindex(columns=["Open", "High", "Low", "Close", "Volume"])
    px.columns = ["open", "high", "low", "close", "volume"]
    write_frame(conn, "prices", _dated(px, asset_id), ["asset_id", "date"])


def write_features(conn, df, asset_id):
    """Wide feature frame -> long (asset_id, date, feature_name, value)."""
    long = df.copy()
    long.index = [str(d.date()) for d in long.index]
    long = long.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="feature_name", value_name="value").dropna(subset=["value"])
    long.insert(0, "asset_id", asset_id)
    write_frame(conn, "features", long, ["asset_id", "date", "feature_name"])


def write_run(conn, run_id, cfg, data_end=None, models_trained=True,
              strategy_name="hmm_walk_forward", notes=""):
    costs = cfg.get("backtest", {}).get("costs", {})
    cost_model = (f"spread + {costs.get('commission_bps', 0)} bps commission per unit turnover; "
                  f"spreads (bps): {json.dumps(costs.get('spread_bps', {}), sort_keys=True)}")
    row = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config_hash": get_config_hash(cfg),
        "git_sha": get_git_sha(),
        "strategy_name": strategy_name,
        "cost_model": cost_model,
        "notes": notes,
        "data_end": None if data_end is None else str(pd.Timestamp(data_end).date()),
        "models_trained": int(bool(models_trained)),
        "config_json": json.dumps(cfg, sort_keys=True, default=str),
    }
    write_frame(conn, "backtest_runs", pd.DataFrame([row]), ["run_id"])
    return run_id


REGIME_COLUMNS = ['raw_label', 'p_bull', 'p_transition', 'p_crisis',
                  'label', 'sp_bull', 'sp_transition', 'sp_crisis', 'target_weight']


def write_regimes(conn, signal_df, asset_id, run_id):
    """Store the signal frame from src.models.signal.compute_signal.
    Only days with a prediction (out-of-sample) are written."""
    sig = signal_df.dropna(subset=['raw_label'])[REGIME_COLUMNS]
    write_frame(conn, "regimes", _dated(sig, asset_id, run_id), ["asset_id", "date", "run_id"])


def write_backtest_daily(conn, btest_df, asset_id, run_id, strategy="hmm", perf_df=None):
    """Daily backtest rows; `perf_df` (from metrics.performance_series) adds
    equity, drawdown and rolling Sharpe for charts."""
    df = pd.DataFrame({
        "weight": btest_df["executed_weight"],
        "gross_return": btest_df["gross_return"],
        "net_return": btest_df["net_return"],
        "turnover": btest_df["turnover"],
        "cost": btest_df["cost"],
    }, index=btest_df.index)
    if perf_df is not None:
        df = df.join(perf_df[["equity", "drawdown", "rolling_sharpe"]])
    out = _dated(df, asset_id, run_id)
    out.insert(2, "strategy", strategy)
    write_frame(conn, "backtest_daily", out, ["run_id", "asset_id", "strategy", "date"])


def write_backtest_metrics(conn, metrics_dict, asset_id, run_id, cycle="full", strategy="hmm"):
    rows = []
    for name, val in metrics_dict.items():
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            continue  # skip labels such as "Asset"
        rows.append({"run_id": run_id, "asset_id": asset_id, "strategy": strategy,
                     "cycle": cycle, "metric_name": name, "metric_value": val_f})
    write_frame(conn, "backtest_metrics", pd.DataFrame(rows),
                ["run_id", "asset_id", "strategy", "cycle", "metric_name"])


def write_evaluation_window(conn, run_id, asset_id, oos_start, oos_end, ann_factor, cost_bps=None):
    """Record the out-of-sample window every strategy for this asset was scored on."""
    write_frame(conn, "evaluation_windows", pd.DataFrame([{
        "run_id": run_id, "asset_id": asset_id,
        "oos_start": pd.Timestamp(oos_start), "oos_end": pd.Timestamp(oos_end),
        "ann_factor": int(ann_factor), "cost_bps": cost_bps,
    }]), ["run_id", "asset_id"])


def write_significance(conn, sig_df, run_id):
    """Store Sharpe-difference test results (output of run_significance_suite)."""
    if sig_df.empty:
        return
    out = sig_df.rename(columns={"asset": "asset_id"})[
        ["asset_id", "strategy", "baseline", "sharpe_diff", "ci_lower", "ci_upper",
         "p_value", "p_value_hac", "p_value_holm"]].copy()
    out.insert(0, "run_id", run_id)
    write_frame(conn, "significance_tests", out, ["run_id", "asset_id", "strategy", "baseline"])


def write_folds(conn, folds_info, asset_id, run_id):
    """Per-fold fit diagnostics and per-state training statistics."""
    fold_rows, state_rows = [], []
    for f in folds_info:
        labels = list(f["state_map"].values())
        starts = f.get("fit", {}).get("starts", [])
        fold_rows.append({
            "run_id": run_id, "asset_id": asset_id, "fold_idx": f["fold_idx"],
            "train_start": f.get("train_start"), "train_end": f["train_end"],
            "test_start": f["test_start"], "test_end": f["test_end"],
            "best_log_likelihood": f.get("fit", {}).get("best_log_likelihood"),
            "n_starts": len(starts),
            "n_converged": sum(bool(s.get("converged")) for s in starts),
            "n_bull_states": labels.count("Bull"),
            "n_transition_states": labels.count("Transition"),
            "n_crisis_states": labels.count("Crisis"),
        })
        for st in f.get("states", []):
            state_rows.append({
                "run_id": run_id, "asset_id": asset_id, "fold_idx": f["fold_idx"],
                "state": st["state"], "label": st["label"],
                "train_days": st["train_days"], "train_share": st["train_share"],
                "ann_return": st["mu"], "ann_vol": st["vol"], "sharpe": st["sharpe"],
            })
    write_frame(conn, "model_folds", pd.DataFrame(fold_rows), ["run_id", "asset_id", "fold_idx"])
    write_frame(conn, "fold_states", pd.DataFrame(state_rows), ["run_id", "asset_id", "fold_idx", "state"])


def write_regime_analysis(conn, asset_id, run_id, profile, transitions, spells):
    """Store outputs of src.analysis.regimes (profile, transitions, spells)."""
    def tag(df):
        out = df.copy()
        out.insert(0, "asset_id", asset_id)
        out.insert(0, "run_id", run_id)
        return out
    write_frame(conn, "regime_profiles", tag(profile), ["run_id", "asset_id", "label", "stat_name"])
    write_frame(conn, "regime_transitions", tag(transitions), ["run_id", "asset_id", "from_label", "to_label"])
    sp = spells.rename(columns={"start": "start_date", "end": "end_date"})[["start_date", "end_date", "label", "days"]]
    write_frame(conn, "regime_spells", tag(sp), ["run_id", "asset_id", "start_date"])
