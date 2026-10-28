"""
Read-only queries against the regime store for the dashboard.

Design choices:
- Read-only connection (SQLite `mode=ro`): the dashboard can never modify
  the results it displays.
- Open, query, close. Holding a connection open would lock the file on
  Windows and block the pipeline's atomic swap of a freshly built database
  (os.replace) while the app is running.
- Every query is pinned to one run_id, so rows from different pipeline runs
  can never be mixed on one chart.
- These functions only reshape stored rows (pivots, sorting). They never
  recompute a metric, a regime or a signal; that is the pipeline's job
  (AGENTS.md rule 6).
"""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.db.store import DEFAULT_DB_PATH, SCHEMA_VERSION
from src.models.signal import REGIME_LABELS


class StoreUnavailable(Exception):
    """The store is missing, empty or built by an older schema."""


def _connect(db_path):
    if not os.path.exists(db_path):
        raise StoreUnavailable(f"No database at `{db_path}`.")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return closing(sqlite3.connect(uri, uri=True))


def _read(db_path, sql, params=(), dates=()):
    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    for col in dates:
        df[col] = pd.to_datetime(df[col])
    return df


def store_version(db_path=DEFAULT_DB_PATH):
    """Modification time of the database file.

    Used as a cache key: when the pipeline publishes a new database, the
    mtime changes and every cached query is reloaded automatically.
    """
    return os.path.getmtime(db_path) if os.path.exists(db_path) else None


def check_store(db_path=DEFAULT_DB_PATH):
    """Raise StoreUnavailable with a readable reason if the store can't be shown."""
    with _connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise StoreUnavailable(
                f"The database uses schema v{version}, but this code expects v{SCHEMA_VERSION}.")
        if conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == 0:
            raise StoreUnavailable("The database has no pipeline runs yet.")


def load_run(db_path=DEFAULT_DB_PATH, run_id=None):
    """The requested run, or the most recent one."""
    if run_id is None:
        df = _read(db_path, "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 1")
    else:
        df = _read(db_path, "SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
    if df.empty:
        raise StoreUnavailable(f"Run `{run_id}` not found.")
    return df.iloc[0].to_dict()


def load_windows(db_path, run_id):
    """Out-of-sample window, annualization and cost per asset (index: asset_id)."""
    df = _read(db_path, "SELECT * FROM evaluation_windows WHERE run_id = ? ORDER BY rowid",
               (run_id,), dates=("oos_start", "oos_end"))
    return df.set_index("asset_id")


def load_latest_signals(db_path, run_id):
    """Each asset's signal on its latest stored date, plus the start of the current regime spell."""
    sql = """
        SELECT r.*, s.start_date AS spell_start, s.days AS spell_days
        FROM regimes r
        JOIN (SELECT asset_id, MAX(date) AS d FROM regimes WHERE run_id = ? GROUP BY asset_id) m
          ON r.asset_id = m.asset_id AND r.date = m.d
        LEFT JOIN regime_spells s
          ON s.run_id = r.run_id AND s.asset_id = r.asset_id AND s.end_date = r.date
        WHERE r.run_id = ?
    """
    df = _read(db_path, sql, (run_id, run_id), dates=("date", "spell_start"))
    return df.set_index("asset_id")


def load_metrics(db_path, run_id, cycle="full"):
    """Wide table: index (asset_id, strategy), one column per metric."""
    df = _read(db_path, """SELECT asset_id, strategy, metric_name, metric_value
                           FROM backtest_metrics WHERE run_id = ? AND cycle = ?""", (run_id, cycle))
    return df.pivot_table(index=["asset_id", "strategy"], columns="metric_name",
                          values="metric_value", aggfunc="first")


def load_period_metrics(db_path, run_id, asset_id):
    """Per-year and per-stress-episode metrics for one asset.

    Returns columns: strategy, kind ('year' / 'episode'), period, then one
    column per metric. 'cycle' is stored as 'year:2020' or 'episode:<name>'.
    """
    df = _read(db_path, """SELECT strategy, cycle, metric_name, metric_value FROM backtest_metrics
                           WHERE run_id = ? AND asset_id = ? AND cycle != 'full'""", (run_id, asset_id))
    if df.empty:
        return pd.DataFrame(columns=["strategy", "kind", "period"])
    wide = df.pivot_table(index=["strategy", "cycle"], columns="metric_name",
                          values="metric_value", aggfunc="first").reset_index()
    wide[["kind", "period"]] = wide["cycle"].str.split(":", n=1, expand=True)
    return wide.drop(columns="cycle")


def load_timeline(db_path, run_id, asset_id):
    """Daily close plus the full stored signal for every out-of-sample day."""
    sql = """
        SELECT r.date, p.close, r.raw_label, r.p_bull, r.p_transition, r.p_crisis,
               r.label, r.sp_bull, r.sp_transition, r.sp_crisis, r.target_weight
        FROM regimes r
        JOIN prices p ON p.asset_id = r.asset_id AND p.date = r.date
        WHERE r.run_id = ? AND r.asset_id = ?
        ORDER BY r.date
    """
    return _read(db_path, sql, (run_id, asset_id), dates=("date",)).set_index("date")


def load_prices(db_path, asset_id):
    """Daily OHLCV for one asset (full history, not only the out-of-sample window)."""
    return _read(db_path, "SELECT date, open, high, low, close, volume FROM prices WHERE asset_id = ? ORDER BY date",
                 (asset_id,), dates=("date",)).set_index("date")


def load_features(db_path, asset_id):
    """The 8 model features for one asset, wide (one column per feature), full history.

    The full history matters for notebooks: model-selection experiments must
    use the first training window, which lies before the out-of-sample period.
    """
    df = _read(db_path, "SELECT date, feature_name, value FROM features WHERE asset_id = ?",
               (asset_id,), dates=("date",))
    return df.pivot(index="date", columns="feature_name", values="value").sort_index()


def load_daily(db_path, run_id, asset_id):
    """Daily backtest rows for every strategy (long format)."""
    sql = """SELECT strategy, date, weight, net_return, turnover, cost, equity, drawdown, rolling_sharpe
             FROM backtest_daily WHERE run_id = ? AND asset_id = ? ORDER BY strategy, date"""
    return _read(db_path, sql, (run_id, asset_id), dates=("date",))


def load_cost_totals(db_path, run_id, asset_id):
    """Total trading cost paid over the window, per strategy (sum of stored daily costs)."""
    df = _read(db_path, """SELECT strategy, SUM(cost) AS total_cost, COUNT(*) AS days
                           FROM backtest_daily WHERE run_id = ? AND asset_id = ?
                           GROUP BY strategy""", (run_id, asset_id))
    return df.set_index("strategy")


def load_spells(db_path, run_id, asset_id):
    return _read(db_path, """SELECT start_date, end_date, label, days FROM regime_spells
                             WHERE run_id = ? AND asset_id = ? ORDER BY start_date""",
                 (run_id, asset_id), dates=("start_date", "end_date"))


def load_profiles(db_path, run_id, asset_id):
    """Regime profile: index = regime label (Bull, Transition, Crisis), columns = stats."""
    df = _read(db_path, """SELECT label, stat_name, value FROM regime_profiles
                           WHERE run_id = ? AND asset_id = ?""", (run_id, asset_id))
    wide = df.pivot_table(index="label", columns="stat_name", values="value", aggfunc="first")
    # Regimes that never occurred (e.g. Gold has no Transition days) carry
    # only empty statistics, so they are left out.
    wide = wide[wide.get("days", pd.Series(1, index=wide.index)) > 0]
    return wide.reindex([lab for lab in REGIME_LABELS if lab in wide.index])


def load_transitions(db_path, run_id, asset_id):
    """Square matrix P(tomorrow = column | today = row), in REGIME_LABELS order."""
    df = _read(db_path, """SELECT from_label, to_label, n, probability FROM regime_transitions
                           WHERE run_id = ? AND asset_id = ?""", (run_id, asset_id))
    probs = df.pivot_table(index="from_label", columns="to_label", values="probability", aggfunc="first")
    counts = df.pivot_table(index="from_label", columns="to_label", values="n", aggfunc="first")
    order = [lab for lab in REGIME_LABELS if lab in probs.index or lab in probs.columns]
    return probs.reindex(index=order, columns=order), counts.reindex(index=order, columns=order)


def load_folds(db_path, run_id, asset_id):
    return _read(db_path, """SELECT * FROM model_folds WHERE run_id = ? AND asset_id = ?
                             ORDER BY fold_idx""", (run_id, asset_id),
                 dates=("train_start", "train_end", "test_start", "test_end"))


def load_fold_states(db_path, run_id, asset_id):
    """Every HMM state of every fold, with its training statistics and assigned label."""
    sql = """SELECT s.*, f.test_start FROM fold_states s
             JOIN model_folds f ON f.run_id = s.run_id AND f.asset_id = s.asset_id
                               AND f.fold_idx = s.fold_idx
             WHERE s.run_id = ? AND s.asset_id = ? ORDER BY s.fold_idx, s.state"""
    df = _read(db_path, sql, (run_id, asset_id), dates=("test_start",))
    df["test_year"] = df["test_start"].dt.year
    return df


def load_significance(db_path, run_id):
    return _read(db_path, "SELECT * FROM significance_tests WHERE run_id = ?", (run_id,))


def load_paper_trades(path="data/paper_trades.csv"):
    """The paper-trading log written by scripts/paper_trade.py (empty if it has never run)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp", ascending=False)
