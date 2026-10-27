import sqlite3
import pandas as pd
import hashlib
import json
import subprocess
import os
from contextlib import contextmanager
from datetime import datetime

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

# Must match "PRAGMA user_version" in schema.sql. Bump both together whenever
# a table's columns or keys change.
SCHEMA_VERSION = 3

def get_git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"

def get_config_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode('utf-8')).hexdigest()

@contextmanager
def get_db_connection(db_path="data/regime_store.db"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()

def init_db(db_path="data/regime_store.db"):
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
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()

def _upsert_many(conn, table, columns, data, conflict_keys):
    """
    Generic idempotent upsert helper.
    """
    if not data: return
    
    cols_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    conflict_str = ", ".join(conflict_keys)
    
    # Update all non-conflict columns
    update_cols = [c for c in columns if c not in conflict_keys]
    if update_cols:
        set_str = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
        sql = f"""
            INSERT INTO {table} ({cols_str}) 
            VALUES ({placeholders})
            ON CONFLICT({conflict_str}) 
            DO UPDATE SET {set_str}
        """
    else:
        # If all columns are part of the PK, just DO NOTHING on conflict
        sql = f"""
            INSERT INTO {table} ({cols_str}) 
            VALUES ({placeholders})
            ON CONFLICT({conflict_str}) 
            DO NOTHING
        """
        
    conn.executemany(sql, data)

def write_asset(conn, asset_id, ticker, name=""):
    sql = """
        INSERT INTO assets (asset_id, ticker, name)
        VALUES (?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET ticker=excluded.ticker, name=excluded.name
    """
    conn.execute(sql, (asset_id, ticker, name))
    conn.commit()

def write_prices(conn, df, asset_id):
    df = df.copy()
    if 'Volume' not in df.columns:
        df['Volume'] = 0.0
    data = []
    for idx, row in df.iterrows():
        date_str = str(idx.date()) if isinstance(idx, pd.Timestamp) else str(idx)
        data.append((asset_id, date_str, row.get('Open', 0.0), row.get('High', 0.0), row.get('Low', 0.0), row.get('Close', 0.0), row['Volume']))
    _upsert_many(conn, 'prices', ['asset_id', 'date', 'open', 'high', 'low', 'close', 'volume'], data, ['asset_id', 'date'])
    conn.commit()

def write_features(conn, df, asset_id):
    # Convert wide df to long format
    # columns are features, index is date
    data = []
    for date_idx, row in df.iterrows():
        date_str = str(date_idx.date()) if isinstance(date_idx, pd.Timestamp) else str(date_idx)
        for col_name, val in row.items():
            if pd.isna(val): continue
            data.append((asset_id, date_str, col_name, float(val)))
            
    _upsert_many(conn, 'features', ['asset_id', 'date', 'feature_name', 'value'], data, ['asset_id', 'date', 'feature_name'])
    conn.commit()

def write_run(conn, run_id, cfg, strategy_name="hmm_walk_forward", notes=""):
    git_sha = get_git_sha()
    config_hash = get_config_hash(cfg)
    created_at = datetime.now().isoformat()
    cost_model = cfg.get("backtest", {}).get("cost_bps", 0)
    
    sql = """
        INSERT INTO backtest_runs (run_id, created_at, config_hash, git_sha, strategy_name, cost_model, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET 
            created_at=excluded.created_at,
            config_hash=excluded.config_hash,
            git_sha=excluded.git_sha,
            strategy_name=excluded.strategy_name,
            cost_model=excluded.cost_model,
            notes=excluded.notes
    """
    conn.execute(sql, (run_id, created_at, config_hash, git_sha, strategy_name, str(cost_model), notes))
    conn.commit()
    return run_id

REGIME_COLUMNS = ['raw_label', 'p_bull', 'p_transition', 'p_crisis',
                  'label', 'sp_bull', 'sp_transition', 'sp_crisis', 'target_weight']


def write_regimes(conn, signal_df, asset_id, run_id):
    """Store the signal frame from src.models.signal.compute_signal.

    Only days with a prediction (out-of-sample) are written.
    """
    data = []
    for date_idx, row in signal_df.dropna(subset=['raw_label']).iterrows():
        date_str = str(date_idx.date()) if isinstance(date_idx, pd.Timestamp) else str(date_idx)
        values = [row[c] if c in ('raw_label', 'label') else float(row[c]) for c in REGIME_COLUMNS]
        data.append((asset_id, date_str, run_id, *values))
    _upsert_many(conn, 'regimes', ['asset_id', 'date', 'run_id'] + REGIME_COLUMNS, data,
                 ['asset_id', 'date', 'run_id'])
    conn.commit()

def write_backtest_daily(conn, btest_df, asset_id, run_id, strategy="hmm"):
    data = []
    for date_idx, row in btest_df.iterrows():
        date_str = str(date_idx.date()) if isinstance(date_idx, pd.Timestamp) else str(date_idx)
        data.append((
            run_id, asset_id, strategy, date_str,
            float(row.get('executed_weight', 0.0)),
            float(row.get('gross_return', 0.0)),
            float(row.get('net_return', 0.0)),
            float(row.get('turnover', 0.0)),
            float(row.get('cost', 0.0))
        ))
    _upsert_many(conn, 'backtest_daily', ['run_id', 'asset_id', 'strategy', 'date', 'weight', 'gross_return', 'net_return', 'turnover', 'cost'], data, ['run_id', 'asset_id', 'strategy', 'date'])
    conn.commit()

def write_backtest_metrics(conn, metrics_dict, asset_id, run_id, cycle="full", strategy="hmm"):
    data = []
    for metric_name, val in metrics_dict.items():
        # skip non-numeric info if any, but all should be floats or similar
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            continue
            
        data.append((run_id, asset_id, strategy, cycle, metric_name, val_f))
        
    _upsert_many(conn, 'backtest_metrics', ['run_id', 'asset_id', 'strategy', 'cycle', 'metric_name', 'metric_value'], data, ['run_id', 'asset_id', 'strategy', 'cycle', 'metric_name'])
    conn.commit()


def write_evaluation_window(conn, run_id, asset_id, oos_start, oos_end, ann_factor):
    """Record the out-of-sample window every strategy for this asset was scored on."""
    _upsert_many(
        conn, 'evaluation_windows',
        ['run_id', 'asset_id', 'oos_start', 'oos_end', 'ann_factor'],
        [(run_id, asset_id, str(pd.Timestamp(oos_start).date()), str(pd.Timestamp(oos_end).date()), int(ann_factor))],
        ['run_id', 'asset_id'])
    conn.commit()


def write_significance(conn, sig_df, run_id):
    """Store Sharpe-difference test results (output of run_significance_suite)."""
    cols = ['asset', 'strategy', 'baseline', 'sharpe_diff', 'ci_lower', 'ci_upper',
            'p_value', 'p_value_hac', 'p_value_holm']
    data = [(run_id,) + tuple(r[c] if isinstance(r[c], str) else float(r[c]) for c in cols)
            for _, r in sig_df.iterrows()]
    _upsert_many(
        conn, 'significance_tests',
        ['run_id', 'asset_id', 'strategy', 'baseline', 'sharpe_diff', 'ci_lower', 'ci_upper',
         'p_value', 'p_value_hac', 'p_value_holm'],
        data, ['run_id', 'asset_id', 'strategy', 'baseline'])
    conn.commit()
