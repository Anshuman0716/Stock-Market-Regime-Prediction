import sqlite3
import pandas as pd
import hashlib
import json
import subprocess
import os
from contextlib import contextmanager
from datetime import datetime

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

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
    with get_db_connection(db_path) as conn:
        with open(SCHEMA_PATH, 'r') as f:
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

def write_regimes(conn, labels_series, probs_df, asset_id, run_id):
    data = []
    # Both should have the same DatetimeIndex
    for date_idx, label in labels_series.items():
        if pd.isna(label): continue
        date_str = str(date_idx.date()) if isinstance(date_idx, pd.Timestamp) else str(date_idx)
        row_probs = probs_df.loc[date_idx]
        data.append((
            asset_id, date_str, run_id, 
            -1, # state_raw (no longer perfectly applicable across folds)
            str(label),
            float(row_probs.get('Bull', 0.0)),
            float(row_probs.get('Transition', 0.0)),
            float(row_probs.get('Crisis', 0.0))
        ))
    _upsert_many(conn, 'regimes', ['asset_id', 'date', 'run_id', 'state_raw', 'label', 'p_bull', 'p_transition', 'p_crisis'], data, ['asset_id', 'date', 'run_id'])
    conn.commit()

def write_backtest_daily(conn, btest_df, asset_id, run_id):
    data = []
    for date_idx, row in btest_df.iterrows():
        date_str = str(date_idx.date()) if isinstance(date_idx, pd.Timestamp) else str(date_idx)
        data.append((
            run_id, asset_id, date_str,
            float(row.get('executed_weight', 0.0)),
            float(row.get('gross_return', 0.0)),
            float(row.get('net_return', 0.0)),
            float(row.get('turnover', 0.0)),
            float(row.get('cost', 0.0))
        ))
    _upsert_many(conn, 'backtest_daily', ['run_id', 'asset_id', 'date', 'weight', 'gross_return', 'net_return', 'turnover', 'cost'], data, ['run_id', 'asset_id', 'date'])
    conn.commit()

def write_backtest_metrics(conn, metrics_dict, asset_id, run_id, cycle="full"):
    data = []
    for metric_name, val in metrics_dict.items():
        # skip non-numeric info if any, but all should be floats or similar
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            continue
            
        data.append((run_id, asset_id, cycle, metric_name, val_f))
        
    _upsert_many(conn, 'backtest_metrics', ['run_id', 'asset_id', 'cycle', 'metric_name', 'metric_value'], data, ['run_id', 'asset_id', 'cycle', 'metric_name'])
    conn.commit()
