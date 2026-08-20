import sys
import os
import uuid
import pandas as pd
from datetime import datetime

# Insert path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.backtest.engine import run_backtest
from src.backtest.risk import compute_target_weight_proba
from src.backtest.metrics import get_all_metrics
from src.db.store import (
    init_db, get_db_connection, write_asset, write_prices, write_features,
    write_run, write_regimes, write_backtest_daily, write_backtest_metrics
)

def print_row_counts(conn, label=""):
    tables = ["assets", "prices", "features", "backtest_runs", "regimes", "backtest_daily", "backtest_metrics"]
    print(f"\n--- Row Counts {label} ---")
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {count}")
    print("-----------------------\n")

def run_backfill(run_id=None):
    cfg = load_config('config/config.yaml')
    data = download_prices(cfg)
    
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
    init_db()
    
    with get_db_connection() as conn:
        write_run(conn, run_id, cfg, notes="Initial backfill")
        
        for asset_name in ['spx', 'nasdaq', 'gold', 'bitcoin']:
            if asset_name not in data:
                continue
                
            print(f"Backfilling {asset_name}...")
            asset = data[asset_name]
            
            # Asset record
            write_asset(conn, asset_name, ticker=cfg['tickers'].get(asset_name, asset_name))
            
            # Prices
            write_prices(conn, asset, asset_name)
            
            # Features
            features = build_features(asset, asset['VIX_Close'])
            features['returns'] = asset['Close'].pct_change()
            features['Close'] = asset['Close']
            features = features.dropna()
            
            # Write features (exclude Close/returns to match modeling scope if desired, but fine to include)
            write_features(conn, features[FEATURE_COLUMNS], asset_name)
            
            # HMM Predictions
            raw_labels, out_probs, folds_info = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name=asset_name)
            
            # Write regimes
            write_regimes(conn, raw_labels, out_probs, asset_name, run_id)
            
            # Smoothing & target weights
            smoothed_prob = out_probs['Bull'].rolling(window=cfg['model']['smoothing_window']).mean().fillna(0)
            target_weights_prob = compute_target_weight_proba(pd.DataFrame({'Bull': smoothed_prob}))
            
            # Backtest engine
            hmm_res = run_backtest(features, target_weights_prob, asset_name=asset_name, use_dynamic_costs=True)
            
            # Write daily backtest tracking
            write_backtest_daily(conn, hmm_res, asset_name, run_id)
            
            # Compute and write metrics
            ann_factor = 252
            m_hmm = get_all_metrics(hmm_res['net_return'], hmm_res['executed_weight'], ann_factor, asset_name.upper() + ' (HMM)')
            # Drop the Asset string column for metric storage
            m_hmm_db = {k: v for k, v in m_hmm.items() if k != 'Asset'}
            write_backtest_metrics(conn, m_hmm_db, asset_name, run_id, cycle="full")
            
    return run_id

if __name__ == "__main__":
    init_db()
    with get_db_connection() as conn:
        print_row_counts(conn, "BEFORE RUN 1")
    
    # We use a deterministic run_id here to prove idempotency across backfill re-runs.
    # In practice, you might generate a new run_id if you want to track a new experiment,
    # but the asset, price, and feature data will be safely upserted without duplication.
    # The backtest_runs, regimes, etc. keyed by this run_id will also be upserted cleanly.
    fixed_run_id = "backfill_phase4"
    run_backfill(fixed_run_id)
    
    with get_db_connection() as conn:
        print_row_counts(conn, "AFTER RUN 1")
        
    print("Running backfill a SECOND time to prove idempotency (no row duplication)...")
    run_backfill(fixed_run_id)
    
    with get_db_connection() as conn:
        print_row_counts(conn, "AFTER RUN 2 (Should perfectly match RUN 1)")
