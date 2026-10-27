"""
Backfill the SQLite store: prices, features, out-of-sample regimes, and the
backtests + metrics of the HMM strategy and its baselines, plus significance
tests. Everything the dashboard and notebooks read comes from here.

Usage:
    python scripts/backfill_db.py                      # normal run
    python scripts/backfill_db.py --check-idempotency  # run twice, compare row counts
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, download_prices
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict
from src.backtest.evaluation import evaluate_asset
from src.stats.significance import run_significance_suite
from src.db.store import (
    init_db, get_db_connection, write_asset, write_prices, write_features,
    write_run, write_regimes, write_backtest_daily, write_backtest_metrics,
    write_evaluation_window, write_significance,
)

TABLES = ["assets", "prices", "features", "backtest_runs", "regimes", "backtest_daily",
          "backtest_metrics", "evaluation_windows", "significance_tests"]

# A fixed run_id keeps the committed database small: re-running the backfill
# upserts over the same rows instead of stacking a new copy each time.
DEFAULT_RUN_ID = "main"


def row_counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def print_row_counts(conn, label=""):
    print(f"\n--- Row Counts {label} ---")
    for t, n in row_counts(conn).items():
        print(f"{t}: {n}")
    print("-----------------------\n")


def run_backfill(run_id=DEFAULT_RUN_ID):
    cfg = load_config('config/config.yaml')
    data = download_prices(cfg)
    init_db()

    returns_for_tests, ann_factors = {}, {}

    with get_db_connection() as conn:
        write_run(conn, run_id, cfg, notes="Out-of-sample evaluation: HMM vs buy_hold vs dma200")

        for asset_name in cfg['backtest']['assets']:
            if asset_name not in data:
                print(f"Skipping {asset_name}: no price data.")
                continue

            print(f"Backfilling {asset_name}...")
            asset = data[asset_name]
            write_asset(conn, asset_name, ticker=cfg['tickers'].get(asset_name, asset_name))
            write_prices(conn, asset, asset_name)

            # Features. 'Close' is carried alongside (not used by the HMM) so
            # the 200-day-average baseline can be computed from the same frame.
            features = build_features(asset, asset['VIX_Close'])
            features['Close'] = asset['Close']
            features = features.dropna()
            write_features(conn, features[FEATURE_COLUMNS], asset_name)

            # Walk-forward HMM: filtered probabilities, NaN inside the
            # initial training window.
            raw_labels, probs, _ = walk_forward_predict(features, FEATURE_COLUMNS, cfg, asset_name=asset_name)
            write_regimes(conn, raw_labels, probs, asset_name, run_id)

            # Evaluate every strategy on the same out-of-sample window.
            ev = evaluate_asset(features, probs, cfg, asset_name)
            write_evaluation_window(conn, run_id, asset_name, ev["oos_start"], ev["oos_end"], ev["ann_factor"])
            for strategy, bt in ev["backtests"].items():
                write_backtest_daily(conn, bt, asset_name, run_id, strategy=strategy)
                m = {k: v for k, v in ev["metrics"][strategy].items() if k != "Asset"}
                write_backtest_metrics(conn, m, asset_name, run_id, cycle="full", strategy=strategy)

            returns_for_tests[asset_name] = {s: bt["net_return"] for s, bt in ev["backtests"].items()}
            ann_factors[asset_name] = ev["ann_factor"]
            print(f"  out-of-sample {ev['oos_start'].date()} -> {ev['oos_end'].date()} "
                  f"({ev['metrics']['hmm']['Years']:.1f} yrs, ann_factor={ev['ann_factor']})")

        # Significance across all assets at once, so Holm can correct for
        # the full family of comparisons.
        sig = run_significance_suite(returns_for_tests, ann_factors)
        write_significance(conn, sig, run_id)

    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--check-idempotency", action="store_true",
                        help="Run the backfill twice and fail if row counts change.")
    args = parser.parse_args()

    run_backfill(args.run_id)
    with get_db_connection() as conn:
        first = row_counts(conn)
        print_row_counts(conn, "AFTER RUN")

    if args.check_idempotency:
        print("Running backfill a SECOND time to check idempotency...")
        run_backfill(args.run_id)
        with get_db_connection() as conn:
            second = row_counts(conn)
        if first != second:
            raise SystemExit(f"Idempotency check FAILED: {first} != {second}")
        print("Idempotency check passed: row counts identical.")


if __name__ == "__main__":
    main()
