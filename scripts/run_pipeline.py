"""
Run the whole pipeline: data -> features -> walk-forward HMM -> signal ->
backtests (HMM + baselines) -> regime analysis -> significance -> SQLite.

Usage:
    python scripts/run_pipeline.py                  # retrain all folds (~5 min)
    python scripts/run_pipeline.py --reuse-models   # fast refresh with saved models
    python scripts/run_pipeline.py --force-refresh  # re-download prices first

Use --reuse-models for a daily update. A full retrain is needed after any
change to config.yaml or the features, and once a new calendar year starts
(the pipeline tells you when).
"""
import argparse
import logging
import os
import sys
import time
import warnings

# joblib (used by scikit-learn) tries to count physical CPU cores with a
# Windows tool that may be missing, then warns "[WinError 2]" and falls back
# to the logical core count. Harmless for us, so hide just that message.
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.portfolio import PORTFOLIO_ID  # noqa: E402
from src.pipeline import run_pipeline, DEFAULT_RUN_ID  # noqa: E402
from src.models.hmm import RetrainRequired  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse-models", action="store_true",
                        help="Skip training; predict with the saved fold models.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-download prices instead of using today's cache.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    # hmmlearn / sklearn emit noisy FutureWarnings unrelated to our results.
    warnings.filterwarnings("ignore", category=FutureWarning)

    t0 = time.perf_counter()
    try:
        summary = run_pipeline(train=not args.reuse_models, run_id=args.run_id,
                               force_refresh=args.force_refresh)
    except RetrainRequired as e:
        sys.exit(f"Cannot reuse models: {e}\nRun without --reuse-models to retrain.")

    print(f"\nDone in {time.perf_counter() - t0:.0f}s "
          f"({'reused saved models' if args.reuse_models else 'retrained all folds'}).")
    # Per-asset rows and the portfolio have different strategies, so each
    # group gets its own header.
    groups = {"assets": {a: r for a, r in summary.items() if a != PORTFOLIO_ID},
              "portfolio (v2.0)": {a: r for a, r in summary.items() if a == PORTFOLIO_ID}}
    print("Sharpe ratio above cash, out-of-sample:")
    for title, rows in groups.items():
        if not rows:
            continue
        strategies = [k[len("sharpe_"):] for k in next(iter(rows.values())) if k.startswith("sharpe_")]
        print(f"\n{title:9s} {'window':25s}" + "".join(f"{name:>14s}" for name in strategies))
        for asset, row in rows.items():
            print(f"{asset:9s} {row['oos_start']} -> {row['oos_end']} "
                  + "".join(f"{row['sharpe_' + name]:14.2f}" for name in strategies))


if __name__ == "__main__":
    main()
