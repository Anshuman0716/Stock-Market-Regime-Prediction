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
    print(f"{'asset':8s} {'out-of-sample':25s} {'Sharpe HMM':>11s} {'Sharpe B&H':>11s}")
    for asset, s in summary.items():
        print(f"{asset:8s} {s['oos_start']} -> {s['oos_end']}  {s['sharpe_hmm']:11.2f} {s['sharpe_buy_hold']:11.2f}")


if __name__ == "__main__":
    main()
