"""
Measure what each Step 6 change did, by re-running the full pipeline under
each setup on the same data.

    python scripts/compare_variants.py          # ~15 min: trains every variant

Variants (each adds one change to the one before):
  1. step5     - labeling "absolute", index returns, cash earns 0, NASDAQ spread 1 bp
  2. labeling  - + labeling "relative"
  3. current   - + tradable funds, T-bill cash, ONEQ spread (= config/config.yaml)

Each variant trains into its own temporary folder, so the live store and
models are never touched. The summary goes to data/variants.json, which
scripts/update_readme.py turns into the README's comparison table.

Why this exists: changing the model and then quoting only the new numbers
would hide how much each change moved the results. Showing every step,
including the old setup, keeps the comparison honest. Note that several
setups have now been evaluated on the same out-of-sample period, so
differences between them are exploratory, not independent evidence.
"""
import copy
from contextlib import closing
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
import warnings
from datetime import datetime

import yaml

warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

OUT = "data/variants.json"
METRICS = ["Sharpe", "CAGR", "Max DD", "Avg. Exposure"]


def variant_configs(cfg):
    """The three setups, each one change away from the previous."""
    step5 = copy.deepcopy(cfg)
    step5["model"]["labeling"]["rule"] = "absolute"
    step5["backtest"].pop("instruments", None)
    step5["backtest"].pop("cash_ticker", None)
    step5["backtest"]["costs"]["spread_bps"]["nasdaq"] = 1.0

    labeling = copy.deepcopy(step5)
    labeling["model"]["labeling"]["rule"] = "relative"

    return [
        ("step5", "Step 5 setup: fixed Sharpe bars, index returns, 0% on cash", step5),
        ("labeling", "+ relative labeling rule", labeling),
        ("current", "+ tradable funds (SPY, ONEQ, GLD) and T-bill cash = current setup", copy.deepcopy(cfg)),
    ]


def summarise(db_path):
    # closing(): sqlite3's own context manager only commits; the file would
    # stay open and Windows could not delete the temporary folder.
    with closing(sqlite3.connect(db_path)) as c:
        rows = c.execute("SELECT asset_id, strategy, metric_name, metric_value FROM backtest_metrics "
                         "WHERE cycle = 'full'").fetchall()
        shares = c.execute("SELECT asset_id, label, value FROM regime_profiles WHERE stat_name = 'share'").fetchall()
        data_end = c.execute("SELECT data_end FROM backtest_runs").fetchone()[0]
    out = {}
    for asset, strategy, name, value in rows:
        if name in METRICS:
            out.setdefault(asset, {}).setdefault(strategy, {})[name] = value
    for asset, label, value in shares:
        out.setdefault(asset, {}).setdefault("regime_share", {})[label] = value
    return out, data_end


def main():
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    results = {"generated_at": datetime.now().isoformat(timespec="seconds"), "variants": []}
    for key, description, vcfg in variant_configs(cfg):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(vcfg, f)
            db_path = os.path.join(tmp, "store.db")
            run_pipeline(config_path=cfg_path, db_path=db_path, model_root=os.path.join(tmp, "models"),
                         train=True)
            assets, data_end = summarise(db_path)
        results["data_end"] = data_end
        results["variants"].append({"key": key, "description": description, "assets": assets})
        print(f"{key:9s} done in {time.perf_counter() - t0:.0f}s")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
