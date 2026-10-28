"""
Measure what each Step 6 change did, by re-running the full pipeline under
each setup on the same data.

    python scripts/compare_variants.py                    # ~25 min: trains every setup
    python scripts/compare_variants.py --only volscale    # (re)run one setup, merge into the file

Variants (each adds one change to the one before):
  1. step5     - labeling "absolute", index returns, cash earns 0, NASDAQ spread 1 bp
  2. labeling  - + labeling "relative" (judged against the asset's own Sharpe)
  3. anchors   - + labeling "sharpe" (best/worst-earning anchors, not calmest/most volatile)
  4. funds     - + tradable funds, T-bill cash, ONEQ spread
  5. current   - + regime-return sizing (= config/config.yaml)
  6. volscale  - + regime-aware volatility scaling (an option, off by default)
  7. btc6      - current + 6 states for Bitcoin (an option, off by default)

Each variant trains into its own temporary folder, so the live store and
models are never touched. The summary goes to data/variants.json, which
scripts/update_readme.py turns into the README's comparison table.

Why this exists: changing the model and then quoting only the new numbers
would hide how much each change moved the results. Showing every step,
including the old setup, keeps the comparison honest. Note that several
setups have now been evaluated on the same out-of-sample period, so
differences between them are exploratory, not independent evidence.
"""
import argparse
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
    """The setups, each one change away from the previous."""
    step5 = copy.deepcopy(cfg)
    step5["model"]["labeling"]["rule"] = "absolute"
    step5["model"]["sizing"] = "p_bull"
    step5["model"]["vol_scaling"] = False
    step5["backtest"].pop("instruments", None)
    step5["backtest"].pop("cash_ticker", None)
    step5["backtest"]["costs"]["spread_bps"]["nasdaq"] = 1.0

    labeling = copy.deepcopy(step5)
    labeling["model"]["labeling"]["rule"] = "relative"

    anchors = copy.deepcopy(step5)
    anchors["model"]["labeling"]["rule"] = "sharpe"

    funds = copy.deepcopy(cfg)
    funds["model"]["sizing"] = "p_bull"
    funds["model"]["vol_scaling"] = False

    current = copy.deepcopy(cfg)
    current["model"]["vol_scaling"] = False

    volscale = copy.deepcopy(cfg)
    volscale["model"]["vol_scaling"] = True

    btc6 = copy.deepcopy(current)
    btc6["model"]["n_states_by_asset"] = {"bitcoin": 6}

    return [
        ("step5", "Step 5 setup: fixed Sharpe bars, index returns, 0% on cash", step5),
        ("labeling", "+ relative labeling rule", labeling),
        ("anchors", "+ Sharpe-anchored labeling rule", anchors),
        ("funds", "+ tradable funds (SPY, ONEQ, GLD) and T-bill cash", funds),
        ("current", "+ regime-return sizing = current setup", current),
        ("volscale", "+ regime-aware volatility scaling (option, off by default)", volscale),
        ("btc6", "current setup with 6 states for Bitcoin (option, off by default)", btc6),
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


def run_variant(key, description, vcfg):
    """Train one setup in a temporary folder and summarise its store."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(vcfg, f)
        db_path = os.path.join(tmp, "store.db")
        run_pipeline(config_path=cfg_path, db_path=db_path, model_root=os.path.join(tmp, "models"), train=True)
        assets, data_end = summarise(db_path)
    print(f"{key:9s} done in {time.perf_counter() - t0:.0f}s")
    return {"key": key, "description": description, "assets": assets}, data_end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated setup keys to (re)run, merged into the existing "
                                       f"{OUT}; runs one setup per call to keep memory low")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    setups = variant_configs(cfg)
    order = [key for key, _, _ in setups]

    if args.only:
        wanted = args.only.split(",")
        unknown = set(wanted) - set(order)
        if unknown:
            sys.exit(f"Unknown setup(s) {sorted(unknown)}; choose from {order}.")
        results = json.load(open(OUT)) if os.path.exists(OUT) else {"variants": []}
    else:
        wanted, results = order, {"variants": []}

    by_key = {v["key"]: v for v in results["variants"]}
    for key, description, vcfg in setups:
        if key in wanted:
            by_key[key], data_end = run_variant(key, description, vcfg)
            if results.get("data_end") not in (None, data_end):
                # Rows trained on different data would not be comparable.
                sys.exit(f"Data now ends {data_end}, but {OUT} was built to {results['data_end']}; "
                         "re-run every setup (no --only).")
            results["data_end"] = data_end

    results["variants"] = [by_key[k] for k in order if k in by_key]
    results["not_measured"] = [k for k in order if k not in by_key]
    results["generated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
