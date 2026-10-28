"""
Report the forward test: results only from days after the design was frozen.

    python scripts/run_pipeline.py --reuse-models   # refresh the store with new data first
    python scripts/forward_test.py

Reads data/regime_store.db and config/freeze.json; recomputes nothing.
Exits with an error if config.yaml no longer matches the frozen version,
because the numbers would then not be a forward test of that version.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.forward import FORWARD_CYCLE, freeze_status, load_freeze  # noqa: E402
from src.dashboard import data  # noqa: E402
from src.dashboard.theme import ASSET_NAMES, STRATEGY_NAMES, STRATEGY_ORDER  # noqa: E402
from src.data.loader import load_config  # noqa: E402
from src.db.store import DEFAULT_DB_PATH, get_config_hash  # noqa: E402


def main():
    freeze = load_freeze()
    status, message = freeze_status(freeze, get_config_hash(load_config()))
    print(message)
    if status != "valid":
        sys.exit(1)

    data.check_store(DEFAULT_DB_PATH)
    run = data.load_run(DEFAULT_DB_PATH)
    m = data.load_metrics(DEFAULT_DB_PATH, run["run_id"], cycle=FORWARD_CYCLE)
    if m.empty:
        print(f"No trading days after {freeze['data_end_at_freeze']} in the store yet "
              f"(data through {run['data_end']}). Refresh with: python scripts/run_pipeline.py --reuse-models")
        return

    rows = []
    for (asset, strategy), r in m.iterrows():
        rows.append({"asset": ASSET_NAMES.get(asset, asset), "strategy": STRATEGY_NAMES.get(strategy, strategy),
                     "order": STRATEGY_ORDER.index(strategy) if strategy in STRATEGY_ORDER else 99,
                     "days": int(r["Days"]), "total return": f"{r['Total Return']:+.2%}",
                     "Sharpe (above cash)": f"{r['Sharpe']:.2f}", "max drawdown": f"{r['Max DD']:.2%}",
                     "avg. exposure": f"{r['Avg. Exposure']:.0%}"})
    table = pd.DataFrame(rows).sort_values(["asset", "order"]).drop(columns="order")
    print(f"Forward test of {freeze['version']}, data through {run['data_end']}:")
    print(table.to_string(index=False))
    print("Short forward windows are noisy: judge after months, not days.")


if __name__ == "__main__":
    main()
