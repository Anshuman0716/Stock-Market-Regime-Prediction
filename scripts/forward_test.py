"""
Report the forward test: results only from days after each design was frozen.

    python scripts/run_pipeline.py --reuse-models   # refresh the store with new data first
    python scripts/forward_test.py

Reads data/regime_store.db and config/freeze.json; recomputes nothing.
Every frozen version is reported on the same new days: v1.0 (each asset's
HMM strategy) and v2.0 (the regime-aware portfolio). A version whose config
no longer matches its freeze record is reported as invalid, and the script
then exits with an error, because those numbers are not a forward test of
that version.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.forward import FORWARD_CYCLE, SCOPE_PORTFOLIO, freeze_status_for, load_freezes  # noqa: E402
from src.backtest.portfolio import PORTFOLIO_ID, PortfolioSettings  # noqa: E402
from src.dashboard import data  # noqa: E402
from src.dashboard.theme import (  # noqa: E402
    ASSET_NAMES, PORTFOLIO_STRATEGY_NAMES, PORTFOLIO_STRATEGY_ORDER, STRATEGY_NAMES, STRATEGY_ORDER,
)
from src.data.loader import load_config  # noqa: E402
from src.db.store import DEFAULT_DB_PATH  # noqa: E402


def version_table(metrics, scope):
    """Forward rows belonging to one version: the portfolio for v2.0, the real assets for v1.0."""
    portfolio = scope == SCOPE_PORTFOLIO
    names = PORTFOLIO_STRATEGY_NAMES if portfolio else STRATEGY_NAMES
    order = PORTFOLIO_STRATEGY_ORDER if portfolio else STRATEGY_ORDER
    rows = []
    for (asset, strategy), r in metrics.iterrows():
        if (asset == PORTFOLIO_ID) != portfolio or strategy not in order:
            continue
        rows.append({"asset": ASSET_NAMES.get(asset, asset), "strategy": names[strategy],
                     "order": order.index(strategy),
                     "days": int(r["Days"]), "total return": f"{r['Total Return']:+.2%}",
                     "Sharpe (above cash)": f"{r['Sharpe']:.2f}", "max drawdown": f"{r['Max DD']:.2%}",
                     "avg. exposure": f"{r['Avg. Exposure']:.0%}"})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["asset", "order"]).drop(columns="order")


def main():
    cfg = load_config()
    freezes = load_freezes()
    if not freezes:
        print("The design has not been frozen (no config/freeze.json).")
        sys.exit(1)

    data.check_store(DEFAULT_DB_PATH)
    run = data.load_run(DEFAULT_DB_PATH)
    m = data.load_metrics(DEFAULT_DB_PATH, run["run_id"], cycle=FORWARD_CYCLE, include_portfolio=True)
    settings = PortfolioSettings.from_config(cfg) or PortfolioSettings()

    any_invalid = False
    for record in freezes:
        status, message = freeze_status_for(record, cfg)
        print(f"\n=== {record['version']} ({record['scope']}) ===")
        print(message)
        if status != "valid":
            any_invalid = True
            continue
        table = None if m.empty else version_table(m, record["scope"])
        if table is None:
            print(f"No trading days after {record['data_end_at_freeze']} in the store yet "
                  f"(data through {run['data_end']}). Refresh with: python scripts/run_pipeline.py --reuse-models")
            continue
        print(f"Forward test of {record['version']}, data through {run['data_end']}:")
        print(table.to_string(index=False))
        if record["scope"] == SCOPE_PORTFOLIO:
            days = int(table["days"].iloc[0])
            if days < settings.forward_min_days:
                print(f"INTERIM: {days} of {settings.forward_min_days} forward trading days. "
                      "No verdict until the window is full (pre-registration, Section 10).")

    print("\nShort forward windows are noisy: judge after months, not days.")
    if any_invalid:
        sys.exit(1)


if __name__ == "__main__":
    main()
