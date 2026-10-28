"""
Cross-asset summaries for the dashboard overview.

These only compare numbers the pipeline already stored (e.g. "is the HMM's
stored Sharpe above buy-and-hold's?"). The headline sentences are generated
from the data on every load rather than written by hand, so the dashboard
can never show a claim that the current results don't support
(AGENTS.md rule 5).
"""
import pandas as pd

from src.dashboard.theme import asset_name

ALPHA = 0.05


def scoreboard_table(metrics, significance, windows):
    """One row per asset comparing the three strategies over its out-of-sample window.

    metrics: index (asset_id, strategy), columns metric names (cycle 'full').
    Percent-type values are returned as fractions; the app formats them.
    """
    rows = []
    for a in windows.index:
        if a not in metrics.index.get_level_values(0):
            continue
        m = metrics.loc[a]
        sig = significance[(significance["asset_id"] == a) & (significance["baseline"] == "buy_hold")]
        rows.append({
            "Asset": asset_name(a),
            "Sharpe · HMM": m.loc["hmm", "Sharpe"],
            "Sharpe · B&H": m.loc["buy_hold", "Sharpe"],
            "Sharpe · 200-DMA": m.loc["dma200", "Sharpe"],
            "CAGR · HMM": m.loc["hmm", "CAGR"],
            "CAGR · B&H": m.loc["buy_hold", "CAGR"],
            "Max DD · HMM": m.loc["hmm", "Max DD"],
            "Max DD · B&H": m.loc["buy_hold", "Max DD"],
            "Avg. exposure · HMM": m.loc["hmm", "Avg. Exposure"],
            "p vs B&H (Holm)": sig["p_value_holm"].iloc[0] if not sig.empty else float("nan"),
        })
    return pd.DataFrame(rows)


def _names(ids):
    names = [asset_name(a) for a in ids]
    return ", ".join(names) if names else "none"


def headline_findings(metrics, significance, assets):
    """Plain-English findings, each computed from the stored results."""
    assets = [a for a in assets if a in metrics.index.get_level_values(0)]
    n = len(assets)

    def better(metric, higher_is_better=True):
        out = []
        for a in assets:
            h, b = metrics.loc[(a, "hmm"), metric], metrics.loc[(a, "buy_hold"), metric]
            if (h > b) if higher_is_better else (h < b):
                out.append(a)
        return out

    sharpe_wins = better("Sharpe")
    # Max DD is stored as a negative number, so "smaller drawdown" = higher value.
    dd_wins = better("Max DD")
    cagr_wins = better("CAGR")

    findings = [
        f"Higher Sharpe ratio than buy-and-hold on **{len(sharpe_wins)} of {n}** assets ({_names(sharpe_wins)}).",
        f"Smaller maximum drawdown than buy-and-hold on **{len(dd_wins)} of {n}** assets ({_names(dd_wins)}).",
        f"Higher annual growth (CAGR) than buy-and-hold on **{len(cagr_wins)} of {n}** assets ({_names(cagr_wins)}).",
    ]

    if not significance.empty:
        n_tests = len(significance)
        sig = significance[significance["p_value_holm"] < ALPHA]
        if sig.empty:
            findings.append(
                f"**None** of the {n_tests} Sharpe-ratio differences is statistically significant after "
                f"correcting for multiple tests (smallest adjusted p = {significance['p_value_holm'].min():.2f}). "
                "The differences above could be luck.")
        else:
            # A significant difference can favour either side: report both directions.
            better, worse = sig[sig["sharpe_diff"] > 0], sig[sig["sharpe_diff"] < 0]

            def pairs(df):
                return ", ".join(f"{asset_name(r.asset_id)} vs {r.baseline}" for r in df.itertuples())

            if not better.empty:
                findings.append(f"HMM Sharpe significantly **higher** (after correcting for {n_tests} tests): "
                                f"{pairs(better)}.")
            else:
                findings.append(f"**No** difference in the HMM's favour is significant after correcting for "
                                f"{n_tests} tests.")
            if not worse.empty:
                findings.append(f"HMM Sharpe significantly **lower** than the baseline: {pairs(worse)}.")
    return findings
