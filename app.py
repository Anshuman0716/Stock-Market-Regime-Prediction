"""
Market Regime Detector — Streamlit dashboard.

Run:  streamlit run app.py

The app only READS results that scripts/run_pipeline.py stored in
data/regime_store.db. Queries live in src/dashboard/data.py, charts in
src/dashboard/charts.py; this file arranges them on the page and formats
numbers for display. It never refits a model or recomputes a metric
(AGENTS.md rules 1 and 6).
"""
import inspect
import json
import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dashboard import charts, data  # noqa: E402
from src.dashboard.summary import headline_findings, scoreboard_table  # noqa: E402
from src.dashboard.theme import (  # noqa: E402
    REGIME_COLORS, REGIME_ICONS, REGIME_LABELS, STRATEGY_NAMES, STRATEGY_ORDER, asset_name, hex_to_rgba,
)
from src.data.loader import load_config  # noqa: E402
from src.db.store import DEFAULT_DB_PATH, get_config_hash  # noqa: E402
from src.stats.significance import block_bootstrap_sharpe_diff, run_significance_suite  # noqa: E402

# Overridable so tests can point the app at a small synthetic store.
DB_PATH = os.environ.get("REGIME_DB_PATH", DEFAULT_DB_PATH)
PAPER_LOG = os.environ.get("REGIME_PAPER_LOG", "data/paper_trades.csv")

st.set_page_config(page_title="Market Regime Detector", page_icon="📈", layout="wide")


# ── Cached loading ───────────────────────────────────────────────────────
# Keyed on the database file's modification time: when the pipeline
# publishes a new store, every query reloads on the next page view.

@st.cache_data(show_spinner=False)
def _load(name, db_path, version, *args):
    return getattr(data, name)(db_path, *args)


def load(name, *args):
    return _load(name, DB_PATH, data.store_version(DB_PATH), *args)


def plot(fig):
    # theme="streamlit" lets text and axes follow the viewer's light/dark theme.
    st.plotly_chart(fig, use_container_width=True, theme="streamlit",
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# ── Display helpers ──────────────────────────────────────────────────────

def as_pct(df, cols):
    """Fractions -> percentages for display (0.123 -> 12.3)."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = out[c] * 100
    return out


def pct_col(label=None, digits=1):
    return st.column_config.NumberColumn(label, format=f"%.{digits}f%%")


def num_col(label=None, digits=2):
    return st.column_config.NumberColumn(label, format=f"%.{digits}f")


def regime_badge(label):
    color = REGIME_COLORS.get(label, "#898781")
    return (f'<span style="background:{hex_to_rgba(color, 0.18)}; border-left:4px solid {color}; '
            f'padding:2px 10px; border-radius:4px; font-weight:600;">'
            f'{REGIME_ICONS.get(label, "")} {label}</span>')


# ── Startup checks ───────────────────────────────────────────────────────

try:
    data.check_store(DB_PATH)
except data.StoreUnavailable as e:
    st.title("📈 Market Regime Detector")
    st.error(f"**The results store can't be shown.** {e}")
    st.markdown("Build it with:\n```\npython scripts/run_pipeline.py\n```")
    st.stop()

run = load("load_run")
RUN_ID = run["run_id"]
run_cfg = json.loads(run["config_json"]) if run.get("config_json") else load_config()
windows = load("load_windows", RUN_ID)
ASSETS = [a for a in run_cfg["backtest"]["assets"] if a in windows.index]
data_end = pd.Timestamp(run["data_end"]).date() if run.get("data_end") else None

# The store records which config produced it; flag it if config.yaml has
# changed since, because the numbers then describe a different setup.
try:
    config_changed = get_config_hash(load_config()) != run["config_hash"]
except (OSError, ValueError):
    # No readable config.yaml (e.g. a deploy with only the store): nothing to compare.
    config_changed = False


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📈 Regime Detector")
    asset = st.radio("Asset", ASSETS, format_func=asset_name,
                     help="Used by every tab except Overview and Statistical evidence.")
    st.divider()
    st.markdown(f"**Data through:** {data_end}")
    st.markdown(f"**Pipeline run:** {str(run['created_at'])[:16].replace('T', ' ')}")
    st.markdown(f"**Models:** {'retrained this run' if run.get('models_trained') else 'reused saved folds'}")
    st.markdown(f"**Code version:** `{str(run['git_sha'])[:8]}`")
    if config_changed:
        st.warning("`config.yaml` has changed since this run. Re-run the pipeline to update the results.")
    with st.expander("Refresh the results"):
        st.markdown("Daily update (≈15 s):")
        st.code("python scripts/run_pipeline.py --reuse-models", language="bash")
        st.markdown("Full retrain (≈4 min), needed after a config change or in a new year:")
        st.code("python scripts/run_pipeline.py", language="bash")

st.title("Market Regime Detector")
st.caption("A 4-state Hidden Markov Model, refit every year on past data only, labels each day as "
           "**Bull**, **Transition** or **Crisis** and sizes a position by the probability of Bull. "
           "Every number here is out-of-sample and after trading costs.")

tabs = st.tabs(["Overview", "Regimes", "Performance", "Risk", "Model insight",
                "Statistical evidence", "Paper trading"])

metrics_full = load("load_metrics", RUN_ID, "full")
significance = load("load_significance", RUN_ID)
win = windows.loc[asset]


def window_caption():
    years = metrics_full.loc[(asset, "hmm"), "Years"]
    st.caption(f"Out-of-sample window {win['oos_start']:%Y-%m-%d} → {win['oos_end']:%Y-%m-%d} "
               f"({years:.1f} years). Costs: {win['cost_bps']:g} bps per unit of turnover. "
               f"Annualized with {int(win['ann_factor'])} trading days per year.")


# ── 1. Overview ──────────────────────────────────────────────────────────

with tabs[0]:
    signals = load("load_latest_signals", RUN_ID)
    st.subheader("Today's signal")
    st.caption(f"Smoothed regime and target exposure at the latest close in the store ({data_end}). "
               "These are exactly the values the backtest trades on; the position is taken two days later.")
    cols = st.columns(len(ASSETS))
    for col, a in zip(cols, ASSETS):
        s = signals.loc[a]
        with col, st.container(border=True):
            st.markdown(f"**{asset_name(a)}**")
            st.markdown(regime_badge(s["label"]), unsafe_allow_html=True)
            st.metric("Target exposure", f"{s['target_weight']:.0%}",
                      help="Share of capital in the asset = smoothed P(Bull).")
            st.caption(" · ".join(f"P({lab}) {s['sp_' + lab.lower()]:.0%}" for lab in REGIME_LABELS))
            if pd.notna(s.get("spell_start")):
                st.caption(f"In {s['label']} since {s['spell_start']:%Y-%m-%d} ({int(s['spell_days'])} trading days)")

    st.subheader("Out-of-sample scoreboard")
    for line in headline_findings(metrics_full, significance, ASSETS):
        st.markdown(f"- {line}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sharpe ratio** (higher is better)")
        plot(charts.asset_dot_plot(metrics_full, "Sharpe", ".2f"))
    with c2:
        st.markdown("**Maximum drawdown** (closer to 0 is better)")
        plot(charts.asset_dot_plot(metrics_full, "Max DD", ".0%"))

    board = scoreboard_table(metrics_full, significance, windows.loc[ASSETS])
    pct = ["CAGR · HMM", "CAGR · B&H", "Max DD · HMM", "Max DD · B&H", "Avg. exposure · HMM"]
    # Short column labels so every column fits without sideways scrolling.
    st.dataframe(
        as_pct(board, pct), hide_index=True, use_container_width=True,
        column_config={**{c: pct_col(c.replace("Avg. exposure", "Exposure"), 1) for c in pct},
                       **{c: num_col(c.replace("200-DMA", "MA")) for c in board.columns if c.startswith("Sharpe")},
                       "p vs B&H (Holm)": num_col("p (Holm)", 3)})
    starts = ", ".join(f"{asset_name(a)} from {windows.loc[a, 'oos_start']:%Y-%m-%d}" for a in ASSETS)
    st.caption(f"Out-of-sample windows: {starts}, all to {data_end}. Within each asset, all three strategies "
               "run on the same dates, start flat and pay the same costs. B&H = buy and hold; MA = 200-day "
               "moving-average filter (invested only when the price is above its 200-day average).")


# ── 2. Regimes ───────────────────────────────────────────────────────────

with tabs[1]:
    st.subheader(f"{asset_name(asset)}: regime timeline")
    raw = st.toggle("Show raw (unsmoothed) HMM probabilities", value=False,
                    help="The strategy trades the 21-day trailing average. Raw values show the model's "
                         "day-by-day output before smoothing.")
    timeline = load("load_timeline", RUN_ID, asset)
    spells = load("load_spells", RUN_ID, asset)
    plot(charts.regime_timeline(timeline, spells, asset, smoothed=not raw))
    st.caption("Top: closing price, shaded by the smoothed regime. Bottom: regime probabilities "
               f"({'raw filtered' if raw else 'trailing-smoothed'}); they sum to 100% every day. "
               "Each fold only uses data up to the day it predicts.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Recent regime spells**")
        recent = spells.sort_values("start_date", ascending=False).head(12).copy()
        recent["label"] = recent["label"].map(lambda lab: f"{REGIME_ICONS[lab]} {lab}")
        st.dataframe(recent.rename(columns={"start_date": "Start", "end_date": "End", "label": "Regime",
                                            "days": "Trading days"}),
                     hide_index=True, use_container_width=True,
                     column_config={"Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                    "End": st.column_config.DateColumn(format="YYYY-MM-DD")})
    with c2:
        st.markdown("**Time spent in each regime**")
        prof = load("load_profiles", RUN_ID, asset)
        never = [lab for lab in REGIME_LABELS if lab not in prof.index]
        share = prof[["share", "n_spells", "avg_spell_days"]].reset_index()
        share["label"] = share["label"].map(lambda lab: f"{REGIME_ICONS[lab]} {lab}")
        st.dataframe(as_pct(share, ["share"]), hide_index=True, use_container_width=True,
                     column_config={"label": "Regime", "share": pct_col("Share of days", 0),
                                    "n_spells": num_col("Spells", 0),
                                    "avg_spell_days": num_col("Avg. spell (days)", 0)})
        if never:
            no_state = [lab for lab in never
                        if load("load_folds", RUN_ID, asset)[f"n_{lab.lower()}_states"].sum() == 0]
            why = (" No yearly fit produced such a state: the middle-volatility states' training Sharpe "
                   "ratios always fell outside the Transition band of the labeling rule."
                   if no_state == ["Transition"] else "")
            st.caption(f"{asset_name(asset)} was never labeled {' or '.join(never)} out-of-sample.{why}")


# ── 3. Performance ───────────────────────────────────────────────────────

with tabs[2]:
    st.subheader(f"{asset_name(asset)}: HMM strategy vs baselines")
    window_caption()
    daily = load("load_daily", RUN_ID, asset)

    rows = ["CAGR", "Ann. Vol", "Sharpe", "Sortino", "Calmar", "Max DD", "Max DD Duration (days)",
            "Hit Rate", "Time in Market", "Avg. Exposure", "Ann. Turnover"]
    pct_rows = {"CAGR", "Ann. Vol", "Max DD", "Hit Rate", "Time in Market", "Avg. Exposure"}
    m = metrics_full.loc[asset].reindex(STRATEGY_ORDER)
    table = pd.DataFrame({
        STRATEGY_NAMES[s]: [(f"{m.loc[s, r]:.1%}" if r in pct_rows else
                             f"{m.loc[s, r]:,.0f}" if "days" in r else f"{m.loc[s, r]:.2f}")
                            for r in rows if r in m.columns]
        for s in STRATEGY_ORDER}, index=[r for r in rows if r in m.columns])
    # Explicit height so every row shows (the default cuts off at ~10 rows).
    st.dataframe(table, use_container_width=True, height=36 * (len(table) + 1) + 3)

    plot(charts.performance_panels(daily))
    st.caption("Growth of $1 after costs (log scale), the fall from the previous peak, and the trailing "
               "one-year Sharpe ratio.")

    periods = load("load_period_metrics", RUN_ID, asset)
    st.markdown("#### Calendar years")
    plot(charts.yearly_returns_chart(periods))
    first_year = windows.loc[asset, "oos_start"]
    st.caption(f"{first_year.year} starts on {first_year:%Y-%m-%d}; the latest year runs to {data_end}.")

    st.markdown("#### Equity stress episodes")
    episodes = run_cfg.get("evaluation", {}).get("stress_episodes", {})
    if (periods["kind"] == "episode").any():
        plot(charts.episode_chart(periods, episodes))
        st.caption("Each episode runs from an S&P 500 closing high to the low that followed. They are "
                   "applied to every asset, so for Gold and Bitcoin they show behaviour during *equity* stress.")
    else:
        st.info("None of the stress episodes falls inside this asset's out-of-sample window.")


# ── 4. Risk ──────────────────────────────────────────────────────────────

with tabs[3]:
    st.subheader(f"{asset_name(asset)}: tail risk, exposure and costs")
    window_caption()
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(charts.tail_risk_chart(metrics_full.loc[asset]))
    with c2:
        st.markdown("**How to read this**")
        st.markdown("- **VaR 95%**: on 1 day in 20, the loss was at least this large.\n"
                    "- **CVaR 95%**: the average loss on those worst 1-in-20 days.\n"
                    "- **99%**: the same for the worst 1 day in 100.\n\n"
                    "Historical values come from the strategy's own out-of-sample daily returns.")
        cf = metrics_full.loc[asset].reindex(STRATEGY_ORDER)[["VaR 95% (CF)", "VaR 99% (CF)"]]
        cf.index = [STRATEGY_NAMES[s] for s in cf.index]
        st.dataframe(cf.map(lambda v: f"{v:.2%}"), use_container_width=True)
        st.caption("CF = Cornish-Fisher VaR, a parametric estimate adjusted for skew and fat tails.")

    st.markdown("#### How much was invested")
    plot(charts.exposure_chart(daily))
    st.caption("Buy & Hold is always fully invested, so it is not drawn. The HMM scales its position "
               "smoothly with P(Bull); the 200-day MA filter is either all in or all out.")

    st.markdown("#### Trading costs")
    costs = load("load_cost_totals", RUN_ID, asset).reindex(STRATEGY_ORDER)
    cost_table = pd.DataFrame({
        "Strategy": [STRATEGY_NAMES[s] for s in STRATEGY_ORDER],
        "Ann. turnover": [metrics_full.loc[(asset, s), "Ann. Turnover"] for s in STRATEGY_ORDER],
        "Total cost paid": costs["total_cost"].to_numpy() * 100,
    })
    st.dataframe(cost_table, hide_index=True, use_container_width=True,
                 column_config={"Ann. turnover": st.column_config.NumberColumn(format="%.2fx"),
                                "Total cost paid": pct_col("Total cost paid (% of capital, summed daily)", 2)})
    st.caption("Turnover of 1x a year = buying or selling the whole portfolio once.")


# ── 5. Model insight ─────────────────────────────────────────────────────

with tabs[4]:
    st.subheader(f"{asset_name(asset)}: what the regimes look like")
    prof = load("load_profiles", RUN_ID, asset)
    feature_cols = [c for c in ["mean_vix", "mean_volatility_20", "mean_rsi", "mean_adx"] if c in prof.columns]
    pt = prof[["share", "avg_spell_days", "ann_return", "ann_vol", "sharpe"] + feature_cols].reset_index()
    pt["label"] = pt["label"].map(lambda lab: f"{REGIME_ICONS[lab]} {lab}")
    st.dataframe(
        as_pct(pt, ["share", "ann_return", "ann_vol", "mean_volatility_20"]), hide_index=True,
        use_container_width=True,
        column_config={"label": "Regime", "share": pct_col("Share of days", 0),
                       "avg_spell_days": num_col("Avg. spell (days)", 0),
                       "ann_return": pct_col("Asset return (ann.)"), "ann_vol": pct_col("Asset vol (ann.)"),
                       "sharpe": num_col("Sharpe"), "mean_vix": num_col("Avg. VIX", 1),
                       "mean_volatility_20": pct_col("Avg. 20-day vol", 2), "mean_rsi": num_col("Avg. RSI", 1),
                       "mean_adx": num_col("Avg. ADX", 1)})
    st.caption("The asset's own return and risk on the days with each smoothed label, out-of-sample. "
               "This describes the regimes; it is not a tradable result.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Transition probabilities (day to day)**")
        probs, counts = load("load_transitions", RUN_ID, asset)
        plot(charts.transition_heatmap(probs, counts))
        st.caption("Read across a row: given today's regime, the chance of each regime tomorrow. "
                   "Large diagonal values mean regimes persist.")
    with c2:
        st.markdown("**How long regimes last**")
        plot(charts.spell_duration_chart(spells))
        st.caption("One dot per spell; the box spans the middle half of spell lengths.")

    st.markdown("#### Walk-forward training: how each year's states were labeled")
    fstates = load("load_fold_states", RUN_ID, asset)
    plot(charts.fold_states_chart(fstates))
    st.caption("Every year the HMM is refit on all data before that year and finds 4 hidden states. Each "
               "state is labeled from its *training* behaviour: lowest volatility → Bull, highest → Crisis, "
               "the rest by their Sharpe ratio. Dot size = share of training days in that state.")
    with st.expander("Fold details"):
        folds = load("load_folds", RUN_ID, asset)
        ft = pd.DataFrame({
            "Trades in": folds["test_start"].dt.year,
            "Trained on": folds["train_start"].dt.strftime("%Y-%m-%d") + " → " + folds["train_end"].dt.strftime("%Y-%m-%d"),
            "Best log-likelihood": folds["best_log_likelihood"],
            "Starts converged": folds["n_converged"].astype(str) + " / " + folds["n_starts"].astype(str),
            "Bull states": folds["n_bull_states"], "Transition states": folds["n_transition_states"],
            "Crisis states": folds["n_crisis_states"],
        })
        st.dataframe(ft, hide_index=True, use_container_width=True,
                     column_config={"Trades in": st.column_config.NumberColumn(format="%d"),
                                    "Best log-likelihood": num_col(digits=0)})


# ── 6. Statistical evidence ──────────────────────────────────────────────

with tabs[5]:
    st.subheader("Is the difference real, or luck?")
    boot = inspect.signature(block_bootstrap_sharpe_diff).parameters
    n_boot = inspect.signature(run_significance_suite).parameters["n_bootstraps"].default
    st.markdown(
        f"For each asset, the HMM's Sharpe ratio is compared with each baseline over the same dates. "
        f"The 95% interval comes from a **moving-block bootstrap** ({boot['block_size'].default}-day blocks, "
        f"{n_boot:,} resamples), which keeps the clustering of volatile days that a simple bootstrap would "
        f"destroy. A second, **HAC** test (robust to autocorrelation) cross-checks the p-value. Because "
        f"{len(significance)} comparisons are made at once, p-values are adjusted with the **Holm** method; "
        "otherwise one \"significant\" result in several tests would be expected by chance alone.")
    plot(charts.significance_forest(significance))
    st.caption("A dot right of zero means the HMM had the higher Sharpe ratio. If the interval crosses zero, "
               "the data can't tell the two strategies apart.")
    st_table = significance.assign(
        Asset=significance["asset_id"].map(asset_name),
        Baseline=significance["baseline"].map(STRATEGY_NAMES),
        _order=significance["asset_id"].map({a: i for i, a in enumerate(ASSETS)}),
    ).sort_values(["_order", "baseline"])[["Asset", "Baseline", "sharpe_diff", "ci_lower", "ci_upper", "p_value", "p_value_hac", "p_value_holm"]]
    st.dataframe(st_table, hide_index=True, use_container_width=True,
                 column_config={"sharpe_diff": num_col("Sharpe diff."), "ci_lower": num_col("95% low"),
                                "ci_upper": num_col("95% high"), "p_value": num_col("p (bootstrap)", 3),
                                "p_value_hac": num_col("p (HAC)", 3), "p_value_holm": num_col("p (Holm)", 3)})


# ── 7. Paper trading ─────────────────────────────────────────────────────

with tabs[6]:
    st.subheader(f"{asset_name(asset)}: paper trading")
    s = load("load_latest_signals", RUN_ID).loc[asset]
    folds = load("load_folds", RUN_ID, asset)
    latest_fold = folds.iloc[-1]
    today = date.today()

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Latest regime**")
    c1.markdown(regime_badge(s["label"]), unsafe_allow_html=True)
    c2.metric("Target exposure", f"{s['target_weight']:.0%}")
    c3.metric("Signal date", f"{s['date']:%Y-%m-%d}")

    age = (today - s["date"].date()).days
    if age > 4:
        st.warning(f"The stored signal is {age} days old. Run `python scripts/run_pipeline.py --reuse-models` "
                   "to update it.")
    if latest_fold["test_start"].year < today.year:
        st.warning(f"The newest model was trained through {latest_fold['train_end']:%Y-%m-%d}. A new year has "
                   "started, so run a full `python scripts/run_pipeline.py` to fit this year's model.")
    else:
        st.success(f"Model for {latest_fold['test_start'].year} is in place "
                   f"(trained through {latest_fold['train_end']:%Y-%m-%d}).")

    st.markdown("**How it works.** `scripts/paper_trade.py` downloads the latest prices, loads the newest "
                "fold model, and computes today's signal with the same code as the backtest (a test checks "
                "the two agree). It then sizes an order so the account holds *target exposure × account value* "
                "in the asset. By default it is a dry run; with Alpaca paper-trading keys and `--live` it "
                "submits the order.")
    # The broker trades a tracking product, not the index itself; SPY is the
    # script's default for the S&P 500. Other assets need their own symbol.
    symbol = "SPY" if asset == "spx" else "<BROKER_SYMBOL>"
    st.code(f"python scripts/paper_trade.py --asset {asset} --symbol {symbol}          # dry run\n"
            f"python scripts/paper_trade.py --asset {asset} --symbol {symbol} --live   # Alpaca paper account",
            language="bash")
    st.caption("Guardrails: it refuses stale data, a missing feature row, and a one-day weight change above "
               "50% (`--max-weight-change`). Set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY for live mode.")

    st.markdown("#### Trade log")
    log = data.load_paper_trades(PAPER_LOG)
    if log.empty:
        st.info("No paper trades logged yet. Each run of `scripts/paper_trade.py` appends a row to "
                f"`{PAPER_LOG}`.")
    else:
        st.dataframe(as_pct(log, ["target_weight", "current_weight", "p_bull"]), hide_index=True,
                     use_container_width=True,
                     column_config={"timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                                    "target_weight": pct_col("Target", 0), "current_weight": pct_col("Held before", 0),
                                    "p_bull": pct_col("Smoothed P(Bull)", 0), "qty_ordered": "Qty ordered",
                                    "asset": "Asset", "side": "Side", "dry_run": "Dry run"})
