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
from src.dashboard.summary import headline_findings, portfolio_results, scoreboard_table  # noqa: E402
from src.dashboard.theme import (  # noqa: E402
    PORTFOLIO_STRATEGY_NAMES, PORTFOLIO_STRATEGY_ORDER, REGIME_COLORS, REGIME_ICONS, REGIME_LABELS,
    STRATEGY_NAMES, STRATEGY_ORDER, asset_name, hex_to_rgba,
)
from src.analysis.forward import (  # noqa: E402
    FORWARD_CYCLE, SCOPE_PORTFOLIO, freeze_status_for, load_freeze, load_freezes,
)
from src.analysis.verdict import FAIL, INTERIM, PASS_RULE, SIGNIFICANT_PASS  # noqa: E402
from src.backtest.portfolio import PORTFOLIO_ID, PortfolioSettings  # noqa: E402
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


@st.cache_data(show_spinner=False)
def _portfolio(db_path, version, run_id, settings):
    return portfolio_results(db_path, run_id, settings)


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
    # Each frozen version is checked against the config of the run shown,
    # using that version's own fingerprint (v1.0 per asset, v2.0 portfolio).
    freezes = load_freezes()
    freeze_states = {f["version"]: freeze_status_for(f, run_cfg) for f in freezes}
    for f in freezes:
        state, msg = freeze_states[f["version"]]
        if state == "valid":
            st.markdown(f"**Frozen:** {f['version']} ({'portfolio' if f['scope'] == SCOPE_PORTFOLIO else 'per asset'}"
                        f", since {f['frozen_on']})")
        else:
            st.warning(msg)
    if config_changed:
        st.warning("`config.yaml` has changed since this run. Re-run the pipeline to update the results.")
    with st.expander("Refresh the results"):
        st.markdown("Daily update (≈15 s):")
        st.code("python scripts/run_pipeline.py --reuse-models", language="bash")
        st.markdown("Full retrain (≈4 min), needed after a config change or in a new year:")
        st.code("python scripts/run_pipeline.py", language="bash")

st.title("Market Regime Detector")
st.caption("A 4-state Hidden Markov Model, refit every year on past data only, labels each day as "
           "**Bull**, **Transition** or **Crisis** and sizes a position by the probability of the regimes "
           "that earned money in training. "
           "Every number here is out-of-sample and after trading costs.")

tabs = st.tabs(["Overview", "Regimes", "Performance", "Risk", "Model insight",
                "Statistical evidence", "Portfolio (v2.0)", "Paper trading"])

metrics_full = load("load_metrics", RUN_ID, "full")
significance = load("load_significance", RUN_ID)
win = windows.loc[asset]
portfolio_settings = PortfolioSettings.from_config(run_cfg) or PortfolioSettings()
portfolio = _portfolio(DB_PATH, data.store_version(DB_PATH), RUN_ID, portfolio_settings)


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
                      help="Share of capital in the asset: each regime's smoothed probability times how much "
                           "of the asset that regime holds (100% if it earned money in training, else 0%).")
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

    board = scoreboard_table(metrics_full, significance, windows.loc[ASSETS]).set_index("Asset")
    # Metrics as rows and assets as columns: four columns always fit, however
    # many baselines the rows compare.
    pct_rows = {"CAGR · HMM", "CAGR · B&H", "Max DD · HMM", "Max DD · B&H", "Avg. exposure · HMM"}

    def fmt(row, v):
        if pd.isna(v):
            return "–"
        return f"{v:.1%}" if row in pct_rows else f"{v:.3f}" if row.startswith("p ") else f"{v:.2f}"

    shown = board.T
    shown = shown.apply(lambda col: [fmt(r, v) for r, v in zip(shown.index, col)])
    shown.index = [r.replace("200-DMA", "MA") for r in shown.index]
    st.dataframe(shown, use_container_width=True, height=36 * (len(shown) + 1) + 3)
    starts = ", ".join(f"{asset_name(a)} from {windows.loc[a, 'oos_start']:%Y-%m-%d}" for a in ASSETS)
    st.caption(f"Out-of-sample windows: {starts}, all to {data_end}. Within each asset, all four strategies "
               "run on the same dates, start flat and pay the same costs. B&H = buy and hold; MA = 200-day "
               "moving-average filter (invested only when the price is above its 200-day average); Vol tgt = "
               "volatility target (holds less when volatility is above its own long-run level).")


    st.subheader("Forward tests")
    if not freezes:
        st.info("The design has not been frozen (no config/freeze.json).")
    fwd = load("load_metrics", RUN_ID, FORWARD_CYCLE)
    for f in freezes:
        state, msg = freeze_states[f["version"]]
        is_portfolio = f["scope"] == SCOPE_PORTFOLIO
        st.markdown(f"**{f['version']}** · {'the regime-aware portfolio' if is_portfolio else 'each asset on its own'}")
        if state != "valid":
            st.info(msg)
            continue
        st.caption(f"Frozen on {f['frozen_on']}; only days after {f['data_end_at_freeze']} count, so this is "
                   "the one result the design work cannot have influenced.")
        ft = fwd.reset_index() if not fwd.empty else pd.DataFrame(columns=["asset_id", "strategy"])
        ft = ft[(ft["asset_id"] == PORTFOLIO_ID) == is_portfolio]
        order, names = ((PORTFOLIO_STRATEGY_ORDER, PORTFOLIO_STRATEGY_NAMES) if is_portfolio
                        else (STRATEGY_ORDER, STRATEGY_NAMES))
        ft = ft[ft["strategy"].isin(order)]
        if ft.empty:
            st.info(f"No trading days since the freeze yet (data through {data_end}). Each "
                    "`python scripts/run_pipeline.py --reuse-models` adds new days.")
            continue
        ft = ft.assign(Asset=ft["asset_id"].map(asset_name), Strategy=ft["strategy"].map(names))
        st.dataframe(as_pct(ft[["Asset", "Strategy", "Days", "Total Return", "Sharpe", "Max DD"]],
                            ["Total Return", "Max DD"]),
                     hide_index=True, use_container_width=True,
                     column_config={"Days": num_col(digits=0), "Total Return": pct_col("Total return", 2),
                                    "Sharpe": num_col("Sharpe (above cash)"), "Max DD": pct_col("Max DD")})
        if is_portfolio and int(ft["Days"].iloc[0]) < portfolio_settings.forward_min_days:
            st.caption(f"Interim: {int(ft['Days'].iloc[0])} of {portfolio_settings.forward_min_days} trading days. "
                       "No verdict until the window is full (pre-registration, Section 10).")
    st.caption("A few weeks of data say almost nothing; judge a forward test after months.")


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
               "smoothly with the probability of the regimes worth holding; the 200-day MA filter is either "
               "all in or all out; the volatility target holds less whenever volatility is above its own norm.")

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
    st.caption("Every year the HMM is refit on all data before that year and finds its hidden states. Each "
               "state is labeled from its *training* behaviour: the best-earning state (highest Sharpe) → Bull, "
               "the worst → Crisis; the others are Bull only if they beat simply holding the asset, Crisis if "
               "they lost money, otherwise Transition. Height = the state's training volatility; dot size = "
               "share of training days in that state.")
    st.markdown("#### Which regimes the strategy held, year by year")
    folds = load("load_folds", RUN_ID, asset)
    if {"w_bull", "w_transition", "w_crisis"} <= set(folds.columns) and folds["w_bull"].notna().any():
        held = pd.DataFrame({"Trades in": folds["test_start"].dt.year,
                             "Bull": folds["w_bull"] > 0, "Transition": folds["w_transition"] > 0,
                             "Crisis": folds["w_crisis"] > 0})
        n = len(held)
        st.caption(f"Each year a regime is held (✓) only if the asset's average return on that regime's "
                   f"*training* days was positive. {asset_name(asset)}: Bull held in {int(held['Bull'].sum())} "
                   f"of {n} years, Transition in {int(held['Transition'].sum())}, Crisis in "
                   f"{int(held['Crisis'].sum())}.")
        st.dataframe(held, hide_index=True, use_container_width=True, height=36 * (n + 1) + 3,
                     column_config={"Trades in": st.column_config.NumberColumn(format="%d"),
                                    **{lab: st.column_config.CheckboxColumn(lab) for lab in REGIME_LABELS}})
    with st.expander("Fold details"):
        ft = pd.DataFrame({
            "Trades in": folds["test_start"].dt.year,
            "Trained on": (folds["train_start"].dt.strftime("%Y-%m-%d") + " → "
                           + folds["train_end"].dt.strftime("%Y-%m-%d")),
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
    ).sort_values(["_order", "baseline"])[["Asset", "Baseline", "sharpe_diff", "ci_lower", "ci_upper",
                                           "p_value", "p_value_hac", "p_value_holm"]]
    st.dataframe(st_table, hide_index=True, use_container_width=True,
                 column_config={"sharpe_diff": num_col("Sharpe diff."), "ci_lower": num_col("95% low"),
                                "ci_upper": num_col("95% high"), "p_value": num_col("p (bootstrap)", 3),
                                "p_value_hac": num_col("p (HAC)", 3), "p_value_holm": num_col("p (Holm)", 3)})


# ── 7. Portfolio (v2.0) ──────────────────────────────────────────────────

VERDICT_STYLE = {FAIL: st.error, SIGNIFICANT_PASS: st.success}

with tabs[6]:
    st.subheader("Portfolio (v2.0): does combining the signals help?")
    st.markdown(
        "The **regime-aware portfolio** holds plain risk-parity weights (1 / 63-day volatility) scaled by each "
        "asset's own v1.0 regime exposure, with the rest in cash. It is compared with **equal weight** and "
        "**plain risk parity** on the same days, assets, costs and cash. Every rule below was written down "
        "and committed in `docs/v2_preregistration.md` before any portfolio result existed.")
    st.markdown("**The pass/fail rule, word for word:**")
    st.markdown(f"> {PASS_RULE}")

    if portfolio is None:
        st.info("**Not run yet.** This store has no portfolio results. They appear here once "
                "`python scripts/run_pipeline.py` builds the portfolio.")
    else:
        v, pwin = portfolio["verdict"], portfolio["window"]
        p_freeze = load_freeze(scope=SCOPE_PORTFOLIO)
        hist_end = p_freeze["data_end_at_freeze"] if p_freeze else data_end
        text = f"**Historical verdict ({pwin['oos_start']:%Y-%m-%d} to {hist_end}): {v.outcome}.**"
        if v.outcome == FAIL:
            text += " Not met: " + "; ".join(v.failed_conditions(PORTFOLIO_STRATEGY_NAMES)) + "."
        VERDICT_STYLE.get(v.outcome, st.info)(text)

        fv = portfolio["forward_verdict"]
        if fv is None:
            st.caption(f"Forward test: no trading days after {hist_end} yet.")
        elif fv.outcome == INTERIM:
            st.caption(f"Forward test: interim, {fv.days} of {fv.min_days} trading days. No verdict yet.")
        else:
            VERDICT_STYLE.get(fv.outcome, st.info)(f"**Forward verdict ({fv.days} trading days): {fv.outcome}.**")

        hist = portfolio["historical"]
        shown = [s for s in PORTFOLIO_STRATEGY_ORDER if s in hist.index]
        table = pd.DataFrame({
            "Strategy": [PORTFOLIO_STRATEGY_NAMES[s] for s in shown],
            "CAGR": hist.loc[shown, "CAGR"].to_numpy(), "Ann. Vol": hist.loc[shown, "Ann. Vol"].to_numpy(),
            "Sharpe": hist.loc[shown, "Sharpe"].to_numpy(), "Max DD": hist.loc[shown, "Max DD"].to_numpy(),
            "Avg. Exposure": hist.loc[shown, "Avg. Exposure"].to_numpy(),
            "Ann. Turnover": hist.loc[shown, "Ann. Turnover"].to_numpy(),
        })
        st.dataframe(as_pct(table, ["CAGR", "Ann. Vol", "Max DD", "Avg. Exposure"]), hide_index=True,
                     use_container_width=True,
                     column_config={"CAGR": pct_col(), "Ann. Vol": pct_col(), "Sharpe": num_col("Sharpe (above cash)"),
                                    "Max DD": pct_col(), "Avg. Exposure": pct_col("Avg. invested", 0),
                                    "Ann. Turnover": num_col("Ann. turnover (x)", 1)})
        st.caption("Historical window: every day up to the v2.0 freeze. The per-asset signals were developed "
                   "on these same years, so this verdict is only half blind; the forward test is the clean one.")

        psig = portfolio["significance"]
        if not psig.empty:
            st.dataframe(psig.assign(Baseline=psig["baseline"].map(PORTFOLIO_STRATEGY_NAMES))[
                ["Baseline", "sharpe_diff", "ci_lower", "ci_upper", "p_value", "p_value_hac", "p_value_holm"]],
                hide_index=True, use_container_width=True,
                column_config={"sharpe_diff": num_col("Sharpe diff. (regime-aware minus baseline)"),
                               "ci_lower": num_col("95% low"), "ci_upper": num_col("95% high"),
                               "p_value": num_col("p (bootstrap)", 3), "p_value_hac": num_col("p (HAC)", 3),
                               "p_value_holm": num_col("p (Holm, 2 tests)", 3)})

        st.markdown("**Value over time** (all stored days, including any after the freeze)")
        plot(charts.performance_panels(load("load_daily", RUN_ID, PORTFOLIO_ID), charts.PORTFOLIO))
        st.markdown("**What each portfolio held** (the empty space above the stack is cash)")
        plot(charts.portfolio_weights_chart(load("load_portfolio_weights", RUN_ID)))


# ── 8. Paper trading ─────────────────────────────────────────────────────

with tabs[7]:
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
        st.dataframe(as_pct(log, ["target_weight", "current_weight", "new_weight", "p_bull"]), hide_index=True,
                     use_container_width=True,
                     column_config={"timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                                    "target_weight": pct_col("Target", 0), "current_weight": pct_col("Held before", 0),
                                    "new_weight": pct_col("Held after", 0), "symbol": "Symbol",
                                    "p_bull": pct_col("Smoothed P(Bull)", 0), "qty_ordered": "Qty ordered",
                                    "asset": "Asset", "side": "Side", "dry_run": "Dry run"})
