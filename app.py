import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import sqlite3
import os

# Insert path to allow imports from src
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.data.loader import load_config
from src.db.store import get_config_hash

st.set_page_config(
    page_title="Market Regime Detector",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Regime Detector")
st.markdown("Detects Bull, Transition and Crisis regimes using a Hidden Markov Model.")

cfg = load_config()

# ── 1. Manifest and Artifact Validation ───────────────────────────
MANIFEST_PATH = "models/manifest.json"

@st.cache_data
def validate_artifacts(cfg):
    if not os.path.exists(MANIFEST_PATH):
        return False, "Models manifest not found. Please run `python scripts/train.py`."
        
    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        return False, f"Error reading manifest: {e}"
        
    current_hash = get_config_hash(cfg)
    if manifest.get('config_hash') != current_hash:
        return False, "Config mismatch. The models were trained under a different config. Please run `python scripts/train.py`."
        
    return True, manifest

valid, msg_or_manifest = validate_artifacts(cfg)
if not valid:
    st.error(f"🚨 **Artifact Error:** {msg_or_manifest}")
    st.stop()

manifest = msg_or_manifest

# ── Sidebar controls ──────────────────────────────────────────────
st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Select Asset",
    ["^GSPC", "^IXIC", "GC=F", "BTC-USD"],
    format_func=lambda x: {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "GC=F":  "Gold",
        "BTC-USD": "Bitcoin"
    }[x]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model trained at:** {manifest.get('created_at', 'Unknown')[:16]}")
st.sidebar.markdown(f"**Git SHA:** {manifest.get('git_sha', 'Unknown')[:8]}")

# Map ticker to config key
ticker_map = {
    "^GSPC": "spx",
    "^IXIC": "nasdaq",
    "GC=F": "gold",
    "BTC-USD": "bitcoin"
}
asset_id = ticker_map[ticker]

# ── DB Loading ───────────────────────────────────────────────────
@st.cache_resource
def get_db_connection():
    return sqlite3.connect("data/regime_store.db", check_same_thread=False)

conn = get_db_connection()

# Every query is pinned to ONE run, so re-running the backfill under a new
# run_id never mixes rows from different experiments.
run_row = conn.execute("SELECT run_id FROM backtest_runs ORDER BY created_at DESC LIMIT 1").fetchone()
if run_row is None:
    st.error("No backtest runs found in the store. Run `python scripts/backfill_db.py`.")
    st.stop()
RUN_ID = run_row[0]

@st.cache_data
def load_timeline_data(asset_id, run_id):
    # Out-of-sample dates only: regimes and backtests both start at the
    # first walk-forward prediction.
    sql = """
        SELECT 
            p.date, 
            p.close, 
            r.label as regime_label,
            r.sp_bull as p_bull, r.sp_transition as p_transition, r.sp_crisis as p_crisis,
            r.target_weight,
            h.net_return as strat_net_return,
            bh.net_return as bh_net_return
        FROM prices p
        JOIN regimes r ON p.asset_id = r.asset_id AND p.date = r.date AND r.run_id = ?
        LEFT JOIN backtest_daily h  ON h.asset_id = p.asset_id AND h.date = p.date
                                   AND h.run_id = ? AND h.strategy = 'hmm'
        LEFT JOIN backtest_daily bh ON bh.asset_id = p.asset_id AND bh.date = p.date
                                   AND bh.run_id = ? AND bh.strategy = 'buy_hold'
        WHERE p.asset_id = ?
        ORDER BY p.date
    """
    df = pd.read_sql_query(sql, conn, params=(run_id, run_id, run_id, asset_id))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

@st.cache_data
def load_metrics(asset_id, run_id):
    sql = """SELECT strategy, metric_name, metric_value FROM backtest_metrics
             WHERE asset_id = ? AND run_id = ? AND cycle = 'full'"""
    df = pd.read_sql_query(sql, conn, params=(asset_id, run_id))
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index='strategy', columns='metric_name', values='metric_value')

@st.cache_data
def load_cycle_performance(asset_id, run_id):
    sql = """SELECT cycle, metric_name, metric_value FROM backtest_metrics
             WHERE asset_id = ? AND run_id = ? AND strategy = 'hmm' AND cycle != 'full'"""
    df = pd.read_sql_query(sql, conn, params=(asset_id, run_id))
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index='cycle', columns='metric_name', values='metric_value')

df = load_timeline_data(asset_id, RUN_ID)
metrics = load_metrics(asset_id, RUN_ID)
cycle_df = load_cycle_performance(asset_id, RUN_ID)

if df.empty:
    st.error(f"No data found in the store for {asset_id}. Ensure you've run the backfill script.")
    st.stop()

# ── 2. Current Regime & Probabilities ─────────────────────────────
st.markdown("### Current Market Regime")

current_row = df.iloc[-1]
current_regime = current_row['regime_label']
current_date   = df.index[-1].strftime("%B %d, %Y")
st.caption(f"As of {current_date} (latest date in the store). "
           f"Out-of-sample period shown: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}.")

color_map = {'Bull': '🟢', 'Transition': '🟡', 'Crisis': '🔴'}
# Smoothed values: the same signal the backtest trades (src/models/signal.py).
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Regime",  f"{color_map.get(current_regime,'⚪')} {current_regime}")
col2.metric("P(Bull)", f"{current_row.get('p_bull', 0.0):.2%}")
col3.metric("P(Transition)", f"{current_row.get('p_transition', 0.0):.2%}")
col4.metric("P(Crisis)", f"{current_row.get('p_crisis', 0.0):.2%}")
col5.metric("Target Exposure", f"{current_row.get('target_weight', 0.0):.0%}")

# ── 3. Regime chart ───────────────────────────────────────────────
st.markdown("### Regime Detection Chart (Walk-Forward)")

colors = {'Bull': '#2ecc71', 'Transition': '#f39c12', 'Crisis': '#e74c3c'}

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df.index, df['close'], color='black', linewidth=0.8, label='Asset Price')

# Faster fill_between logic
for regime in colors.keys():
    mask = df['regime_label'] == regime
    if mask.any():
        ax.fill_between(df.index, df['close'].min(), df['close'].max(),
                        where=mask, color=colors[regime], alpha=0.3)

patches = [mpatches.Patch(color=c, alpha=0.6, label=l) for l, c in colors.items()]
ax.legend(handles=patches, loc='upper left')
ax.set_title(f"{ticker} Price with Market Regimes")
ax.set_ylabel("Price")
ax.set_yscale('log')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── 4. Cumulative Performance vs Baselines ────────────────────────
st.markdown("### Strategy Performance (Post-Cost)")
st.markdown("Both lines cover the same out-of-sample dates and pay the same spread + commission costs.")

if not df['strat_net_return'].isna().all():
    df['cumulative_strat'] = (1 + df['strat_net_return'].fillna(0)).cumprod()
    # Buy & Hold comes from the store too (same engine, lag and costs),
    # rather than being recomputed here from prices.
    df['cumulative_market'] = (1 + df['bh_net_return'].fillna(0)).cumprod()

    fig2, ax2 = plt.subplots(figsize=(16, 4))
    ax2.plot(df.index, df['cumulative_market'], label='Buy & Hold',       linewidth=1.2)
    ax2.plot(df.index, df['cumulative_strat'],  label='Regime Strategy',  linewidth=1.2, linestyle='--')
    ax2.fill_between(df.index, 1, df['cumulative_market'].max(),
                     where=df['regime_label'] == 'Crisis',
                     color='red', alpha=0.1, label='Crisis Period')
    ax2.set_title("Cumulative Returns — Buy & Hold vs Regime Strategy")
    ax2.set_ylabel("Portfolio Value ($1 start)")
    ax2.set_yscale('log')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── 5. VaR/CVaR Panel & Headline Metrics ──────────────────────────
st.markdown("### Headline Risk & Performance Metrics")

if not metrics.empty:
    st.dataframe(metrics.style.format(precision=4), use_container_width=True)
else:
    st.info("Metrics not found in DB.")

# ── 6. Per-Cycle Table ────────────────────────────────────────────
st.markdown("### Per-Market-Cycle Breakdown")
if not cycle_df.empty:
    st.dataframe(cycle_df.style.format(precision=4), use_container_width=True)
else:
    st.info("Cycle metrics not found in DB.")
