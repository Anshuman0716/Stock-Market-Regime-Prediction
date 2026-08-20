import streamlit as st
import pandas as pd
import numpy as np
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
        return False, f"Config mismatch. The models were trained under a different config. Please run `python scripts/train.py`."
        
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

@st.cache_data
def load_timeline_data(asset_id):
    # Join prices and regimes and backtest_daily
    sql = """
        SELECT 
            p.date, 
            p.close, 
            r.label as regime_label, 
            r.p_bull, r.p_transition, r.p_crisis,
            b.net_return as strat_net_return
        FROM prices p
        JOIN regimes r ON p.asset_id = r.asset_id AND p.date = r.date
        LEFT JOIN backtest_daily b ON p.asset_id = b.asset_id AND p.date = b.date
        WHERE p.asset_id = ?
        ORDER BY p.date
    """
    df = pd.read_sql_query(sql, conn, params=(asset_id,))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

@st.cache_data
def load_metrics(asset_id):
    sql = "SELECT metric_name, metric_value FROM backtest_metrics WHERE asset_id = ? AND cycle = 'full'"
    df = pd.read_sql_query(sql, conn, params=(asset_id,))
    return df.set_index('metric_name')['metric_value'].to_dict()

@st.cache_data
def load_cycle_performance(asset_id):
    sql = "SELECT cycle, metric_name, metric_value FROM backtest_metrics WHERE asset_id = ? AND cycle != 'full'"
    df = pd.read_sql_query(sql, conn, params=(asset_id,))
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index='cycle', columns='metric_name', values='metric_value')

df = load_timeline_data(asset_id)
metrics = load_metrics(asset_id)
cycle_df = load_cycle_performance(asset_id)

if df.empty:
    st.error(f"No data found in the store for {asset_id}. Ensure you've run the backfill script.")
    st.stop()

# ── 2. Current Regime & Probabilities ─────────────────────────────
st.markdown("### Current Market Regime")

current_row = df.iloc[-1]
current_regime = current_row['regime_label']
current_date   = df.index[-1].strftime("%B %d, %Y")

color_map = {'Bull': '🟢', 'Transition': '🟡', 'Crisis': '🔴'}
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Regime",  f"{color_map.get(current_regime,'⚪')} {current_regime}")
col2.metric("P(Bull)", f"{current_row.get('p_bull', 0.0):.2%}")
col3.metric("P(Transition)", f"{current_row.get('p_transition', 0.0):.2%}")
col4.metric("P(Crisis)", f"{current_row.get('p_crisis', 0.0):.2%}")

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
st.markdown("Returns are fully out-of-sample and factor in dynamic bid-ask spread and commission costs.")

if not df['strat_net_return'].isna().all():
    df['cumulative_strat'] = (1 + df['strat_net_return'].fillna(0)).cumprod()
    
    # We compute Buy & Hold directly from price ratio to match
    df['cumulative_market'] = df['close'] / df['close'].iloc[0]

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

if metrics:
    m_df = pd.DataFrame([metrics])
    st.dataframe(m_df.style.format(precision=4), use_container_width=True)
else:
    st.info("Metrics not found in DB.")

# ── 6. Per-Cycle Table ────────────────────────────────────────────
st.markdown("### Per-Market-Cycle Breakdown")
if not cycle_df.empty:
    st.dataframe(cycle_df.style.format(precision=4), use_container_width=True)
else:
    st.info("Cycle metrics not found in DB.")
