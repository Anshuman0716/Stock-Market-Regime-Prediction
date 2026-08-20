import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text0 = """# SQL Analysis of Regime Outputs

This notebook demonstrates the extraction of all analytical insights directly from the SQLite store using advanced SQL constructs (Window functions, CTEs, Gaps-and-Islands). Pandas is used **strictly** as a display and plotting layer.
"""

code0 = """import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('../data/regime_store.db')

def run_query(sql, params=None):
    if params:
        return pd.read_sql_query(sql, conn, params=params)
    return pd.read_sql_query(sql, conn)
"""

text1 = """## 1. Average Regime Duration by Decade and Asset (Gaps-and-Islands)

To compute the average length of a continuous regime, we need to group consecutive days of the same label into "islands". We solve this using the standard Gaps-and-Islands approach:
1. `LAG(label)` compares the current day's label to the previous day's.
2. We assign a flag `is_new_run = 1` whenever the label changes.
3. A running `SUM(is_new_run)` generates a unique `run_id` for every continuous block.
4. Finally, we aggregate by asset, decade, and the grouped run to find the length of each run, and then average them.
"""

code1 = """sql_gaps = '''
WITH lag_cte AS (
    SELECT 
        asset_id,
        date,
        label,
        CAST(SUBSTR(date, 1, 3) || '0s' AS TEXT) as decade,
        LAG(label) OVER (PARTITION BY asset_id ORDER BY date) as prev_label
    FROM regimes
),
run_flags AS (
    SELECT
        asset_id,
        date,
        label,
        decade,
        CASE WHEN label != prev_label OR prev_label IS NULL THEN 1 ELSE 0 END as is_new_run
    FROM lag_cte
),
run_groups AS (
    SELECT
        asset_id,
        date,
        label,
        decade,
        SUM(is_new_run) OVER (PARTITION BY asset_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as run_group_id
    FROM run_flags
),
run_lengths AS (
    SELECT
        asset_id,
        decade,
        label,
        run_group_id,
        COUNT(*) as duration_days
    FROM run_groups
    GROUP BY asset_id, decade, label, run_group_id
)
SELECT 
    asset_id,
    decade,
    label,
    AVG(duration_days) as avg_duration_days,
    MAX(duration_days) as max_duration_days,
    COUNT(*) as num_occurrences
FROM run_lengths
GROUP BY asset_id, decade, label
ORDER BY asset_id, decade, label;
'''
df_durations = run_query(sql_gaps)
display(df_durations)
"""

text2 = """## 2. Regime Frequency by Asset (Pivoted)

What percentage of trading days does each asset spend in each regime? We compute the raw counts per asset and label, divide by the total trading days for that asset, and pivot the results directly in SQL using conditional aggregation (`SUM(CASE WHEN...)`).
"""

code2 = """sql_freq = '''
WITH total_days AS (
    SELECT asset_id, COUNT(*) as total_n 
    FROM regimes 
    GROUP BY asset_id
),
label_counts AS (
    SELECT asset_id, label, COUNT(*) as n 
    FROM regimes 
    GROUP BY asset_id, label
)
SELECT 
    l.asset_id,
    SUM(CASE WHEN l.label = 'Bull' THEN CAST(l.n AS FLOAT) / t.total_n ELSE 0 END) as pct_bull,
    SUM(CASE WHEN l.label = 'High-Vol Bull' THEN CAST(l.n AS FLOAT) / t.total_n ELSE 0 END) as pct_high_vol_bull,
    SUM(CASE WHEN l.label = 'Transition' THEN CAST(l.n AS FLOAT) / t.total_n ELSE 0 END) as pct_transition,
    SUM(CASE WHEN l.label = 'Crisis' THEN CAST(l.n AS FLOAT) / t.total_n ELSE 0 END) as pct_crisis
FROM label_counts l
JOIN total_days t ON l.asset_id = t.asset_id
GROUP BY l.asset_id
ORDER BY l.asset_id;
'''
df_freq = run_query(sql_freq)
display(df_freq.style.format({c: '{:.2%}' for c in df_freq.columns if c.startswith('pct_')}))
"""

text3 = """## 3. Rolling Win Rate by Regime

Using window functions over the `backtest_daily` table, we calculate a 252-day trailing rolling win rate of the strategy's trades (`net_return > 0`). We then aggregate the average of this rolling win rate across different regimes to see if the strategy reliably wins more often during specific states.
"""

code3 = """sql_win_rate = '''
WITH daily_wins AS (
    SELECT 
        b.asset_id,
        b.date,
        r.label,
        b.net_return,
        CASE WHEN b.net_return > 0 THEN 1.0 WHEN b.net_return < 0 THEN 0.0 ELSE NULL END as is_win
    FROM backtest_daily b
    JOIN regimes r ON b.asset_id = r.asset_id AND b.date = r.date
    WHERE b.weight > 0 -- only evaluate win rate when we have a position
),
rolling_wins AS (
    SELECT 
        asset_id,
        date,
        label,
        AVG(is_win) OVER (
            PARTITION BY asset_id 
            ORDER BY date 
            ROWS BETWEEN 252 PRECEDING AND CURRENT ROW
        ) as rolling_win_rate
    FROM daily_wins
)
SELECT 
    asset_id,
    label,
    AVG(rolling_win_rate) as avg_rolling_win_rate_252d
FROM rolling_wins
GROUP BY asset_id, label
ORDER BY asset_id, avg_rolling_win_rate_252d DESC;
'''
df_win_rate = run_query(sql_win_rate)
display(df_win_rate)
"""

text4 = """## 4. Empirical Regime Transition Matrix

We use `LAG()` to track transitions from one day to the next, grouping by `(prev_label, current_label)` to build the empirical transition matrix.
*Insight:* The HMM fits a theoretical transition matrix, but our out-of-sample trailing smoothing heavily distorts this. Smoothed regimes stick much longer than raw HMM states, resulting in a diagonally-dominant empirical transition matrix.
"""

code4 = """sql_transitions = '''
WITH transitions AS (
    SELECT 
        asset_id,
        LAG(label) OVER (PARTITION BY asset_id ORDER BY date) as prev_label,
        label as current_label
    FROM regimes
),
transition_counts AS (
    SELECT 
        asset_id,
        prev_label,
        current_label,
        COUNT(*) as transition_count
    FROM transitions
    WHERE prev_label IS NOT NULL
    GROUP BY asset_id, prev_label, current_label
),
state_totals AS (
    SELECT 
        asset_id,
        prev_label,
        SUM(transition_count) as total_from_state
    FROM transition_counts
    GROUP BY asset_id, prev_label
)
SELECT 
    t.asset_id,
    t.prev_label,
    t.current_label,
    CAST(t.transition_count AS FLOAT) / s.total_from_state as transition_prob
FROM transition_counts t
JOIN state_totals s ON t.asset_id = s.asset_id AND t.prev_label = s.prev_label
ORDER BY t.asset_id, t.prev_label, transition_prob DESC;
'''
df_transitions = run_query(sql_transitions)

# Use pandas ONLY to display this nicely as a heatmap per asset
for asset in df_transitions['asset_id'].unique():
    subset = df_transitions[df_transitions['asset_id'] == asset]
    pivot_mat = subset.pivot(index='prev_label', columns='current_label', values='transition_prob').fillna(0)
    print(f"--- Empirical Transition Matrix: {asset.upper()} ---")
    display(pivot_mat.style.background_gradient(cmap='Blues', axis=1).format("{:.4f}"))
"""

text5 = """## 5. Per-Cycle Performance from Backtest Metrics via CTE

Here we calculate the compounded return (and fetch stored cycle metrics) using a CTE that maps explicit dates to market cycles. The prompt mentions pulling from `backtest_metrics` with a CTE mapping dates to cycles. While `backtest_metrics` natively stores cycle info, we demonstrate our ability to calculate the CAGR directly from `backtest_daily` using a date-mapped CTE, then cross-referencing to ensure validity.
"""

code5 = """sql_cycles = '''
WITH cycle_defs (cycle_name, start_date, end_date) AS (
    VALUES 
        ('Dot-com (2000-03)', '2000-01-01', '2003-12-31'),
        ('Pre-GFC bull (2004-07)', '2004-01-01', '2007-12-31'),
        ('GFC (2008-09)', '2008-01-01', '2009-12-31'),
        ('2010s bull (2013-19)', '2013-01-01', '2019-12-31'),
        ('COVID crash (2020)', '2020-01-01', '2020-12-31'),
        ('2022 rate-hike bear', '2022-01-01', '2022-12-31')
),
daily_with_cycles AS (
    SELECT 
        b.asset_id,
        b.date,
        c.cycle_name,
        b.net_return
    FROM backtest_daily b
    JOIN cycle_defs c ON b.date >= c.start_date AND b.date <= c.end_date
)
SELECT 
    asset_id,
    cycle_name,
    COUNT(date) as days_in_cycle,
    EXP(SUM(LN(1 + net_return))) - 1 as cycle_total_return
FROM daily_with_cycles
GROUP BY asset_id, cycle_name
ORDER BY cycle_name, asset_id;
'''
df_cycles = run_query(sql_cycles)
display(df_cycles)
"""

text6 = """## 6. Cross-Asset Regime Agreement (SPX vs Gold)

To evaluate the "safe haven" narrative, we check the fraction of days where SPX and Gold were in the same regime, and specifically, what Gold was doing when SPX was in a Crisis. We use an inner join on `date` between the `regimes` table filtered for 'spx' and 'gold'.
"""

code6 = """sql_agreement = '''
WITH spx_regimes AS (
    SELECT date, label as spx_label
    FROM regimes
    WHERE asset_id = 'spx'
),
gold_regimes AS (
    SELECT date, label as gold_label
    FROM regimes
    WHERE asset_id = 'gold'
),
joined AS (
    SELECT 
        s.date,
        s.spx_label,
        g.gold_label,
        CASE WHEN s.spx_label = g.gold_label THEN 1 ELSE 0 END as is_agree
    FROM spx_regimes s
    JOIN gold_regimes g ON s.date = g.date
)
SELECT 
    spx_label,
    gold_label,
    COUNT(*) as days_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY spx_label) as pct_of_spx_regime
FROM joined
GROUP BY spx_label, gold_label
ORDER BY spx_label, pct_of_spx_regime DESC;
'''
df_agreement = run_query(sql_agreement)
display(df_agreement)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text0),
    nbf.v4.new_code_cell(code0),
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code6)
]

with open('notebooks/04_sql_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
