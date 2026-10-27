-- Regime Store Schema
-- 
-- Design choices:
-- 1. We use a LONG format for features (asset_id, date, feature_name, value). 
--    Tradeoff: While a wide format (one column per feature) is slightly faster for 
--    pandas to load directly without a pivot, the long format makes ablation 
--    queries far nicer and allows us to add or remove feature columns in 
--    the future without running ALTER TABLE commands.
-- 2. Idempotent upserts are handled via ON CONFLICT DO UPDATE using explicit 
--    UNIQUE constraints (or PRIMARY KEYs).
-- 3. Backtest tables carry a `strategy` column (hmm / buy_hold / dma200) so the
--    HMM and its baselines live side by side for the same run and dates.
-- 4. PRAGMA user_version is the schema version. store.init_db() rebuilds the
--    database when it does not match, because CREATE TABLE IF NOT EXISTS
--    would silently keep an old table layout.

PRAGMA user_version = 3;

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    name TEXT,
    calendar TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    asset_id TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (asset_id, date),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS features (
    asset_id TEXT,
    date TEXT,
    feature_name TEXT,
    value REAL,
    PRIMARY KEY (asset_id, date, feature_name),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    cost_model TEXT,
    notes TEXT
);

-- The trading signal (src/models/signal.py), one row per out-of-sample day.
-- raw_label / p_*  : unsmoothed filtered HMM output
-- label / sp_*     : trailing-smoothed, what the strategy and dashboard use
-- target_weight    : position decided at that day's close (= sp_bull)
CREATE TABLE IF NOT EXISTS regimes (
    asset_id TEXT,
    date TEXT,
    run_id TEXT,
    raw_label TEXT,
    p_bull REAL,
    p_transition REAL,
    p_crisis REAL,
    label TEXT,
    sp_bull REAL,
    sp_transition REAL,
    sp_crisis REAL,
    target_weight REAL,
    PRIMARY KEY (asset_id, date, run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
);

CREATE TABLE IF NOT EXISTS backtest_daily (
    run_id TEXT,
    asset_id TEXT,
    strategy TEXT NOT NULL,
    date TEXT,
    weight REAL,
    gross_return REAL,
    net_return REAL,
    turnover REAL,
    cost REAL,
    PRIMARY KEY (run_id, asset_id, strategy, date),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id TEXT,
    asset_id TEXT,
    strategy TEXT NOT NULL,
    cycle TEXT,
    metric_name TEXT,
    metric_value REAL,
    PRIMARY KEY (run_id, asset_id, strategy, cycle, metric_name),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- The out-of-sample window each asset was evaluated on. Every strategy for
-- a (run, asset) is scored over exactly this window.
CREATE TABLE IF NOT EXISTS evaluation_windows (
    run_id TEXT,
    asset_id TEXT,
    oos_start TEXT NOT NULL,
    oos_end TEXT NOT NULL,
    ann_factor INTEGER NOT NULL,
    PRIMARY KEY (run_id, asset_id),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Sharpe-difference tests of a strategy against each baseline.
CREATE TABLE IF NOT EXISTS significance_tests (
    run_id TEXT,
    asset_id TEXT,
    strategy TEXT,
    baseline TEXT,
    sharpe_diff REAL,
    ci_lower REAL,
    ci_upper REAL,
    p_value REAL,
    p_value_hac REAL,
    p_value_holm REAL,
    PRIMARY KEY (run_id, asset_id, strategy, baseline),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Indexes to speed up range scans on dates for specific assets
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_features_date ON features(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_regimes_date ON regimes(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_backtest_daily_date ON backtest_daily(asset_id, date);
