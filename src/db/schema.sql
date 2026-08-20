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

CREATE TABLE IF NOT EXISTS regimes (
    asset_id TEXT,
    date TEXT,
    run_id TEXT,
    state_raw INTEGER,
    label TEXT,
    p_bull REAL,
    p_transition REAL,
    p_crisis REAL,
    PRIMARY KEY (asset_id, date, run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
);

CREATE TABLE IF NOT EXISTS backtest_daily (
    run_id TEXT,
    asset_id TEXT,
    date TEXT,
    weight REAL,
    gross_return REAL,
    net_return REAL,
    turnover REAL,
    cost REAL,
    PRIMARY KEY (run_id, asset_id, date),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id TEXT,
    asset_id TEXT,
    cycle TEXT,
    metric_name TEXT,
    metric_value REAL,
    PRIMARY KEY (run_id, asset_id, cycle, metric_name),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Indexes to speed up range scans on dates for specific assets
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_features_date ON features(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_regimes_date ON regimes(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_backtest_daily_date ON backtest_daily(asset_id, date);
