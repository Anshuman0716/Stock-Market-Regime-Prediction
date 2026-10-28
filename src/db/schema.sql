-- Regime Store Schema
--
-- Design choices:
-- 1. We use a LONG format for features (asset_id, date, feature_name, value)
--    and for regime profiles / metrics (name, value). A wide format is a bit
--    faster to load, but the long format lets us add or remove features and
--    metrics without ALTER TABLE.
-- 2. Idempotent upserts are handled via ON CONFLICT DO UPDATE on PRIMARY KEYs.
-- 3. Backtest tables carry a `strategy` column (hmm / buy_hold / dma200) so the
--    HMM and its baselines live side by side for the same run and dates.
--    The v2.0 portfolio is stored as one more asset, asset_id 'portfolio',
--    with strategies regime_aware / equal_weight / risk_parity, so every
--    per-asset table and chart helper works for it unchanged.
-- 4. PRAGMA user_version is the schema version. store.init_db() rebuilds the
--    database when it does not match, because CREATE TABLE IF NOT EXISTS
--    would silently keep an old table layout.
-- 5. Everything the dashboard shows is precomputed here by the pipeline, so
--    the app only reads; it never re-derives results.

PRAGMA user_version = 7;

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

-- One row per pipeline run. models_trained = 1 when the HMMs were refit,
-- 0 when saved models were reused for a fast refresh.
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    cost_model TEXT,
    notes TEXT,
    data_end TEXT,
    models_trained INTEGER,
    config_json TEXT
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

-- Daily backtest per strategy, plus chart-ready series:
-- equity ($1 start), drawdown, trailing one-year Sharpe (above cash).
-- cash_return is what uninvested capital earned that day (T-bill rate), so
-- excess returns (net_return - cash_return) can be recomputed from the table.
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
    equity REAL,
    drawdown REAL,
    rolling_sharpe REAL,
    cash_return REAL,
    PRIMARY KEY (run_id, asset_id, strategy, date),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- cycle = 'full', 'year:2020' or 'episode:covid_crash_2020'
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
-- a (run, asset) is scored over exactly this window, trading `instrument`
-- (e.g. SPY for the S&P 500) with idle cash earning `cash_ticker`.
CREATE TABLE IF NOT EXISTS evaluation_windows (
    run_id TEXT,
    asset_id TEXT,
    oos_start TEXT NOT NULL,
    oos_end TEXT NOT NULL,
    ann_factor INTEGER NOT NULL,
    cost_bps REAL,
    instrument TEXT,
    cash_ticker TEXT,
    PRIMARY KEY (run_id, asset_id),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Sharpe-difference tests of a strategy against each baseline.
-- family: which tests were Holm-corrected together. 'asset' = each asset's
-- HMM vs its baselines (v1.0); 'portfolio' = the regime-aware portfolio vs
-- the two portfolio baselines (v2.0). Kept apart because they answer
-- different questions (docs/v2_preregistration.md, Section 9).
CREATE TABLE IF NOT EXISTS significance_tests (
    run_id TEXT,
    asset_id TEXT,
    strategy TEXT,
    baseline TEXT,
    family TEXT NOT NULL DEFAULT 'asset',
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

-- Weight held in each asset by each portfolio strategy on each day of the
-- portfolio window (the weight that earned that day's return, i.e. after
-- the two-day delay). The rest of the portfolio was in cash.
CREATE TABLE IF NOT EXISTS portfolio_weights (
    run_id TEXT,
    strategy TEXT NOT NULL,
    date TEXT,
    asset_id TEXT,
    weight REAL,
    PRIMARY KEY (run_id, strategy, date, asset_id),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- One row per walk-forward fold (one refit per year).
CREATE TABLE IF NOT EXISTS model_folds (
    run_id TEXT,
    asset_id TEXT,
    fold_idx INTEGER,
    train_start TEXT,
    train_end TEXT,
    test_start TEXT,
    test_end TEXT,
    best_log_likelihood REAL,
    n_starts INTEGER,
    n_converged INTEGER,
    n_bull_states INTEGER,
    n_transition_states INTEGER,
    n_crisis_states INTEGER,
    -- share of the asset held in each regime (from the training window)
    w_bull REAL,
    w_transition REAL,
    w_crisis REAL,
    PRIMARY KEY (run_id, asset_id, fold_idx),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Each raw HMM state of each fold: its TRAINING-window behaviour and the
-- regime label the labeling rule gave it.
CREATE TABLE IF NOT EXISTS fold_states (
    run_id TEXT,
    asset_id TEXT,
    fold_idx INTEGER,
    state INTEGER,
    label TEXT,
    train_days INTEGER,
    train_share REAL,
    ann_return REAL,
    ann_vol REAL,
    sharpe REAL,
    PRIMARY KEY (run_id, asset_id, fold_idx, state),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Out-of-sample description of each smoothed regime (long format):
-- days, share, n_spells, avg_spell_days, ann_return, ann_vol, sharpe,
-- mean_<feature>.
CREATE TABLE IF NOT EXISTS regime_profiles (
    run_id TEXT,
    asset_id TEXT,
    label TEXT,
    stat_name TEXT,
    value REAL,
    PRIMARY KEY (run_id, asset_id, label, stat_name),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Empirical P(regime tomorrow | regime today) from the smoothed labels.
CREATE TABLE IF NOT EXISTS regime_transitions (
    run_id TEXT,
    asset_id TEXT,
    from_label TEXT,
    to_label TEXT,
    n INTEGER,
    probability REAL,
    PRIMARY KEY (run_id, asset_id, from_label, to_label),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Consecutive runs of one smoothed regime (for timeline bands and durations).
CREATE TABLE IF NOT EXISTS regime_spells (
    run_id TEXT,
    asset_id TEXT,
    start_date TEXT,
    end_date TEXT,
    label TEXT,
    days INTEGER,
    PRIMARY KEY (run_id, asset_id, start_date),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Indexes to speed up range scans on dates for specific assets
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_features_date ON features(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_regimes_date ON regimes(asset_id, date);
CREATE INDEX IF NOT EXISTS idx_backtest_daily_date ON backtest_daily(asset_id, date);
