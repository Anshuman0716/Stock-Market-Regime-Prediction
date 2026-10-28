"""
End-to-end tests of src/pipeline.py on the small synthetic sample data,
plus unit tests for the regime analysis helpers.

The download step is replaced by a fake so the tests never touch the network.
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import yaml

import src.pipeline as pipeline
from src.analysis.regimes import regime_spells, transition_matrix, period_breakdown
from src.models.hmm import RetrainRequired, predict_from_artifacts
from src.backtest.portfolio import PORTFOLIO_ID, PORTFOLIO_STRATEGIES, PortfolioSettings
from src.dashboard import data
from src.stats.significance import adjust_pvalues, block_bootstrap_sharpe_diff

CFG = {
    "tickers": {"spx": "^GSPC", "gold": "GC=F", "vix": "^VIX"},
    "date_range": {"start": "2015-01-01", "end": None},
    "model": {"n_states": 3, "covariance_type": "diag", "n_iter": 20, "n_init": 2,
              "smoothing_window": 5, "min_train_days": 400},
    "backtest": {
        "assets": ["spx", "gold"],
        "annualization": {"default": 252},
        "costs": {"commission_bps": 0.5, "spread_bps": {"default": 2.0, "spx": 1.0}},
    },
    "evaluation": {"stress_episodes": {"covid_crash_2020": ["2020-02-19", "2020-03-23"],
                                       "before_data": ["2000-01-01", "2000-06-01"]}},
}

TABLES = ["assets", "prices", "features", "backtest_runs", "regimes", "backtest_daily",
          "backtest_metrics", "evaluation_windows", "significance_tests", "model_folds",
          "fold_states", "regime_profiles", "regime_transitions", "regime_spells"]


@pytest.fixture
def env(tmp_path, monkeypatch, sample_spx, sample_vix):
    """Temp config/db/models paths and a fake download returning sample data."""
    spx = sample_spx.join(sample_vix["Close"].rename("VIX_Close"))
    # A second asset with a different path so the two aren't identical.
    gold = spx.copy()
    gold[["Open", "High", "Low", "Close"]] = gold[["Open", "High", "Low", "Close"]].iloc[::-1].values
    fake = {"spx": spx, "gold": gold}
    monkeypatch.setattr(pipeline, "download_prices", lambda cfg, force_refresh=False: fake)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG))
    return {"config_path": str(cfg_path), "db_path": str(tmp_path / "store.db"),
            "model_root": str(tmp_path / "models"), "tmp": tmp_path}


def _run(env, **kw):
    return pipeline.run_pipeline(config_path=env["config_path"], db_path=env["db_path"],
                                 model_root=env["model_root"], **kw)


def _read(env, sql):
    with sqlite3.connect(env["db_path"]) as c:
        return pd.read_sql(sql, c)


def test_pipeline_fills_every_table_and_publishes_cleanly(env):
    summary = _run(env, train=True)
    assert set(summary) == {"spx", "gold"}
    for t in TABLES:
        assert _read(env, f"SELECT COUNT(*) n FROM {t}")["n"][0] > 0, t

    # Models, manifest and DB are published; no staging/building leftovers.
    assert os.path.exists(os.path.join(env["model_root"], "spx", "hmm_fold_0.joblib"))
    assert not os.path.exists(os.path.join(env["model_root"], ".staging"))
    assert not os.path.exists(env["db_path"] + ".building")
    manifest = json.load(open(os.path.join(env["model_root"], "manifest.json")))
    assert set(manifest["assets"]) == {"spx", "gold"}

    # Every strategy has full, yearly and episode metrics; an episode
    # outside the data window is skipped rather than stored empty.
    cycles = set(_read(env, "SELECT DISTINCT cycle FROM backtest_metrics")["cycle"])
    assert "full" in cycles and "episode:covid_crash_2020" in cycles
    assert any(c.startswith("year:") for c in cycles)
    assert "episode:before_data" not in cycles

    # Run record carries the real cost model, not a placeholder "0".
    run = _read(env, "SELECT cost_model, models_trained FROM backtest_runs")
    assert "commission" in run["cost_model"][0] and run["models_trained"][0] == 1

    # Equity curves start near $1 and drawdowns are never positive.
    daily = _read(env, "SELECT equity, drawdown FROM backtest_daily")
    assert (daily["drawdown"] <= 1e-12).all()


def test_instrument_and_cash_reach_the_store(env, monkeypatch):
    """With a traded fund and a cash rate configured, P&L follows the fund,
    idle cash earns the rate, and the store records both."""
    spx_index = pipeline.download_prices(None)["spx"]
    fund = spx_index["Close"] * 0.1 * (1 + 0.0001) ** np.arange(len(spx_index))  # a fund that slowly drifts
    fake = {"spx": spx_index.assign(Trade_Close=fund, Cash_Yield=4.0)}
    monkeypatch.setattr(pipeline, "download_prices", lambda cfg, force_refresh=False: fake)
    cfg = dict(CFG, backtest=dict(CFG["backtest"], assets=["spx"], instruments={"spx": "SPY"},
                                  cash_ticker="^IRX"))
    with open(env["config_path"], "w") as f:
        yaml.safe_dump(cfg, f)
    _run(env, train=True)

    win = _read(env, "SELECT instrument, cash_ticker FROM evaluation_windows")
    assert win.iloc[0].tolist() == ["SPY", "^IRX"]
    daily = _read(env, "SELECT strategy, date, weight, net_return, cost, cash_return FROM backtest_daily "
                       "WHERE strategy = 'buy_hold' ORDER BY date")
    assert np.allclose(daily["cash_return"].iloc[1:], 4.0 / 100 / 252)
    # Fully invested days earn the fund's return, not the index's.
    fund_r = fund.pct_change().reindex(pd.to_datetime(daily["date"])).to_numpy()
    full = (daily["weight"] == 1.0).to_numpy()
    assert np.allclose((daily["net_return"] + daily["cost"]).to_numpy()[full], fund_r[full])

    # Every baseline is tested against the HMM, Holm-adjusted together.
    sig = _read(env, "SELECT baseline FROM significance_tests")
    assert sorted(sig["baseline"]) == ["buy_hold", "dma200", "vol_target"]


def test_regime_sizing_is_stored_and_traded(env):
    """With regime-return sizing, every stored target weight equals the
    probability-weighted exposure using that year's stored regime weights."""
    cfg = dict(CFG, model=dict(CFG["model"], sizing="regime_returns", labeling={"rule": "sharpe"}))
    with open(env["config_path"], "w") as f:
        yaml.safe_dump(cfg, f)
    _run(env, train=True)

    folds = _read(env, "SELECT asset_id, test_start, w_bull, w_transition, w_crisis FROM model_folds")
    assert folds[["w_bull", "w_transition", "w_crisis"]].isin([0.0, 1.0]).all().all()
    reg = _read(env, "SELECT asset_id, date, sp_bull, sp_transition, sp_crisis, target_weight FROM regimes")
    reg["year"] = reg["date"].str[:4]
    folds["year"] = folds["test_start"].str[:4]
    m = reg.merge(folds, on=["asset_id", "year"])
    expected = (m.sp_bull * m.w_bull + m.sp_transition * m.w_transition + m.sp_crisis * m.w_crisis).clip(0, 1)
    expected = expected.where(expected >= 1e-3, 0.0)
    assert len(m) == len(reg) and np.allclose(m.target_weight, expected)


def test_forward_test_scores_only_days_after_the_freeze(env):
    """A freeze record in the middle of the sample data: days after it are
    stored as cycle "forward", matching the daily rows exactly."""
    freeze_path = env["tmp"] / "freeze.json"
    freeze_path.write_text(json.dumps({"version": "vT", "frozen_on": "2020-07-01",
                                       "data_end_at_freeze": "2020-06-30", "config_hash": "x"}))
    pipeline.run_pipeline(config_path=env["config_path"], db_path=env["db_path"], model_root=env["model_root"],
                          train=True, freeze_path=str(freeze_path))
    fwd = _read(env, "SELECT asset_id, strategy, metric_value FROM backtest_metrics "
                     "WHERE cycle = 'forward' AND metric_name = 'Days'")
    daily = _read(env, "SELECT asset_id, strategy, COUNT(*) n FROM backtest_daily WHERE date > '2020-06-30' "
                       "GROUP BY 1, 2")
    m = fwd.merge(daily, on=["asset_id", "strategy"])
    assert len(m) == len(daily) > 0 and (m["metric_value"] == m["n"]).all()


def test_reuse_models_reproduces_training_exactly(env):
    _run(env, train=True)
    trained_regimes = _read(env, "SELECT * FROM regimes ORDER BY asset_id, date")
    trained_metrics = _read(env, "SELECT * FROM backtest_metrics ORDER BY 1,2,3,4,5")

    _run(env, train=False)
    reused_regimes = _read(env, "SELECT * FROM regimes ORDER BY asset_id, date")
    reused_metrics = _read(env, "SELECT * FROM backtest_metrics ORDER BY 1,2,3,4,5")

    pd.testing.assert_frame_equal(trained_regimes, reused_regimes)
    pd.testing.assert_frame_equal(trained_metrics, reused_metrics)
    assert _read(env, "SELECT models_trained FROM backtest_runs")["models_trained"][0] == 0


def test_reuse_refuses_after_config_change(env):
    _run(env, train=True)
    changed = dict(CFG, model=dict(CFG["model"], smoothing_window=10))
    with open(env["config_path"], "w") as f:
        yaml.safe_dump(changed, f)
    with pytest.raises(RetrainRequired):
        _run(env, train=False)


def test_failed_run_leaves_published_store_untouched(env, monkeypatch):
    _run(env, train=True)
    before = _read(env, "SELECT COUNT(*) n FROM regimes")["n"][0]

    def boom(*a, **k):
        raise RuntimeError("simulated crash mid-run")
    monkeypatch.setattr(pipeline, "write_significance", boom)
    with pytest.raises(RuntimeError):
        _run(env, train=True)
    # The live DB and models are still the previous, complete ones.
    assert _read(env, "SELECT COUNT(*) n FROM regimes")["n"][0] == before
    assert os.path.exists(os.path.join(env["model_root"], "spx", "hmm_fold_0.joblib"))


def test_predict_from_artifacts_requires_a_fold_for_every_year(env, sample_spx):
    _run(env, train=True)
    manifest = json.load(open(os.path.join(env["model_root"], "manifest.json")))
    folds = manifest["assets"]["spx"]["folds"][:-1]  # drop the latest year's fold
    feats = pipeline.build_asset_features(
        sample_spx.join(pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "sample_vix.csv"),
                                    index_col="Date", parse_dates=True)["Close"].rename("VIX_Close")))
    with pytest.raises(RetrainRequired):
        predict_from_artifacts(feats, pipeline.FEATURE_COLUMNS, folds, os.path.join(env["model_root"], "spx"))


# --- regime analysis helpers -------------------------------------------------

def _labels(seq):
    return pd.Series(seq, index=pd.bdate_range("2021-01-01", periods=len(seq)))


def test_regime_spells():
    sp = regime_spells(_labels(["Bull", "Bull", "Crisis", "Crisis", "Crisis", "Bull"]))
    assert sp["label"].tolist() == ["Bull", "Crisis", "Bull"]
    assert sp["days"].tolist() == [2, 3, 1]


def test_transition_matrix_rows_sum_to_one():
    tm = transition_matrix(_labels(["Bull", "Bull", "Crisis", "Bull", "Transition", "Transition"]))
    sums = tm.groupby("from_label")["probability"].sum()
    np.testing.assert_allclose(sums.dropna(), 1.0)
    bull_to_bull = tm.query("from_label == 'Bull' and to_label == 'Bull'")
    assert bull_to_bull["n"].item() == 1 and bull_to_bull["probability"].item() == pytest.approx(1 / 3)


def test_period_breakdown_years_and_episodes():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    bt = pd.DataFrame({"net_return": 0.001, "executed_weight": 1.0}, index=idx)
    out = period_breakdown({"hmm": bt}, 252, {"covid": ("2020-02-19", "2020-03-23")})
    cycles = {c for _, c, _ in out}
    assert cycles == {"year:2020", "year:2021", "episode:covid"}
    covid = next(m for _, c, m in out if c == "episode:covid")
    assert covid["Days"] == len(bt.loc["2020-02-19":"2020-03-23"])
    assert covid["Total Return"] == pytest.approx(1.001 ** covid["Days"] - 1)


# --- v2.0 portfolio step -----------------------------------------------------

TWO_ASSET_PORTFOLIO = PortfolioSettings(assets=("spx", "gold"))
V1_TABLES = ["regimes", "backtest_daily", "backtest_metrics", "evaluation_windows", "significance_tests",
             "model_folds", "fold_states"]


def _v1_rows(env, table):
    """Every row of a table that belongs to a real asset, in a stable order."""
    df = _read(env, f"SELECT * FROM {table} WHERE asset_id != '{PORTFOLIO_ID}'")
    if "family" in df:
        df = df.drop(columns="family")
    return df.sort_values(list(df.columns[:4])).reset_index(drop=True)


def test_adding_the_portfolio_changes_no_v1_number(env):
    """The portfolio only reads the per-asset results. Every per-asset row,
    including the v1.0 Holm-adjusted p-values, must be identical with and
    without it."""
    _run(env, train=True, portfolio=None)
    without = {t: _v1_rows(env, t) for t in V1_TABLES}
    _run(env, train=False, portfolio=TWO_ASSET_PORTFOLIO)
    for t in V1_TABLES:
        pd.testing.assert_frame_equal(without[t], _v1_rows(env, t), obj=t)
    assert _read(env, f"SELECT COUNT(*) n FROM backtest_daily WHERE asset_id = '{PORTFOLIO_ID}'")["n"][0] > 0


def test_regime_aware_portfolio_trades_the_stored_signals(env):
    """Held regime-aware weight = held risk-parity weight x the asset's stored
    target_weight from two trading days earlier, on every day and asset."""
    _run(env, train=True, portfolio=TWO_ASSET_PORTFOLIO)
    w = _read(env, "SELECT strategy, date, asset_id, weight FROM portfolio_weights")
    w = w.pivot_table(index=["asset_id", "date"], columns="strategy", values="weight").reset_index()
    reg = _read(env, "SELECT asset_id, date, target_weight FROM regimes")
    for asset, g in w.groupby("asset_id"):
        g = g.sort_values("date").reset_index(drop=True)
        tw = reg[reg.asset_id == asset].set_index("date")["target_weight"]
        # Exposure decided two calendar rows before each held day.
        decided = tw.reindex(g["date"].shift(2)).to_numpy()
        expected = g["risk_parity"].to_numpy() * np.nan_to_num(decided)
        assert np.allclose(g["regime_aware"].to_numpy()[2:], expected[2:]), asset
        assert (g["regime_aware"] <= g["risk_parity"] + 1e-12).all()
        # Not a trivial pass: the signal really scales the position on some days
        # and leaves it on others.
        assert (g["regime_aware"] < g["risk_parity"] - 1e-6).any(), asset
        assert (g["regime_aware"] > 0).any(), asset

    # Equal weight: 50/50 once in, never more than 100% in total anywhere.
    eq = w.pivot_table(index="date", columns="asset_id", values="equal_weight").iloc[2:]
    assert np.allclose(eq.to_numpy(), 0.5)
    totals = _read(env, "SELECT strategy, date, SUM(weight) s FROM portfolio_weights GROUP BY 1, 2")
    assert (totals["s"] <= 1 + 1e-9).all()


def test_portfolio_window_metrics_and_significance_family(env):
    _run(env, train=True, portfolio=TWO_ASSET_PORTFOLIO)
    win = _read(env, f"SELECT * FROM evaluation_windows WHERE asset_id = '{PORTFOLIO_ID}'").iloc[0]
    first_oos = _read(env, "SELECT MIN(oos_start) d FROM evaluation_windows "
                           f"WHERE asset_id != '{PORTFOLIO_ID}'")["d"][0]
    assert win["oos_start"] == first_oos and win["ann_factor"] == 252

    # All three strategies are scored on the same dates, with full, yearly
    # and episode metrics, and no forward cycle before the v2.0 freeze.
    daily = _read(env, f"SELECT strategy, COUNT(*) n, MIN(date) a, MAX(date) b FROM backtest_daily "
                       f"WHERE asset_id = '{PORTFOLIO_ID}' GROUP BY strategy")
    assert set(daily["strategy"]) == set(PORTFOLIO_STRATEGIES)
    assert daily["n"].nunique() == 1 and daily["a"].nunique() == 1 and daily["b"].nunique() == 1
    cycles = set(_read(env, f"SELECT DISTINCT cycle FROM backtest_metrics WHERE asset_id = '{PORTFOLIO_ID}'")["cycle"])
    assert "full" in cycles and "episode:covid_crash_2020" in cycles and "forward" not in cycles

    # Two portfolio tests, Holm-corrected with each other only.
    sig = _read(env, "SELECT * FROM significance_tests WHERE family = 'portfolio'")
    assert sorted(sig["baseline"]) == ["equal_weight", "risk_parity"]
    assert (sig["strategy"] == "regime_aware").all() and (sig["asset_id"] == PORTFOLIO_ID).all()
    np.testing.assert_allclose(sig["p_value_holm"], adjust_pvalues(sig["p_value"].to_numpy()))
    per_asset = _read(env, "SELECT * FROM significance_tests WHERE family = 'asset'")
    assert len(per_asset) > 0 and PORTFOLIO_ID not in set(per_asset["asset_id"])


def test_existing_readers_do_not_see_the_portfolio(env):
    """Dashboard / README loaders list real assets and v1.0 tests only,
    unless the portfolio is asked for explicitly."""
    _run(env, train=True, portfolio=TWO_ASSET_PORTFOLIO)
    run_id = pipeline.DEFAULT_RUN_ID
    assert PORTFOLIO_ID not in data.load_windows(env["db_path"], run_id).index
    assert PORTFOLIO_ID not in data.load_metrics(env["db_path"], run_id).index.get_level_values(0)
    assert PORTFOLIO_ID in data.load_metrics(env["db_path"], run_id, include_portfolio=True).index.get_level_values(0)
    assert PORTFOLIO_ID in data.load_windows(env["db_path"], run_id, include_portfolio=True).index
    assert set(data.load_significance(env["db_path"], run_id)["family"]) == {"asset"}
    assert len(data.load_significance(env["db_path"], run_id, family="portfolio")) == 2
    w = data.load_portfolio_weights(env["db_path"], run_id, strategy="risk_parity")
    assert list(w.columns) == ["gold", "spx"] and np.allclose(w.iloc[2:].sum(axis=1), 1.0)


def test_portfolio_is_skipped_when_an_asset_is_missing(env, caplog):
    """The default (pre-registered) portfolio needs Bitcoin; this run has
    none, so the step is skipped with a warning instead of building a
    different portfolio than the one that was registered."""
    with caplog.at_level("WARNING"):
        summary = _run(env, train=True, portfolio=PortfolioSettings())
    assert PORTFOLIO_ID not in summary
    assert "Portfolio skipped" in caplog.text and "bitcoin" in caplog.text
    assert _read(env, "SELECT COUNT(*) n FROM portfolio_weights")["n"][0] == 0


def _write_cfg(env, cfg):
    with open(env["config_path"], "w") as f:
        yaml.safe_dump(cfg, f)


PORTFOLIO_SECTION = {"assets": ["spx", "gold"], "calendar_asset": "spx", "risk_parity_lookback": 63,
                     "max_signal_staleness": 5, "execution_delay": 2, "ann_factor": 252,
                     "significance": {"block_size": 60, "n_bootstraps": 200, "seed": 42, "alpha": 0.05},
                     "forward_min_days": 252}


def test_portfolio_section_in_config_is_used_and_needs_no_retrain(env):
    """Train without a portfolio section, then add one: the saved models are
    reused (the per-asset fingerprint ignores the section) and the portfolio
    is built from the config by default."""
    _run(env, train=True)
    assert _read(env, "SELECT COUNT(*) n FROM portfolio_weights")["n"][0] == 0
    _write_cfg(env, dict(CFG, portfolio=PORTFOLIO_SECTION))
    summary = _run(env, train=False)            # would raise RetrainRequired if the hash had changed
    assert PORTFOLIO_ID in summary
    assert _read(env, "SELECT COUNT(*) n FROM portfolio_weights")["n"][0] > 0


def _freeze_both(env, data_end, portfolio_hash="x"):
    path = env["tmp"] / "freeze.json"
    record = {"frozen_on": "2020-07-01", "data_end_at_freeze": data_end}
    path.write_text(json.dumps({"versions": [
        {"version": "v1", "scope": "assets", "config_hash": "x", **record},
        {"version": "v2", "scope": "portfolio", "config_hash": portfolio_hash, **record}]}))
    return str(path)


@pytest.mark.parametrize("min_days, has_forward_tests", [(130, True), (10_000, False)])
def test_portfolio_historical_and_forward_windows(env, min_days, has_forward_tests):
    """With a v2.0 freeze in the middle of the data: "historical" covers the
    days up to it, "forward" the days after it, the historical Sharpe tests
    use only historical days, and forward tests exist only once the forward
    window has forward_min_days."""
    settings = PortfolioSettings(assets=("spx", "gold"), n_bootstraps=200, forward_min_days=min_days)
    freeze_path = _freeze_both(env, "2019-12-31")
    pipeline.run_pipeline(config_path=env["config_path"], db_path=env["db_path"], model_root=env["model_root"],
                          train=True, freeze_path=freeze_path, portfolio=settings)

    daily = _read(env, f"SELECT strategy, date, net_return, cash_return FROM backtest_daily "
                       f"WHERE asset_id = '{PORTFOLIO_ID}' ORDER BY date")
    metrics = _read(env, f"SELECT strategy, cycle, metric_name, metric_value FROM backtest_metrics "
                         f"WHERE asset_id = '{PORTFOLIO_ID}'")

    def metric(strategy, cycle, name):
        row = metrics.query("strategy == @strategy and cycle == @cycle and metric_name == @name")
        return row["metric_value"].item()

    one = daily[daily.strategy == "regime_aware"]
    n_hist = int((one.date <= "2019-12-31").sum())
    n_fwd = int((one.date > "2019-12-31").sum())
    assert n_hist > 0 and n_fwd > 0
    assert metric("regime_aware", "historical", "Years") == pytest.approx(n_hist / 252)
    assert metric("regime_aware", "forward", "Days") == n_fwd
    assert metric("regime_aware", "full", "Years") == pytest.approx((n_hist + n_fwd) / 252)

    # The stored historical test is exactly the bootstrap on historical days.
    excess = {s: g.set_index("date").eval("net_return - cash_return")
              for s, g in daily[daily.date <= "2019-12-31"].groupby("strategy")}
    expected = block_bootstrap_sharpe_diff(excess["regime_aware"], excess["risk_parity"], 252, n_bootstraps=200)
    sig = _read(env, "SELECT family, baseline, sharpe_diff, p_value FROM significance_tests "
                     f"WHERE asset_id = '{PORTFOLIO_ID}'")
    hist = sig[(sig.family == "portfolio") & (sig.baseline == "risk_parity")].iloc[0]
    assert hist["sharpe_diff"] == pytest.approx(expected["diff"])
    assert hist["p_value"] == pytest.approx(expected["p_value"])
    assert set(sig.family) == ({"portfolio", "portfolio_forward"} if has_forward_tests else {"portfolio"})
    if has_forward_tests:
        assert len(sig[sig.family == "portfolio_forward"]) == 2
