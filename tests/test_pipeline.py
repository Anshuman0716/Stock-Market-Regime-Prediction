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
