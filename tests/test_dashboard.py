"""
Tests for the dashboard: read-only queries, chart builders, generated
headlines, and a headless run of the real app.py against a small synthetic
store built by the real pipeline (download faked, no network).
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import yaml

import src.pipeline as pipeline
from src.dashboard import charts, data
from src.backtest.portfolio import PORTFOLIO_ID, PortfolioSettings
from src.dashboard.summary import headline_findings, portfolio_results, scoreboard_table
from src.dashboard.theme import PORTFOLIO_STRATEGY_ORDER, STRATEGY_ORDER

HERE = os.path.dirname(__file__)
APP = os.path.join(HERE, "..", "app.py")

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
    # Deliberately not in date order, to test chronological sorting.
    "evaluation": {"stress_episodes": {"covid_crash_2020": ["2020-02-19", "2020-03-23"],
                                       "q4_selloff_2018": ["2018-09-20", "2018-12-24"]}},
}


# The v2.0 portfolio over the two synthetic assets (fewer bootstrap draws to keep tests fast).
PORTFOLIO_CFG = {"assets": ["spx", "gold"], "calendar_asset": "spx", "risk_parity_lookback": 63,
                 "max_signal_staleness": 5, "execution_delay": 2, "ann_factor": 252,
                 "significance": {"block_size": 60, "n_bootstraps": 200, "seed": 42, "alpha": 0.05},
                 "forward_min_days": 252}


def _build_store(tmp, cfg):

    def read(name):
        return pd.read_csv(os.path.join(HERE, "data", name), index_col="Date", parse_dates=True)

    spx = read("sample_spx.csv").join(read("sample_vix.csv")["Close"].rename("VIX_Close"))
    gold = spx.copy()
    gold[["Open", "High", "Low", "Close"]] = gold[["Open", "High", "Low", "Close"]].iloc[::-1].values

    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "download_prices", lambda cfg, force_refresh=False: {"spx": spx, "gold": gold})
        pipeline.run_pipeline(config_path=str(cfg_path), db_path=str(tmp / "store.db"),
                              model_root=str(tmp / "models"), train=True)
    return str(tmp / "store.db")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Build one synthetic store with the real pipeline, shared by every test here.
    No portfolio section: this is also the 'portfolio not run yet' case."""
    return _build_store(tmp_path_factory.mktemp("dash"), CFG)


@pytest.fixture(scope="module")
def portfolio_store(tmp_path_factory):
    """The same store with the v2.0 portfolio built."""
    return _build_store(tmp_path_factory.mktemp("dash_portfolio"), dict(CFG, portfolio=PORTFOLIO_CFG))


def _raw(db, sql):
    with sqlite3.connect(db) as c:
        return pd.read_sql(sql, c)


# ── Data access ──────────────────────────────────────────────────────────

def test_loaders_return_exactly_what_is_stored(store):
    data.check_store(store)
    run = data.load_run(store)
    assert run["run_id"] == "main"

    # Metrics: the pivot holds the stored values unchanged.
    m = data.load_metrics(store, "main")
    raw = _raw(store, "SELECT * FROM backtest_metrics WHERE cycle='full' AND asset_id='spx' "
                      "AND strategy='hmm' AND metric_name='Sharpe'")
    assert m.loc[("spx", "hmm"), "Sharpe"] == raw["metric_value"][0]

    # Latest signal = the last regimes row of each asset, with its spell start.
    sig = data.load_latest_signals(store, "main")
    assert set(sig.index) == {"spx", "gold"}
    last = _raw(store, "SELECT * FROM regimes WHERE asset_id='spx' ORDER BY date DESC LIMIT 1").iloc[0]
    assert sig.loc["spx", "date"] == pd.Timestamp(last["date"])
    assert sig.loc["spx", "target_weight"] == last["target_weight"]
    assert sig.loc["spx", "spell_start"] <= sig.loc["spx", "date"]

    # Timeline covers every out-of-sample day, with prices joined in.
    tl = data.load_timeline(store, "main", "spx")
    assert len(tl) == _raw(store, "SELECT COUNT(*) n FROM regimes WHERE asset_id='spx'")["n"][0]
    assert tl["close"].notna().all() and tl.index.is_monotonic_increasing

    probs, _ = data.load_transitions(store, "main", "spx")
    assert np.allclose(probs.dropna(how="all").sum(axis=1), 1.0)

    prof = data.load_profiles(store, "main", "spx")
    assert (prof["days"] > 0).all()  # regimes that never occurred are left out

    periods = data.load_period_metrics(store, "main", "spx")
    assert set(periods["kind"]) <= {"year", "episode"}

    costs = data.load_cost_totals(store, "main", "spx")
    assert set(costs.index) == set(STRATEGY_ORDER)


def test_store_is_read_only(store):
    with data._connect(store) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM regimes")


def test_check_store_rejects_missing_or_outdated_database(tmp_path):
    with pytest.raises(data.StoreUnavailable):
        data.check_store(str(tmp_path / "nope.db"))
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as c:
        c.execute("PRAGMA user_version = 3")
    with pytest.raises(data.StoreUnavailable, match="schema"):
        data.check_store(str(old))


def test_paper_log_missing_is_empty(tmp_path):
    assert data.load_paper_trades(str(tmp_path / "none.csv")).empty


# ── Charts ───────────────────────────────────────────────────────────────

def test_charts_build_from_stored_rows(store):
    tl = data.load_timeline(store, "main", "spx")
    spells = data.load_spells(store, "main", "spx")
    fig = charts.regime_timeline(tl, spells, "spx")
    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["Close", "Bull", "Transition", "Crisis"]
    assert len(fig.layout.shapes) == len(spells)  # one shaded band per regime spell

    raw_fig = charts.regime_timeline(tl, spells, "spx", smoothed=False)
    assert np.allclose(raw_fig.data[1].y, tl["p_bull"])  # the toggle switches to raw probabilities

    daily = data.load_daily(store, "main", "spx")
    perf = charts.performance_panels(daily)
    assert len(perf.data) == 3 * len(STRATEGY_ORDER)  # each strategy: equity, drawdown, rolling Sharpe

    m = data.load_metrics(store, "main")
    for fig in [charts.tail_risk_chart(m.loc["spx"]), charts.exposure_chart(daily),
                charts.spell_duration_chart(spells),
                charts.fold_states_chart(data.load_fold_states(store, "main", "spx")),
                charts.transition_heatmap(*data.load_transitions(store, "main", "spx")),
                charts.asset_dot_plot(m, "Sharpe", ".2f"),
                charts.significance_forest(data.load_significance(store, "main")),
                charts.yearly_returns_chart(data.load_period_metrics(store, "main", "spx"))]:
        assert isinstance(fig, go.Figure) and len(fig.data) > 0


def test_episode_chart_is_chronological():
    rows = []
    for ep in ["covid_crash_2020", "q4_selloff_2018"]:
        for s in ["hmm", "buy_hold", "dma200"]:
            rows.append({"strategy": s, "kind": "episode", "period": ep, "Total Return": -0.1,
                         "Avg. Exposure": 0.5})
    fig = charts.episode_chart(pd.DataFrame(rows), CFG["evaluation"]["stress_episodes"])
    order = list(fig.layout.yaxis.categoryarray)
    # Top of a horizontal bar chart is the END of the category array: oldest on top.
    assert order[-1].startswith("Q4 selloff 2018") and order[0].startswith("Covid crash 2020")


# ── Generated headlines ──────────────────────────────────────────────────

def _metrics(values):
    idx = pd.MultiIndex.from_tuples([(a, s) for a in values for s in values[a]], names=["asset_id", "strategy"])
    return pd.DataFrame([values[a][s] for a, s in idx], index=idx)


def test_headline_counts_and_significance():
    base = {"Sharpe": 0.5, "Max DD": -0.3, "CAGR": 0.08, "Years": 10, "Avg. Exposure": 1.0}
    m = _metrics({
        "spx": {"hmm": {**base, "Sharpe": 0.7, "Max DD": -0.2}, "buy_hold": base, "dma200": base},
        "gold": {"hmm": {**base, "Sharpe": 0.4, "Max DD": -0.4}, "buy_hold": base, "dma200": base},
    })
    sig = pd.DataFrame({"asset_id": ["spx", "gold"], "baseline": ["buy_hold", "buy_hold"],
                        "sharpe_diff": [0.2, -0.1], "p_value_holm": [0.30, 1.0]})
    lines = headline_findings(m, sig, ["spx", "gold"])
    assert "**1 of 2** assets (S&P 500)" in lines[0]   # Sharpe
    assert "**1 of 2** assets (S&P 500)" in lines[1]   # smaller drawdown
    assert "**0 of 2** assets (none)" in lines[2]      # CAGR
    assert lines[3].startswith("**None**") and "0.30" in lines[3]

    sig.loc[0, "p_value_holm"] = 0.01
    lines = headline_findings(m, sig, ["spx", "gold"])
    assert "significantly **higher**" in lines[3] and "S&P 500" in lines[3]

    # A significant result AGAINST the HMM must never read as a win.
    sig.loc[0, "sharpe_diff"] = -0.2
    lines = headline_findings(m, sig, ["spx", "gold"])
    assert lines[3].startswith("**No** difference in the HMM's favour")
    assert "significantly **lower**" in lines[4]

    windows = pd.DataFrame({"oos_start": pd.to_datetime(["2010-01-04", "2010-01-04"])}, index=["spx", "gold"])
    board = scoreboard_table(m, sig, windows)
    assert list(board["Asset"]) == ["S&P 500", "Gold"]
    assert board.loc[0, "p vs B&H (Holm)"] == 0.01


# ── The real app, headless ───────────────────────────────────────────────

def test_app_renders_every_tab_for_every_asset(store, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REGIME_DB_PATH", store)
    monkeypatch.setenv("REGIME_PAPER_LOG", str(tmp_path / "no_log.csv"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8
    for asset in ["spx", "gold"]:
        at.sidebar.radio[0].set_value(asset).run()
        assert not at.exception, (asset, [e.value for e in at.exception])
    at.toggle[0].set_value(True).run()  # raw probabilities
    assert not at.exception


def _app_text(at):
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption]
    for kind in (at.info, at.error, at.success, at.warning):
        parts += [m.value for m in kind]
    return "\n".join(parts)


def test_app_says_portfolio_not_run_yet_without_results(store, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REGIME_DB_PATH", store)
    monkeypatch.setenv("REGIME_PAPER_LOG", str(tmp_path / "no_log.csv"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _app_text(at)
    assert "Not run yet" in text and "Historical verdict" not in text
    assert "strictly higher Sharpe ratio" in text      # the rule is shown even before any result


def test_app_shows_the_portfolio_verdict_and_charts(portfolio_store, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REGIME_DB_PATH", portfolio_store)
    monkeypatch.setenv("REGIME_PAPER_LOG", str(tmp_path / "no_log.csv"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8
    settings = PortfolioSettings.from_config(dict(CFG, portfolio=PORTFOLIO_CFG))
    expected = portfolio_results(portfolio_store, "main", settings)
    text = _app_text(at)
    assert f"{expected['verdict'].outcome}.**" in text and "Not run yet" not in text


def test_portfolio_results_are_read_and_judged_from_the_store(store, portfolio_store):
    settings = PortfolioSettings.from_config(dict(CFG, portfolio=PORTFOLIO_CFG))
    assert portfolio_results(store, "main", settings) is None
    res = portfolio_results(portfolio_store, "main", settings)
    assert list(res["historical"].index.sort_values()) == sorted(PORTFOLIO_STRATEGY_ORDER)
    assert len(res["significance"]) == 2 and res["forward_verdict"] is None
    # The verdict uses exactly the stored historical numbers.
    h = res["historical"]
    higher = all(h.loc["regime_aware", "Sharpe"] > h.loc[b, "Sharpe"] for b in ("equal_weight", "risk_parity"))
    shallower = all(h.loc["regime_aware", "Max DD"] >= h.loc[b, "Max DD"] for b in ("equal_weight", "risk_parity"))
    assert (res["verdict"].outcome != "Fail") == (higher and shallower)


def test_portfolio_charts_build_from_stored_rows(portfolio_store):
    daily = data.load_daily(portfolio_store, "main", PORTFOLIO_ID)
    fig = charts.performance_panels(daily, charts.PORTFOLIO)
    assert {t.name for t in fig.data} == {"Regime-aware portfolio", "Equal weight", "Risk parity"}
    weights = data.load_portfolio_weights(portfolio_store, "main")
    fig = charts.portfolio_weights_chart(weights)
    assert isinstance(fig, go.Figure) and {t.name for t in fig.data} == {"S&P 500", "Gold"}
    # Stacked weights never exceed 100%: the rest is cash.
    assert (weights.sum(axis=1) <= 1 + 1e-9).all()


def test_app_explains_a_missing_store_instead_of_crashing(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REGIME_DB_PATH", str(tmp_path / "missing.db"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("can't be shown" in e.value for e in at.error)


# ── README generator (scripts/update_readme.py) ──────────────────────────────

def test_readme_blocks_are_generated_from_the_store(store, tmp_path):
    import importlib.util
    path = os.path.join(HERE, "..", "scripts", "update_readme.py")
    spec = importlib.util.spec_from_file_location("update_readme", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    template = "\n\n".join(f"<!-- generated:{n} -->\nold text\n<!-- /generated:{n} -->" for n in mod.BLOCKS)
    st = mod.Store(db=store, variants=str(tmp_path / "missing.json"))
    out = mod.render(template, st)
    assert "old text" not in out and "nan" not in out.lower()
    assert "S&P 500" in out and "| **HMM regime strategy** |" in out
    assert mod.render(out, st) == out  # rendering is idempotent: --check passes right after a write


def _readme_module():
    import importlib.util
    path = os.path.join(HERE, "..", "scripts", "update_readme.py")
    spec = importlib.util.spec_from_file_location("update_readme", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_portfolio_block_has_no_numbers_before_the_portfolio_is_built(store, tmp_path):
    """AGENTS.md rule 5: no result in the README that code has not produced."""
    mod = _readme_module()
    block = mod.block_portfolio(mod.Store(db=store, variants=str(tmp_path / "missing.json")))
    assert "Not run yet" in block and "[TODO]" in block
    assert "Historical verdict" not in block and "%" not in block and "|" not in block


def test_readme_portfolio_block_quotes_the_rule_and_the_stored_verdict(portfolio_store, tmp_path):
    from src.analysis.verdict import PASS_RULE
    mod = _readme_module()
    st = mod.Store(db=portfolio_store, variants=str(tmp_path / "missing.json"))
    block = mod.block_portfolio(st)
    verdict = portfolio_results(portfolio_store, st.run_id, st.portfolio_settings)["verdict"]
    assert f"> {PASS_RULE}" in block
    assert f"): {verdict.outcome}.**" in block
    assert "| **Regime-aware portfolio** |" in block and "| Equal weight |" in block
    assert "nan" not in block.lower() and "Not run yet" not in block
