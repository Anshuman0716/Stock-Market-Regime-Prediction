"""
The end-to-end pipeline, in the one direction the project allows:

    loader -> features -> walk-forward HMM -> signal -> evaluation
           -> regime analysis -> portfolio (v2.0) -> significance
           -> SQLite store (+ models/manifest)

The portfolio step runs after every asset and only REUSES their signals:
it trains nothing, so adding it cannot change any v1.0 number.

Why one pipeline instead of separate train / backfill scripts
-------------------------------------------------------------
Previously train.py fit every model and wrote the manifest, then
backfill_db.py fit every model AGAIN to fill the database. That doubled the
run time and meant the models on disk and the numbers in the database came
from two different fits. Now one run trains once and writes both, or, with
train=False, reuses the saved models for a fast daily refresh.

Two safety properties
---------------------
1. Models are trained into a staging folder and only replace the live
   models/ folders once EVERY asset has succeeded, so a crash halfway cannot
   leave a mix of old and new fold files.
2. The database is built as a separate file and swapped in at the end, so
   the dashboard never reads a half-written store.
"""
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.data.loader import load_config, download_prices, get_ann_factor, get_cost_bps
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict, predict_from_artifacts, regime_weight_frame, RetrainRequired
from src.backtest.evaluation import BASELINES, evaluate_asset, trading_returns
from src.backtest.metrics import performance_series
from src.backtest.portfolio import (
    PORTFOLIO_BASELINES, PORTFOLIO_ID, PORTFOLIO_STRATEGY, PortfolioSettings, as_backtest_frame,
    evaluate_portfolio, portfolio_metrics, slice_backtest,
)
from src.analysis.forward import (
    FREEZE_PATH, HISTORICAL_CYCLE, SCOPE_ASSETS, SCOPE_PORTFOLIO, forward_start, freeze_status_for,
    historical_end, load_freeze,
)
from src.analysis.regimes import regime_profile, transition_matrix, regime_spells, period_breakdown
from src.stats.significance import run_significance_suite
from src.db.store import (
    DEFAULT_DB_PATH, init_db, get_db_connection, get_config_hash, get_git_sha,
    write_asset, write_prices, write_features, write_run, write_regimes,
    write_backtest_daily, write_backtest_metrics, write_evaluation_window,
    write_significance, write_folds, write_regime_analysis, write_portfolio_weights,
)

logger = logging.getLogger(__name__)

DEFAULT_RUN_ID = "main"


def build_asset_features(asset_df):
    """Model features plus the index Close the 200-day baseline needs."""
    features = build_features(asset_df, asset_df["VIX_Close"])
    features["Close"] = asset_df["Close"]
    return features.dropna()


def _load_manifest(model_root):
    path = os.path.join(model_root, "manifest.json")
    if not os.path.exists(path):
        raise RetrainRequired(f"No manifest at {path}; run the pipeline with training first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _replace_file(src, dst):
    """Atomically move src over dst, with a clear message if dst is locked."""
    try:
        os.replace(src, dst)
    except PermissionError as e:
        raise PermissionError(
            f"Could not replace {dst} (is the dashboard or another program holding it open?). "
            f"The finished file is kept at {src}; close the other program and re-run.") from e


def run_pipeline(config_path="config/config.yaml", db_path=DEFAULT_DB_PATH, model_root="models",
                 train=True, run_id=DEFAULT_RUN_ID, force_refresh=False, freeze_path=FREEZE_PATH,
                 portfolio="config"):
    """
    Run everything and write the store (and, when training, the models).

    Parameters
    ----------
    train : bool
        True  -> refit every walk-forward fold (slow; needed after config or
                 feature changes, or when a new calendar year starts).
        False -> reuse saved fold models (fast daily refresh). Refuses if
                 the saved models were trained under a different config.
    force_refresh : bool
        Re-download prices instead of using today's parquet cache.
    portfolio : "config", PortfolioSettings or None
        The v2.0 portfolio to build from the per-asset signals. "config"
        (default) reads config.yaml's `portfolio:` section, and skips the
        portfolio if there is none; None skips it. It is also skipped, with
        a warning, if one of its assets has no data in this run.

    Returns
    -------
    dict with each asset's out-of-sample window and every strategy's Sharpe
    (above cash), plus a "portfolio" entry when the portfolio was built.
    """
    cfg = load_config(config_path)
    # The per-asset fingerprint: what the saved models were trained under.
    # It ignores the portfolio section, so portfolio edits never force a retrain.
    config_hash = get_config_hash(cfg)
    if isinstance(portfolio, str) and portfolio == "config":
        portfolio = PortfolioSettings.from_config(cfg)

    # Each frozen version is checked against its own fingerprint (forward.py).
    freeze = load_freeze(freeze_path, SCOPE_ASSETS)
    portfolio_freeze = load_freeze(freeze_path, SCOPE_PORTFOLIO)
    for record in (freeze, portfolio_freeze):
        status, message = freeze_status_for(record, cfg)
        if status == "invalid":
            logger.warning(message)

    manifest = None
    if not train:
        manifest = _load_manifest(model_root)
        if manifest.get("config_hash") != config_hash:
            raise RetrainRequired("Saved models were trained under a different config; retrain.")

    data = download_prices(cfg, force_refresh=force_refresh)
    assets = [a for a in cfg["backtest"]["assets"] if a in data]
    missing = set(cfg["backtest"]["assets"]) - set(assets)
    if missing:
        logger.warning(f"No price data for {sorted(missing)}; skipping them.")
    data_end = max(data[a].index[-1] for a in assets)

    staging_models = os.path.join(model_root, ".staging")
    if train and os.path.exists(staging_models):
        shutil.rmtree(staging_models)

    tmp_db = db_path + ".building"
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    init_db(tmp_db)

    run = _RunState(cfg=cfg, run_id=run_id, train=train, manifest=manifest, model_root=model_root,
                    staging_models=staging_models, forward_start=forward_start(freeze),
                    portfolio_freeze=portfolio_freeze,
                    new_manifest={"created_at": None, "config_hash": config_hash,
                                  "git_sha": get_git_sha(), "assets": {}})
    summary = {}

    with get_db_connection(tmp_db) as conn:
        write_run(conn, run_id, cfg, data_end=data_end, models_trained=train,
                  notes=f"HMM vs {', '.join(BASELINES)}, out-of-sample only")

        for asset in assets:
            try:
                summary[asset] = _run_asset(conn, run, asset, data[asset])
            except RetrainRequired:
                raise
            except Exception as e:
                # Name the asset: "KeyError: 'Close'" alone says nothing about where.
                raise RuntimeError(f"Pipeline failed for asset '{asset}': {type(e).__name__}: {e}") from e

        if portfolio is not None:
            missing_assets = [a for a in portfolio.assets if a not in run.portfolio_inputs]
            if missing_assets:
                logger.warning(f"Portfolio skipped: no results for {missing_assets} in this run.")
            else:
                try:
                    summary[PORTFOLIO_ID] = _run_portfolio(conn, run, portfolio)
                except Exception as e:
                    raise RuntimeError(f"Pipeline failed for the portfolio: {type(e).__name__}: {e}") from e

        # Significance across all assets at once so Holm can correct for
        # the full family of comparisons. The portfolio is its own family
        # (docs/v2_preregistration.md, Section 9), corrected separately.
        sig = [run_significance_suite(run.excess_returns, run.ann_factors, family="asset")]
        for family, excess in run.portfolio_excess.items():
            sig.append(run_significance_suite(
                {PORTFOLIO_ID: excess}, {PORTFOLIO_ID: portfolio.ann_factor},
                strategy=PORTFOLIO_STRATEGY, baselines=PORTFOLIO_BASELINES, family=family,
                block_size=portfolio.block_size, n_bootstraps=portfolio.n_bootstraps, seed=portfolio.seed))
        write_significance(conn, pd.concat(sig, ignore_index=True), run_id)

    _publish(run, assets, tmp_db, db_path)
    return summary


@dataclass
class _RunState:
    """What one pipeline run shares across assets."""
    cfg: dict
    run_id: str
    train: bool
    manifest: Optional[dict]
    model_root: str
    staging_models: str
    new_manifest: dict
    forward_start: Optional[object] = None  # first day of the forward test, if the design is frozen
    portfolio_freeze: Optional[dict] = None  # the v2.0 freeze record, if the portfolio is frozen
    excess_returns: dict = field(default_factory=dict)  # asset -> strategy -> daily returns above cash
    ann_factors: dict = field(default_factory=dict)
    # What the portfolio step needs from each asset, collected as assets finish.
    portfolio_inputs: dict = field(default_factory=dict)  # asset -> {price, signal, entry, asset_df}
    # Significance family -> portfolio strategy -> daily returns above cash.
    portfolio_excess: dict = field(default_factory=dict)


def _run_asset(conn, run, asset, asset_df):
    """Features -> walk-forward HMM -> evaluation -> store, for one asset."""
    cfg, run_id = run.cfg, run.run_id
    logger.info(f"[{asset}] features")
    features = build_asset_features(asset_df)

    write_asset(conn, asset, ticker=cfg["tickers"].get(asset, asset))
    write_prices(conn, asset_df, asset)
    write_features(conn, features[FEATURE_COLUMNS], asset)

    if run.train:
        logger.info(f"[{asset}] training walk-forward folds")
        _, probs, folds = walk_forward_predict(
            features, FEATURE_COLUMNS, cfg, asset_name=asset,
            model_dir=os.path.join(run.staging_models, asset))
    else:
        logger.info(f"[{asset}] predicting with saved folds")
        _, probs, folds = predict_from_artifacts(
            features, FEATURE_COLUMNS, run.manifest["assets"][asset]["folds"],
            os.path.join(run.model_root, asset))
    run.new_manifest["assets"][asset] = {
        "training_start": str(features.index[0].date()),
        "training_end": str(features.index[-1].date()),
        "folds": folds,
    }

    # Signal + every strategy on the same out-of-sample window, earning the
    # traded instrument's returns and the cash rate.
    ann = get_ann_factor(cfg, asset)
    asset_returns, cash_returns = trading_returns(asset_df, features.index, ann)
    ev = evaluate_asset(features, probs, cfg, asset, asset_returns=asset_returns, cash_returns=cash_returns,
                        regime_weights=regime_weight_frame(features.index, folds))
    bt_cfg = cfg.get("backtest", {})
    write_regimes(conn, ev["signal"], asset, run_id)
    write_evaluation_window(conn, run_id, asset, ev["oos_start"], ev["oos_end"], ann,
                            cost_bps=get_cost_bps(cfg, asset),
                            instrument=(bt_cfg.get("instruments") or {}).get(asset, cfg["tickers"].get(asset)),
                            cash_ticker=bt_cfg.get("cash_ticker"))
    for strategy, bt in ev["backtests"].items():
        write_backtest_daily(conn, bt, asset, run_id, strategy=strategy,
                             perf_df=performance_series(bt["net_return"], ann, bt["cash_return"]))
        m = {k: v for k, v in ev["metrics"][strategy].items() if k != "Asset"}
        write_backtest_metrics(conn, m, asset, run_id, cycle="full", strategy=strategy)

    # Per-year, per-stress-episode and forward-test breakdown for every strategy.
    episodes = cfg.get("evaluation", {}).get("stress_episodes", {})
    for strategy, cycle, metrics in period_breakdown(ev["backtests"], ann, episodes, run.forward_start):
        write_backtest_metrics(conn, metrics, asset, run_id, cycle=cycle, strategy=strategy)

    # What the regimes looked like out-of-sample.
    labels = ev["signal"]["label"].loc[ev["oos_start"]:]
    write_regime_analysis(
        conn, asset, run_id,
        profile=regime_profile(labels, features, FEATURE_COLUMNS, ann),
        transitions=transition_matrix(labels),
        spells=regime_spells(labels))
    write_folds(conn, folds, asset, run_id)

    run.excess_returns[asset] = {s: bt["net_return"] - bt["cash_return"] for s, bt in ev["backtests"].items()}
    run.ann_factors[asset] = ann
    # The portfolio reuses exactly what the HMM strategy traded: the same
    # target weights, from the first out-of-sample day only.
    run.portfolio_inputs[asset] = {
        "price": asset_df["Trade_Close"] if "Trade_Close" in asset_df else asset_df["Close"],
        "signal": ev["signal"]["target_weight"].loc[ev["oos_start"]:],
        "entry": ev["oos_start"],
        "asset_df": asset_df,
    }
    return {
        "oos_start": str(ev["oos_start"].date()), "oos_end": str(ev["oos_end"].date()),
        **{f"sharpe_{s}": ev["metrics"][s]["Sharpe"] for s in ev["metrics"]},
    }


def _run_portfolio(conn, run, settings):
    """Build the three v2.0 portfolios from the per-asset results and store them.

    The calendar is the trading days of the calendar asset's instrument
    (SPY), including the history before the window so risk parity has a
    volatility estimate on day one. Cash is that asset's T-bill series,
    turned into a daily rate with 252 because the portfolio lives on the
    stock calendar.
    """
    cfg, run_id = run.cfg, run.run_id
    logger.info(f"[{PORTFOLIO_ID}] building from {', '.join(settings.assets)}")
    inputs = {a: run.portfolio_inputs[a] for a in settings.assets}
    cal_df = run.portfolio_inputs[settings.calendar_asset]["asset_df"]
    cal_price = cal_df["Trade_Close"] if "Trade_Close" in cal_df else cal_df["Close"]
    calendar = cal_price.dropna().index
    _, cash = trading_returns(cal_df, calendar, settings.ann_factor)

    res = evaluate_portfolio(
        prices={a: x["price"] for a, x in inputs.items()},
        signals={a: x["signal"] for a, x in inputs.items()},
        entry_dates={a: x["entry"] for a, x in inputs.items()},
        calendar=calendar, cash_returns=cash,
        cost_bps={a: get_cost_bps(cfg, a) for a in settings.assets}, settings=settings)

    bt_cfg = cfg.get("backtest", {})
    instruments = bt_cfg.get("instruments") or {}
    traded = "+".join(instruments.get(a, cfg["tickers"].get(a, a)) for a in settings.assets)
    write_asset(conn, PORTFOLIO_ID, ticker=traded, name="Portfolio (v2.0)")
    write_evaluation_window(conn, run_id, PORTFOLIO_ID, res["oos_start"], res["oos_end"], settings.ann_factor,
                            cost_bps=None, instrument=traded, cash_ticker=bt_cfg.get("cash_ticker"))
    # The two windows the pre-registration judges on (Sections 6 and 10):
    # "historical" = days up to the v2.0 freeze, "forward" = days after it.
    # Without a v2.0 freeze every day is historical and there is no forward.
    hist_end = historical_end(run.portfolio_freeze)
    fwd_start = forward_start(run.portfolio_freeze)

    frames, hist_excess, fwd_excess = {}, {}, {}
    for strategy, bt in res["backtests"].items():
        frame = as_backtest_frame(bt)
        frames[strategy] = frame
        write_backtest_daily(conn, frame, PORTFOLIO_ID, run_id, strategy=strategy,
                             perf_df=performance_series(frame["net_return"], settings.ann_factor,
                                                        frame["cash_return"]))
        m = {k: v for k, v in res["metrics"][strategy].items() if k != "Asset"}
        write_backtest_metrics(conn, m, PORTFOLIO_ID, run_id, cycle="full", strategy=strategy)
        hist = portfolio_metrics(slice_backtest(bt, end=hist_end), settings.ann_factor)
        write_backtest_metrics(conn, {k: v for k, v in hist.items() if k != "Asset"}, PORTFOLIO_ID, run_id,
                               cycle=HISTORICAL_CYCLE, strategy=strategy)
        write_portfolio_weights(conn, bt.held_weights, run_id, strategy)

        excess = frame["net_return"] - frame["cash_return"]
        hist_excess[strategy] = excess.loc[:hist_end]
        if fwd_start is not None:
            fwd_excess[strategy] = excess.loc[fwd_start:]

    # The historical Sharpe tests use the historical window only, so later
    # (forward) days can never move the historical verdict. The forward
    # tests run once the forward window reaches forward_min_days; before
    # that a Sharpe difference is mostly noise (pre-registration, Section 10).
    run.portfolio_excess["portfolio"] = hist_excess
    n_forward = len(next(iter(fwd_excess.values()))) if fwd_excess else 0
    if n_forward >= settings.forward_min_days:
        run.portfolio_excess["portfolio_forward"] = fwd_excess

    # Years, stress episodes and the forward window, which for the
    # portfolio counts from its OWN freeze (v2.0), not v1.0's.
    episodes = cfg.get("evaluation", {}).get("stress_episodes", {})
    for strategy, cycle, metrics in period_breakdown(frames, settings.ann_factor, episodes, fwd_start):
        write_backtest_metrics(conn, metrics, PORTFOLIO_ID, run_id, cycle=cycle, strategy=strategy)

    return {
        "oos_start": str(res["oos_start"].date()), "oos_end": str(res["oos_end"].date()),
        **{f"sharpe_{s}": res["metrics"][s]["Sharpe"] for s in res["metrics"]},
    }


def _publish(run, assets, tmp_db, db_path):
    """Swap in the new models, then the manifest, then the database."""
    if run.train:
        for asset in assets:
            live_dir = os.path.join(run.model_root, asset)
            if os.path.exists(live_dir):
                shutil.rmtree(live_dir)
            shutil.move(os.path.join(run.staging_models, asset), live_dir)
        shutil.rmtree(run.staging_models, ignore_errors=True)

        run.new_manifest["created_at"] = datetime.now().isoformat()
        tmp_manifest = os.path.join(run.model_root, "manifest.json.building")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(run.new_manifest, f, indent=2, default=str)
        _replace_file(tmp_manifest, os.path.join(run.model_root, "manifest.json"))

    _replace_file(tmp_db, db_path)
