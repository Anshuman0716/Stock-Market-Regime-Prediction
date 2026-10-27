"""
The end-to-end pipeline, in the one direction the project allows:

    loader -> features -> walk-forward HMM -> signal -> evaluation
           -> regime analysis -> significance -> SQLite store (+ models/manifest)

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
from datetime import datetime

from src.data.loader import load_config, download_prices, get_cost_bps
from src.features.engineering import build_features, FEATURE_COLUMNS
from src.models.hmm import walk_forward_predict, predict_from_artifacts, RetrainRequired
from src.backtest.evaluation import evaluate_asset
from src.backtest.metrics import performance_series
from src.analysis.regimes import regime_profile, transition_matrix, regime_spells, period_breakdown
from src.stats.significance import run_significance_suite
from src.db.store import (
    DEFAULT_DB_PATH, init_db, get_db_connection, get_config_hash, get_git_sha,
    write_asset, write_prices, write_features, write_run, write_regimes,
    write_backtest_daily, write_backtest_metrics, write_evaluation_window,
    write_significance, write_folds, write_regime_analysis,
)

logger = logging.getLogger(__name__)

DEFAULT_RUN_ID = "main"


def build_asset_features(asset_df):
    """Model features plus the Close column the 200-day baseline needs."""
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
                 train=True, run_id=DEFAULT_RUN_ID, force_refresh=False):
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

    Returns
    -------
    dict with per-asset out-of-sample windows and HMM / buy-and-hold Sharpe.
    """
    cfg = load_config(config_path)
    config_hash = get_config_hash(cfg)

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

    new_manifest = {"created_at": None, "config_hash": config_hash, "git_sha": get_git_sha(), "assets": {}}
    returns_for_tests, ann_factors, summary = {}, {}, {}

    with get_db_connection(tmp_db) as conn:
        write_run(conn, run_id, cfg, data_end=data_end, models_trained=train,
                  notes="HMM vs buy_hold vs dma200, out-of-sample only")

        for asset in assets:
            logger.info(f"[{asset}] features")
            asset_df = data[asset]
            features = build_asset_features(asset_df)

            write_asset(conn, asset, ticker=cfg["tickers"].get(asset, asset))
            write_prices(conn, asset_df, asset)
            write_features(conn, features[FEATURE_COLUMNS], asset)

            if train:
                logger.info(f"[{asset}] training walk-forward folds")
                _, probs, folds = walk_forward_predict(
                    features, FEATURE_COLUMNS, cfg, asset_name=asset,
                    model_dir=os.path.join(staging_models, asset))
            else:
                logger.info(f"[{asset}] predicting with saved folds")
                folds = manifest["assets"][asset]["folds"]
                _, probs, folds = predict_from_artifacts(
                    features, FEATURE_COLUMNS, folds, os.path.join(model_root, asset))
            new_manifest["assets"][asset] = {
                "training_start": str(features.index[0].date()),
                "training_end": str(features.index[-1].date()),
                "folds": folds,
            }

            # Signal + every strategy on the same out-of-sample window.
            ev = evaluate_asset(features, probs, cfg, asset)
            ann = ev["ann_factor"]
            write_regimes(conn, ev["signal"], asset, run_id)
            write_evaluation_window(conn, run_id, asset, ev["oos_start"], ev["oos_end"], ann,
                                    cost_bps=get_cost_bps(cfg, asset))
            for strategy, bt in ev["backtests"].items():
                write_backtest_daily(conn, bt, asset, run_id, strategy=strategy,
                                     perf_df=performance_series(bt["net_return"], ann))
                m = {k: v for k, v in ev["metrics"][strategy].items() if k != "Asset"}
                write_backtest_metrics(conn, m, asset, run_id, cycle="full", strategy=strategy)

            # Per-year and per-stress-episode breakdown for every strategy.
            episodes = cfg.get("evaluation", {}).get("stress_episodes", {})
            for strategy, cycle, metrics in period_breakdown(ev["backtests"], ann, episodes):
                write_backtest_metrics(conn, metrics, asset, run_id, cycle=cycle, strategy=strategy)

            # What the regimes looked like out-of-sample.
            labels = ev["signal"]["label"].loc[ev["oos_start"]:]
            write_regime_analysis(
                conn, asset, run_id,
                profile=regime_profile(labels, features, FEATURE_COLUMNS, ann),
                transitions=transition_matrix(labels),
                spells=regime_spells(labels))
            write_folds(conn, folds, asset, run_id)

            returns_for_tests[asset] = {s: bt["net_return"] for s, bt in ev["backtests"].items()}
            ann_factors[asset] = ann
            summary[asset] = {
                "oos_start": str(ev["oos_start"].date()), "oos_end": str(ev["oos_end"].date()),
                "sharpe_hmm": ev["metrics"]["hmm"]["Sharpe"],
                "sharpe_buy_hold": ev["metrics"]["buy_hold"]["Sharpe"],
            }

        # Significance across all assets at once so Holm can correct for
        # the full family of comparisons.
        write_significance(conn, run_significance_suite(returns_for_tests, ann_factors), run_id)

    # --- Publish: models first, then manifest, then the database ----------
    if train:
        for asset in assets:
            live_dir = os.path.join(model_root, asset)
            if os.path.exists(live_dir):
                shutil.rmtree(live_dir)
            shutil.move(os.path.join(staging_models, asset), live_dir)
        shutil.rmtree(staging_models, ignore_errors=True)

        new_manifest["created_at"] = datetime.now().isoformat()
        tmp_manifest = os.path.join(model_root, "manifest.json.building")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(new_manifest, f, indent=2, default=str)
        _replace_file(tmp_manifest, os.path.join(model_root, "manifest.json"))

    _replace_file(tmp_db, db_path)
    return summary
