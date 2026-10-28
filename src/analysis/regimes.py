"""
Descriptive analysis of the out-of-sample regimes, computed once by the
pipeline and stored, so the dashboard and notebooks only read results.

Everything here describes what ALREADY happened in the out-of-sample
period, using the smoothed regime labels the strategy actually used. None
of it feeds back into the model or the trading signal.
"""
import numpy as np
import pandas as pd

from src.backtest.metrics import get_period_metrics
from src.models.signal import REGIME_LABELS


def regime_spells(labels):
    """
    Consecutive runs of the same regime.

    Returns one row per spell: label, start, end, days. A "spell" is how
    long the model stayed in one regime before switching, which tells you
    how persistent (or jumpy) the regimes are.
    """
    labels = labels.dropna()
    if labels.empty:
        return pd.DataFrame(columns=["label", "start", "end", "days"])
    spell_id = (labels != labels.shift()).cumsum()
    grouped = labels.groupby(spell_id)
    return pd.DataFrame({
        "label": grouped.first().values,
        "start": grouped.apply(lambda s: s.index[0]).values,
        "end": grouped.apply(lambda s: s.index[-1]).values,
        "days": grouped.size().values,
    })


def transition_matrix(labels):
    """
    Empirical day-to-day regime transitions: P(regime tomorrow | regime today).

    Counted from the smoothed labels, so it shows how the strategy's regime
    actually moved, not the HMM's internal (per-fold) transition matrix.
    Rows with no observations get probability NaN rather than a fake 0.

    Returns long format: from_label, to_label, n, probability.
    """
    labels = labels.dropna()
    pairs = pd.DataFrame({"from_label": labels.values[:-1], "to_label": labels.values[1:]})
    counts = pd.crosstab(pairs["from_label"], pairs["to_label"]).reindex(
        index=REGIME_LABELS, columns=REGIME_LABELS, fill_value=0)
    row_tot = counts.sum(axis=1)
    probs = counts.div(row_tot.where(row_tot > 0), axis=0)
    out = counts.stack().rename("n").to_frame()
    out["probability"] = probs.stack()
    out = out.reset_index()
    out.columns = ["from_label", "to_label", "n", "probability"]
    return out


def regime_profile(labels, features, feature_cols, ann_factor):
    """
    What each regime looked like: how often it occurred, how long its spells
    lasted, the asset's own return/volatility while in it, and the average
    value of each model feature.

    The return statistics are CONTEMPORANEOUS (return on the same day as the
    label). They describe the regimes; they are not a tradable result (the
    tradable result is the backtest, which applies a 2-day execution lag).

    Returns long format: label, stat_name, value.
    """
    labels = labels.dropna()
    feats = features.loc[labels.index]
    spells = regime_spells(labels)
    rows = []
    for label in REGIME_LABELS:
        mask = labels == label
        n = int(mask.sum())
        r = feats.loc[mask, "returns"]
        label_spells = spells[spells["label"] == label]
        stats = {
            "days": n,
            "share": n / len(labels) if len(labels) else np.nan,
            "n_spells": int(len(label_spells)),
            "avg_spell_days": float(label_spells["days"].mean()) if len(label_spells) else np.nan,
            "ann_return": float(r.mean() * ann_factor) if n else np.nan,
            "ann_vol": float(r.std(ddof=1) * np.sqrt(ann_factor)) if n > 1 else np.nan,
        }
        stats["sharpe"] = (stats["ann_return"] / stats["ann_vol"]
                           if n > 1 and stats["ann_vol"] > 0 else np.nan)
        for col in feature_cols:
            stats[f"mean_{col}"] = float(feats.loc[mask, col].mean()) if n else np.nan
        rows += [{"label": label, "stat_name": k, "value": v} for k, v in stats.items()]
    return pd.DataFrame(rows)


def period_breakdown(backtests, ann_factor, episodes=None, forward_start=None):
    """
    Metrics for every strategy, per calendar year and per stress episode.

    Parameters
    ----------
    backtests : dict[str, DataFrame]
        strategy -> output of run_backtest (out-of-sample window only).
    ann_factor : int
    episodes : dict[str, (start, end)] or None
        Named date windows (config: evaluation.stress_episodes). Episodes
        that don't overlap the out-of-sample window are skipped; partly
        overlapping ones are clipped to it and flagged with "Days".
    forward_start : Timestamp or None
        First day of the forward test (src/analysis/forward.py). Days from
        then on are scored as cycle "forward" once there are any.

    Returns
    -------
    list of (strategy, cycle, metrics_dict). cycle is "year:2020" or
    "episode:covid_2020".
    """
    out = []
    for strategy, bt in backtests.items():
        def metrics(chunk):
            return get_period_metrics(chunk["net_return"], chunk["executed_weight"], ann_factor,
                                      cash_returns=chunk.get("cash_return"))

        for year, chunk in bt.groupby(bt.index.year):
            out.append((strategy, f"year:{year}", metrics(chunk)))
        for name, (start, end) in (episodes or {}).items():
            chunk = bt.loc[start:end]
            if chunk.empty:
                continue
            out.append((strategy, f"episode:{name}", metrics(chunk)))
        if forward_start is not None:
            chunk = bt.loc[forward_start:]
            if not chunk.empty:
                out.append((strategy, "forward", metrics(chunk)))
    return out
