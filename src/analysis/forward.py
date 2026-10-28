"""
The forward test: results only from days AFTER a design was frozen.

Why: Steps 1-8 each made design decisions after looking at results from the
same 2010-2026 out-of-sample period. Every decision was justified from
training data, but repeated rounds still bend a design towards the period
being looked at. The only result that design work cannot have influenced is
one on data that did not exist yet. config/freeze.json records when each
design was frozen and which configuration was frozen; the pipeline scores
every strategy on the days after that point as cycle "forward".

Several versions, side by side
------------------------------
freeze.json holds a LIST of frozen versions. Each has a `scope`:

- "assets"    (v1.0): each asset's HMM strategy on its own. Checked against
               the per-asset fingerprint (store.get_config_hash), which
               ignores the `portfolio:` config section.
- "portfolio" (v2.0): the regime-aware portfolio. Checked against the
               whole-config fingerprint (store.get_portfolio_hash).

So adding or freezing a portfolio version leaves v1.0's forward test valid,
and both are judged on the same new days.

The freeze file lives outside config.yaml on purpose: recording a freeze
must not change the configuration's hash (that would force a retrain), and
changing config.yaml afterwards must be detectable (the hashes then differ
and the forward test is reported as invalid).
"""
import json
import os

import pandas as pd

from src.db.store import get_config_hash, get_portfolio_hash

FREEZE_PATH = "config/freeze.json"
FORWARD_CYCLE = "forward"
# Portfolio days up to the v2.0 freeze: the window its historical verdict
# is read on (docs/v2_preregistration.md, Section 6).
HISTORICAL_CYCLE = "historical"

SCOPE_ASSETS = "assets"
SCOPE_PORTFOLIO = "portfolio"


def load_freezes(path=FREEZE_PATH):
    """Every frozen version, oldest first ([] if nothing is frozen).

    Reads both layouts: the list ({"versions": [...]}) and the single record
    v1.0 was frozen with, which counts as an "assets" version.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["versions"] if "versions" in raw else [raw]
    return [{"scope": SCOPE_ASSETS, **r} for r in records]


def load_freeze(path=FREEZE_PATH, scope=SCOPE_ASSETS):
    """The latest frozen version of one scope, or None.

    Defaults to the per-asset scope, so code written for v1.0 (one freeze
    record) keeps reading v1.0's record unchanged.
    """
    matching = [r for r in load_freezes(path) if r["scope"] == scope]
    return matching[-1] if matching else None


def config_fingerprint(cfg, scope):
    """The fingerprint a freeze of `scope` is checked against."""
    if scope == SCOPE_PORTFOLIO:
        return get_portfolio_hash(cfg)
    return get_config_hash(cfg)


def forward_start(freeze):
    """First calendar day that counts for the forward test (the day after the
    last data the design saw), or None without a freeze."""
    if not freeze:
        return None
    return pd.Timestamp(freeze["data_end_at_freeze"]) + pd.Timedelta(days=1)


def historical_end(freeze):
    """Last day of the window the design was built on, or None without a freeze."""
    if not freeze:
        return None
    return pd.Timestamp(freeze["data_end_at_freeze"])


def freeze_status(freeze, config_hash):
    """('none' | 'valid' | 'invalid', explanation) for the current configuration.

    `config_hash` must be the fingerprint matching the record's scope
    (config_fingerprint); freeze_status_for picks it for you.
    """
    if not freeze:
        return "none", "The design has not been frozen (no config/freeze.json)."
    if freeze.get("config_hash") != config_hash:
        return "invalid", (f"config.yaml has changed since {freeze['version']} was frozen on "
                           f"{freeze['frozen_on']}, so these results are no longer a forward test of "
                           f"{freeze['version']}. Start a new version with a new freeze record instead.")
    return "valid", (f"{freeze['version']} frozen on {freeze['frozen_on']}; forward test counts days "
                     f"after {freeze['data_end_at_freeze']}.")


def freeze_status_for(freeze, cfg):
    """freeze_status, checked against the fingerprint of the record's own scope."""
    if not freeze:
        return freeze_status(None, None)
    return freeze_status(freeze, config_fingerprint(cfg, freeze.get("scope", SCOPE_ASSETS)))
