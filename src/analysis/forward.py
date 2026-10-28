"""
The forward test: results only from days AFTER the design was frozen.

Why: Steps 1-8 each made design decisions after looking at results from the
same 2010-2026 out-of-sample period. Every decision was justified from
training data, but repeated rounds still bend a design towards the period
being looked at. The only result that design work cannot have influenced is
one on data that did not exist yet. config/freeze.json records when the
design was frozen and which configuration was frozen; the pipeline scores
every strategy on the days after that point as cycle "forward".

The freeze file lives outside config.yaml on purpose: recording a freeze
must not change the configuration's hash (that would force a retrain), and
changing config.yaml afterwards must be detectable (the hashes then differ
and the forward test is reported as invalid).
"""
import json
import os

import pandas as pd

FREEZE_PATH = "config/freeze.json"
FORWARD_CYCLE = "forward"


def load_freeze(path=FREEZE_PATH):
    """The freeze record, or None if the design has not been frozen."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def forward_start(freeze):
    """First calendar day that counts for the forward test (the day after the
    last data the design saw), or None without a freeze."""
    if not freeze:
        return None
    return pd.Timestamp(freeze["data_end_at_freeze"]) + pd.Timedelta(days=1)


def freeze_status(freeze, config_hash):
    """('none' | 'valid' | 'invalid', explanation) for the current configuration."""
    if not freeze:
        return "none", "The design has not been frozen (no config/freeze.json)."
    if freeze.get("config_hash") != config_hash:
        return "invalid", (f"config.yaml has changed since {freeze['version']} was frozen on "
                           f"{freeze['frozen_on']}, so these results are no longer a forward test of "
                           f"{freeze['version']}. Start a new version with a new freeze record instead.")
    return "valid", (f"{freeze['version']} frozen on {freeze['frozen_on']}; forward test counts days "
                     f"after {freeze['data_end_at_freeze']}.")
