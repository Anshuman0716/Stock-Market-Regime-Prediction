"""
Guards on the real config/config.yaml and config/freeze.json.

Why test the config itself: the v2.0 rules were pre-registered
(docs/v2_preregistration.md) before any result existed, and the frozen
versions in freeze.json promise a forward test of an exact configuration.
These tests fail loudly if config.yaml drifts away from either, instead of
leaving it to someone noticing an "invalid" line in a report.
"""
import pytest

from src.analysis.forward import freeze_status_for, load_freezes
from src.backtest.portfolio import PortfolioSettings
from src.data.loader import load_config
from src.db.store import get_config_hash


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/config.yaml")


def test_portfolio_section_matches_the_preregistration(cfg):
    """config.yaml's portfolio section must equal the pre-registered values
    (PortfolioSettings defaults = pre-registration Section 13)."""
    assert PortfolioSettings.from_config(cfg) == PortfolioSettings()
    expected = PortfolioSettings()
    assert expected.assets == ("spx", "gold", "bitcoin")
    assert (expected.risk_parity_lookback, expected.max_signal_staleness, expected.execution_delay) == (63, 5, 2)
    assert (expected.ann_factor, expected.block_size, expected.n_bootstraps, expected.seed) == (252, 60, 2000, 42)
    assert (expected.alpha, expected.forward_min_days) == (0.05, 252)


def test_every_frozen_version_matches_the_current_config(cfg):
    """If this fails, config.yaml was changed after a version was frozen. Do
    not edit the freeze record to make it pass: add a new version instead."""
    versions = load_freezes("config/freeze.json")
    assert [v["version"] for v in versions] == ["v1.0", "v2.0"]
    for v in versions:
        status, message = freeze_status_for(v, cfg)
        assert status == "valid", message


def test_v1_fingerprint_survived_the_portfolio_section(cfg):
    """Adding the portfolio section left v1.0's frozen per-asset hash unchanged."""
    assert "portfolio" in cfg
    assert get_config_hash(cfg) == "5855083a7597a86e46a96e91f7680222"


def test_unknown_portfolio_setting_is_an_error(cfg):
    bad = dict(cfg, portfolio=dict(cfg["portfolio"], risk_parity_lookbak=63))
    with pytest.raises(KeyError, match="risk_parity_lookbak"):
        PortfolioSettings.from_config(bad)
    assert PortfolioSettings.from_config({"model": {}}) is None
