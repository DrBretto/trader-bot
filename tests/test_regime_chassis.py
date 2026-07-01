"""PKT-3 — regime chassis activation regression tests.

Locks the three fixes that took the chassis from silently inert to live:
  1. load_brain_config merges config/regime_compatibility.json (the socket wire).
  2. assert_regime_chassis_loaded is a fail-loud gate: empty/missing table under
     native_two_stage ABORTS; no-op for other engines.
  3. build_forecast_bundle applies regime_score_mult != 1.0 and re-ranks mu; and
     health=None resolves to the explicit UNKNOWN_HEALTH_DEFAULT (not a silent
     .get(sym, 1.0)), honoring a reported score.
"""
import pandas as pd
import pytest

from src.brain import (
    load_brain_config,
    assert_regime_chassis_loaded,
    RegimeChassisInertError,
)
from src.brain.forecast_adapter import build_forecast_bundle, UNKNOWN_HEALTH_DEFAULT


def test_load_brain_config_merges_regime_compatibility():
    cfg = load_brain_config()
    table = cfg.get("regime_compatibility")
    assert table, "load_brain_config must merge a non-empty regime_compatibility"
    # the live regime label seen in production must resolve inside the table
    assert "choppy" in table


def test_startup_assertion_fires_on_empty_table_native_two_stage():
    cfg = load_brain_config()
    assert cfg.get("engine") == "native_two_stage"
    assert_regime_chassis_loaded(cfg)  # loaded -> passes

    empty = dict(cfg)
    empty["regime_compatibility"] = {}
    with pytest.raises(RegimeChassisInertError):
        assert_regime_chassis_loaded(empty)

    missing = dict(cfg)
    missing.pop("regime_compatibility", None)
    with pytest.raises(RegimeChassisInertError):
        assert_regime_chassis_loaded(missing)


def test_startup_assertion_noop_for_non_native_engine():
    cfg = load_brain_config()
    other = dict(cfg)
    other["engine"] = "tilt_adapter"
    other["regime_compatibility"] = {}
    other["genome"] = "tilt_adapter"
    # a non-two-stage engine legitimately carries no table — must NOT raise
    assert_regime_chassis_loaded(other)


def test_regime_tilt_reranks_mu():
    cfg = load_brain_config()
    table = cfg["regime_compatibility"]
    uni = pd.read_csv("config/universe.csv")
    # near-equal mu on a bond and an equity so the choppy tilt (equity 0.95,
    # bond 1.05) flips their order.
    mu = {"SPY": 0.010, "AGG": 0.0099}
    tilt = build_forecast_bundle("2026-07-01", mu, None, "choppy", uni,
                                 health_map={}, regime_compat=table)
    raw = build_forecast_bundle("2026-07-01", mu, None, "choppy", uni,
                                health_map={}, regime_compat=None)
    assert all(abs(v - 1.0) < 1e-12 for v in raw.regime_score_mult.values())
    assert any(abs(v - 1.0) > 1e-9 for v in tilt.regime_score_mult.values())
    raw_order = sorted(mu, key=lambda s: -raw.mu_M1[s])
    tilt_order = sorted(mu, key=lambda s: -(tilt.mu_M1[s] * tilt.regime_score_mult[s]))
    assert raw_order == ["SPY", "AGG"]
    assert tilt_order == ["AGG", "SPY"]  # bond boosted over near-equal equity


def test_health_none_uses_explicit_default_and_honors_reported():
    cfg = load_brain_config()
    uni = pd.read_csv("config/universe.csv")
    f = build_forecast_bundle("2026-07-01", {"SPY": 0.01, "XLK": 0.01}, None,
                              "choppy", uni, health_map={"SPY": 0.4},
                              regime_compat=cfg["regime_compatibility"])
    assert f.health["SPY"] == 0.4                      # reported score honored
    assert f.health["XLK"] == UNKNOWN_HEALTH_DEFAULT   # None -> explicit default
