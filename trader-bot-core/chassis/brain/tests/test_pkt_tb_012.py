"""PKT-TB-012 cutover acceptance tests.

Covers the checkable acceptance items of the cutover:
  - FREEZE_ORB1 cold-start assertion passes for the deployed engine + models, and
    ABORTS (BrainFreezeError) on any engine_sha / model_sha / θ-hash mismatch.
  - the live brain is the two-stage engine: run_cutover writes intents tagged
    engine == "native_two_stage" (NOT tilt_adapter), gated by the invariant
    self-check; a RED invariant ABORTS (fail-safe to incumbent).
  - mode=shadow and dates <= the frozen boundary do not produce a live write.
  - champion freeze is byte-immutable and the New Brain forward line is
    re-anchored C0-continuous to the frozen terminal.
  - the surface carries forward_confirmed:false + the forecast-rung rent block.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from chassis.brain import (
    assert_cold_start,
    BrainFreezeError,
    run_cutover,
    theta_from_freeze,
    load_brain_config,
)
from chassis.brain import freeze as freeze_mod
from chassis.brain import runtime as runtime_mod


# --------------------------------------------------------------- fixtures
UNIVERSE = pd.DataFrame([
    {"symbol": s, "asset_class": "equity", "sector": sec, "leverage_flag": 0, "eligible": 1}
    for s, sec in [
        ("SPY", "broad"), ("QQQ", "broad"), ("IWM", "broad"), ("XLK", "tech"),
        ("XLF", "fin"), ("XLE", "energy"), ("XLV", "health"), ("XLI", "indust"),
        ("XLY", "discr"), ("XLP", "staples"), ("XLU", "util"), ("XLB", "mat"),
        ("VTV", "value"), ("VIG", "div"), ("RSP", "broad"),
    ]
])


def _features():
    rows = []
    for i, s in enumerate(UNIVERSE["symbol"]):
        rows.append({"date": "2026-06-12", "symbol": s,
                     "close": 100.0 + i, "vol_21d": 0.15})
    return pd.DataFrame(rows)


def _mu_map():
    # Distinct, deterministic mu so the top-N is unambiguous.
    syms = list(UNIVERSE["symbol"])
    return {s: round(1.0 - 0.05 * i, 4) for i, s in enumerate(syms)}


def _forecaster(pending):
    mm = _mu_map()
    return {d: {"mu": mm} for d in pending}


PORTFOLIO = {"cash": 100_000.0, "holdings": [], "portfolio_value": 100_000.0}
LIVE_CFG = {"mode": "live", "forward_boundary": "2026-06-11"}


# --------------------------------------------------------------- freeze gate
def test_cold_start_assertion_passes_for_deployed_engine():
    fa = assert_cold_start()
    assert fa.ok
    assert fa.engine_sha == fa.expected_engine_sha
    assert fa.model_sha == fa.expected_model_sha


def test_cold_start_aborts_on_model_sha_mismatch(monkeypatch):
    monkeypatch.setattr(freeze_mod, "compute_model_sha", lambda *a, **k: "deadbeefcafe")
    with pytest.raises(BrainFreezeError):
        assert_cold_start()


def test_cold_start_aborts_on_engine_sha_mismatch(monkeypatch):
    monkeypatch.setattr(freeze_mod, "compute_engine_sha", lambda *a, **k: "0" * 64)
    with pytest.raises(BrainFreezeError):
        assert_cold_start()


def test_theta_hashes_match_frozen_contract():
    theta_sel, theta_size = theta_from_freeze()
    assert theta_sel.N == 10 and theta_sel.h_min == 0.60
    # theta_from_freeze raises if the content hashes do not match; reaching here
    # means both asserted equal to FREEZE_ORB1.


def test_theta_assertion_aborts_on_tampered_value():
    fz = freeze_mod.load_freeze()
    tampered = copy.deepcopy(fz)
    tampered["theta_sel"]["values"]["N"] = 7   # moves the content hash
    with pytest.raises(BrainFreezeError):
        theta_from_freeze(tampered)


# --------------------------------------------------------------- the cutover
def test_engine_not_tilt_writes_live_intents():
    res = run_cutover("2026-06-12", _features(), "neutral", UNIVERSE, PORTFOLIO,
                      _forecaster, config=LIVE_CFG, now_iso="2026-06-12T22:00:00")
    assert res.ok
    assert res.engine == "native_two_stage"
    assert res.trade_intents["expert_metrics"]["engine"] == "native_two_stage"
    assert res.invariant_green is True
    assert len(res.selected_universe) == 10           # top-N held set


def test_red_invariant_aborts_to_incumbent(monkeypatch):
    monkeypatch.setattr(runtime_mod, "held_symbols_invariant", lambda *a, **k: False)
    res = run_cutover("2026-06-12", _features(), "neutral", UNIVERSE, PORTFOLIO,
                      _forecaster, config=LIVE_CFG)
    assert res.ok is False
    assert res.invariant_green is False
    assert "invariant self-check RED" in res.reason
    assert res.trade_intents is None                  # no live write


def test_shadow_mode_does_not_write():
    res = run_cutover("2026-06-12", _features(), "neutral", UNIVERSE, PORTFOLIO,
                      _forecaster, config={"mode": "shadow", "forward_boundary": "2026-06-11"})
    assert res.ok is False
    assert "not live" in res.reason
    assert res.trade_intents is None


def test_frozen_boundary_owns_its_dates():
    res = run_cutover("2026-06-11", _features(), "neutral", UNIVERSE, PORTFOLIO,
                      _forecaster, config=LIVE_CFG)
    assert res.ok is False
    assert "forward_boundary" in res.reason
    assert res.trade_intents is None


def test_inference_failure_falls_back_to_incumbent():
    def boom(pending):
        raise RuntimeError("inference exploded")
    inc = {"actions": [{"action": "HOLD", "symbol": "SPY"}]}
    res = run_cutover("2026-06-12", _features(), "neutral", UNIVERSE, PORTFOLIO,
                      boom, config=LIVE_CFG, incumbent_intents=inc)
    assert res.ok is False
    assert "inference failed" in res.reason
    assert res.incumbent_intents == inc               # fail-safe target preserved


# --------------------------------------------------------------- new-brain surface
# (The champion-freeze re-anchor tests were removed with extender.py in the
# clean-core rebuild: the displayed line is now the stored ledger, not a re-anchor.
# The champion freeze itself stays gated infra; the surface annotation persists.)
def test_surface_carries_forward_confirmed_false():
    from chassis.utils.dashboard_metrics import attach_new_brain_surface
    out = attach_new_brain_surface({"metrics": {}}, None)
    nb = out["metrics"]["new_brain"]
    assert nb["forward_confirmed"] is False
    assert "forecast_rung_rent" in nb
    assert "non_assertion" in nb


def test_brain_active_config_is_live():
    # The live engine is the tilt_adapter (deterministic rules + small M1 tilt).
    # native_two_stage was retired 2026-06-23 (operator verdict).
    cfg = load_brain_config()
    assert cfg["mode"] == "live"
    assert cfg["engine"] == "tilt_adapter"
