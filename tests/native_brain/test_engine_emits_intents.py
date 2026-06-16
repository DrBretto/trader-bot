"""PKT-TB-008 acceptance: the two-stage module emits intents for a frozenset.

Verifies the engine emits ``trade_intents.json`` in the exact schema the morning
executor consumes, for exactly the selected (held) frozenset.
"""
from __future__ import annotations

import json

from src.brain.engine import run_engine, select

# Minimal field set the morning executor needs from every BUY intent.
_MORNING_REQUIRED = {"action", "symbol", "shares", "price"}
_INTENTS_REQUIRED_KEYS = {
    "generated_date", "generated_timestamp", "regime", "actions",
    "buy_candidates", "expert_metrics", "expires_after_days",
}


def test_emits_canonical_schema(forecast, theta_sel, theta_size, portfolio):
    out = run_engine(forecast, theta_sel, theta_size, portfolio, now_iso="2026-06-16T20:00:00")
    ti = out.trade_intents
    assert set(ti.keys()) == _INTENTS_REQUIRED_KEYS
    assert ti["generated_date"] == "2026-06-16"
    assert ti["regime"] == "risk_on_trend"
    assert ti["expires_after_days"] == 3
    assert isinstance(ti["actions"], list) and ti["actions"]


def test_intents_cover_exactly_the_frozenset(forecast, theta_sel, theta_size, portfolio):
    sel = select(forecast, theta_sel)
    out = run_engine(forecast, theta_sel, theta_size, portfolio)

    # The selected set is the expected frozenset (eligibility + gates applied).
    expected = frozenset({"ITA", "SOXX", "XRT", "RSP", "TLT", "AGG", "MUB", "FXI"})
    assert sel.selected_set == expected
    # Gated-out names are absent: SHY (ineligible), FXE/USO (health), KRE (M4 block).
    for gated in ("SHY", "FXE", "USO", "KRE"):
        assert gated not in sel.selected_set

    # held_symbols == selected_set minus the (here empty) lot-infeasible set.
    assert out.allocation.lot_infeasible == frozenset()
    assert out.allocation.held_symbols == expected

    # Every held name gets exactly one BUY/HOLD/REDUCE action (positions empty -> BUY).
    held_actions = {
        a["symbol"] for a in out.trade_intents["actions"]
        if a["action"] in ("BUY", "HOLD", "REDUCE")
    }
    assert held_actions == expected


def test_every_action_is_morning_executor_compatible(forecast, theta_sel, theta_size, portfolio):
    out = run_engine(forecast, theta_sel, theta_size, portfolio)
    for action in out.trade_intents["actions"]:
        missing = _MORNING_REQUIRED - set(action.keys())
        assert not missing, f"{action.get('symbol')} missing {missing}"
        assert isinstance(action["shares"], int)
        # dollars is reconstructable; when present it must be shares*price-consistent.
        if "dollars" in action and action["price"] > 0:
            assert abs(action["dollars"] - action["shares"] * action["price"]) < 0.01


def test_writes_files_to_out_dir(tmp_path, forecast, theta_sel, theta_size, portfolio):
    incumbent = {"generated_date": "2026-06-16", "actions": [{"action": "BUY", "symbol": "SPY"}]}
    run_engine(
        forecast, theta_sel, theta_size, portfolio,
        out_dir=tmp_path, incumbent_intents=incumbent,
    )
    intents_path = tmp_path / "trade_intents.json"
    incumbent_path = tmp_path / "trade_intents.incumbent.json"
    assert intents_path.exists() and incumbent_path.exists()
    written = json.loads(intents_path.read_text())
    assert written["regime"] == "risk_on_trend"
    # incumbent reference preserved for the ledgers.
    assert json.loads(incumbent_path.read_text())["actions"][0]["symbol"] == "SPY"
