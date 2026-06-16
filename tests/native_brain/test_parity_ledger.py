"""PKT-TB-008: the nightly parity residual ledger is produced by the closed loop.

Verifies the closed-loop controller behaviour (proportional ex-post correction)
and that the engine appends a well-formed nightly residual ledger row -- the F8
disclosure form LIVE_PREREG pre-registers as a forward read.
"""
from __future__ import annotations

from src.brain.engine import ParityController, ParityLedger, run_engine

_LEDGER_KEYS = {
    "date", "target_gross_frac", "realized_gross_frac", "gross_residual_frac",
    "realized_beta", "realized_sigma", "kappa", "lot_residual_dollars",
    "lot_infeasible", "n_held",
}


def test_controller_uncorrected_on_first_night():
    c = ParityController(gain=0.5)
    assert c.corrected_target(1.0, None) == 1.0


def test_controller_pulls_drift_back():
    """If last night realized BELOW target, the controller raises the new target;
    if ABOVE, it lowers it -- a proportional pull, not a one-signed re-leak.
    """
    c = ParityController(gain=0.5)
    # realized short of target -> push up
    assert c.corrected_target(1.0, 0.8) == 1.0 + (1.0 - 0.8) * 0.5
    # realized over target -> pull down
    assert c.corrected_target(1.0, 1.2) == 1.0 + (1.0 - 1.2) * 0.5
    # clipped to [lo, hi]
    assert c.corrected_target(1.0, -5.0) <= c.hi


def test_engine_appends_nightly_ledger_row(forecast, theta_sel, theta_size, portfolio):
    out = run_engine(forecast, theta_sel, theta_size, portfolio)
    entry = out.parity_ledger_entry
    assert set(entry.keys()) == _LEDGER_KEYS
    assert entry["date"] == "2026-06-16"
    # residual is realized - target, reported (not assumed away)
    assert abs(
        entry["gross_residual_frac"]
        - (entry["realized_gross_frac"] - entry["target_gross_frac"])
    ) < 1e-6


def test_persistent_ledger_round_trips_and_feeds_back(tmp_path, forecast, theta_sel, theta_size, portfolio):
    path = tmp_path / "parity_ledger.jsonl"
    ledger = ParityLedger(path=path)
    run_engine(forecast, theta_sel, theta_size, portfolio, parity_ledger=ledger)
    assert path.exists()
    assert ledger.last_realized_gross_frac() is not None

    # A fresh ledger reading the same file recovers the feedback term.
    reloaded = ParityLedger(path=path)
    assert reloaded.last_realized_gross_frac() == ledger.last_realized_gross_frac()

    # Second night consumes the prior realized as the controller feedback.
    out2 = run_engine(forecast, theta_sel, theta_size, portfolio, parity_ledger=ledger)
    assert len(ledger.entries) == 2
    assert out2.corrected_gross_frac > 0.0
