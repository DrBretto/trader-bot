"""PKT-TB-007 — C5 lot-fix unit tests (BUILD_SPEC_007 §3.3 — must pass before
§8 step 3, the B0-EXPR exactness chain).

(i)   SCHD 2x738-share lots, mark $25, cash $10k => adapter w_prev/held_shares
      see 1,476 shares;
(ii)  full-exit SELL of 1,476 leaves 0 lots and credits 1476 x open;
(iii) BUY of a held symbol yields ONE lot, summed shares, weighted entry;
(iv)  regression: single-lot behavior byte-identical pre/post patch on a
      3-day synthetic replay.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.three_line_replay import replay_engine as RE          # noqa: E402
from src.utils.three_line_replay.replay_engine import Portfolio, Position  # noqa: E402
from lot_fix_007 import (aggregate_book, _execute_intents_lotfix,    # noqa: E402
                         harness_lot_patch)


def _lot(sym, shares, entry, peak=None, **kw):
    return Position(symbol=sym, shares=float(shares), entry_price=float(entry),
                    entry_date=kw.pop("entry_date", "2026-01-31"),
                    peak_price=float(peak if peak is not None else entry),
                    asset_class=kw.pop("asset_class", "equity"),
                    sector=kw.pop("sector", "broad"),
                    leverage_flag=kw.pop("leverage_flag", 0), **kw)


def _schd_two_lot_portfolio():
    return Portfolio(cash=10_000.0,
                     positions=[_lot("SCHD", 738, 24.0), _lot("SCHD", 738, 26.0)])


PARAMS = {"min_order_dollars": 250, "trailing_stop_base": 0.10,
          "trailing_stop_leveraged": 0.06}


# ---------------------------------------------------------------- (i) adapter side
def test_adapter_aggregates_multilot_w_prev_and_held_shares():
    pf = _schd_two_lot_portfolio()
    held, w_prev, nav, missing = aggregate_book(pf.positions, {"SCHD": 25.0},
                                                pf.cash)
    assert held["SCHD"] == pytest.approx(1476.0)
    assert nav == pytest.approx(10_000.0 + 1476 * 25.0)
    assert w_prev["SCHD"] == pytest.approx(1476 * 25.0 / nav)
    assert missing == []


# ---------------------------------------------------------------- (ii) full exit
def test_full_exit_sell_clears_all_lots_and_credits_open():
    pf = _schd_two_lot_portfolio()
    ohlc = {"SCHD": {"open": 25.50, "close": 25.40}}
    executed = _execute_intents_lotfix(
        pf, [{"symbol": "SCHD", "action": "SELL", "shares": 1476,
              "price": 25.45, "reason": "FULL_EXIT"}],
        ohlc, PARAMS, "2026-02-02")
    assert len(executed) == 1
    assert executed[0]["shares"] == pytest.approx(1476.0)
    assert executed[0]["dollars"] == pytest.approx(round(1476 * 25.50, 2))
    assert [p for p in pf.positions if p.symbol == "SCHD"] == []
    assert pf.cash == pytest.approx(10_000.0 + 1476 * 25.50)


def test_unfixed_harness_sells_only_last_lot_documenting_the_defect():
    """The defect the patch exists for: original _execute_intents clamps the
    SELL to the last lot (position_map last-lot-wins)."""
    pf = _schd_two_lot_portfolio()
    ohlc = {"SCHD": {"open": 25.50, "close": 25.40}}
    executed = RE._execute_intents(
        pf, [{"symbol": "SCHD", "action": "SELL", "shares": 1476,
              "price": 25.45, "reason": "FULL_EXIT"}],
        ohlc, PARAMS, "2026-02-02")
    assert executed[0]["shares"] == pytest.approx(738.0)        # the bug
    assert len([p for p in pf.positions if p.symbol == "SCHD"]) == 1


# ---------------------------------------------------------------- (iii) BUY merge
def test_buy_into_held_symbol_increments_single_lot_weighted_entry():
    pf = Portfolio(cash=50_000.0, positions=[_lot("ITA", 100, 140.0, peak=150.0)])
    ohlc = {"ITA": {"open": 145.0, "close": 146.0}}
    executed = _execute_intents_lotfix(
        pf, [{"symbol": "ITA", "action": "BUY", "shares": 50, "dollars": 7250.0,
              "price": 145.0, "reason": "ORB1_TILT_BUY"}],
        ohlc, PARAMS, "2026-02-03")
    lots = [p for p in pf.positions if p.symbol == "ITA"]
    assert len(lots) == 1                                       # ONE lot
    assert lots[0].shares == pytest.approx(150.0)               # summed
    assert lots[0].entry_price == pytest.approx((100 * 140.0 + 50 * 145.0) / 150)
    assert lots[0].peak_price == pytest.approx(150.0)           # max kept
    assert executed[0]["shares"] == pytest.approx(50.0)


# ---------------------------------------------------------------- (iv) regression
def _three_day_synthetic(execute):
    """3-day synthetic replay over single-lot books; returns serialized bytes."""
    pf = Portfolio(cash=100_000.0,
                   positions=[_lot("SPY", 50, 600.0, peak=640.0),
                              _lot("TLT", 100, 90.0, peak=95.0)])
    days = [
        ("2026-02-02",
         [{"symbol": "XLB", "action": "BUY", "shares": 40, "dollars": 3600.0,
           "price": 90.0, "reason": "SCORE_0.70"},
          {"symbol": "TLT", "action": "REDUCE", "shares": 100, "price": 91.0,
           "reason": "EXPOSURE_TRIM"}],
         {"XLB": {"open": 90.5, "close": 91.0}, "TLT": {"open": 91.2, "close": 91.5},
          "SPY": {"open": 610.0, "close": 612.0}}),
        ("2026-02-03",
         [{"symbol": "SPY", "action": "SELL", "shares": 50, "price": 612.0,
           "reason": "STOP_HIT"},
          {"symbol": "USO", "action": "BUY", "shares": 30, "dollars": 2400.0,
           "price": 80.0, "reason": "SCORE_0.66"}],
         {"SPY": {"open": 560.0, "close": 558.0}, "USO": {"open": 80.4, "close": 81.0},
          "XLB": {"open": 91.1, "close": 91.3}, "TLT": {"open": 91.6, "close": 91.4}}),
        ("2026-02-04",
         [{"symbol": "XLB", "action": "SELL", "shares": 40, "price": 91.3,
           "reason": "HEALTH_COLLAPSE"},
          {"symbol": "FAKE", "action": "BUY", "shares": 10, "dollars": 100.0,
           "price": 10.0, "reason": "SUB_MIN"}],
         {"XLB": {"open": 91.0, "close": 90.8}, "USO": {"open": 81.2, "close": 81.5},
          "TLT": {"open": 91.3, "close": 91.2}, "FAKE": {"open": 10.0, "close": 10.0}}),
    ]
    log = []
    for date, intents, ohlc in days:
        log.append(execute(pf, deepcopy(intents), ohlc, PARAMS, date))
    state = {"cash": round(pf.cash, 6),
             "positions": [{"symbol": p.symbol, "shares": p.shares,
                            "entry_price": round(p.entry_price, 8),
                            "peak_price": p.peak_price,
                            "entry_date": p.entry_date}
                           for p in pf.positions]}
    return json.dumps({"executed": log, "state": state},
                      sort_keys=True, default=str).encode()


def test_single_lot_regression_byte_identical_pre_post_patch():
    before = _three_day_synthetic(RE._execute_intents)
    after = _three_day_synthetic(_execute_intents_lotfix)
    assert before == after


def test_harness_lot_patch_swaps_and_restores():
    orig = RE._execute_intents
    with harness_lot_patch(RE):
        assert RE._execute_intents is _execute_intents_lotfix
    assert RE._execute_intents is orig
