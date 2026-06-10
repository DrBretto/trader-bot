"""PKT-TB-006 — replay-wiring tests (deliverable #5).

Covers: Δw -> intent conversion math (never REDUCE, min_order, no_trade_band,
full-exit exception), the preliminary same-day-row guard (+ prices fallback),
the holdout guard refusal, and cost-overlay determinism + zero-cost == raw
identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_replay as RR                     # noqa: E402
from strategy_adapter import delta_to_intents, guarded_marks  # noqa: E402

SYMS = ["SPY", "QQQ", "TLT", "GLD"]
UNI = {s: {"symbol": s, "asset_class": "equity", "sector": "broad",
           "leverage_flag": 0} for s in SYMS}
MARKS = {"SPY": 500.0, "QQQ": 400.0, "TLT": 90.0, "GLD": 200.0}


# ------------------------------------------------------------- intent conversion
def test_buy_sell_math_and_never_reduce():
    nav = 100_000.0
    w_tgt = np.array([0.10, 0.00, 0.05, 0.02])
    w_prev = np.array([0.05, 0.04, 0.05, 0.00])
    held = {"QQQ": 10.0, "TLT": 55.0}
    intents = delta_to_intents(w_tgt, w_prev, nav, SYMS, MARKS, held, UNI,
                               no_trade_band=0.010, min_order=250.0)
    by = {i["symbol"]: i for i in intents}
    assert set(by) == {"SPY", "QQQ", "GLD"}          # TLT delta 0 -> band
    assert all(i["action"] in ("BUY", "SELL") for i in intents)  # NEVER REDUCE
    # BUY: delta 0.05 * 100k = $5000 @ 500 -> 10 integer shares
    assert by["SPY"]["action"] == "BUY"
    assert by["SPY"]["shares"] == 10
    assert by["SPY"]["dollars"] == pytest.approx(5000.0)
    assert by["SPY"]["price"] == pytest.approx(500.0)
    # full exit: w_tgt == 0 -> SELL ALL held shares (explicit)
    assert by["QQQ"]["action"] == "SELL"
    assert by["QQQ"]["shares"] == pytest.approx(10.0)
    # GLD BUY 0.02*100k = $2000 @ 200 -> 10 shares
    assert by["GLD"]["shares"] == 10
    # metadata routed from universe
    assert by["SPY"]["sector"] == "broad" and by["SPY"]["leverage_flag"] == 0


def test_partial_trim_is_sell_with_explicit_shares():
    nav = 100_000.0
    w_tgt = np.array([0.05, 0.0, 0.0, 0.0])
    w_prev = np.array([0.10, 0.0, 0.0, 0.0])
    held = {"SPY": 20.0}
    (it,) = delta_to_intents(w_tgt, w_prev, nav, SYMS, MARKS, held, UNI,
                             no_trade_band=0.010, min_order=250.0)
    assert it["action"] == "SELL"
    assert it["shares"] == pytest.approx(0.05 * nav / 500.0)   # 10 shares
    assert it["shares"] < 20.0                                  # partial, not all


def test_min_order_and_band_filters():
    nav = 100_000.0
    # delta 0.0011 -> $110 < min_order 250 -> dropped (band 0.001)
    w_tgt = np.array([0.0011, 0.0, 0.0, 0.0])
    w_prev = np.zeros(4)
    out = delta_to_intents(w_tgt, w_prev, nav, SYMS, MARKS, {}, UNI,
                           no_trade_band=0.001, min_order=250.0)
    assert out == []
    # inside the band -> dropped even though dollars large
    w_tgt = np.array([0.009, 0.0, 0.0, 0.0])
    out = delta_to_intents(w_tgt, w_prev, nav, SYMS, MARKS, {}, UNI,
                           no_trade_band=0.010, min_order=250.0)
    assert out == []
    # full exit below min_order still emitted (position must be closeable)
    w_tgt = np.zeros(4)
    w_prev = np.array([0.002, 0.0, 0.0, 0.0])
    held = {"SPY": 0.4}                                  # $200 position
    (it,) = delta_to_intents(w_tgt, w_prev, nav, SYMS, MARKS, held, UNI,
                             no_trade_band=0.001, min_order=250.0)
    assert it["action"] == "SELL" and it["shares"] == pytest.approx(0.4)


def test_sell_without_position_and_bad_price_skipped():
    nav = 100_000.0
    w_tgt = np.array([0.0, 0.0, 0.0, 0.10])
    w_prev = np.array([0.05, 0.0, 0.0, 0.0])
    marks = dict(MARKS)
    marks["GLD"] = 0.0                                   # bad price -> skip BUY
    out = delta_to_intents(w_tgt, w_prev, nav, SYMS, marks, {}, UNI,
                           no_trade_band=0.010, min_order=250.0)
    assert out == []                                     # no held SPY, no GLD price


# ------------------------------------------------------------- preliminary rows
def _features(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "close"])


def test_preliminary_same_day_rows_dropped():
    df = _features([("2026-02-19", "SPY", 690.0),       # D-1 row (good)
                    ("2026-02-20", "SPY", 999.0)])      # same-day preliminary row
    marks, src = guarded_marks(df, "2026-02-20", cache=None)
    assert src == "features"
    assert marks["SPY"] == pytest.approx(690.0)          # preliminary row ignored


def test_quirk_dir_falls_back_to_prices():
    df = _features([("2026-02-04", "SPY", 686.11)])      # ENTIRELY same-day (quirk)

    class FakeCache:
        def get_parquet(self, key):
            assert key == "daily/2026-02-04/prices.parquet"
            return _features([("2026-02-02", "SPY", 688.0),
                              ("2026-02-03", "SPY", 689.53),
                              ("2026-02-04", "SPY", 686.11)])

    marks, src = guarded_marks(df, "2026-02-04", cache=FakeCache())
    assert src == "prices_fallback"
    assert marks["SPY"] == pytest.approx(689.53)         # D-1 close, not preliminary


# ------------------------------------------------------------- holdout guard
def test_holdout_guard_refuses_without_env():
    for window in ("holdout", "full"):
        with pytest.raises(SystemExit):
            RR.check_holdout_authorization(window, smoke=False, env={})
    # authorized -> passes
    RR.check_holdout_authorization("holdout", smoke=False,
                                   env={RR.ENV_FLAG: "1"})
    # smoke never needs the flag
    RR.check_holdout_authorization("full", smoke=True, env={})


def test_smoke_window_structurally_pre_holdout():
    class FakeCache:
        cache_dir = PROTO / "cache" / "s3"
    dates = RR.build_trading_dates(FakeCache(), "full", smoke=True)
    assert dates[0] == "2026-02-03"
    assert max(dates) < RR.HOLDOUT_START
    assert all(d <= RR.SMOKE_LAST_DECISION for d in dates[1:-1])


# ------------------------------------------------------------- cost overlay
def _fake_result():
    actions = [
        {"date": "2026-02-04", "symbol": "SPY", "action": "BUY",
         "shares": 10.0, "price": 500.0, "dollars": 5000.0},
        {"date": "2026-02-04", "symbol": "TLT", "action": "SELL",
         "shares": 50.0, "price": 90.0, "dollars": 4500.0},
        {"date": "2026-02-05", "symbol": "QQQ", "action": "BUY",
         "shares": 5.0, "price": 400.0, "dollars": 2000.0},
    ]
    dvm = {"2026-02-04": 100_000.0, "2026-02-05": 100_500.0,
           "2026-02-06": 100_400.0}
    return {"actions": actions, "date_value_map": dvm}


def _uni_df():
    return pd.DataFrame([{"symbol": s, "sector": "broad",
                          "asset_class": "equity"} for s in SYMS])


def test_cost_overlay_deterministic():
    a = RR.cost_overlay(_fake_result(), _uni_df(), seed=4242)
    b = RR.cost_overlay(_fake_result(), _uni_df(), seed=4242)
    assert a == b
    c = RR.cost_overlay(_fake_result(), _uni_df(), seed=7)
    assert c["total_cost_dollars"] != a["total_cost_dollars"]


def test_cost_overlay_zero_cost_identity():
    zero_cfg = {"spread_bps": {}, "asset_class_default_bps":
                {"equity": 0.0, "bond": 0.0, "commodity": 0.0, "fx": 0.0,
                 "vol": 0.0},
                "slippage_range_bps": 0.0}
    o = RR.cost_overlay(_fake_result(), _uni_df(), seed=4242,
                        cost_config=zero_cfg)
    assert o["total_cost_dollars"] == pytest.approx(0.0)
    assert o["cost_adjusted"] == pytest.approx(o["raw"])


def test_cost_overlay_subtracts_cumulative_cashflow():
    o = RR.cost_overlay(_fake_result(), _uni_df(), seed=4242)
    day1 = sum(r["cost_dollars"] for r in o["trade_costs"]
               if r["date"] == "2026-02-04")
    day2 = sum(r["cost_dollars"] for r in o["trade_costs"]
               if r["date"] == "2026-02-05")
    assert o["cost_adjusted"][0] == pytest.approx(o["raw"][0] - day1)
    assert o["cost_adjusted"][1] == pytest.approx(o["raw"][1] - day1 - day2)
    assert o["cost_adjusted"][2] == pytest.approx(o["raw"][2] - day1 - day2)
    assert all(r["cost_bps"] >= 0 for r in o["trade_costs"])
