"""PKT-TB-006 — replay-wiring tests (deliverable #5).

Covers: Δw -> intent conversion math (never REDUCE, min_order, no_trade_band,
full-exit exception), the preliminary same-day-row guard (+ prices fallback),
the holdout guard refusal, cost-overlay determinism + zero-cost == raw
identity, and the battery knobs: the R08 equal-trust executive bypass
(tau=1/M over active members, f=0.7), the R07 sigma-source swap
(trailing21 vs risknet, deterministic + recorded), the R13/R14 --cost-seed
override, and the per-arm nightly audit copy.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ea                                   # noqa: E402
import run_replay as RR                     # noqa: E402
import strategy_adapter as SA               # noqa: E402
from strategy_adapter import delta_to_intents, guarded_marks, make_syn1_strategy  # noqa: E402
from src.utils.three_line_replay.strategies import StrategyContext  # noqa: E402

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
    # same seed -> byte-identical serialized overlay (R13/R14 contract)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = RR.cost_overlay(_fake_result(), _uni_df(), seed=7)
    assert c["total_cost_dollars"] != a["total_cost_dollars"]
    # different seed -> a DIFFERENT cost-adjusted series, same raw series
    assert c["cost_adjusted"] != a["cost_adjusted"]
    assert c["raw"] == a["raw"]


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


# ------------------------------------------------------------- runner knobs CLI
def test_runner_cli_knob_defaults_and_overrides():
    ap = RR.build_parser()
    a = ap.parse_args(["--arm", "syn1", "--out", "x"])
    assert a.cost_seed == RR.COST_SEED == 4242
    assert a.exec_mode == "learned"
    assert a.sigma_source == "trailing21"        # the landed/R01 behavior, named
    b = ap.parse_args(["--arm", "syn1", "--out", "x", "--cost-seed", "4243",
                       "--exec-mode", "equal_trust", "--sigma-source", "risknet"])
    assert (b.cost_seed, b.exec_mode, b.sigma_source) == \
        (4243, "equal_trust", "risknet")
    with pytest.raises(SystemExit):
        ap.parse_args(["--arm", "syn1", "--out", "x", "--exec-mode", "bogus"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--arm", "syn1", "--out", "x", "--sigma-source", "bogus"])


# ------------------------------------------------------------- nightly audit copy
def test_copy_nightly_audit_fresh_files_only(tmp_path):
    nightly = tmp_path / "nightly"
    out = tmp_path / "out"
    dates = ["2026-02-04", "2026-02-05"]
    for d in dates:
        dd = nightly / d
        dd.mkdir(parents=True)
        (dd / "meta_decision.json").write_text(json.dumps({"date": d}))
        (dd / "trade_intents.json").write_text(json.dumps({"date": d,
                                                           "intents": []}))
    # date 1's files are STALE (written by an earlier arm) -> must be skipped
    old = time.time() - 3600
    for name in ("meta_decision.json", "trade_intents.json"):
        os.utime(nightly / dates[0] / name, (old, old))
    n = RR.copy_nightly_audit(nightly, out, dates, since_ts=time.time() - 60)
    assert n == 2
    assert (out / "nightly_audit" / dates[1] / "meta_decision.json").exists()
    assert (out / "nightly_audit" / dates[1] / "trade_intents.json").exists()
    assert not (out / "nightly_audit" / dates[0]).exists()
    # copied bytes identical to the source
    assert (out / "nightly_audit" / dates[1] / "meta_decision.json").read_bytes() \
        == (nightly / dates[1] / "meta_decision.json").read_bytes()


# ------------------------------------------------------------- battery knobs
D_FIX = "2026-02-20"
MEMBERS = ("cast", "gbm_cond", "event_head")


def _nightly_fixture(tmp_path, books_by_member, mu_val=0.5,
                     sigma_proxy=0.10, sigma_e4=5.0):
    """Minimal store/nightly/<D> + ledger + universe csv for adapter tests."""
    nd = tmp_path / "nightly"
    day = nd / D_FIX
    day.mkdir(parents=True, exist_ok=True)
    dates = ["2026-02-17", "2026-02-18", "2026-02-19", D_FIX]
    pd.DataFrame([{"date": d, "member": m, "u_realized": 0.001,
                   "w_rec_raw": 0.5} for m in MEMBERS for d in dates]
                 ).to_parquet(nd / "ledger.parquet", index=False)
    (nd / "ledger_meta.json").write_text(json.dumps({"u_std": 1.0}))
    pd.concat([pd.DataFrame({"member": m, "symbol": SYMS, "mu": mu_val,
                             "sigma": 0.1, "c": 0.5}) for m in MEMBERS]
              ).to_parquet(day / "expert_opinions.parquet", index=False)
    pd.concat([pd.DataFrame({"member": m, "symbol": SYMS,
                             "w": books_by_member[m]}) for m in MEMBERS]
              ).to_parquet(day / "solo_books.parquet", index=False)
    pd.DataFrame({"symbol": SYMS, "sigma_hat": sigma_e4, "beta_hat": 1.0,
                  "book_vol_hat": 2.0, "sigma_hat_proxy": sigma_proxy,
                  "book_vol_hat_proxy": 0.1}
                 ).to_parquet(day / "risknet.parquet", index=False)
    (day / "exec_inputs.json").write_text(json.dumps({
        "date": D_FIX, "r": [[0.0] * 4] * 3, "z": [0.0] * 24, "c": [0.5] * 3,
        "agree": [0.0] * 3, "g": [0.0] * 3,
        "book_vol_hat_proxy": 0.1, "book_vol_hat_e4": 2.0,
        "members": list(MEMBERS)}))
    uni = tmp_path / "universe.csv"
    pd.DataFrame([{"symbol": s, "asset_class": "equity", "sector": "broad",
                   "leverage_flag": 0} for s in SYMS]).to_csv(uni, index=False)
    return nd, uni


def _ctx():
    feat = pd.DataFrame([("2026-02-19", s, MARKS[s]) for s in SYMS],
                        columns=["date", "symbol", "close"])
    port = SimpleNamespace(cash=100_000.0, positions=[])
    return StrategyContext(inputs_date=D_FIX, portfolio=port, variant_config={},
                           expert_signals={}, expert_metrics={}, decisions={},
                           panic_streak=0, last_regime=None, features_df=feat,
                           inference={}, llm_risks={})


def _equal_trust_strategy(tmp_path, nd, uni, genome=None, **kw):
    return make_syn1_strategy(genome or ea.Genome.b0(),
                              exec_weights_dir=tmp_path / "no_exec_weights",
                              nightly_dir=nd, cache=None, universe_csv=uni,
                              exec_mode="equal_trust", **kw)


def test_equal_trust_forward_math_tau_and_f(tmp_path):
    """R08 bypass: tau = 1/M, f = 0.7, w_tgt = 0.7 * mean(books) (no rails)."""
    books = {m: np.full(4, 0.1) for m in MEMBERS}
    nd, uni = _nightly_fixture(tmp_path, books)
    strat = _equal_trust_strategy(tmp_path, nd, uni)
    intents = strat.post_decision(_ctx(), [])
    meta = json.loads((nd / D_FIX / "meta_decision.json").read_text())
    assert meta["exec_mode"] == "equal_trust"
    assert meta["deployment_fraction"] == pytest.approx(0.7)
    for m in MEMBERS:
        assert meta["trust"][m] == pytest.approx(1.0 / 3.0)
        assert meta["blend_coef"][m] == pytest.approx(1.0 / 3.0)
    assert meta["n_exec_seeds"] == 0                 # learned exec never loaded
    assert meta["rails"] == []
    assert meta["gross_target_applied"] == pytest.approx(0.7 * 0.4)
    # w_tgt = 0.7 * 0.1 per symbol -> $7000 BUYs
    by = {i["symbol"]: i for i in intents}
    assert set(by) == set(SYMS)
    for s in SYMS:
        assert by[s]["action"] == "BUY"
        assert by[s]["dollars"] == pytest.approx(7000.0)
        exp = int(7000.0 / MARKS[s])
        assert by[s]["shares"] in (exp, exp - 1)     # int-truncation float edge


def test_equal_trust_over_active_members_only(tmp_path):
    """tau = 1/M over ACTIVE members: gated member at exactly 0, rest 1/2."""
    books = {"cast": np.array([0.2, 0.0, 0.0, 0.0]),
             "gbm_cond": np.array([0.0, 0.2, 0.0, 0.0]),
             "event_head": np.array([0.0, 0.0, 0.2, 0.0])}
    nd, uni = _nightly_fixture(tmp_path, books)
    g = ea.Genome.b0()
    g.member_gate = [1, 0, 1]
    strat = _equal_trust_strategy(tmp_path, nd, uni, genome=g)
    intents = strat.post_decision(_ctx(), [])
    meta = json.loads((nd / D_FIX / "meta_decision.json").read_text())
    assert meta["trust"] == pytest.approx(
        {"cast": 0.5, "gbm_cond": 0.0, "event_head": 0.5})
    # blend = 0.5*cast + 0.5*event books -> SPY/TLT at 0.7*0.1, no QQQ/GLD
    by = {i["symbol"]: i for i in intents}
    assert set(by) == {"SPY", "TLT"}
    assert by["SPY"]["dollars"] == pytest.approx(7000.0)
    # all members gated off refuses at build time
    g2 = ea.Genome.b0()
    g2.member_gate = [0, 0, 0]
    with pytest.raises(ValueError, match="gated off"):
        _equal_trust_strategy(tmp_path, nd, uni, genome=g2)


def test_sigma_source_swap_hits_vol_cap_path_and_is_recorded(tmp_path):
    """R07: 'risknet' selects the E4 sigma_hat column (here huge -> vol cap
    binds); 'trailing21' keeps the proxy (no rail). Both named in meta."""
    books = {m: np.full(4, 0.1) for m in MEMBERS}
    nd, uni = _nightly_fixture(tmp_path, books, sigma_proxy=0.10, sigma_e4=5.0)
    g = ea.Genome.b0()
    strat_t = _equal_trust_strategy(tmp_path, nd, uni, sigma_source="trailing21")
    strat_t.post_decision(_ctx(), [])
    meta_t = json.loads((nd / D_FIX / "meta_decision.json").read_text())
    assert meta_t["sigma_source"] == "trailing21"
    assert "vol_cap" not in meta_t["rails"]
    assert meta_t["est_book_vol"] == pytest.approx(0.7 * 0.4 * 0.10, abs=1e-9)
    strat_r = _equal_trust_strategy(tmp_path, nd, uni, sigma_source="risknet")
    strat_r.post_decision(_ctx(), [])
    meta_r = json.loads((nd / D_FIX / "meta_decision.json").read_text())
    assert meta_r["sigma_source"] == "risknet"
    assert "vol_cap" in meta_r["rails"]              # 0.28*5.0 >> vol_target 0.10
    assert meta_r["est_book_vol"] == pytest.approx(g.vol_target_ann, abs=1e-9)


def test_sigma_swap_deterministic_byte_identical(tmp_path):
    """Two fresh strategies, same knobs -> byte-identical decision artifacts."""
    books = {m: np.full(4, 0.1) for m in MEMBERS}
    nd, uni = _nightly_fixture(tmp_path, books)
    outs = []
    for _ in range(2):
        strat = _equal_trust_strategy(tmp_path, nd, uni, sigma_source="risknet")
        strat.post_decision(_ctx(), [])
        outs.append(((nd / D_FIX / "meta_decision.json").read_bytes(),
                     (nd / D_FIX / "trade_intents.json").read_bytes()))
    assert outs[0] == outs[1]


def test_unknown_knob_values_refused():
    with pytest.raises(ValueError, match="exec_mode"):
        make_syn1_strategy(ea.Genome.b0(), "x", "y", exec_mode="bogus")
    with pytest.raises(ValueError, match="sigma_source"):
        make_syn1_strategy(ea.Genome.b0(), "x", "y", sigma_source="bogus")
    # the SIGMA_SOURCES map is the manifest contract
    assert SA.SIGMA_SOURCES["trailing21"] == ("sigma_hat_proxy",
                                              "book_vol_hat_proxy")
    assert SA.SIGMA_SOURCES["risknet"] == ("sigma_hat", "book_vol_hat_e4")
