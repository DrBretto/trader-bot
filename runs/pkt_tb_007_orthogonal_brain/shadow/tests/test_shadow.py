"""PKT-TB-007 dual forward shadow — tests.

Covers the pre-registered engineering invariants:
  1. forward-only refusal (nothing <= 2026-06-10 is ever processed)
  2. catch-up idempotence (process 3 dates, re-run, byte-identical state)
  3. determinism (fresh reprocess reproduces all non-timestamp content)
  4. forecast-record timestamp-before-outcome invariant (stale records are
     excluded from the IC ledger, never scored)
  5. published JSON schema
plus book mechanics (tilt divergence, costs, normalization anchoring).

The organ-inference step is injected (stub) — the real inference pipeline is
validated separately by the parity self-check against the frozen prototype
store/nightly_007 files (forward_inference._parity), which runs every night.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SHADOW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW))

import shadow_lib as SL                                    # noqa: E402
import shadow_nightly as SN                                # noqa: E402

DATES = ["2026-06-11", "2026-06-12", "2026-06-15", "2026-06-16"]
# 8+ symbols: the IC scorer keeps the registered >=8 cross-section bar
DRIFT = {"SPY": 1, "AAA": 4, "BBB": -4, "CCC": 3, "DDD": -3, "EEE": 2,
         "FFF": -2, "GGG": 0}
SYMS = sorted(DRIFT)
BASE_PX = {s: 50.0 + 10 * i for i, s in enumerate(SYMS)}
STUB_MU = {s: DRIFT[s] / 10.0 for s in SYMS}      # aligned with drift => IC>0


# ---------------------------------------------------------------- fixture
def _cal(n_back: int = 12, n_fwd: int = 10):
    """Weekday calendar around the fixture dates."""
    start = pd.Timestamp("2026-06-11") - pd.tseries.offsets.BDay(n_back)
    return [d.strftime("%Y-%m-%d") for d in
            pd.bdate_range(start, periods=n_back + n_fwd)]


def _px(sym: str, date: str) -> float:
    i = _cal().index(date)
    return BASE_PX[sym] * (1.0 + 0.0005 * i * DRIFT[sym])


def make_fixture(root: Path, n_dirs: int = 4, with_morning: bool = True):
    cal = _cal()
    for k, D in enumerate(DATES[:n_dirs]):
        ddir = root / "daily" / D
        ddir.mkdir(parents=True, exist_ok=True)
        di = cal.index(D)
        hist = cal[max(di - 10, 0):di]            # bars through D-1
        rows = []
        for d in hist:
            for s in SYMS:
                p = _px(s, d)
                rows.append({"date": d, "symbol": s, "open": p * 0.999,
                             "high": p * 1.01, "low": p * 0.99,
                             "close": p, "volume": 1e6})
        pd.DataFrame(rows).to_parquet(ddir / "prices.parquet", index=False)
        feats = pd.DataFrame([{"date": hist[-1], "symbol": s,
                               "close": _px(s, hist[-1])} for s in SYMS])
        feats.to_parquet(ddir / "features.parquet", index=False)
        intents = {"generated_date": D, "regime": "calm_uptrend",
                   "actions": ([{"action": "BUY", "symbol": "AAA",
                                 "shares": 50, "price": _px("AAA", hist[-1]),
                                 "dollars": 50 * _px("AAA", hist[-1]),
                                 "asset_class": "equity", "sector": "tech",
                                 "leverage_flag": 0, "reason": "TEST"}]
                               if k == 0 else [])}
        (ddir / "trade_intents.json").write_text(json.dumps(intents))
        (ddir / "portfolio_state.json").write_text(json.dumps(
            {"cash": 100000.0, "holdings": [],
             "portfolio_value": 100000.0}))
        if with_morning:
            p_rows = [{"date": D, "symbol": s, "open": _px(s, D) * 0.999,
                       "high": _px(s, D), "low": _px(s, D) * 0.99,
                       "close": _px(s, D), "volume": 1e5} for s in SYMS]
            pd.DataFrame(p_rows).to_parquet(ddir / "morning_prices.parquet",
                                            index=False)
    # dashboard live line
    dash = {"equity_curve": [{"date": d, "value": 110000.0 + 10 * i}
                             for i, d in enumerate(cal)]}
    (root / "dashboard").mkdir(parents=True, exist_ok=True)
    (root / "dashboard" / "dashboard.json").write_text(json.dumps(dash))


def make_ohlcv(state_root: Path):
    cal = _cal()
    d = state_root / "cache" / "ohlcv"
    d.mkdir(parents=True, exist_ok=True)
    for s in SYMS:
        rows = [{"date": c, "open": _px(s, c) * 0.999, "high": _px(s, c),
                 "low": _px(s, c) * 0.99, "close": _px(s, c),
                 "adj_close": _px(s, c), "volume": 1e6,
                 "source": "fixture", "adj_factor": 1.0} for c in cal]
        pd.DataFrame(rows).to_parquet(d / f"{s}.parquet", index=False)


def stub_infer_factory(fixed_ts: str = None):
    def infer(pending, ctx):
        out = {}
        for D in pending:
            mu = dict(STUB_MU)
            out[D] = {"date": D,
                      "recorded_at": fixed_ts or dt.datetime.now(
                          dt.timezone.utc).isoformat(timespec="seconds"),
                      "mu": mu,
                      "rank": {sym: r + 1 for r, sym in enumerate(
                          sorted(mu, key=lambda x: -mu[x]))},
                      "n_valid": len(mu), "q": 0.5,
                      "feature_sha": "stub", "model_sha": "stub",
                      "code_sha": "stub"}
            SL.write_json(ctx.organ_dir / f"{D}.json",
                          {"date": D, "organs": {"M1": {"mu": mu, "q": 0.5}}})
        return out
    return infer


def stub_strategies_factory(ctx, log_dir=None):
    """Deterministic tilt stubs over the ladder rungs that carry a tilt:
    F (forecast) adds a $600 BBB buy, E/U add $300 (E=F+M4 damp, U=E+universe).
    I and R carry no tilt (R is the regime throttle, identity in this fixture's
    'calm_uptrend' regime), so R==I exactly."""
    def make(amount):
        def post(c, intents):
            out = [dict(it) for it in intents]
            px = float(c.features_df[c.features_df["symbol"] == "BBB"]
                       ["close"].iloc[-1])
            sh = int(amount / px)
            out.append({"action": "BUY", "symbol": "BBB", "shares": sh,
                        "dollars": sh * px, "price": px,
                        "asset_class": "equity", "sector": "fin",
                        "leverage_flag": 0, "reason": "STUB_TILT"})
            return out
        return SimpleNamespace(post_decision=post)
    return {"F": make(600.0), "E": make(300.0), "U": make(300.0)}


def make_ctx(tmp: Path, publish: bool = False) -> SN.Ctx:
    src_root = tmp / "source"
    state_root = tmp / "state"
    if not (src_root / "daily").exists():
        make_fixture(src_root)
    make_ohlcv(state_root)
    uni = pd.DataFrame([{"symbol": s, "sector": "tech",
                         "asset_class": "equity", "leverage_flag": 0}
                        for s in SYMS])
    return SN.Ctx(root=state_root, source=SL.LocalSource(src_root),
                  infer_fn=stub_infer_factory(),
                  universe_df=uni,
                  decision_params={"min_order_dollars": 250.0},
                  logf=tmp / "test.log", publish=publish,
                  strategies_factory=stub_strategies_factory)


def state_digest(ctx, volatile=("recorded_at", "scored_at")) -> str:
    """Digest of state.json + ledgers; volatile=None keeps timestamps."""
    h = hashlib.sha256()
    for p in sorted([ctx.state_json] + list(ctx.ledgers.glob("*.jsonl"))):
        if not p.exists():
            continue
        txt = p.read_text()
        if volatile:
            if p.suffix == ".jsonl":
                lines = []
                for ln in txt.splitlines():
                    if not ln.strip():
                        continue
                    o = json.loads(ln)
                    for k in volatile:
                        o.pop(k, None)
                    lines.append(json.dumps(o, sort_keys=True))
                txt = "\n".join(lines)
            else:
                o = json.loads(txt)
                for k in volatile:
                    o.pop(k, None)
                txt = json.dumps(o, sort_keys=True)
        h.update(p.name.encode())
        h.update(txt.encode())
    return h.hexdigest()


# ================================================================ tests
def test_forward_only_refusal(tmp_path):
    # the selector never yields dates <= the boundary
    dates = ["2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"]
    assert SL.pending_dates(dates, None) == ["2026-06-11", "2026-06-12"]
    with pytest.raises(RuntimeError, match="FORWARD-ONLY"):
        SL.assert_forward_only("2026-06-10")
    with pytest.raises(RuntimeError, match="FORWARD-ONLY"):
        SL.assert_forward_only("2025-01-02")
    SL.assert_forward_only("2026-06-11")          # no raise

    # end-to-end: a pre-boundary dir in the source is never touched
    ctx = make_ctx(tmp_path)
    old = tmp_path / "source" / "daily" / "2026-06-10"
    shutil.copytree(tmp_path / "source" / "daily" / "2026-06-11", old)
    summary = SN.run_night(ctx)
    assert "2026-06-10" not in summary["pending"]
    state = SN.load_state(ctx)
    assert state["start_date"] == "2026-06-11"
    fore = SL.read_jsonl(ctx.ledgers / "forecast_ledger.jsonl")
    assert all(r["date"] > "2026-06-10" for r in fore)


def test_catchup_idempotence_byte_identical(tmp_path):
    ctx = make_ctx(tmp_path)
    s1 = SN.run_night(ctx)
    # 4 dirs: first 3 settleable (each has a successor), newest provisional
    assert s1["settled"] == DATES[:3]
    d1 = state_digest(ctx, volatile=None)         # full bytes incl timestamps
    s2 = SN.run_night(ctx)                        # re-run: nothing pending
    assert s2["settled"] == []
    d2 = state_digest(ctx, volatile=None)
    assert d1 == d2, "re-run must leave state byte-identical"


def test_fresh_reprocess_determinism(tmp_path):
    ctx = make_ctx(tmp_path)
    SN.run_night(ctx)
    d1 = state_digest(ctx)                        # timestamps canonicalized
    # wipe state (keep source fixture), reprocess from scratch
    shutil.rmtree(ctx.root)
    make_ohlcv(ctx.root)
    ctx2 = make_ctx(tmp_path)
    SN.run_night(ctx2)
    d2 = state_digest(ctx2)
    assert d1 == d2, "fresh catch-up must reproduce identical content"


def test_books_diverge_and_costs(tmp_path):
    ctx = make_ctx(tmp_path)
    SN.run_night(ctx)
    rows = [r for r in SL.read_jsonl(ctx.ledgers / "equity_ledger.jsonl")
            if not r.get("skipped")]
    assert len(rows) == 3
    # F (forecast rung) bought more BBB than E than I; BBB drifts down here.
    assert rows[-1]["nav_F"] != rows[-1]["nav_I"]
    assert rows[-1]["nav_E"] != rows[-1]["nav_I"]
    # R (regime throttle) is identity in 'calm_uptrend' -> R == I exactly
    assert rows[-1]["nav_R"] == rows[-1]["nav_I"]
    acts_F = SL.read_jsonl(ctx.ledgers / "actions_F.jsonl")
    acts_I = SL.read_jsonl(ctx.ledgers / "actions_I.jsonl")
    assert len(acts_F) > len(acts_I)
    assert any(a["reason"] == "STUB_TILT" for a in acts_F)


def test_timestamp_before_outcome_invariant(tmp_path):
    ctx = make_ctx(tmp_path)
    # forge a STALE record: recorded after its own maturity date
    stale_ts = "2026-07-01T00:00:00+00:00"
    ctx.infer_fn = stub_infer_factory(fixed_ts=stale_ts)
    SN.run_night(ctx)
    ic = SL.read_jsonl(ctx.ledgers / "ic_ledger.jsonl")
    scored_weeks = [r for r in ic if r["date"] == "2026-06-11"]
    assert scored_weeks, "weekly grid row for D0 must exist"
    assert scored_weeks[0]["ic"] is None
    assert scored_weeks[0]["reason"] == "stale_record_excluded"

    # honest record (stamped before maturity) scores normally
    tmp2 = tmp_path / "honest"
    ctx2 = make_ctx(tmp2)
    ctx2.infer_fn = stub_infer_factory(
        fixed_ts="2026-06-11T03:00:00+00:00")
    SN.run_night(ctx2)
    ic2 = [r for r in SL.read_jsonl(ctx2.ledgers / "ic_ledger.jsonl")
           if r["date"] == "2026-06-11"]
    assert ic2 and ic2[0]["ic"] is not None
    assert ic2[0]["n"] == len(SYMS)
    # mu ranks AAA > SPY > BBB and fixture returns AAA,SPY up / BBB down
    assert ic2[0]["ic"] > 0


def test_published_json_schema(tmp_path):
    ctx = make_ctx(tmp_path, publish=True)        # LocalSource captures
    SN.run_night(ctx)
    payload = ctx.source.published[SL.S3_DASHBOARD_KEY]
    required = ["as_of", "live_line", "shadow_A", "shadow_B", "ic_series",
                "stats", "prereg_pointer"]
    for k in required:
        assert k in payload, f"missing key {k}"
    st = payload["stats"]
    for k in ["days_accrued", "mean_ic", "ic_t", "utility_diff_bp_day",
              "ci"]:
        assert k in st, f"missing stats key {k}"
    assert payload["start_date"] == "2026-06-11"
    # series shape [date, value]
    for ln in ["shadow_A", "shadow_B", "live_line"]:
        for pt in payload[ln]:
            assert len(pt) == 2 and isinstance(pt[0], str)
    # normalization: all books anchored to the live value at start date
    live0 = dict(payload["live_line"])["2026-06-11"]
    assert abs(payload["shadow_A"][0][1] - live0) < 1e-6
    assert abs(payload["shadow_B"][0][1] - live0) < 1e-6
    # provisional point for the newest (unsettled) date
    assert payload["provisional_date"] == DATES[3]
    assert payload["shadow_A"][-1][0] == DATES[3]
    # ic_series rows are [date, ic, n]
    for row in payload["ic_series"]:
        assert len(row) == 3
    # state mirror landed on the allowed prefix only
    for key in ctx.source.published:
        assert (key == SL.S3_DASHBOARD_KEY
                or key.startswith(SL.S3_STATE_PREFIX))


def test_legacy_state_migration(tmp_path):
    """Pre-PKT-011 state (legacy I/A/B books + legacy ledgers) self-heals to
    the I,R,F,E,U ladder on load: A->F, B->E (evolved books carry over), R
    parent-seeded from I, U parent-seeded from E. Idempotent + the migrated
    state then settles forward without KeyError, ending byte-identical on
    re-run."""
    ctx = make_ctx(tmp_path)
    ctx.ledgers.mkdir(parents=True, exist_ok=True)
    # forge an evolved legacy state at last_settled 2026-06-12
    bookI = {"cash": 111.0, "positions": [
        {"symbol": "AAA", "shares": 10.0, "entry_price": 50.0,
         "entry_date": "2026-06-11", "peak_price": 51.0,
         "asset_class": "equity", "sector": "tech", "leverage_flag": 0,
         "last_close": 50.5}]}
    bookA = {"cash": 222.0, "positions": list(bookI["positions"])}
    bookB = {"cash": 333.0, "positions": list(bookI["positions"])}
    legacy = {"schema": "shadow_state.v1",
              "forward_boundary": SL.FORWARD_BOUNDARY,
              "start_date": "2026-06-11", "baseline": None,
              "last_settled_date": "2026-06-12", "n_settled": 2,
              "books": {"I": bookI, "A": bookA, "B": bookB}}
    SL.write_json(ctx.state_json, legacy)
    # legacy ledgers
    for r in [{"date": "2026-06-11", "nav_I": 100.0, "nav_A": 101.0,
               "nav_B": 102.0, "n_actions_I": 1, "n_actions_A": 2,
               "n_actions_B": 3},
              {"date": "2026-06-12", "nav_I": 105.0, "nav_A": 106.0,
               "nav_B": 107.0, "n_actions_I": 0, "n_actions_A": 1,
               "n_actions_B": 1}]:
        SL.append_jsonl(ctx.ledgers / "equity_ledger.jsonl", r)
    for b in ("I", "A", "B"):
        SL.append_jsonl(ctx.ledgers / f"actions_{b}.jsonl",
                        {"action": "BUY", "symbol": "AAA", "shares": 10.0,
                         "price": 50.0, "dollars": 500.0, "date": "2026-06-11",
                         "asset_class": "equity", "sector": "tech",
                         "leverage_flag": 0, "reason": "TEST"})

    state = SN.load_state(ctx)
    # ladder shape; legacy keys gone
    assert set(state["books"]) == set(SL.BOOKS)
    # evolved books carried over by alias
    assert state["books"]["F"]["cash"] == 222.0      # A -> F
    assert state["books"]["E"]["cash"] == 333.0      # B -> E
    assert state["books"]["I"]["cash"] == 111.0
    # new rungs parent-seeded
    assert state["books"]["R"]["cash"] == state["books"]["I"]["cash"]   # R<-I
    assert state["books"]["U"]["cash"] == state["books"]["E"]["cash"]   # U<-E

    # ledgers migrated
    eq = SL.read_jsonl(ctx.ledgers / "equity_ledger.jsonl")
    assert all("nav_A" not in r and "nav_B" not in r for r in eq)
    assert eq[0]["nav_F"] == 101.0 and eq[0]["nav_E"] == 102.0
    assert eq[0]["nav_R"] == eq[0]["nav_I"] and eq[0]["n_actions_R"] == 0
    assert eq[0]["nav_U"] == eq[0]["nav_E"]
    assert (ctx.ledgers / "actions_F.jsonl").exists()
    assert (ctx.ledgers / "actions_E.jsonl").exists()
    assert (ctx.ledgers / "actions_R.jsonl").exists()   # copied from I
    assert (ctx.ledgers / "actions_U.jsonl").exists()   # copied from E
    assert not (ctx.ledgers / "actions_A.jsonl").exists()
    assert not (ctx.ledgers / "actions_B.jsonl").exists()

    # idempotent: a second migration is a no-op
    assert SL.migrate_state_books(state, ctx.ledgers) is False

    # and the migrated state settles forward without KeyError 'R'
    summary = SN.run_night(ctx)
    assert "settled" in summary


def test_empty_armed_skeleton(tmp_path):
    """No pending decision dates => clean 'armed' skeleton publish."""
    src_root = tmp_path / "source"
    (src_root / "daily" / "2026-06-10").mkdir(parents=True)  # pre-boundary
    (src_root / "dashboard").mkdir(parents=True)
    (src_root / "dashboard" / "dashboard.json").write_text(
        json.dumps({"equity_curve": []}))
    state_root = tmp_path / "state"
    make_ohlcv(state_root)
    uni = pd.DataFrame([{"symbol": s, "sector": "tech",
                         "asset_class": "equity"} for s in SYMS])
    ctx = SN.Ctx(root=state_root, source=SL.LocalSource(src_root),
                 infer_fn=stub_infer_factory(), universe_df=uni,
                 decision_params={"min_order_dollars": 250.0},
                 logf=tmp_path / "t.log", publish=True,
                 strategies_factory=stub_strategies_factory)
    summary = SN.run_night(ctx)
    assert summary["settled"] == [] and summary["forecasts_new"] == 0
    payload = ctx.source.published[SL.S3_DASHBOARD_KEY]
    assert payload["shadow_A"] == [] and payload["start_date"] is None
    assert payload["stats"]["days_accrued"] == 0
