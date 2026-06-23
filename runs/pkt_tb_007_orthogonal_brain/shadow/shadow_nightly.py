"""PKT-TB-007 dual forward shadow — the idempotent nightly job.

  .venv/bin/python shadow_nightly.py            # the nightly (catch-up) run
  .venv/bin/python shadow_nightly.py --no-publish   # local only (debug)

Per pending decision date D (> 2026-06-10, strictly forward-only):
  1. pull daily/<D> artifacts (skip-existing),
  2. splice new bars into the shadow OHLCV store; refresh GDELT/CBOE,
  3. M1 (+disp_z, M4-A) deploy inference -> organ_inputs/<D>.json +
     append-only forecast ledger (timestamp + feature hash),
  4. reconstruct the three paper books ON TOP of the chassis's actual
     intents (I = no tilt, A = a-priori genome, B = a-priori + M4-A damp),
     filled at D's open from the NEXT daily dir's prices (harness D/D+1
     convention; the newest date publishes provisionally from
     morning_prices.parquet and settles next night),
  5. score matured weekly forecasts (D+5 cross-sectional outcome -> IC),
  6. persist state + ledgers locally AND mirror to
     s3://investment-system-data/shadow/pkt_tb_007/ (overwrite-in-place),
  7. publish dashboard/shadow_timeseries.json (live line mirrored from
     dashboard/dashboard.json equity_curve — never recomputed).

Fail-soft: any missing input skips the date with a logged reason; ALERT
lines in the log are the alerting channel (operator SNS out of scope).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SHADOW = Path(__file__).resolve().parent
if str(SHADOW) not in sys.path:
    sys.path.append(str(SHADOW))

import shadow_lib as SL                                   # noqa: E402
from shadow_lib import (BOOKS, FORWARD_BOUNDARY, HORIZON_TD, IC_STEP_TD,
                        PREREG_POINTER, S3_DASHBOARD_KEY, S3_STATE_PREFIX,
                        append_jsonl, assert_forward_only, book_from_dict,
                        book_to_dict, compute_stats, is_decision_dir,
                        jdump, log_line, overlay_books, pending_dates,
                        process_book_date, read_jsonl, robust_value,
                        seed_book_from_state, write_json)

NY_TZ = "America/New_York"


# ---------------------------------------------------------------- context
@dataclass
class Ctx:
    root: Path                          # state dir
    source: Any                         # S3Source | LocalSource
    infer_fn: Callable                  # (pending, ctx) -> {date: record}
    universe_df: Any
    decision_params: Dict[str, Any]
    logf: Optional[Path] = None
    publish: bool = True
    strategies_factory: Optional[Callable] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def cache_daily(self) -> Path:
        return self.root / "cache" / "daily"

    @property
    def ohlcv_dir(self) -> Path:
        return self.root / "cache" / "ohlcv"

    @property
    def organ_dir(self) -> Path:
        return self.root / "organ_inputs"

    @property
    def ledgers(self) -> Path:
        return self.root / "ledgers"

    @property
    def state_json(self) -> Path:
        return self.root / "state.json"

    @property
    def published_dir(self) -> Path:
        return self.root / "published"


def load_state(ctx: Ctx) -> dict:
    if ctx.state_json.exists():
        state = json.loads(ctx.state_json.read_text())
    else:
        return SL.default_state()
    # self-heal pre-PKT-011 state (legacy I/A/B -> ladder I,R,F,E,U). Idempotent:
    # a no-op once the full ladder is present. Persist if a migration was applied
    # so the rewrite is durable (and mirrored to S3 on the publish step).
    if SL.migrate_state_books(state, ctx.ledgers):
        save_state(ctx, state)
        log_line("migrated legacy state books (I/A/B) -> ladder "
                 "(I,R,F,E,U): R seeded from I, U seeded from E", ctx.logf)
    return state


def save_state(ctx: Ctx, state: dict) -> None:
    write_json(ctx.state_json, state)


# ---------------------------------------------------------------- calendar
def trading_calendar(ctx: Ctx) -> List[str]:
    import pandas as pd
    p = ctx.ohlcv_dir / "SPY.parquet"
    df = pd.read_parquet(p, columns=["date"])
    return sorted(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").unique())


# ---------------------------------------------------------------- books step
def successor_prices(ctx: Ctx, date: str, all_dates: List[str]):
    """The next daily dir AFTER `date` carrying prices.parquet (it holds
    D's own OHLC bar — the harness 2-day convention). Fetches on demand."""
    import pandas as pd
    for d in all_dates:
        if d <= date:
            continue
        got = ctx.source.fetch_daily(d, ctx.cache_daily)
        if got.get("prices.parquet"):
            return d, pd.read_parquet(ctx.cache_daily / d / "prices.parquet")
    return None, None


def _load_selected_universe(ctx: Ctx, date: str) -> Optional[set]:
    """The live engine's selected set for ``date`` (PKT-TB-012), if published.

    Read from daily/<D>/brain_selected_universe.json (the cutover writes it). When
    present, the U rung restricts E's trades to this set, so the U-E universe rung
    is wired to the live selected_set instead of reading ~0 (U==E). Sets the
    ctx.extras flag so compute_ladder_stats clears the 'not yet wired' caveat."""
    p = ctx.cache_daily / date / "brain_selected_universe.json"
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sel = doc.get("selected_universe") or []
    if not sel:
        return None
    ctx.extras["selected_universe"] = True
    return set(sel)


def settle_dates(ctx: Ctx, state: dict, decision_dates: List[str],
                 all_dates: List[str]) -> List[str]:
    """Process decision dates in order until one cannot be priced. Mutates
    state (books, last_settled_date). Returns settled dates."""
    import pandas as pd
    from src.utils.three_line_replay import replay_engine as RE

    settled = []
    strategies = None
    for D in decision_dates:
        assert_forward_only(D)
        ddir = ctx.cache_daily / D
        required = ["trade_intents.json", "portfolio_state.json",
                    "features.parquet"]
        missing = [f for f in required if not (ddir / f).exists()]
        if missing:
            log_line(f"SKIP {D}: missing inputs {missing} (never fabricated)",
                     ctx.logf)
            append_jsonl(ctx.ledgers / "equity_ledger.jsonl",
                         {"date": D, "skipped": True,
                          "reason": f"missing_inputs:{','.join(missing)}"})
            state["last_settled_date"] = D
            continue
        succ, succ_px = successor_prices(ctx, D, all_dates)
        if succ is None:
            break                              # newest date — provisional only
        if state["books"] is None:
            pstate = json.loads((ddir / "portfolio_state.json").read_text())
            books = {b: seed_book_from_state(pstate) for b in BOOKS}
            state["books"] = {b: book_to_dict(books[b]) for b in BOOKS}
            state["start_date"] = D
            log_line(f"books seeded at {D} from daily/{D}/portfolio_state.json"
                     f" (cash {pstate['cash']:.2f}, "
                     f"{len(pstate.get('holdings', []))} holdings)", ctx.logf)
        books = {b: book_from_dict(state["books"][b]) for b in BOOKS}
        if strategies is None:
            factory = ctx.strategies_factory or _default_strategies_factory
            strategies = factory(ctx, log_dir=SHADOW / "logs")
        ohlc = RE._ohlc_for_date(succ_px, D)
        intents_doc = json.loads((ddir / "trade_intents.json").read_text())
        intents = list(intents_doc.get("actions", []))
        regime = intents_doc.get("regime")
        if not ohlc or "SPY" not in ohlc:
            log_line(f"SKIP-FLAT {D}: no OHLC bar in daily/{succ}/prices "
                     f"(half-day/outage?) — books flat-hold", ctx.logf)
            row = {"date": D, "skipped": True, "reason": "no_ohlc_bar"}
            for b in BOOKS:
                nav, _ = robust_value(books[b], {})
                row[f"nav_{b}"] = round(nav, 2)
            append_jsonl(ctx.ledgers / "equity_ledger.jsonl", row)
        else:
            feats = pd.read_parquet(ddir / "features.parquet")
            res = process_book_date(books, D, intents, ohlc, feats,
                                    ctx.decision_params, strategies,
                                    ctx.universe_df, regime,
                                    selected_universe=_load_selected_universe(ctx, D))
            append_jsonl(ctx.ledgers / "equity_ledger.jsonl", res["row"])
            for b in BOOKS:
                for a in res["actions"][b]:
                    append_jsonl(ctx.ledgers / f"actions_{b}.jsonl", a)
        state["books"] = {b: book_to_dict(books[b]) for b in BOOKS}
        state["last_settled_date"] = D
        state["n_settled"] = int(state.get("n_settled", 0)) + 1
        save_state(ctx, state)
        settled.append(D)
        log_line(f"settled {D} (fills from daily/{succ})", ctx.logf)
    return settled


def _default_strategies_factory(ctx: Ctx, log_dir: Optional[Path]):
    from risk_stats_007 import RiskStats
    risk = RiskStats(ohlcv_dir=ctx.ohlcv_dir)
    cache = SL.LocalDailyCache(ctx.cache_daily)
    return SL.make_strategies(ctx.organ_dir, risk, cache, log_dir)


# ---------------------------------------------------------------- provisional
def provisional_marks(ctx: Ctx, state: dict, date: str) -> Optional[dict]:
    """Provisional valuation of the newest unsettled decision date from its
    own morning_prices.parquet (open fill + intraday mark). Computed on
    DEEP COPIES — state is never mutated; it settles for real next night."""
    import pandas as pd
    from src.utils.three_line_replay import replay_engine as RE
    ddir = ctx.cache_daily / date
    mp = ddir / "morning_prices.parquet"
    if state["books"] is None or not mp.exists():
        return None
    if not (ddir / "trade_intents.json").exists():
        return None
    try:
        px = pd.read_parquet(mp)
        ohlc = RE._ohlc_for_date(px, date)
        if not ohlc:
            return None
        books = {b: book_from_dict(deepcopy(state["books"][b]))
                 for b in BOOKS}
        factory = ctx.strategies_factory or _default_strategies_factory
        strategies = factory(ctx, log_dir=None)
        feats = pd.read_parquet(ddir / "features.parquet")
        intents_doc = json.loads((ddir / "trade_intents.json").read_text())
        res = process_book_date(books, date, intents_doc.get("actions", []),
                                ohlc, feats, ctx.decision_params, strategies,
                                ctx.universe_df, intents_doc.get("regime"),
                                selected_universe=_load_selected_universe(ctx, date))
        row = res["row"]
        row["provisional"] = True
        return row
    except Exception as e:                                   # noqa: BLE001
        log_line(f"provisional valuation failed for {date}: "
                 f"{type(e).__name__}: {e}", ctx.logf)
        return None


# ---------------------------------------------------------------- IC scoring
def open_panel(ctx: Ctx, symbols: List[str]) -> Dict[str, Dict[str, float]]:
    import pandas as pd
    out: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        p = ctx.ohlcv_dir / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["date", "open"])
        ds = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for d, o in zip(ds, df["open"]):
            if o == o:                                       # not NaN
                out.setdefault(d, {})[sym] = float(o)
    return out


def score_matured(ctx: Ctx, state: dict) -> int:
    """Score weekly forecast records whose +5td outcome exists. The
    timestamp-before-outcome invariant is ENFORCED here: a record stamped
    on/after its maturity date is never scored (ALERT instead)."""
    import numpy as np
    from scipy.stats import spearmanr

    if state.get("start_date") is None:
        return 0
    fore = {r["date"]: r
            for r in read_jsonl(ctx.ledgers / "forecast_ledger.jsonl")}
    ic_rows = read_jsonl(ctx.ledgers / "ic_ledger.jsonl")
    scored = {r["date"] for r in ic_rows}
    cal = trading_calendar(ctx)
    if state["start_date"] not in cal:
        log_line(f"ALERT start_date {state['start_date']} not in trading "
                 f"calendar", ctx.logf)
        return 0
    i0 = cal.index(state["start_date"])
    weekly = cal[i0::IC_STEP_TD]
    symbols = sorted({s for r in fore.values() for s in r["mu"]})
    opens = open_panel(ctx, symbols) if symbols else {}
    n_new = 0
    for wd in weekly:
        if wd in scored:
            continue
        ci = cal.index(wd)
        if ci + HORIZON_TD >= len(cal):
            continue                                         # not matured
        mat_date = cal[ci + HORIZON_TD]
        rec = fore.get(wd)
        if rec is None:
            append_jsonl(ctx.ledgers / "ic_ledger.jsonl",
                         {"date": wd, "ic": None, "n": 0,
                          "reason": "no_forecast_record",
                          "maturity_date": mat_date})
            n_new += 1
            continue
        rec_at = dt.datetime.fromisoformat(rec["recorded_at"]).date()
        if str(rec_at) >= mat_date:
            log_line(f"ALERT stale forecast record {wd} (recorded "
                     f"{rec_at} >= maturity {mat_date}) — EXCLUDED from "
                     f"the IC ledger", ctx.logf)
            append_jsonl(ctx.ledgers / "ic_ledger.jsonl",
                         {"date": wd, "ic": None, "n": 0,
                          "reason": "stale_record_excluded",
                          "maturity_date": mat_date})
            n_new += 1
            continue
        o0, o5 = opens.get(wd, {}), opens.get(mat_date, {})
        mu, r5 = [], []
        for s, m in rec["mu"].items():
            if s in o0 and s in o5 and o0[s] > 0:
                mu.append(float(m))
                r5.append(o5[s] / o0[s] - 1.0)
        if len(mu) < 8:
            append_jsonl(ctx.ledgers / "ic_ledger.jsonl",
                         {"date": wd, "ic": None, "n": len(mu),
                          "reason": "insufficient_outcomes",
                          "maturity_date": mat_date})
            n_new += 1
            continue
        r5 = np.asarray(r5) - float(np.mean(r5))             # y5_raw
        ic = float(spearmanr(np.asarray(mu), r5).statistic)
        late = bool(rec.get("late_record", False))
        append_jsonl(ctx.ledgers / "ic_ledger.jsonl",
                     {"date": wd, "ic": round(ic, 4), "n": len(mu),
                      "maturity_date": mat_date, "late_record": late,
                      "scored_at": dt.datetime.now(dt.timezone.utc)
                      .isoformat(timespec="seconds")})
        n_new += 1
    return n_new


# ---------------------------------------------------------------- publish
def build_payload(ctx: Ctx, state: dict, provisional_row: Optional[dict]
                  ) -> dict:
    equity_rows = [r for r in read_jsonl(ctx.ledgers / "equity_ledger.jsonl")
                   if not r.get("skipped")]
    actions = {b: read_jsonl(ctx.ledgers / f"actions_{b}.jsonl")
               for b in BOOKS}
    ic_rows = [r for r in read_jsonl(ctx.ledgers / "ic_ledger.jsonl")
               if r.get("ic") is not None]

    # live line — mirrored from what the dashboard already uses
    live_line: List[List[Any]] = []
    try:
        dash = ctx.source.get_json("dashboard/dashboard.json")
        for pt in dash.get("equity_curve", []):
            if state.get("start_date") and pt["date"] >= state["start_date"]:
                live_line.append([pt["date"], round(float(pt["value"]), 2)])
    except Exception as e:                                   # noqa: BLE001
        log_line(f"live line mirror unavailable: {type(e).__name__}: {e}",
                 ctx.logf)

    payload: Dict[str, Any] = {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "schema": "shadow_timeseries.v2",
        "prereg_pointer": PREREG_POINTER,
        "live_prereg_pointer": "LIVE_PREREG.md (the LIVE-arm forward contract; PKT-TB-010)",
        "start_date": state.get("start_date"),
        "forward_boundary": FORWARD_BOUNDARY,
        "live_line": live_line,
        # ladder lines (PKT-TB-011): I,R,F,E,U; U = the deployed New Brain line
        "shadow_I": [], "shadow_R": [], "shadow_F": [], "shadow_E": [],
        "shadow_U": [],
        # legacy display aliases kept for the existing PerformanceChart overlay
        "shadow_A": [], "shadow_B": [],
        "provisional_date": None,
        "ic_series": [[r["date"], r["ic"], r["n"]] for r in ic_rows],
        "organ_ledger": [],          # the 5-rung rent ledger (RentLedger.tsx)
        "forecast_leg": {},          # IC-only certified-skill card
        "stats": {},
    }

    series: Dict[str, List[float]] = {}
    dates: List[str] = []
    if equity_rows:
        overlays = overlay_books(actions, equity_rows, ctx.universe_df)
        dates = overlays["I"]["dates"]
        # normalization: each book anchored to the live line's NAV at D0
        live_at_start = None
        for d, v in live_line:
            if d >= state["start_date"]:
                live_at_start = v
                break
        for b in BOOKS:
            adj = overlays[b]["cost_adjusted"]
            series[b] = adj
            base = adj[0] if adj and adj[0] else None
            scale = (live_at_start / base) if (live_at_start and base) else 1.0
            line = [[d, round(v * scale, 2)] for d, v in zip(dates, adj)]
            if provisional_row is not None and f"nav_{b}" in provisional_row:
                cum_cost = ((overlays[b]["raw"][-1]
                             - overlays[b]["cost_adjusted"][-1])
                            if overlays[b]["raw"] else 0.0)
                pv = (provisional_row[f"nav_{b}"] - cum_cost) * scale
                line.append([provisional_row["date"], round(pv, 2)])
                payload["provisional_date"] = provisional_row["date"]
            payload[f"shadow_{b}"] = line
        # legacy display aliases (PerformanceChart overlay): A=F (M1 tilt), B=E
        payload["shadow_A"] = payload.get("shadow_F", [])
        payload["shadow_B"] = payload.get("shadow_E", [])
    elif provisional_row is not None:
        payload["provisional_date"] = provisional_row["date"]

    factor_prices = factor_price_series(ctx, dates)
    payload["stats"] = SL.compute_ladder_stats(
        ic_rows, dates, series, factor_prices=factor_prices,
        selected_universe_active=bool(ctx.extras.get("selected_universe")))
    payload["stats"]["n_settled"] = int(state.get("n_settled", 0))
    # surface the two v2 additions at the top level for the frontend
    payload["organ_ledger"] = payload["stats"].get("organ_ledger", [])
    payload["forecast_leg"] = payload["stats"].get("forecast_leg", {})
    return payload


def factor_price_series(ctx: Ctx, dates: List[str]
                        ) -> Dict[str, List[float]]:
    """Close-price series for the multi-factor exposure strip (C6), aligned to
    ``dates``: mkt=SPY, duration=TLT, commodity=USO. A factor with incomplete
    coverage over the window is dropped (the strip flags the missing factor)."""
    import pandas as pd
    factor_syms = {"mkt": "SPY", "duration": "TLT", "commodity": "USO"}
    out: Dict[str, List[float]] = {}
    if not dates:
        return out
    for fname, sym in factor_syms.items():
        p = ctx.ohlcv_dir / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["date", "close"])
        ds = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        by_date = {d: float(c) for d, c in zip(ds, df["close"]) if c == c}
        series = [by_date.get(d) for d in dates]
        if all(v is not None for v in series):
            out[fname] = series
    return out


def mirror_to_s3(ctx: Ctx, payload: dict) -> List[str]:
    keys = []
    ctx.source.put_json(S3_DASHBOARD_KEY, payload)
    keys.append(S3_DASHBOARD_KEY)
    write_json(ctx.published_dir / "shadow_timeseries.json", payload)
    if ctx.state_json.exists():
        ctx.source.put_file(S3_STATE_PREFIX + "state.json", ctx.state_json)
        keys.append(S3_STATE_PREFIX + "state.json")
    for name in ["forecast_ledger.jsonl", "ic_ledger.jsonl",
                 "equity_ledger.jsonl"] + [f"actions_{b}.jsonl"
                                           for b in BOOKS]:
        p = ctx.ledgers / name
        if p.exists():
            ctx.source.put_file(S3_STATE_PREFIX + f"ledgers/{name}", p)
            keys.append(S3_STATE_PREFIX + f"ledgers/{name}")
    for p in sorted(ctx.organ_dir.glob("*.json")):
        ctx.source.put_file(S3_STATE_PREFIX + f"organ_inputs/{p.name}", p)
    prereg = SHADOW / "SHADOW_PREREG.md"
    if prereg.exists():
        ctx.source.put_file(S3_STATE_PREFIX + "SHADOW_PREREG.md", prereg)
        keys.append(S3_STATE_PREFIX + "SHADOW_PREREG.md")
    return keys


# ---------------------------------------------------------------- main run
def run_night(ctx: Ctx) -> dict:
    t0 = dt.datetime.now()
    log_line("=== shadow night run start ===", ctx.logf)
    state = load_state(ctx)
    all_dates = ctx.source.list_daily_dates()
    pend = pending_dates(all_dates, state.get("last_settled_date"))
    summary: Dict[str, Any] = {"pending": list(pend), "settled": [],
                               "forecasts_new": 0, "ic_new": 0}

    decision_dates: List[str] = []
    for D in pend:
        assert_forward_only(D)
        got = ctx.source.fetch_daily(D, ctx.cache_daily)
        if is_decision_dir(got):
            decision_dates.append(D)
        else:
            log_line(f"non-decision dir daily/{D} "
                     f"(missing {[f for f, ok in got.items() if not ok]})",
                     ctx.logf)

    if not decision_dates:
        log_line("no pending decision dates — shadow armed, publishing "
                 "current series", ctx.logf)
    else:
        # forecast leg (independent of book settlement)
        fore_have = {r["date"] for r in
                     read_jsonl(ctx.ledgers / "forecast_ledger.jsonl")}
        to_infer = [d for d in decision_dates if d not in fore_have]
        if to_infer:
            try:
                records = ctx.infer_fn(to_infer, ctx)
                for d in sorted(records):
                    rec = records[d]
                    rec["late_record"] = _is_late(rec)
                    append_jsonl(ctx.ledgers / "forecast_ledger.jsonl", rec)
                    summary["forecasts_new"] += 1
            except Exception as e:                           # noqa: BLE001
                log_line(f"ALERT inference failed "
                         f"({type(e).__name__}: {e}) — no forecasts written "
                         f"this night; books fall back to neutral organ "
                         f"contract", ctx.logf)

        # utility leg
        summary["settled"] = settle_dates(ctx, state, decision_dates,
                                          all_dates)

    # IC leg
    try:
        summary["ic_new"] = score_matured(ctx, state)
    except Exception as e:                                   # noqa: BLE001
        log_line(f"ALERT IC scoring failed ({type(e).__name__}: {e})",
                 ctx.logf)

    save_state(ctx, state)

    # provisional point for the newest unsettled decision date
    prov = None
    unsettled = [d for d in decision_dates if d not in summary["settled"]]
    if unsettled:
        prov = provisional_marks(ctx, state, unsettled[0])
        if prov is not None:
            log_line(f"provisional marks for {unsettled[0]} "
                     f"(settles next night)", ctx.logf)

    payload = build_payload(ctx, state, prov)
    if ctx.publish:
        keys = mirror_to_s3(ctx, payload)
        log_line(f"published {len(keys)} S3 objects "
                 f"(incl. {S3_DASHBOARD_KEY})", ctx.logf)
    else:
        write_json(ctx.published_dir / "shadow_timeseries.json", payload)

    summary["stats"] = payload["stats"]
    summary["wall_clock_s"] = round(
        (dt.datetime.now() - t0).total_seconds(), 1)
    log_line(f"=== shadow night run done {jdump(summary)} ===", ctx.logf)
    return summary


def _is_late(rec: dict) -> bool:
    """Record stamped after its decision date's NYSE open — transparency
    flag for catch-up records (inputs are still point-in-time S3 artifacts)."""
    try:
        from zoneinfo import ZoneInfo
        rec_at = dt.datetime.fromisoformat(rec["recorded_at"])
        open_et = dt.datetime.fromisoformat(rec["date"] + "T09:30:00") \
            .replace(tzinfo=ZoneInfo(NY_TZ))
        return rec_at >= open_et
    except Exception:                                        # noqa: BLE001
        return True


# ---------------------------------------------------------------- production
def production_infer_fn(pending: List[str], ctx: Ctx) -> Dict[str, dict]:
    import forward_inference as FI
    FI.ensure_seed_caches(ctx.logf)
    for d in pending:
        FI.extend_ohlcv(d, ctx.logf)
    FI.gdelt_forward(ctx.logf)
    FI.cboe_forward(ctx.logf)
    FI.build_panel(pending, ctx.logf)
    return FI.run_inference(pending, ctx.logf)


def build_production_ctx(publish: bool = True) -> Ctx:
    import pandas as pd
    source = SL.S3Source()
    # the chassis's ACTIVE decision params (what the live system trades)
    try:
        active = source.get_json("config/decision_params.active.json")
        dparams = active["decision_params"]
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(f"cannot read decision_params.active.json: {e}")
    universe_df = pd.read_csv(SL.REPO / "config" / "universe.csv")
    logf = SHADOW / "logs" / f"shadow_{dt.date.today().isoformat()}.log"
    # seed caches BEFORE anything needs the SPY calendar
    import forward_inference as FI
    FI.ensure_seed_caches(logf)
    return Ctx(root=SL.STATE, source=source, infer_fn=production_infer_fn,
               universe_df=universe_df, decision_params=dparams, logf=logf,
               publish=publish)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-publish", action="store_true",
                    help="skip all S3 writes (local debug)")
    args = ap.parse_args(argv)
    ctx = build_production_ctx(publish=not args.no_publish)
    try:
        run_night(ctx)
        return 0
    except Exception as e:                                   # noqa: BLE001
        log_line(f"ALERT night run FAILED: {type(e).__name__}: {e}",
                 ctx.logf)
        import traceback
        log_line(traceback.format_exc(), ctx.logf)
        return 1


if __name__ == "__main__":
    sys.exit(main())
