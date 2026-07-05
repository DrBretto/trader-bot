"""THE KEYSTONE — true history replay on the clean spine (P4).

``replay(D0..D1)`` marches each settled NY trading day D and, for each one,
**rebuilds the substrate as-of D** and runs the **same** ``forecast(D) -> engine(D)``
spine the forward night runs, marks the resulting picks at that day's settled
prices, and appends ONE ledger leaf. The ONLY difference between replay and the
forward night is the **date driver** and **where the marks come from** —
selection/allocation is literally the same ``run_engine`` call.

This is what kills "**re-marked, not replayed**". The broken PKT-5 replay took the
old engine's historical holdings (``daily/<D>/portfolio_state.json``) and re-priced
them; the forward leaf-append (``src/canon/equity_append.append_settled_point_for_publish``)
does the same — it marks whatever holdings the morning executor left behind. Here
history is **RECOMPUTED** by the corrected engine (fresh P1 prices + P2 GDELT + P3
spine): each day's picks come out of ``run_engine`` on the as-of-D substrate, and
the return engine is the recomputed book chained day-over-day *within the replay*.
No stored historical holding is ever read.

AS-OF-D IS ENFORCED BY THE PANEL, NOT BY MUTATING THE STORE. ``forecast.inference``'s
``build_panel([D])`` bounds the panel at ``end=D`` and date D's feature window is
D-63..D-1 (NaN bar AT D — the training convention). Future bars physically present
in the seed OHLCV store are never read for date D. The correctness of that bound is
not asserted in a comment — ``run_inference``'s parity self-check RECOMPUTES the
overlap dates against the frozen forward prototype (``store/nightly_007/<D>.json``)
and ABORTS on divergence. So mu-parity vs forward and no-future-leak are the SAME
gate, and it is reality-checked on every replayed date that overlaps the frozen
reference.

REGIME. Each replayed day computes its as-of-D fused regime via the ONE picker
``forecast.regime.regime(D)`` (PKT-TRADER-BOT-REGIME-AS-OF-D) — the SAME picker the
forward path (``decide.cutover.run_cutover``) now calls — and feeds that label into
``build_forecast_bundle`` so ``regime_score_mult`` reflects the real regime tilt
instead of the constant ``'neutral'`` (which is absent from
``config/regime_compatibility.json`` and collapses every mult to 1.0 = raw-mu
ranking). The picker is bounded ``end=D`` (no future leak), deterministic, and
seeded (OHLCV + FRED + CBOE). It is the DETERMINISTIC ``baseline_regime_model ->
decide_regime_v3`` fused path; the legacy trained-ensemble regime label is NOT a
deterministic function of as-of-D seed data (moving deployed vintage + non-seeded
gdelt + S3-pinned config) and is out of scope for the deterministic spine —
surfaced in the run receipt, never stubbed to neutral.

SINK. The append-only, content-addressed equity ledger
(``lines/ledger.EquityLedger`` — KEEP-verbatim, relocated from ``src/canon`` in P5;
the publish gates live in ``publish/``) is injected as ``ledger``. Every day appends
exactly ONE frontier leaf via ``ledger.append`` (append-only + supersede;
``_assert_cache_not_shrunk``; ``G-APPEND-ONLY-FRONTIER``); no settled leaf is ever
overwritten. One leaf carries all three displayed lines: ``value`` = the canon
engine book, ``benchmark`` = the div-reinvested SPY line, ``comparison`` = the
shadow/challenger M1-tilt line (the dotted line the leaf schema already reserves).

CHALLENGER. Exactly ONE challenger mechanism: the shadow line is the M1-conviction
tilt of the SAME ``run_engine`` selection — same mu (0.0-diff), same selected grid,
**tilt only** (Stage-2 weights re-shaped by mu conviction, membership untouched).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
CORE_ROOT = _THIS.parents[1]                       # trader-bot-core
SEED_OHLCV = CORE_ROOT / "store" / "seeds" / "cache" / "ohlcv"
UNIVERSE_CSV = CORE_ROOT / "config" / "universe.csv"

# Frozen-champion terminal anchor (src/canon/equity_seed.py): the displayed line
# through 2026-06-11 is 114772.39. Used only as the DEFAULT genesis anchor; the
# reality-test seeds its own anchor for a mechanism proof over the seed+parity
# window (<= 2026-06-09), which precedes the champion boundary. Real prod seeding
# with the correct segment/boundary is P6, not this packet.
FROZEN_CHAMPION_TERMINAL = 114772.39

# SPY benchmark dividend accrual (mirrors src/steps/paper_trader.py:368) — a
# ~1.3%/yr daily drip reinvested into the buy-and-hold benchmark.
SPY_ANNUAL_DIV = 0.013
TRADING_DAYS_YR = 252


# --------------------------------------------------------------------------- #
# OHLCV access (settled prices, straight from the seed store — no S3, no future)
# --------------------------------------------------------------------------- #
class _OHLCVStore:
    """Cached reader over the seed OHLCV parquets. The seed closes cross-check the
    live S3 daily snapshots to ~1e-4 (store/seeds/cache/ohlcv/coverage_report.json),
    so marking off them reproduces the settled S3 marks."""

    def __init__(self, root: Path = SEED_OHLCV):
        self.root = Path(root)
        self._cache: Dict[str, pd.DataFrame] = {}

    def frame(self, sym: str) -> Optional[pd.DataFrame]:
        if sym not in self._cache:
            p = self.root / f"{sym}.parquet"
            if not p.exists():
                self._cache[sym] = None
            else:
                df = pd.read_parquet(p, columns=["date", "open", "close"])
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                self._cache[sym] = df.sort_values("date").reset_index(drop=True)
        return self._cache[sym]

    def trading_days(self, anchor_sym: str = "SPY") -> List[str]:
        df = self.frame(anchor_sym)
        return [d.strftime("%Y-%m-%d") for d in df["date"]]

    def bar(self, sym: str, date: str) -> Optional[dict]:
        df = self.frame(sym)
        if df is None:
            return None
        ts = pd.Timestamp(date)
        row = df[df["date"] == ts]
        if row.empty:
            return None
        r = row.iloc[-1]
        return {"open": float(r["open"]), "close": float(r["close"])}

    def prev_trading_day(self, date: str, anchor_sym: str = "SPY") -> Optional[str]:
        days = self.trading_days(anchor_sym)
        if date not in days:
            # nearest earlier settled day present in the calendar
            earlier = [d for d in days if d < date]
            return earlier[-1] if earlier else None
        i = days.index(date)
        return days[i - 1] if i > 0 else None

    def close_asof(self, sym: str, date: str) -> Optional[float]:
        """Most recent close on/before `date` (the last-known mark at decision time)."""
        df = self.frame(sym)
        if df is None:
            return None
        sub = df[df["date"] <= pd.Timestamp(date)]
        if sub.empty:
            return None
        return float(sub.iloc[-1]["close"])

    def realized_vol_asof(self, sym: str, date: str, window: int = 21) -> float:
        """21-day realized vol of daily close returns through `date` (idio_vol
        tiebreak input; deterministic, backward-only)."""
        df = self.frame(sym)
        if df is None:
            return 0.0
        sub = df[df["date"] <= pd.Timestamp(date)].tail(window + 1)
        if len(sub) < 5:
            return 0.0
        r = sub["close"].to_numpy()
        rets = r[1:] / r[:-1] - 1.0
        v = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_YR))
        return v if v == v else 0.0


# --------------------------------------------------------------------------- #
# the recomputed book (chained WITHIN the replay — never read from stored state)
# --------------------------------------------------------------------------- #
@dataclass
class Book:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)

    def value(self, closes: Mapping[str, float]) -> float:
        """cash + sum(shares * settled close). A symbol with no close for the day
        contributes 0 (it will have been sold or is untradable) — the same
        never-drop-cash rule as the shadow marker (robust_value)."""
        v = float(self.cash)
        for sym, sh in self.positions.items():
            if sh and sym in closes:
                v += sh * float(closes[sym])
        return v

    def apply(self, intents: List[dict], fill: Mapping[str, float]) -> None:
        """Execute the recomputed intents at the settled OPEN (the fill), moving
        cash and shares. Cash is conserved (no external flow); the book value is
        continuous through the trade. Mirrors the morning executor's open-fill /
        close-mark cycle without pulling its S3-coupled machinery."""
        for it in intents:
            sym = str(it.get("symbol", ""))
            act = str(it.get("action", "")).upper()
            px = fill.get(sym)
            if px is None or px <= 0:
                # no fill price for the day -> the trade cannot execute; leave the
                # position untouched (it stays marked at its last close).
                continue
            if act == "BUY":
                sh = int(it.get("shares", 0))
                self.positions[sym] = self.positions.get(sym, 0) + sh
                self.cash -= sh * px
            elif act in ("SELL",):
                sh = int(self.positions.get(sym, 0))
                self.positions[sym] = 0
                self.cash += sh * px
            elif act in ("REDUCE",):
                sh = int(it.get("reduce_shares", it.get("shares", 0)))
                have = int(self.positions.get(sym, 0))
                sh = min(sh, have)
                self.positions[sym] = have - sh
                self.cash += sh * px
            # HOLD: no-op
        # prune flat positions
        self.positions = {s: sh for s, sh in self.positions.items() if sh}


# --------------------------------------------------------------------------- #
# as-of-D substrate rebuild + the shared forecast->engine spine
# --------------------------------------------------------------------------- #
def _ensure_substrate(state_dir: Optional[str]) -> None:
    """Seed the writable STATE tree and build gdelt_features FROM SEEDS ONLY (no
    forward network fetch). The panel's end=D bound + the inference parity
    self-check make the physically-present future rows inert for date D."""
    if state_dir:
        os.environ["BRAIN_STATE_DIR"] = str(state_dir)
    from forecast import inference as FI
    from forecast import shadow_lib as SL
    from forecast import features_gdelt as FG

    SL.STATE.mkdir(parents=True, exist_ok=True)
    FI.ensure_seed_caches()
    out = SL.STORE / "gdelt_features.parquet"
    if not out.exists():
        FG.build_gdelt_features(daily_dir=SL.GDELT_DAILY_DIR, out_path=out)


def mu_asof(date: str) -> Tuple[Dict[str, float], str]:
    """Rebuild the panel as-of `date` (end=date) and run the SAME inference the
    forward night runs. Returns (full-universe mu, mu_sha16). ABORTS via the
    inference parity self-check if the as-of-D recompute diverges from the frozen
    forward reference (the reality gate that mu-parity vs forward is 0.0 and no
    future bar leaked)."""
    from forecast import inference as FI

    FI.build_panel([date])
    recs = FI.run_inference([date], parity_check=True)
    rec = recs.get(date)
    if not rec or not rec.get("mu"):
        raise RuntimeError(f"as-of-{date} inference produced no mu — substrate "
                           f"cannot be rebuilt for this date (STOP; do NOT re-mark)")
    mu = {str(s): float(v) for s, v in rec["mu"].items()}
    return mu, mu_sha16(mu)


def mu_sha16(mu: Mapping[str, float]) -> str:
    return hashlib.sha256(json.dumps(
        {k: round(float(v), 8) for k, v in sorted(mu.items())},
        sort_keys=True).encode()).hexdigest()[:16]


def _features_df_asof(date: str, universe_df, ohlcv: _OHLCVStore):
    """A one-row-per-symbol features frame carrying the decision-time mark
    (close as-of D-1 — the last known price) + realized vol. This is what
    build_forecast_bundle/build_portfolio_state read for idio_vol + marks."""
    prev = ohlcv.prev_trading_day(date) or date
    rows = []
    for sym in universe_df["symbol"].astype(str):
        c = ohlcv.close_asof(sym, prev)
        if c is None or c <= 0:
            continue
        rows.append({"symbol": sym, "date": prev, "close": c,
                     "vol_21d": ohlcv.realized_vol_asof(sym, prev)})
    return pd.DataFrame(rows)


def _engine_for_date(date: str, mu: Mapping[str, float], book: Book,
                     universe_df, features_df, theta_sel, theta_size,
                     regime_compat: Optional[Mapping] = None,
                     regime_label: str = "neutral"):
    """Build f + portfolio state from the CARRIED book and run the SAME run_engine
    the forward path runs. Returns the EngineOutput (canon picks)."""
    from adapter.forecast_adapter import build_forecast_bundle, build_portfolio_state
    from engine import run_engine

    prev = None
    for _, r in features_df.iterrows():
        prev = r["date"]
        break
    holdings = []
    for sym, sh in book.positions.items():
        cp = float(features_df.loc[features_df["symbol"] == sym, "close"].iloc[0]) \
            if (features_df["symbol"] == sym).any() else 0.0
        holdings.append({"symbol": sym, "shares": int(sh), "current_price": cp})
    portfolio_state = {"cash": float(book.cash), "holdings": holdings,
                       "date": prev}

    f = build_forecast_bundle(date, mu, features_df, regime_label, universe_df,
                              health_map={}, regime_compat=regime_compat)
    portfolio = build_portfolio_state(portfolio_state, features_df, universe_df)
    return run_engine(f, theta_sel, theta_size, portfolio), f


# The ONE challenger mechanism (M1-conviction tilt) lives in ``publish/challenger.py``
# — its single canonical home. Imported at the call site (below) so there is exactly
# one copy of the tilt in ``trader-bot-core/``.


# --------------------------------------------------------------------------- #
# the replay loop
# --------------------------------------------------------------------------- #
@dataclass
class DayResult:
    date: str
    mu_sha16: str
    regime_label: str
    canon_selected: List[str]
    canon_value: float
    shadow_value: float
    benchmark_value: float
    canon_return: float
    shadow_return: float
    benchmark_return: float
    leaf: dict


@dataclass
class ReplayResult:
    days: List[DayResult]
    terminal_leaf: dict
    selected_by_date: Dict[str, List[str]]


def replay(
    d0: str,
    d1: str,
    *,
    ledger,
    state_dir: Optional[str] = None,
    issued_by: str = "true-replay-keystone@trader-bot",
    genesis_anchor: float = 100_000.0,
    genesis_date: Optional[str] = None,
    freeze_mu_dates: Optional[List[str]] = None,
    selected_universe_sink: Optional[str] = None,
    regime_fn: Optional[Callable[[str], str]] = None,
    ohlcv: Optional["_OHLCVStore"] = None,
    write_genesis: bool = True,
    bench_anchor: Optional[float] = None,
    comparison_anchor: Optional[float] = None,
    segment: str = "new_brain",
    source: str = "native_two_stage",
    model_id_prefix: str = "replay@",
    leaf_extra: Optional[Dict] = None,
) -> ReplayResult:
    """Replay [d0..d1] by RECOMPUTING each settled day's picks on the as-of-D
    substrate and appending ONE settled leaf per day to `ledger`.

    `ledger` is any object exposing the KEEP-verbatim EquityLedger interface
    (.append(...), .read_manifest()). The genesis anchor leaf (at `genesis_date`,
    default = the trading day before d0) seeds the frontier; each replayed day
    appends value=canon, benchmark=SPY, comparison=shadow.

    `freeze_mu_dates`: if given, the mu computed on the FIRST of these dates is
    reused (frozen) for all of them — the deliberate frozen-mu sub-window that
    ``monitors.check_substrate_fresh`` must flag stale. This never re-marks; it
    injects a frozen forecast to prove the freeze cannot pass silently.
    """
    from forecast.freeze import load_freeze
    from forecast.regime import regime
    from decide.cutover import theta_from_freeze, load_brain_config

    _ensure_substrate(state_dir)
    ohlcv = ohlcv if ohlcv is not None else _OHLCVStore()
    universe_df = pd.read_csv(UNIVERSE_CSV)
    theta_sel, theta_size = theta_from_freeze(load_freeze())
    try:
        regime_compat = load_brain_config().get("regime_compatibility") or {}
    except Exception:  # noqa: BLE001 — absent table -> identity ranking (neutral)
        regime_compat = {}

    all_days = ohlcv.trading_days()
    window = [d for d in all_days if d0 <= d <= d1]
    if not window:
        raise RuntimeError(f"no settled trading days in [{d0}..{d1}] in the seed "
                           f"calendar (STOP; the window is not replayable)")

    gdate = genesis_date or ohlcv.prev_trading_day(window[0])
    if gdate is None:
        raise RuntimeError("no trading day precedes the window; cannot anchor")

    # Per-line anchors. In the P4 keystone proof all three start equal at
    # ``genesis_anchor``. In the P6 continuous-anchor mode the three displayed
    # lines each start at their own pre-split terminal (canon/comparison at the
    # frozen-champion terminal, benchmark at its own SPY terminal), so the
    # reconstruction joins the byte-unchanged pre-split line with NO seam.
    anchor_canon = genesis_anchor
    anchor_bench = bench_anchor if bench_anchor is not None else genesis_anchor
    anchor_shadow = comparison_anchor if comparison_anchor is not None else genesis_anchor

    # genesis anchor leaf (frontier seed) — only when seeding a fresh ledger. In
    # continuous-anchor mode (write_genesis=False) the ledger ALREADY carries the
    # byte-unchanged pre-split leaves through ``gdate``; the reconstruction appends
    # onto that existing frontier without minting (or disturbing) an anchor leaf.
    if write_genesis:
        ledger.append(date=gdate, value=anchor_canon, benchmark=anchor_bench,
                      comparison=anchor_shadow, segment=segment,
                      model_id="genesis-anchor", source="replay-genesis",
                      issued_by=issued_by)

    canon = Book(cash=anchor_canon)
    shadow = Book(cash=anchor_shadow)
    spy_close0 = ohlcv.close_asof("SPY", gdate)
    bm_shares = anchor_bench / spy_close0 if spy_close0 else 0.0

    prev_canon_v = anchor_canon
    prev_shadow_v = anchor_shadow
    prev_bm_v = anchor_bench
    disp_canon = anchor_canon
    disp_shadow = anchor_shadow
    disp_bm = anchor_bench

    freeze_set = set(freeze_mu_dates or [])
    frozen_mu: Optional[Dict[str, float]] = None
    results: List[DayResult] = []
    selected_by_date: Dict[str, List[str]] = {}

    for D in window:
        prevD = ohlcv.prev_trading_day(D)
        closes_prev = {s: c for s in universe_df["symbol"].astype(str)
                       if (c := ohlcv.close_asof(s, prevD)) is not None}
        closes_D = {s: b["close"] for s in universe_df["symbol"].astype(str)
                    if (b := ohlcv.bar(s, D)) is not None}
        open_D = {s: b["open"] for s in universe_df["symbol"].astype(str)
                  if (b := ohlcv.bar(s, D)) is not None}

        # --- as-of-D forecast (RECOMPUTED; parity-gated) ---
        if D in freeze_set and frozen_mu is not None:
            mu, msha = frozen_mu, mu_sha16(frozen_mu)          # deliberate freeze
        else:
            mu, msha = mu_asof(D)
            if D in freeze_set and frozen_mu is None:
                frozen_mu = dict(mu)

        features_df = _features_df_asof(D, universe_df, ohlcv)

        # --- regime label for D ---
        # Default (P4 keystone + forward path): the ONE deterministic as-of-D
        # picker, shared with ``decide.cutover.run_cutover``. P6 reconstruction:
        # ``regime_fn`` injects the RECORDED regime the algorithm consumed (locked
        # operator scope #3 — the deterministic picker is FORWARD-only).
        reg_label = regime_fn(D) if regime_fn is not None else regime(D)

        # --- the SAME run_engine as forward (canon) ---
        engine_out, f = _engine_for_date(
            D, mu, canon, universe_df, features_df, theta_sel, theta_size,
            regime_compat=regime_compat, regime_label=reg_label)
        canon_intents = engine_out.trade_intents["actions"]
        selected = sorted(engine_out.allocation.held_symbols)
        selected_by_date[D] = selected

        # --- shadow: the ONE challenger, M1-tilt of the same selection ---
        from publish.challenger import shadow_tilt_intents
        shadow_intents = shadow_tilt_intents(engine_out, f)

        # --- mark RECOMPUTED picks at settled prices (chained WITHIN replay) ---
        # value of the carried book BEFORE today's trades, marked at prevD close
        prev_canon_v = canon.value(closes_prev) if results or canon.positions else prev_canon_v
        prev_shadow_v = shadow.value(closes_prev) if results or shadow.positions else prev_shadow_v
        canon.apply(canon_intents, open_D)
        shadow.apply(shadow_intents, open_D)
        canon_v = canon.value(closes_D)
        shadow_v = shadow.value(closes_D)

        # SPY benchmark: div-reinvested buy-and-hold
        spy_D = closes_D.get("SPY") or ohlcv.close_asof("SPY", D)
        if spy_D and prevD:
            daily_div = spy_D * (SPY_ANNUAL_DIV / TRADING_DAYS_YR)
            bm_shares += (bm_shares * daily_div) / spy_D if spy_D else 0.0
        bm_v = bm_shares * spy_D if spy_D else prev_bm_v

        canon_ret = canon_v / prev_canon_v - 1.0 if prev_canon_v else 0.0
        shadow_ret = shadow_v / prev_shadow_v - 1.0 if prev_shadow_v else 0.0
        bm_ret = bm_v / prev_bm_v - 1.0 if prev_bm_v else 0.0

        disp_canon *= (1.0 + canon_ret)
        disp_shadow *= (1.0 + shadow_ret)
        disp_bm *= (1.0 + bm_ret)

        # --- append ONE settled leaf (append-only + supersede; never overwrite) ---
        leaf = ledger.append(
            date=D, value=disp_canon, benchmark=disp_bm, comparison=disp_shadow,
            segment=segment, model_id=f"{model_id_prefix}{msha}",
            source=source, issued_by=issued_by, extra=leaf_extra)

        if selected_universe_sink:
            _emit_selected_universe(selected_universe_sink, D, selected, msha)

        results.append(DayResult(
            date=D, mu_sha16=msha, regime_label=reg_label, canon_selected=selected,
            canon_value=disp_canon, shadow_value=disp_shadow,
            benchmark_value=disp_bm, canon_return=canon_ret,
            shadow_return=shadow_ret, benchmark_return=bm_ret, leaf=leaf))
        prev_canon_v, prev_shadow_v, prev_bm_v = canon_v, shadow_v, bm_v

    return ReplayResult(days=results,
                        terminal_leaf=results[-1].leaf if results else {},
                        selected_by_date=selected_by_date)


def _emit_selected_universe(sink: str, date: str, selected: List[str], msha: str) -> None:
    """Write the day's brain_selected_universe.json into a local daily/<D>/ tree —
    the S3-observable fingerprint monitors.check_substrate_fresh keys on. A frozen
    mu yields byte-identical selected_universe across days (the stale signature)."""
    d = Path(sink) / "daily" / date
    d.mkdir(parents=True, exist_ok=True)
    (d / "brain_selected_universe.json").write_text(json.dumps(
        {"date": date, "selected_universe": selected, "mu_sha16": msha}, indent=2))


# --------------------------------------------------------------------------- #
# CLI — the local governed invoke surface (P6 can wrap this in an AWS route)
# --------------------------------------------------------------------------- #
def _main() -> int:
    import argparse
    import sys
    sys.path.insert(0, str(CORE_ROOT))
    sys.path.insert(0, str(CORE_ROOT.parent))       # src.*
    ap = argparse.ArgumentParser(prog="replay.driver",
                                 description="true history replay on the clean spine")
    ap.add_argument("--d0", required=True)
    ap.add_argument("--d1", required=True)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--sink", default=None, help="dir for brain_selected_universe leaves")
    args = ap.parse_args()

    from lines.ledger import EquityLedger        # KEEP-verbatim ledger, relocated (P5)
    from replay._fake_s3 import FakeS3            # local in-mem S3 (reality-test sink)

    ledger = EquityLedger(FakeS3(), bucket="investment-system-data")
    res = replay(args.d0, args.d1, ledger=ledger, state_dir=args.state_dir,
                 selected_universe_sink=args.sink)
    print(json.dumps({
        "window": [args.d0, args.d1],
        "n_days": len(res.days),
        "terminal_leaf": {k: res.terminal_leaf.get(k)
                          for k in ("date", "value", "benchmark", "comparison")},
        "picks_by_date": {r.date: r.canon_selected[:10] for r in res.days},
        "mu_sha_by_date": {r.date: r.mu_sha16 for r in res.days},
        "regime_by_date": {r.date: r.regime_label for r in res.days},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
