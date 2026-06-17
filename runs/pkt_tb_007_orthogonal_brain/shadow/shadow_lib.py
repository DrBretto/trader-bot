"""PKT-TB-007 dual forward shadow — library (state, caches, paper books).

Plan: docs/plans/2026-06-11-dual-forward-shadow.md (binding).
Verdict contract: SHADOW_PREREG.md (this directory).

Two legs, both written nightly BEFORE outcomes exist:
  FORECAST LEG  — M1 deploy-inference per-symbol mu/ranks, appended to an
                  append-only ledger with a wall-clock timestamp + feature
                  hash; scored at +5 trading days (weekly cadence, the
                  gates_007.windowed_ics convention).
  UTILITY LEG   — three paper books reconstructed ON TOP of the chassis's
                  actual decisions (daily/<D>/trade_intents.json):
                    I = incumbent paper book (actual intents, no tilt)
                    A = a-priori genome tilt   (genomes/shadow_A.json)
                    B = a-priori + M4-A damping (genomes/shadow_B.json)
                  filled at D's open from daily/<D+1>/prices.parquet (the
                  harness's own D/D+1 convention), marked to D's close,
                  post-hoc cost overlay seed 4242 (TB-006 adjudication #2).

REUSED (imported, not forked):
  - src/utils/three_line_replay/replay_engine.py : Position, Portfolio,
    _ohlc_for_date, _mark_to_close, _apply_cluster_cap
  - runs/pkt_tb_007_orthogonal_brain/prototype/tilt_adapter.py (the ONE
    expression channel), genome_007, lot_fix_007 (_execute_intents_lotfix),
    risk_stats_007.RiskStats, run_replay_007.cost_overlay
  - runs/pkt_tb_006_clean_sheet_brain/prototype/feature_store.FeatureStore,
    features_gdelt.build_gdelt_features, gdelt_backfill (dirs re-pointed)

COPIED MINIMALLY (module boundaries did not allow import; each noted):
  - seed_portfolio (replay_engine hardcodes START_PORTFOLIO_DATE; ours is
    parameterized by date)  -> seed_book_from_state()
  - evt_ctl / disp_lags / cal_dummies blocks from make_targets_007.py (they
    live inside a monolithic main())  -> forward_inference.build_p7_extras()
  - trailing_pct from precompute_nightly_007.py (module imports torch at
    top level via main(); 5 lines)    -> forward_inference.trailing_pct()

FORWARD-ONLY: the shadow refuses to process any decision date <= 2026-06-10.

Known instrumentation notes (recorded, not silently absorbed):
  - forward OHLCV rows are spliced from S3 daily/<D>/prices.parquet with
    adj_close := close (no future-dividend back-adjustment); ex-dividend
    days appear as real price drops in forward features, a small fidelity
    gap vs the dividend-adjusted training history. Splits are NOT auto-
    adjusted: a >35% overnight move on any symbol raises an ALERT line.
  - CBOE staleness > 5 calendar days nulls disp_z (honest B-disp fallback);
    GDELT staleness uses the trained gdelt_available masking convention.
  - alerting = ALERT lines in the nightly log only (operator SNS is out of
    scope for this packet).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SHADOW = Path(__file__).resolve().parent
RUN_ROOT = SHADOW.parent                       # runs/pkt_tb_007_orthogonal_brain
PROTO7 = RUN_ROOT / "prototype"
REPO = RUN_ROOT.parents[1]
TB6_PROTO = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
for _p in (str(SHADOW), str(PROTO7), str(REPO), str(TB6_PROTO)):
    if _p not in sys.path:
        sys.path.append(_p)

# ---------------------------------------------------------------- constants
FORWARD_BOUNDARY = "2026-06-10"      # nothing <= this date is ever processed
SHADOW_PANEL_START = "2025-05-01"    # gives full 252d trailing windows at D0
BUCKET = "investment-system-data"
AWS_PROFILE = "personal"
S3_STATE_PREFIX = "shadow/pkt_tb_007/"
S3_DASHBOARD_KEY = "dashboard/shadow_timeseries.json"
COST_SEED = 4242
IC_STEP_TD = 5                       # weekly cadence (gates_007.windowed_ics)
HORIZON_TD = 5                       # forecast maturity
SPLIT_ALERT_RET = 0.35               # |1d move| above this => ALERT (split?)
CBOE_STALE_DAYS = 5                  # calendar days; staler => disp_z null
PREREG_POINTER = ("runs/pkt_tb_007_orthogonal_brain/shadow/SHADOW_PREREG.md"
                  " (mirrored at s3://investment-system-data/shadow/pkt_tb_007/"
                  "SHADOW_PREREG.md)")

STATE = SHADOW / "state"
CACHE_DAILY = STATE / "cache" / "daily"
CACHE_OHLCV = STATE / "cache" / "ohlcv"
CACHE_CBOE = STATE / "cache" / "cboe"
CACHE_FRED = STATE / "cache" / "fred"
CACHE_COT = STATE / "cache" / "cot"
GDELT_DAILY_DIR = STATE / "gdelt" / "daily"
GDELT_RECORDS_DIR = STATE / "gdelt" / "records"
STORE = STATE / "store"
ORGAN_DIR = STATE / "organ_inputs"
LEDGERS = STATE / "ledgers"
STATE_JSON = STATE / "state.json"
LOGS = SHADOW / "logs"

DAILY_FILES_REQUIRED = ["prices.parquet", "features.parquet",
                        "trade_intents.json", "portfolio_state.json"]
# brain_selected_universe.json (PKT-TB-012): the live engine's selected set,
# consumed by the U rung (_restrict_universe) so the U-E universe-choice rung is
# wired to the live selected_set instead of reading ~0 (U==E).
DAILY_FILES_OPTIONAL = ["morning_prices.parquet", "brain_selected_universe.json"]

GENOME_A = SHADOW / "genomes" / "shadow_A.json"
GENOME_B = SHADOW / "genomes" / "shadow_B.json"

# ---------------------------------------------------------------- rent ladder
# PKT-TB-011: the 2-organ A/B side-car (I, A=M1 tilt, B=A+M4 damp) is
# generalized into the 5-rung leave-one-organ-out ladder
# (DESIGN_DOSSIER §2 STAGE 4; reports/04 §1.2). Each ADJACENT pair is a clean
# single-organ marginal:
#   I  raw chassis intents                              baseline    --
#   R  I + regime gate (exposure throttle)              R-I  regime
#   F  R + M1 forecast tilt                             F-R  forecast (signal)
#   E  F + M4 event-damp                                E-F  event-damping
#   U  E + universe-choice (selected vs full)           U-E  universe-choice
# U is the deployed New Brain line. The decomposition is additive in
# LOG-return space: (U-I) = (R-I)+(F-R)+(E-F)+(U-E)  (C8).
LADDER = (
    {"book": "I", "parent": None, "organs": (),                                 "component": None},
    {"book": "R", "parent": "I",  "organs": ("regime",),                        "component": "regime"},
    {"book": "F", "parent": "R",  "organs": ("regime", "M1"),                   "component": "forecast"},
    {"book": "E", "parent": "F",  "organs": ("regime", "M1", "M4"),             "component": "event"},
    {"book": "U", "parent": "E",  "organs": ("regime", "M1", "M4", "universe"), "component": "universe"},
)
BOOKS = tuple(r["book"] for r in LADDER)            # ("I","R","F","E","U")
RUNGS = tuple(r for r in LADDER if r["parent"] is not None)
# back-compat aliases: the old A/B books map onto the ladder (A->F, B->E).
LEGACY_BOOK_ALIAS = {"A": "F", "B": "E"}

# Per-rung tilt genome. R carries NO per-name tilt (regime is a book-level
# exposure throttle applied in process_book_date, not a gene). F carries the
# frozen a-priori M1 genome (= old shadow_A); E adds the M4-A damp (= old
# shadow_B); U reuses E's genome and adds a universe-restriction post-filter.
LADDER_GENOMES = {
    "M1": {"tilt_gain": 0.5, "conviction_temp": 1.0, "dead_zone": 0.05,
           "disp_gain": 1.0, "cap_core": 0.0125, "cap_conditional": 0.00625,
           "defensive_fraction": 0.25, "event_damp_strength": 0.0,
           "organ_trust": {"M1": 1.0}},
    "M1_M4": {"tilt_gain": 0.5, "conviction_temp": 1.0, "dead_zone": 0.05,
              "disp_gain": 1.0, "cap_core": 0.0125, "cap_conditional": 0.00625,
              "defensive_fraction": 0.25, "event_damp_strength": 0.5,
              "organ_trust": {"M1": 1.0, "M4": 1.0}},
}
# book -> genome key (None = no tilt strategy)
RUNG_GENOME = {"I": None, "R": None, "F": "M1", "E": "M1_M4", "U": "M1_M4"}

# Regime -> book-level gross-exposure multiplier (the regime GATE as an
# exposure throttle). Identity (1.0) for unlisted regimes so the R rung is
# honestly ~zero when the regime does not bite. Frozen by hand.
REGIME_EXPOSURE = {"panic": 0.50, "force_sell": 0.50, "defensive": 0.75,
                   "caution": 0.85, "neutral": 1.0, "constructive": 1.0,
                   "risk_on": 1.0}

# Multi-factor exposure-strip basis (C6): SPY + duration + broad-commodity.
# The strip uses REALIZED multi-factor regression betas of each rung's daily
# return-difference on these factor returns -- NOT the static gross-weighted
# beta_proxy the Analyst proved misses the duration/FX/commodity sleeve where
# the F8 leak lives -- and also prints the realized gross-differential term.
STRIP_FACTORS = (("mkt", "SPY"), ("duration", "TLT"), ("commodity", "USO"))
# rungs whose marginal is intrinsically an exposure move: the regime throttle
# changes gross exposure by construction, so its GROSS column is the
# meaningful one and a full strip is not claimed (C7 non-strippable label).
EXPOSURE_RUNGS = {"regime"}

# three-valued verdict band (SHADOW_PREREG §2 / TB-006 §7) + BH-FDR target
MATERIALITY_BP = 1.5
FDR_Q = 0.10
BOOKS_LEGACY = ("I", "A", "B")      # the pre-PKT-011 payload books (display compat)


# ---------------------------------------------------------------- small utils
def log_line(msg: str, logf: Optional[Path] = None) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    if logf is not None:
        logf.parent.mkdir(parents=True, exist_ok=True)
        with logf.open("a") as fh:
            fh.write(line + "\n")


def jdump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=float)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(jdump(row) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


# ---------------------------------------------------------------- S3 layer
def s3_client():
    import boto3
    return boto3.Session(profile_name=AWS_PROFILE,
                         region_name="us-east-1").client("s3")


class S3Source:
    """The real daily-artifact source. Tests substitute LocalSource."""

    def __init__(self, client=None):
        self.s3 = client or s3_client()

    def list_daily_dates(self) -> List[str]:
        pg = self.s3.get_paginator("list_objects_v2")
        out = set()
        for p in pg.paginate(Bucket=BUCKET, Prefix="daily/", Delimiter="/"):
            for cp in p.get("CommonPrefixes", []) or []:
                out.add(cp["Prefix"].split("/")[-2])
        return sorted(out)

    def fetch_daily(self, date: str, dest: Path) -> Dict[str, bool]:
        """Download required+optional daily files into dest/<date>/.
        Returns {filename: present}. Skip-existing (idempotent)."""
        got = {}
        for fname in DAILY_FILES_REQUIRED + DAILY_FILES_OPTIONAL:
            p = dest / date / fname
            if p.exists():
                got[fname] = True
                continue
            try:
                body = self.s3.get_object(
                    Bucket=BUCKET, Key=f"daily/{date}/{fname}")["Body"].read()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
                got[fname] = True
            except Exception:
                got[fname] = False
        return got

    def get_json(self, key: str) -> Any:
        return json.loads(self.s3.get_object(Bucket=BUCKET, Key=key)
                          ["Body"].read())

    def put_json(self, key: str, obj: Any) -> None:
        assert key == S3_DASHBOARD_KEY or key.startswith(S3_STATE_PREFIX), \
            f"S3 write outside allowed surface: {key}"
        self.s3.put_object(Bucket=BUCKET, Key=key,
                           Body=json.dumps(obj, sort_keys=True,
                                           default=float).encode(),
                           ContentType="application/json")

    def put_file(self, key: str, path: Path) -> None:
        assert key.startswith(S3_STATE_PREFIX), \
            f"S3 write outside allowed surface: {key}"
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=path.read_bytes())


class LocalSource:
    """Filesystem daily-artifact source for tests (no network)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.published: Dict[str, Any] = {}

    def list_daily_dates(self) -> List[str]:
        return sorted(d.name for d in (self.root / "daily").iterdir()
                      if d.is_dir())

    def fetch_daily(self, date: str, dest: Path) -> Dict[str, bool]:
        got = {}
        for fname in DAILY_FILES_REQUIRED + DAILY_FILES_OPTIONAL:
            src = self.root / "daily" / date / fname
            p = dest / date / fname
            if p.exists():
                got[fname] = True
                continue
            if src.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, p)
                got[fname] = True
            else:
                got[fname] = False
        return got

    def get_json(self, key: str) -> Any:
        p = self.root / key
        if not p.exists():
            raise FileNotFoundError(key)
        return json.loads(p.read_text())

    def put_json(self, key: str, obj: Any) -> None:
        self.published[key] = obj

    def put_file(self, key: str, path: Path) -> None:
        self.published[key] = path.read_bytes()


# ---------------------------------------------------------------- pending dates
def is_decision_dir(files_present: Dict[str, bool]) -> bool:
    return all(files_present.get(f) for f in DAILY_FILES_REQUIRED)


def pending_dates(all_dates: List[str], last_settled: Optional[str]) -> List[str]:
    """Dates strictly after FORWARD_BOUNDARY and after last_settled.
    The forward-only refusal lives here AND in process-time guards."""
    out = [d for d in all_dates if d > FORWARD_BOUNDARY]
    if last_settled:
        out = [d for d in out if d > last_settled]
    return sorted(out)


def assert_forward_only(date: str) -> None:
    if date <= FORWARD_BOUNDARY:
        raise RuntimeError(
            f"FORWARD-ONLY REFUSAL: decision date {date} <= "
            f"{FORWARD_BOUNDARY} can never be processed by the shadow")


# ---------------------------------------------------------------- state
def default_state() -> dict:
    return {"schema": "shadow_state.v1",
            "forward_boundary": FORWARD_BOUNDARY,
            "start_date": None,            # D0, set on first processed date
            "baseline": None,              # {live_value, nav} at D0 close
            "last_settled_date": None,
            "n_settled": 0,
            "books": None}                 # serialized books after last settle


def load_state() -> dict:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return default_state()


def save_state(state: dict) -> None:
    write_json(STATE_JSON, state)


# ---------------------------------------------------------------- paper books
def _replay_engine():
    from src.utils.three_line_replay import replay_engine as RE
    return RE


def seed_book_from_state(portfolio_state: dict):
    """COPIED MINIMALLY from replay_engine.seed_portfolio (which hardcodes
    START_PORTFOLIO_DATE + an S3Cache); parameterized by the state dict."""
    RE = _replay_engine()
    positions = []
    for h in portfolio_state.get("holdings", []):
        positions.append(RE.Position(
            symbol=h["symbol"], shares=float(h["shares"]),
            entry_price=float(h["entry_price"]),
            entry_date=h.get("entry_date", ""),
            peak_price=float(h.get("peak_price",
                                   h.get("current_price", h["entry_price"]))),
            asset_class=h.get("asset_class", "equity"),
            sector=h.get("sector", "broad"),
            leverage_flag=int(h.get("leverage_flag", 0) or 0),
            last_close=h.get("current_price"),
        ))
    return RE.Portfolio(cash=float(portfolio_state["cash"]),
                        positions=positions)


def book_to_dict(book) -> dict:
    return {"cash": book.cash,
            "positions": [{
                "symbol": p.symbol, "shares": p.shares,
                "entry_price": p.entry_price, "entry_date": p.entry_date,
                "peak_price": p.peak_price, "asset_class": p.asset_class,
                "sector": p.sector, "leverage_flag": p.leverage_flag,
                "last_close": p.last_close} for p in book.positions]}


def book_from_dict(d: dict):
    RE = _replay_engine()
    return RE.Portfolio(
        cash=float(d["cash"]),
        positions=[RE.Position(
            symbol=p["symbol"], shares=float(p["shares"]),
            entry_price=float(p["entry_price"]), entry_date=p["entry_date"],
            peak_price=float(p["peak_price"]), asset_class=p["asset_class"],
            sector=p["sector"], leverage_flag=int(p["leverage_flag"]),
            last_close=p.get("last_close")) for p in d["positions"]])


def robust_value(book, ohlc: Dict[str, Dict[str, float]]
                 ) -> Tuple[float, List[str]]:
    """Cash + sum(shares x close), falling back to last_close/peak_price for
    symbols missing a bar (the replay's provisional-day valuation rule —
    never drop a holding from the valuation)."""
    missing = []
    v = float(book.cash)
    for p in book.positions:
        q = ohlc.get(p.symbol, {})
        mark = q.get("close") or p.last_close or p.peak_price
        if p.symbol not in ohlc:
            missing.append(p.symbol)
        v += p.shares * float(mark)
    return v, missing


def make_strategies(organ_dir: Path, risk, cache, log_dir: Optional[Path]):
    """Tilt strategies for the ladder rungs that carry a per-name tilt.

    PKT-TB-011: generalized from the 2-book (A,B) case to the 5-rung ladder.
    Only F (M1), E (M1+M4) and U (M1+M4, universe-restricted) carry a tilt
    genome; I and R carry none (R is a book-level regime throttle applied in
    ``process_book_date``). The genomes are the frozen a-priori M1 / M1+M4
    values (identical to the old shadow_A / shadow_B), passed inline.
    """
    from tilt_adapter import make_tilt_strategy
    strats = {}
    for book, gkey in RUNG_GENOME.items():
        if gkey is None:
            continue
        ld = (log_dir / f"expression_{book}") if log_dir else None
        if ld:
            ld.mkdir(parents=True, exist_ok=True)
        strats[book] = make_tilt_strategy(
            LADDER_GENOMES[gkey], organ_dir, log_dir=ld, cache=cache, risk=risk)
    return strats


def _regime_throttle(intents: List[dict], regime: Optional[str]) -> List[dict]:
    """The regime GATE as a book-level exposure throttle (the R rung organ).

    Scales the size of risk-ADDING legs (BUY/ADD/INCREASE) by the frozen
    ``REGIME_EXPOSURE`` multiplier for the day's regime. Identity (mult 1.0)
    for unlisted / neutral regimes, so R == I exactly when the regime does not
    bite and the R-I rung is honestly ~zero. This is an EXPOSURE move by
    construction (its gross column is the meaningful one; C7 labels it
    'exposure-strip incomplete')."""
    mult = REGIME_EXPOSURE.get((regime or "").lower(), 1.0)
    if mult == 1.0:
        return [dict(it) for it in intents]
    add = {"BUY", "ADD", "INCREASE", "OPEN"}
    out = []
    for it in intents:
        it = dict(it)
        if str(it.get("action", "")).upper() in add:
            for fld in ("shares", "dollars", "target_shares", "target_dollars",
                        "weight", "target_weight"):
                if isinstance(it.get(fld), (int, float)):
                    it[fld] = it[fld] * mult
        out.append(it)
    return out


def _restrict_universe(intents: List[dict],
                       selected_universe: Optional[set]) -> List[dict]:
    """The universe-choice organ (the U rung): keep only legs whose symbol is
    in the engine's SELECTED set (the New Brain universe) vs the full set E
    trades. ``selected_universe is None`` => identity (the live engine's
    selected_set is wired at the PKT-TB-012 cutover; until then U==E and the
    U-E rung is honestly ~zero, labeled on the surface)."""
    if not selected_universe:
        return [dict(it) for it in intents]
    return [dict(it) for it in intents
            if it.get("symbol") in selected_universe]


class LocalDailyCache:
    """tilt_adapter.guarded_marks fallback interface over the local daily
    cache (get_parquet only)."""

    def __init__(self, daily_dir: Path):
        self.daily_dir = Path(daily_dir)

    def get_parquet(self, key: str):
        import pandas as pd
        p = self.daily_dir / Path(key).relative_to("daily")
        return pd.read_parquet(p)


_BOOK_ORGANS = {r["book"]: r["organs"] for r in LADDER}


def process_book_date(books: Dict[str, Any], date: str,
                      intents: List[dict], ohlc: Dict[str, Dict[str, float]],
                      features_df, decision_params: dict,
                      strategies: Dict[str, Any], universe_df,
                      regime: Optional[str],
                      selected_universe: Optional[set] = None) -> Dict[str, Any]:
    """One settled decision date across the FIVE ladder books. Mutates books.
    Returns the equity-ledger row (raw NAVs; costs overlaid separately).

    PKT-TB-011: generalized from (I,A,B) to the leave-one-organ-out ladder.
    Per book the organs in its ``LADDER`` row are stacked, in order:
      regime  -> book-level exposure throttle (``_regime_throttle``)
      M1/M4   -> the tilt strategy (``strategies[book]``; F=M1, E/U=M1+M4)
      universe-> restrict to the engine's selected set (``_restrict_universe``)
    so I<R<F<E<U nest and each adjacent pair is one clean organ marginal.
    """
    import pandas as pd
    RE = _replay_engine()
    from lot_fix_007 import _execute_intents_lotfix
    from src.utils.three_line_replay.strategies import StrategyContext

    sector_by_symbol = dict(zip(universe_df["symbol"], universe_df["sector"]))
    current_marks = RE._latest_close_per_symbol(features_df)
    row: Dict[str, Any] = {"date": date, "provisional": False}
    actions_by_book: Dict[str, List[dict]] = {}
    for bname in BOOKS:
        book = books[bname]
        organs = _BOOK_ORGANS.get(bname, ())
        my_intents = [dict(it) for it in intents]
        if "regime" in organs:                      # R rung organ (and above)
            my_intents = _regime_throttle(my_intents, regime)
        if bname in strategies:                      # M1 / M4 tilt organs
            ctx = StrategyContext(
                inputs_date=date, portfolio=book,
                variant_config={"decision_params": dict(decision_params)},
                expert_signals={}, expert_metrics={}, decisions={},
                panic_streak=0, last_regime=None,
                features_df=features_df, inference={}, llm_risks={})
            my_intents = strategies[bname].post_decision(ctx, my_intents)
        if "universe" in organs:                     # U rung organ
            my_intents = _restrict_universe(my_intents, selected_universe)
        marks_ohlc = {s: {"close": v} for s, v in current_marks.items()}
        nav_pre, _ = robust_value(book, marks_ohlc)
        my_intents = RE._apply_cluster_cap(
            my_intents, book, current_marks, sector_by_symbol, {},
            decision_params.get("max_sector_weight"), nav_pre,
            float(decision_params.get("min_order_dollars", 250)))
        executed = _execute_intents_lotfix(book, my_intents, ohlc,
                                           decision_params, date,
                                           entry_regime=regime)
        RE._mark_to_close(book, ohlc)
        nav, missing = robust_value(book, ohlc)
        row[f"nav_{bname}"] = round(nav, 2)
        row[f"n_actions_{bname}"] = len(executed)
        if missing:
            row.setdefault("missing_marks", {})[bname] = missing
        actions_by_book[bname] = executed
    row["regime"] = regime
    return {"row": row, "actions": actions_by_book}


def overlay_books(actions_ledgers: Dict[str, List[dict]],
                  equity_rows: List[dict], universe_df) -> Dict[str, Any]:
    """Post-hoc cost overlay per book over the FULL action history (ONE
    shared rng sequence per book, seed 4242 — TB-006 adjudication #2,
    reused from run_replay_007.cost_overlay)."""
    from run_replay_007 import cost_overlay
    out = {}
    for bname in BOOKS:
        result = {"actions": actions_ledgers.get(bname, []),
                  "date_value_map": {r["date"]: r[f"nav_{bname}"]
                                     for r in equity_rows}}
        out[bname] = cost_overlay(result, universe_df, seed=COST_SEED)
    return out


# ---------------------------------------------------------------- statistics
def daily_diff_bp(dates: List[str], a: List[float], b: List[float]
                  ) -> List[float]:
    """Paired daily return differences (a minus b), basis points."""
    out = []
    for i in range(1, len(dates)):
        if a[i - 1] and b[i - 1]:
            out.append((a[i] / a[i - 1] - b[i] / b[i - 1]) * 1e4)
    return out


def hac_se(x, max_lag: int = 5) -> float:
    """Newey-West HAC standard error of the mean (Bartlett kernel)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(max_lag, n - 1) + 1):
        w = 1.0 - k / (max_lag + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return float((max(s, 0.0) / n) ** 0.5)


def compute_stats(ic_rows: List[dict], dates: List[str],
                  series: Dict[str, List[float]]) -> dict:
    import numpy as np
    stats: Dict[str, Any] = {"days_accrued": max(len(dates) - 1, 0),
                             "n_weeks_ic": len(ic_rows)}
    ics = [r["ic"] for r in ic_rows]
    if ics:
        m = float(np.mean(ics))
        se = (float(np.std(ics, ddof=1) / np.sqrt(len(ics)))
              if len(ics) > 1 else float("nan"))
        stats["mean_ic"] = round(m, 4)
        stats["ic_t"] = round(m / se, 3) if se and se > 0 else None
    else:
        stats["mean_ic"] = None
        stats["ic_t"] = None
    # Legacy 2-organ fields, remapped onto the ladder for display back-compat:
    #   utility_diff_bp_day = U-I (the deployed New Brain vs incumbent)
    #   m4a_diff_bp_day     = E-F (the M4 event-damp marginal, old B-A)
    for key, (x, y) in {"utility_diff_bp_day": ("U", "I"),
                        "m4a_diff_bp_day": ("E", "F")}.items():
        sx = series.get(x, series.get(LEGACY_BOOK_ALIAS.get(x, x), []))
        sy = series.get(y, series.get(LEGACY_BOOK_ALIAS.get(y, y), []))
        d = daily_diff_bp(dates, sx, sy)
        if len(d) >= 3:
            m = float(np.mean(d))
            se = hac_se(d)
            stats[key] = round(m, 4)
            stats[key + "_ci"] = [round(m - 1.96 * se, 4),
                                  round(m + 1.96 * se, 4)]
            if key == "utility_diff_bp_day":
                stats["ci"] = stats[key + "_ci"]
        else:
            stats[key] = None
            stats[key + "_ci"] = None
            if key == "utility_diff_bp_day":
                stats["ci"] = None
    return stats


# ============================================================ rent ladder math
# PKT-TB-011 attribution spine. Every function here operates on the per-book
# cost-adjusted NAV series the overlay produces; none reaches into the engine.
def _simple_returns(navs: List[float]) -> List[float]:
    out = []
    for i in range(1, len(navs)):
        out.append(navs[i] / navs[i - 1] - 1.0 if navs[i - 1] else 0.0)
    return out


def _log_returns(navs: List[float]) -> List[float]:
    import numpy as np
    out = []
    for i in range(1, len(navs)):
        if navs[i - 1] and navs[i] and navs[i] > 0 and navs[i - 1] > 0:
            out.append(float(np.log(navs[i] / navs[i - 1])))
        else:
            out.append(0.0)
    return out


def _norm_sf(z: float) -> float:
    """One-sided upper-tail standard-normal survival function."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _ols(y, X):
    """Least-squares fit of y on [1, X]; returns (intercept, slope_vector)."""
    import numpy as np
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), [float(c) for c in coef[1:]]


def multifactor_strip(gross_daily: List[float],
                      factor_daily: Dict[str, List[float]]) -> Dict[str, Any]:
    """C6: strip a rung's daily GROSS return-difference (bp) of its REALIZED
    multi-factor exposure (SPY + duration + broad-commodity), using realized
    regression betas -- NOT the static gross-weighted beta_proxy the Analyst
    proved misses the duration/FX/commodity sleeve where the F8 leak lives --
    and report the realized gross-differential (exposure) term explicitly.

    Returns the stripped daily series (selection skill), the per-factor betas,
    and the exposure vs stripped bp/day split so the F8 leak is visible: if
    gross is dominated by exposure and stripped straddles zero, the surface
    says so.
    """
    import numpy as np
    g = np.asarray(gross_daily, dtype=float)
    names = [n for n, v in factor_daily.items() if len(v) == len(g)]
    if len(g) < 3 or not names:
        return {"stripped_daily": list(g), "betas": {}, "exposure_daily": [0.0] * len(g),
                "exposure_bp_day": 0.0, "stripped_bp_day": (float(g.mean()) if len(g) else None),
                "factors_used": names}
    X = np.column_stack([np.asarray(factor_daily[n], dtype=float) for n in names])
    _intercept, slopes = _ols(g, X)                 # realized multi-factor betas
    betas = {n: round(b, 4) for n, b in zip(names, slopes)}
    exposure = X @ np.asarray(slopes, dtype=float)   # realized exposure term/day
    stripped = g - exposure                          # selection skill residual+alpha
    return {"stripped_daily": [float(s) for s in stripped],
            "betas": betas,
            "exposure_daily": [float(e) for e in exposure],
            "exposure_bp_day": round(float(exposure.mean()), 4),
            "stripped_bp_day": round(float(stripped.mean()), 4),
            "gross_diff_bp_day": round(float(g.mean()), 4),
            "factors_used": names}


def bh_fdr(pvals: List[Optional[float]], q: float = FDR_Q) -> List[bool]:
    """Benjamini-Hochberg survivors at false-discovery rate ``q``. None p-values
    (no power yet) never survive. Returns a survivor flag per input position."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    survivors = [False] * len(pvals)
    if not idx:
        return survivors
    ordered = sorted(idx, key=lambda i: pvals[i])
    m = len(ordered)
    kmax = 0
    for rank, i in enumerate(ordered, start=1):
        if pvals[i] <= (rank / m) * q:
            kmax = rank
    for rank, i in enumerate(ordered, start=1):
        if rank <= kmax:
            survivors[i] = True
    return survivors


def three_valued_verdict(mean: Optional[float], ci: Optional[List[float]],
                         t_onesided: Optional[float]) -> str:
    """positive | zero (measured) | indeterminate at available power.

    Never 'paying rent' on a point estimate alone (SHADOW_PREREG §2 band)."""
    if mean is None or ci is None:
        return "indeterminate at available power"
    if t_onesided is not None and t_onesided >= 2.0:
        return "positive"
    if ci[0] > -MATERIALITY_BP and ci[1] < MATERIALITY_BP:
        return "zero (measured at materiality scale)"
    return "indeterminate at available power"


def _stat_block(d: List[float]) -> Dict[str, Any]:
    """mean / HAC(5) 95% CI / one-sided t / two-sided p for a daily-diff series."""
    import numpy as np
    if len(d) < 3:
        return {"mean": None, "ci": None, "t": None, "p": None, "n": len(d)}
    m = float(np.mean(d))
    se = hac_se(d)
    t = (m / se) if se and se > 0 else None
    import math
    p = (math.erfc(abs(t) / math.sqrt(2.0)) if t is not None else None)  # two-sided
    return {"mean": round(m, 4),
            "ci": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
            "t": (round(t, 3) if t is not None else None),
            "p": (round(p, 5) if p is not None else None),
            "n": len(d)}


def _m4_subwindow(dates: List[str], series: Dict[str, List[float]]) -> Dict[str, Any]:
    """C9: the M4-A event-damp read (E-F, the old B-A) computed ONLY on the
    largest contiguous sub-window where the base tilt's running estimate
    (F-I) is positive -- the discriminating condition that excludes the
    'damping a losing tilt scores positive by construction' confound
    (SHADOW_PREREG §3, carried verbatim into LIVE_PREREG)."""
    import numpy as np
    base = daily_diff_bp(dates, series.get("F", []), series.get("I", []))
    m4 = daily_diff_bp(dates, series.get("E", []), series.get("F", []))
    n = min(len(base), len(m4))
    if n < 1:
        return {"valid": False, "reason": "no settled days", **_stat_block([])}
    cum = np.cumsum(base[:n])
    best_a = best_b = -1
    a = None
    for i in range(n):
        if cum[i] > 0:
            if a is None:
                a = i
            if best_a < 0 or (i - a) > (best_b - best_a):
                best_a, best_b = a, i
        else:
            a = None
    if best_a < 0:
        return {"valid": False,
                "reason": "no sub-window with base tilt (F-I) > 0",
                **_stat_block([])}
    sub = m4[best_a:best_b + 1]
    blk = _stat_block(sub)
    return {"valid": True,
            "window": [dates[best_a + 1], dates[best_b + 1]],
            "base_tilt_positive": True, **blk}


def compute_ladder_stats(ic_rows: List[dict], dates: List[str],
                         series: Dict[str, List[float]],
                         factor_prices: Optional[Dict[str, List[float]]] = None,
                         selected_universe_active: bool = False) -> dict:
    """The PKT-TB-011 attribution spine: the 5-rung leave-one-organ-out ladder
    with GROSS + multi-factor-STRIPPED columns (C6), log-space additivity +
    printed bp/day residual (C8), per-rung divergence ledger + caveats (C7),
    and BH-FDR(10%) three-valued verdicts incl. the M4 sub-window (C9).

    Superset of ``compute_stats`` -- carries the legacy flat fields too.
    """
    import numpy as np
    stats = compute_stats(ic_rows, dates, series)   # legacy fields + IC leg

    # factor daily returns (aligned to dates[1:]); missing factor -> dropped.
    factor_daily: Dict[str, List[float]] = {}
    if factor_prices:
        for fname, px in factor_prices.items():
            if px and len(px) == len(dates):
                factor_daily[fname] = _simple_returns([p * 1e4 for p in px])
    mkt_daily = factor_daily.get("mkt", [])

    organ_ledger: List[dict] = []
    pvals: List[Optional[float]] = []
    sum_bp = 0.0
    sum_log_bp = 0.0
    per_rung_divergence: Dict[str, float] = {}
    for rung in RUNGS:
        P, Q, comp = rung["book"], rung["parent"], rung["component"]
        sp, sq = series.get(P, []), series.get(Q, [])
        gross = daily_diff_bp(dates, sp, sq)
        gblk = _stat_block(gross)
        # multi-factor strip (C6)
        strip = multifactor_strip(gross, factor_daily) if len(gross) >= 3 else \
            {"stripped_daily": gross, "betas": {}, "exposure_bp_day": None,
             "stripped_bp_day": gblk["mean"], "gross_diff_bp_day": gblk["mean"],
             "factors_used": []}
        sblk = _stat_block(strip["stripped_daily"])
        # log-space additive marginal (C8): exact-telescoping cumulative
        lP, lQ = _log_returns(sp), _log_returns(sq)
        cum_log_bp = (float(np.sum(lP) - np.sum(lQ)) * 1e4
                      if lP and lQ and len(lP) == len(lQ) else None)
        if gblk["mean"] is not None:
            sum_bp += gblk["mean"]
        if cum_log_bp is not None:
            sum_log_bp += cum_log_bp
        # divergence ledger (C7): cumulative |path| of the rung's daily diff
        per_rung_divergence[comp] = (round(float(np.sum(np.abs(gross))), 4)
                                     if gross else 0.0)
        strippable = comp not in EXPOSURE_RUNGS
        caveat = []
        if not strippable:
            caveat.append("exposure-strip incomplete -- this rung is an "
                          "exposure move by construction; the GROSS column is "
                          "the meaningful one (residual factor risk).")
        if comp == "universe" and not selected_universe_active:
            caveat.append("universe restriction not yet wired to the live "
                          "engine's selected_set (PKT-TB-012); U==E until then "
                          "-- this rung reads ~0 and is shown for completeness.")
        if strip["factors_used"] and len(strip["factors_used"]) < len(STRIP_FACTORS):
            missing = [n for n, _ in STRIP_FACTORS if n not in strip["factors_used"]]
            caveat.append(f"strip missing factor(s) {missing} this window "
                          "-- residual factor risk.")
        # verdict on the STRIPPED column (selection skill), one-sided t
        t1 = (sblk["t"] if sblk["t"] is not None else None)
        verdict = three_valued_verdict(sblk["mean"], sblk["ci"], t1)
        pvals.append(sblk["p"])
        organ_ledger.append({
            "component": comp, "book_pair": f"{P}-{Q}",
            "gross_bp_day": gblk["mean"], "gross_ci": gblk["ci"],
            "stripped_bp_day": sblk["mean"], "stripped_ci": sblk["ci"],
            "t": sblk["t"], "n_days": sblk["n"],
            "exposure_bp_day": strip.get("exposure_bp_day"),
            "gross_diff_bp_day": strip.get("gross_diff_bp_day"),
            "betas": strip.get("betas", {}),
            "factors_used": strip.get("factors_used", []),
            "cum_log_rent_bp": (round(cum_log_bp, 4) if cum_log_bp is not None else None),
            "verdict": verdict, "strippable": strippable,
            "caveat": " ".join(caveat) if caveat else None,
        })

    # M4 sub-window read -- an FDR family member (C9)
    m4 = _m4_subwindow(dates, series)
    m4_verdict = three_valued_verdict(m4.get("mean"), m4.get("ci"), m4.get("t"))
    pvals.append(m4.get("p"))

    # BH-FDR across the family = 4 stripped rungs + the M4 sub-window
    survivors = bh_fdr(pvals, FDR_Q)
    n_members = sum(1 for p in pvals if p is not None)
    for i, row in enumerate(organ_ledger):
        row["fdr_survivor"] = bool(survivors[i])
        row["fdr_p"] = pvals[i]
    m4_survivor = survivors[-1] if survivors else False
    # "1 of N" badge: a lone positive line not yet an FDR survivor
    positives = [r for r in organ_ledger if r["verdict"] == "positive"]
    if len(positives) == 1 and not positives[0].get("fdr_survivor"):
        positives[0]["badge"] = ("1 of N -- not narratable as a discovery "
                                 "alone (awaiting BH-FDR survival)")

    # additivity (C8): exact in log space; bp/day residual printed
    total_bp = stats.get("utility_diff_bp_day")     # U-I
    lU, lI = _log_returns(series.get("U", [])), _log_returns(series.get("I", []))
    log_total_bp = (float(np.sum(lU) - np.sum(lI)) * 1e4
                    if lU and lI and len(lU) == len(lI) else None)

    stats["organ_ledger"] = organ_ledger
    stats["forecast_leg"] = {
        "mean_ic": stats.get("mean_ic"), "ic_t": stats.get("ic_t"),
        "n_weeks": stats.get("n_weeks_ic", 0),
        "certified": False,
        "note": ("forecast skill receipt (realized weekly rank-IC); conversion "
                 "is the F-R rung, reported separately and pre-registered, never "
                 "IC x notional."),
    }
    stats["m4_subwindow"] = {**m4, "verdict": m4_verdict,
                             "fdr_survivor": bool(m4_survivor),
                             "in_fdr_family": True}
    stats["additivity"] = {
        "bp_day_total_U_minus_I": total_bp,
        "sum_bp_day_rungs": round(sum_bp, 4),
        "bp_day_residual": (round((total_bp - sum_bp), 4)
                            if total_bp is not None else None),
        "bp_day_note": ("bp/day rung-sums are APPROXIMATE: daily-return "
                        "differences do not telescope across pairs with "
                        "different denominators; the residual is the one-signed "
                        "path-divergence term. Use the log-space columns for "
                        "exact additivity."),
        "log_total_U_minus_I_bp": (round(log_total_bp, 4)
                                   if log_total_bp is not None else None),
        "sum_log_rungs_bp": round(sum_log_bp, 4),
        "log_residual_bp": (round((log_total_bp - sum_log_bp), 4)
                            if log_total_bp is not None else None),
        "log_note": "cumulative log-return rungs telescope exactly: residual ~ 0.",
    }
    stats["divergence"] = {
        "order": [r["component"] for r in RUNGS],
        "order_dependence_note": ("the ladder is a NESTED leave-one-organ-out "
                                  "decomposition in the fixed order "
                                  "regime->forecast->event->universe; a "
                                  "different organ order yields different "
                                  "single-organ marginals. The counterfactual "
                                  "books (R,F,E) path-diverge from U exactly as "
                                  "the TB-007 tilt diverged from the incumbent "
                                  "(C7); each rung carries a stripped-column "
                                  "caveat above."),
        "per_rung_path_bp": per_rung_divergence,
    }
    stats["fdr"] = {"q": FDR_Q, "family": [r["component"] for r in RUNGS] + ["m4_subwindow"],
                    "n_members": n_members, "n_survivors": sum(1 for s in survivors if s)}
    stats["materiality_bp"] = MATERIALITY_BP
    return stats
