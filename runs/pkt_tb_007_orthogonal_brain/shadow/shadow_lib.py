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
DAILY_FILES_OPTIONAL = ["morning_prices.parquet"]

GENOME_A = SHADOW / "genomes" / "shadow_A.json"
GENOME_B = SHADOW / "genomes" / "shadow_B.json"
BOOKS = ("I", "A", "B")


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
    """Books A and B tilt strategies over the frozen shadow genomes."""
    from tilt_adapter import make_tilt_strategy
    strats = {}
    for name, gpath in (("A", GENOME_A), ("B", GENOME_B)):
        ld = (log_dir / f"expression_{name}") if log_dir else None
        if ld:
            ld.mkdir(parents=True, exist_ok=True)
        strats[name] = make_tilt_strategy(
            gpath, organ_dir, log_dir=ld, cache=cache, risk=risk)
    return strats


class LocalDailyCache:
    """tilt_adapter.guarded_marks fallback interface over the local daily
    cache (get_parquet only)."""

    def __init__(self, daily_dir: Path):
        self.daily_dir = Path(daily_dir)

    def get_parquet(self, key: str):
        import pandas as pd
        p = self.daily_dir / Path(key).relative_to("daily")
        return pd.read_parquet(p)


def process_book_date(books: Dict[str, Any], date: str,
                      intents: List[dict], ohlc: Dict[str, Dict[str, float]],
                      features_df, decision_params: dict,
                      strategies: Dict[str, Any], universe_df,
                      regime: Optional[str]) -> Dict[str, Any]:
    """One settled decision date across the three books. Mutates books.
    Returns the equity-ledger row (raw NAVs; costs overlaid separately)."""
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
        my_intents = [dict(it) for it in intents]
        if bname in strategies:
            ctx = StrategyContext(
                inputs_date=date, portfolio=book,
                variant_config={"decision_params": dict(decision_params)},
                expert_signals={}, expert_metrics={}, decisions={},
                panic_streak=0, last_regime=None,
                features_df=features_df, inference={}, llm_risks={})
            my_intents = strategies[bname].post_decision(ctx, my_intents)
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
    for key, (x, y) in {"utility_diff_bp_day": ("A", "I"),
                        "m4a_diff_bp_day": ("B", "A")}.items():
        d = daily_diff_bp(dates, series.get(x, []), series.get(y, []))
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
