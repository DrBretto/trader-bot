"""The live New-Brain cutover runtime (PKT-TB-012).

This is the thing that makes the **two-stage engine (PKT-TB-008) — not
tilt_adapter** — write the live ``trade_intents.json``. The Attack-5 ruling
(DESIGN_DOSSIER §3) is binding: nothing is branded "New Brain" unless this
engine writes the intents AND its invariant self-check is green.

``run_cutover`` orchestrates one decision date with the full abort-never-degrade
discipline (DESIGN_DOSSIER reports/03 §1, §3):

  1. read ``config/brain.active.json`` (the live/shadow switch; raises if missing)
  2. FREEZE_ORB1 cold-start assertion (engine_sha + model_sha + θ content hashes)
  3. obtain full-universe M1 ``mu`` from the frozen forward inference
  4. build the engine contracts (forecast_adapter)
  5. **invariant self-check gate** — held_symbols_invariant() across the θ_size
     grid must be True (this IS the PKT-TB-009 spine, run nightly); a False
     result ABORTS
  6. run_engine -> canonical trade_intents (engine == "native_two_stage")

Any failure in 1–6 returns ``ok=False`` with a reason; the caller leaves the
incumbent intents in place (fail-safe to incumbent) and raises an SNS CRITICAL.
The runtime itself NEVER writes S3 and NEVER degrades silently.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from .engine import (
    SelectionParams,
    SizingParams,
    LotPolicy,
    held_symbols_invariant,
    run_engine,
)
from .forecast_adapter import build_forecast_bundle, build_portfolio_state
from .freeze import BrainFreezeError, assert_cold_start, load_freeze

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]

# Mirrors the shadow's forward boundary: nothing on/before this date is ever a
# live New-Brain decision date (the frozen champion line owns <= 2026-06-11).
DEFAULT_FORWARD_BOUNDARY = "2026-06-11"


class BrainConfigError(RuntimeError):
    """config/brain.active.json missing or malformed (raises, never defaults)."""


@dataclass(frozen=True)
class CutoverResult:
    ok: bool
    mode: str
    reason: str
    date: str
    trade_intents: Optional[dict] = None
    incumbent_intents: Optional[dict] = None
    selected_universe: Optional[List[str]] = None
    engine: str = ""
    invariant_green: bool = False
    freeze_detail: str = ""
    parity_record: Mapping[str, object] = field(default_factory=dict)


def _config_path() -> Path:
    env = os.environ.get("BRAIN_ACTIVE_PATH")
    if env:
        return Path(env)
    task_root = os.environ.get("LAMBDA_TASK_ROOT")
    if task_root and (Path(task_root) / "config" / "brain.active.json").exists():
        return Path(task_root) / "config" / "brain.active.json"
    return _REPO_ROOT / "config" / "brain.active.json"


def load_brain_config() -> dict:
    """Read the live/shadow switch. Raises on missing (like
    config/decision_params.active.json, handler.py:90-112)."""
    p = _config_path()
    if not p.exists():
        raise BrainConfigError(
            f"Missing required brain switch at {p} — the cutover refuses to "
            "guess live/shadow mode")
    cfg = json.loads(p.read_text())
    if "mode" not in cfg:
        raise BrainConfigError(f"brain.active.json at {p} has no 'mode'")
    return cfg


def theta_from_freeze(freeze: Optional[dict] = None) -> tuple:
    """Reconstruct (θ_sel, θ_size) from FREEZE_ORB1 and ASSERT their content
    hashes match the frozen contract (a θ that doesn't hash-match is a no-mid-
    stream violation — abort)."""
    freeze = freeze or load_freeze()
    ts = freeze["theta_sel"]["values"]
    lp = ts["lot_policy"]
    theta_sel = SelectionParams(
        N=int(ts["N"]),
        h_min=float(ts["h_min"]),
        core_fraction=float(ts["core_fraction"]),
        regime_admissibility={
            k: tuple(v) for k, v in (ts.get("regime_admissibility") or {}).items()
        },
        tiebreak=tuple(ts["tiebreak"]),
        lot_policy=LotPolicy(min_order=float(lp["min_order"]),
                             reference_nav=float(lp["reference_nav"])),
    )
    sz = freeze["theta_size"]["values"]
    theta_size = SizingParams(
        gross_target=float(sz["gross_target"]),
        max_position_weight=float(sz["max_position_weight"]),
        max_cluster_weight=float(sz["max_cluster_weight"]),
        cash_reserve_pct=float(sz["cash_reserve_pct"]),
        regime_exposure_multiplier={
            k: float(v) for k, v in (sz.get("regime_exposure_multiplier") or {}).items()
        },
        parity_gain=float(sz["parity_gain"]),
    )
    exp_sel = freeze["theta_sel"].get("content_hash", "")
    exp_size = freeze["theta_size"].get("content_hash", "")
    got_sel = theta_sel.content_hash()
    got_size = theta_size.content_hash()
    problems = []
    if exp_sel and got_sel != exp_sel:
        problems.append(f"θ_sel content_hash {got_sel[:12]}… != frozen {exp_sel[:12]}…")
    if exp_size and got_size != exp_size:
        problems.append(f"θ_size content_hash {got_size[:12]}… != frozen {exp_size[:12]}…")
    if problems:
        raise BrainFreezeError("FREEZE_ORB1 θ assertion FAILED: " + "; ".join(problems))
    return theta_sel, theta_size


def forward_boundary(config: Optional[dict] = None) -> str:
    if config and config.get("forward_boundary"):
        return str(config["forward_boundary"])
    return DEFAULT_FORWARD_BOUNDARY


def is_live(config: dict, date: str) -> tuple:
    """(enabled, reason). Live iff mode=='live' AND date strictly after the
    forward boundary. The champion line owns <= the boundary; the New Brain owns
    forward of it."""
    mode = str(config.get("mode", "shadow"))
    if mode != "live":
        return False, f"brain.active mode='{mode}' (not live) — incumbent writes intents"
    fb = forward_boundary(config)
    if date <= fb:
        return False, f"date {date} <= forward_boundary {fb} — frozen champion owns this date"
    return True, "live"


def run_cutover(
    date: str,
    features_df,
    regime_label: str,
    universe_df,
    portfolio_state: dict,
    forecaster: Callable[[List[str]], Dict[str, dict]],
    health_map: Optional[Mapping[str, float]] = None,
    config: Optional[dict] = None,
    now_iso: Optional[str] = None,
    incumbent_intents: Optional[dict] = None,
    parity_ledger=None,
) -> CutoverResult:
    """Run the New-Brain cutover for one date. Never raises for an operational
    failure — returns ok=False with a reason so the caller falls back to the
    incumbent intents (fail-safe). Only programmer errors propagate.

    ``forecaster(pending)`` returns ``{date: {"mu": {sym: val}, ...}}`` — the
    frozen brain's full-universe M1 forecast (forecast-ledger record shape).
    """
    config = config if config is not None else load_brain_config()
    mode = str(config.get("mode", "shadow"))
    enabled, why = is_live(config, date)
    if not enabled:
        return CutoverResult(ok=False, mode=mode, reason=why, date=date,
                             incumbent_intents=incumbent_intents)

    # (2) cold-start freeze assertion + θ hash assertion (abort-never-degrade)
    try:
        freeze = load_freeze()
        freeze_assert = assert_cold_start()
        theta_sel, theta_size = theta_from_freeze(freeze)
    except BrainFreezeError as e:
        return CutoverResult(ok=False, mode=mode, date=date,
                             reason=f"FREEZE assertion failed (ABORT): {e}",
                             incumbent_intents=incumbent_intents)

    # (3) full-universe M1 forecast from the frozen brain
    try:
        records = forecaster([date])
        rec = records.get(date) if records else None
        mu_map = dict(rec.get("mu", {})) if rec else {}
    except Exception as e:  # noqa: BLE001 — operational, fall back to incumbent
        return CutoverResult(ok=False, mode=mode, date=date,
                             reason=f"forward inference failed (ABORT): "
                                    f"{type(e).__name__}: {e}",
                             incumbent_intents=incumbent_intents)
    if not mu_map:
        return CutoverResult(ok=False, mode=mode, date=date,
                             reason="no M1 forecast for date (ABORT)",
                             incumbent_intents=incumbent_intents)

    # (4) build the engine contracts
    # The chassis regime_compatibility table (config bundle, handler.py:97)
    # restores the regime picker into Stage-1 ranking via per-symbol
    # regime_score_mult (PROPOSAL_ORTHOGONALITY §0 — the chassis socket).
    f = build_forecast_bundle(date, mu_map, features_df, regime_label,
                              universe_df, health_map=health_map,
                              regime_compat=config.get("regime_compatibility"))
    portfolio = build_portfolio_state(portfolio_state, features_df, universe_df)

    # (5) invariant self-check GATE (the PKT-TB-009 spine, run nightly).
    #     A False result ABORTS the night (abort-never-degrade); no live write.
    try:
        invariant_green = held_symbols_invariant(f, theta_sel, portfolio)
    except Exception as e:  # noqa: BLE001
        return CutoverResult(ok=False, mode=mode, date=date,
                             reason=f"invariant self-check raised (ABORT): "
                                    f"{type(e).__name__}: {e}",
                             freeze_detail=freeze_assert.detail,
                             incumbent_intents=incumbent_intents)
    if not invariant_green:
        return CutoverResult(ok=False, mode=mode, date=date, invariant_green=False,
                             reason="invariant self-check RED — held_symbols not "
                                    "invariant under θ_size (ABORT; no New Brain "
                                    "write, no brand)",
                             freeze_detail=freeze_assert.detail,
                             incumbent_intents=incumbent_intents)

    # (6) run the engine — the two-stage native pipeline writes the intents.
    try:
        out = run_engine(f, theta_sel, theta_size, portfolio,
                         parity_ledger=parity_ledger, now_iso=now_iso,
                         incumbent_intents=incumbent_intents)
    except Exception as e:  # noqa: BLE001
        return CutoverResult(ok=False, mode=mode, date=date,
                             reason=f"engine run failed (ABORT): "
                                    f"{type(e).__name__}: {e}",
                             freeze_detail=freeze_assert.detail,
                             incumbent_intents=incumbent_intents)

    selected = sorted(out.allocation.held_symbols)
    return CutoverResult(
        ok=True, mode=mode, date=date,
        reason="live: two-stage engine wrote intents (invariant green)",
        trade_intents=out.trade_intents,
        incumbent_intents=incumbent_intents,
        selected_universe=selected,
        engine=out.trade_intents.get("expert_metrics", {}).get("engine", ""),
        invariant_green=True,
        freeze_detail=freeze_assert.detail,
        parity_record=dict(out.allocation.parity_record),
    )


# ---------------------------------------------------------- production forecaster
S3_BUCKET = "investment-system-data"


def _ohlcv_watermark(SL):
    """Return ``(max_date_str_or_None, row_count)`` for the in-store SPY panel —
    the substrate-currency watermark the freshness gate keys on."""
    import pandas as pd
    spy = SL.CACHE_OHLCV / "SPY.parquet"
    if not spy.exists():
        return None, 0
    df = pd.read_parquet(spy, columns=["date"])
    d = pd.to_datetime(df["date"])
    return (d.max().strftime("%Y-%m-%d") if len(d) else None), int(len(df))


def _extend_ohlcv_from_s3(SL, FI) -> dict:
    """Bring the brain's OHLCV store current: fetch the chassis's published
    daily/<D>/prices.parquet for every date newer than the seed store and splice
    them in (extend_ohlcv). Uses the Lambda execution role's default creds (NOT
    the laptop 'personal' profile that SL.S3Source assumes).

    Returns a diagnostic dict (store watermark before/after, gap, bars added,
    per-date errors). Failures are LOGGED with the exact exception — NEVER
    silently swallowed (ISSUE-01/F-D1). The aggregate substrate-currency gate in
    ``production_forecaster`` is what ABORTS a stale night; per-date faults are
    collected here so that gate can name them."""
    import boto3
    import traceback
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    last_ohlcv, n_before = _ohlcv_watermark(SL)
    result = {"last_ohlcv_before": last_ohlcv, "rows_before": n_before,
              "dates_listed": 0, "gap": [], "downloaded": 0, "extended": 0,
              "bars_added": 0, "errors": []}

    try:
        pg = s3.get_paginator("list_objects_v2")
        dates = set()
        for page in pg.paginate(Bucket=S3_BUCKET, Prefix="daily/", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []) or []:
                d = cp["Prefix"].split("/")[-2]
                if len(d) == 10 and d[4] == "-":
                    dates.add(d)
    except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
        msg = f"S3 list daily/ FAILED: {type(e).__name__}: {e}"
        print(f"  [FRESHNESS] {msg}")
        result["errors"].append(msg)
        dates = set()
    result["dates_listed"] = len(dates)

    gap = sorted(d for d in dates if last_ohlcv is None or d > last_ohlcv)
    result["gap"] = gap
    print(f"  [FRESHNESS] store SPY max={last_ohlcv} rows={n_before}; listed "
          f"{len(dates)} daily dates; gap={len(gap)} "
          f"[{gap[0] if gap else '-'}..{gap[-1] if gap else '-'}]")

    for d in gap:
        dst = SL.CACHE_DAILY / d / "prices.parquet"
        if not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(S3_BUCKET, f"daily/{d}/prices.parquet", str(dst))
                result["downloaded"] += 1
            except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
                msg = f"download daily/{d}/prices.parquet FAILED: {type(e).__name__}: {e}"
                print(f"  [FRESHNESS] {msg}")
                result["errors"].append(msg)
                continue
        try:
            FI.extend_ohlcv(d)
            result["extended"] += 1
        except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
            print(f"  [FRESHNESS] extend_ohlcv({d}) FAILED: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()}")
            result["errors"].append(f"extend_ohlcv({d}): {type(e).__name__}: {e}")
            continue

    last_after, n_after = _ohlcv_watermark(SL)
    result["last_ohlcv_after"] = last_after
    result["rows_after"] = n_after
    result["bars_added"] = n_after - n_before
    print(f"  [FRESHNESS] after extend: SPY max={last_after} rows={n_after} "
          f"(+{result['bars_added']} bars; {result['extended']}/{len(gap)} dates "
          f"extended; {result['downloaded']} downloaded; {len(result['errors'])} errors)")
    return result


# ----------------------------------------------------- fail-loud staleness gate
# The single worst failure this project has shipped (ISSUE-01/02): the in-Lambda
# OHLCV store silently stopped advancing, so `mu` froze and the SAME concentrated
# line published every night for two weeks with zero alarm — because every monitor
# keyed on PUBLISH recency (fresh leaf written nightly) not SUBSTRATE currency (the
# OHLCV panel behind mu). This gate keys on substrate currency: if the store's max
# settled bar lags the run date, or new daily dates were available but zero bars
# spliced (the frozen-substrate signature), the night ABORTS + ALARMS instead of
# shipping a frozen forecast.
_STALE_TOLERANCE_TD = 3   # trading days the store may lag the run date before ABORT


class StaleSubstrateError(RuntimeError):
    """Raised by the freshness gate when the OHLCV substrate is not current.
    run_cutover's forecaster try/except catches it -> ok=False -> the incumbent
    intents are retained (fail-safe) and an SNS CRITICAL is raised."""


def _weekday_trading_days_between(d0: str, d1: str) -> int:
    """Weekday count strictly between two YYYY-MM-DD dates (a cheap NYSE proxy;
    holidays make this CONSERVATIVE — it never fires falsely loud). 0 if d1<=d0."""
    import datetime as _dt
    try:
        a = _dt.date.fromisoformat(str(d0)[:10])
        b = _dt.date.fromisoformat(str(d1)[:10])
    except Exception:  # noqa: BLE001
        return 0
    if b <= a:
        return 0
    n, cur = 0, a
    while cur < b:
        cur += _dt.timedelta(days=1)
        if cur.weekday() < 5:
            n += 1
    return n


def _latest_settled_trading_day(today: Optional[str] = None) -> str:
    """Latest weekday on/before `today` (NY). The night forecasts off the last
    settled session; the OHLCV store's max bar should track this."""
    import datetime as _dt
    d = _dt.date.fromisoformat(today[:10]) if today else _dt.datetime.now(
        _dt.timezone.utc).astimezone(_dt.timezone(-_dt.timedelta(hours=5))).date()
    while d.weekday() >= 5:
        d -= _dt.timedelta(days=1)
    return d.isoformat()


def _freshness_gate_verdict(run_date: str, ohlcv_max_date: Optional[str],
                            fresh: dict) -> dict:
    """Substrate-currency verdict. Two independent invariants, either trips ABORT:
      1. max settled OHLCV bar within K trading days of the run date, AND
      2. bar-count advanced when the S3 gap was non-empty (mu is a pure function of
         the price/vol OHLCV panel -> bars-advanced is the causal proxy for
         mu-freshness; a no-op splice IS the frozen-`mu` signature)."""
    reasons: List[str] = []
    if not ohlcv_max_date:
        reasons.append("OHLCV store empty — no SPY watermark to trust")
    else:
        lag = _weekday_trading_days_between(ohlcv_max_date, run_date)
        if lag > _STALE_TOLERANCE_TD:
            reasons.append(
                f"OHLCV substrate STALE: max settled bar {ohlcv_max_date} is {lag} "
                f"trading days behind run date {run_date} (tolerance {_STALE_TOLERANCE_TD})")
    gap = fresh.get("gap") or []
    if gap and fresh.get("bars_added", 0) <= 0:
        reasons.append(
            f"OHLCV extend NO-OP: {len(gap)} newer daily date(s) available "
            f"[{gap[0]}..{gap[-1]}] but 0 bars spliced — the frozen-substrate signature")
    return {"stale": bool(reasons), "reasons": reasons,
            "run_date": run_date, "ohlcv_max_date": ohlcv_max_date,
            "bars_added": fresh.get("bars_added", 0), "gap_len": len(gap)}


def _alert_stale_substrate(verdict: dict, fresh: dict) -> None:
    reason = "; ".join(verdict.get("reasons") or ["stale"])
    body = (
        "FAIL-LOUD STALENESS GATE FIRED — the night was ABORTED before shipping a "
        "forecast over a stale substrate (the frozen-`mu` failure can no longer ship "
        "silently).\n\n"
        f"run_date                 = {verdict.get('run_date')}\n"
        f"OHLCV store max bar      = {verdict.get('ohlcv_max_date')}\n"
        f"bars spliced this run    = {verdict.get('bars_added')}\n"
        f"newer daily dates (gap)  = {verdict.get('gap_len')}\n"
        f"reason                   = {reason}\n\n"
        "extend errors:\n  " + ("\n  ".join(fresh.get("errors") or ["(none)"])) + "\n\n"
        "The incumbent intents are retained (fail-safe). Check CloudWatch "
        "/aws/lambda/investment-system-daily-pipeline for the [FRESHNESS] lines.")
    try:
        from src.utils.sns_alerts import send_alert
        send_alert(
            subject="[TraderBot] CRITICAL: forecast substrate STALE — night ABORTED (fail-loud gate)",
            body=body)
    except Exception as e:  # noqa: BLE001 — alerting must never crash the check
        print(f"  [FRESHNESS] stale-substrate alert failed (non-fatal): {e}")


def _assert_substrate_current(run_date: str, SL, fresh: dict) -> None:
    """The enforced gate. Raises StaleSubstrateError (+ SNS alert) when the OHLCV
    substrate is not current, so a frozen forecast cannot ship."""
    ohlcv_max, _rows = _ohlcv_watermark(SL)
    verdict = _freshness_gate_verdict(run_date, ohlcv_max, fresh)
    print(f"  [FRESHNESS] gate verdict: {verdict}")
    if verdict["stale"]:
        _alert_stale_substrate(verdict, fresh)
        raise StaleSubstrateError("; ".join(verdict["reasons"]))


def diagnose_forecast_freshness(pending: Optional[List[str]] = None,
                                force_stale: bool = False) -> dict:
    """Governed, NON-DESTRUCTIVE in-Lambda probe of forecast substrate currency.

    Runs the same forward path as production_forecaster (extend OHLCV -> panel ->
    inference) but writes NOTHING to S3 (all work in /tmp), then returns a
    diagnostic dict: the OHLCV watermark before/after the S3 extend, the exact
    per-date errors (the root of the swallow), and the resulting mu hash / top-10.
    ``force_stale=True`` SKIPS the extend to prove the gate fires over a
    deliberately frozen store."""
    import sys
    import hashlib
    candidates = []
    env_subset = os.environ.get("BRAIN_RUNTIME_SUBSET")
    if env_subset:
        candidates.append(Path(env_subset))
    candidates.append(_REPO_ROOT / "runs" / "pkt_tb_007_orthogonal_brain" / "shadow")
    for p in candidates:
        sp = str(p)
        if Path(sp).exists() and sp not in sys.path:
            sys.path.append(sp)
    import shadow_lib as SL  # type: ignore
    import forward_inference as FI  # type: ignore

    SL.STATE.mkdir(parents=True, exist_ok=True)
    FI.ensure_seed_caches()

    if force_stale:
        before, rows = _ohlcv_watermark(SL)
        fresh = {"forced_stale": True, "last_ohlcv_before": before, "rows_before": rows,
                 "gap": ["__forced__"], "bars_added": 0, "errors": []}
        print("  [FRESHNESS] force_stale=True — SKIPPING extend to exercise the gate")
    else:
        fresh = _extend_ohlcv_from_s3(SL, FI)

    if pending is None:
        pending = [_latest_settled_trading_day()]
    run_date = pending[-1]

    try:
        FI.gdelt_forward()
    except Exception as e:  # noqa: BLE001
        print(f"  [FRESHNESS] gdelt_forward soft-fail: {type(e).__name__}: {e}")
    try:
        FI.cboe_forward()
    except Exception as e:  # noqa: BLE001
        print(f"  [FRESHNESS] cboe_forward soft-fail: {type(e).__name__}: {e}")

    FI.build_panel(pending)
    records = FI.run_inference(pending)
    rec = records.get(run_date) if records else None
    mu = dict(rec.get("mu", {})) if rec else {}
    top10 = [s for s, _ in sorted(mu.items(), key=lambda kv: -kv[1])[:10]]
    mu_sha = hashlib.sha256(json.dumps(
        {k: round(float(v), 8) for k, v in sorted(mu.items())},
        sort_keys=True).encode()).hexdigest()[:16]

    ohlcv_max, ohlcv_rows = _ohlcv_watermark(SL)
    verdict = _freshness_gate_verdict(run_date, ohlcv_max, fresh)
    out = {"pending": pending, "settled_trading_day": run_date,
           "ohlcv_max_date": ohlcv_max, "ohlcv_rows": ohlcv_rows,
           "freshness": fresh, "n_mu": len(mu), "mu_top10": top10,
           "mu_sha16": mu_sha, "gate": verdict}
    print(f"  [FRESHNESS] DIAG: settled={run_date} ohlcv_max={ohlcv_max} "
          f"rows={ohlcv_rows} n_mu={len(mu)} mu_sha={mu_sha}\n"
          f"  [FRESHNESS] DIAG: top10={top10}\n"
          f"  [FRESHNESS] DIAG: gate={verdict}")
    return out


def production_forecaster(pending: List[str]) -> Dict[str, dict]:
    """Run the frozen brain's forward inference INSIDE the night Lambda (the baked
    runtime subset). Returns ``{date: {"mu": {sym: val}, ...}}``.

    The runtime subset is baked mirroring the repo layout (Dockerfile.lambda) so
    every module path constant resolves; the writable state tree is BRAIN_STATE_DIR
    (/tmp in Lambda). Imported lazily so unit tests inject a forecaster without the
    heavy ML deps. forward inference's parity self-check ABORTS on any divergence
    from the frozen reference — the correctness gate.
    """
    import sys
    candidates = []
    env_subset = os.environ.get("BRAIN_RUNTIME_SUBSET")
    if env_subset:
        candidates.append(Path(env_subset))
    candidates.append(_REPO_ROOT / "runs" / "pkt_tb_007_orthogonal_brain" / "shadow")
    for p in candidates:
        sp = str(p)
        if Path(sp).exists() and sp not in sys.path:
            sys.path.append(sp)

    import shadow_lib as SL  # type: ignore
    import forward_inference as FI  # type: ignore

    # writable state tree (/tmp in Lambda) + frozen read-only seeds copied in
    SL.STATE.mkdir(parents=True, exist_ok=True)
    FI.ensure_seed_caches()
    _fresh = _extend_ohlcv_from_s3(SL, FI)
    # FAIL-LOUD substrate-currency staleness gate (ISSUE-01/02/13): a stale OHLCV
    # store ABORTS the night here instead of shipping a frozen mu. run_cutover's
    # forecaster try/except catches this -> ok=False -> incumbent retained + alarm.
    _assert_substrate_current(pending[-1] if pending else _latest_settled_trading_day(),
                              SL, _fresh)
    # external feeds are fail-soft (GDELT masking / CBOE staleness-null are
    # registered honest fallbacks); never let them abort the night.
    try:
        FI.gdelt_forward()
    except Exception:
        pass
    try:
        FI.cboe_forward()
    except Exception:
        pass
    FI.build_panel(pending)
    return FI.run_inference(pending)
