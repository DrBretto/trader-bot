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


def _regime_compat_path() -> Path:
    """Locate config/regime_compatibility.json alongside brain.active.json
    (honors LAMBDA_TASK_ROOT in the baked image, then the repo root)."""
    env = os.environ.get("REGIME_COMPAT_PATH")
    if env:
        return Path(env)
    task_root = os.environ.get("LAMBDA_TASK_ROOT")
    if task_root and (Path(task_root) / "config" / "regime_compatibility.json").exists():
        return Path(task_root) / "config" / "regime_compatibility.json"
    return _REPO_ROOT / "config" / "regime_compatibility.json"


def _load_regime_compatibility() -> dict:
    """Load config/regime_compatibility.json — the chassis regime x sector/asset
    multiplier table. Returns {} when the file is ABSENT (so shadow / tilt_adapter
    modes stay usable), but a PRESENT-but-malformed file RAISES: a broken table
    must never silently degrade to {} (= inert chassis). The native_two_stage
    startup assertion (``assert_regime_chassis_loaded``) is what turns an
    empty/absent table into a loud ABORT on the live engine."""
    p = _regime_compat_path()
    if not p.exists():
        return {}
    try:
        table = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 — malformed table is a loud config error
        raise BrainConfigError(
            f"regime_compatibility.json at {p} is present but unparseable: "
            f"{type(e).__name__}: {e}")
    if not isinstance(table, dict):
        raise BrainConfigError(
            f"regime_compatibility.json at {p} is not a JSON object "
            "(regime -> {sector/asset_class: multiplier})")
    return table


def load_brain_config() -> dict:
    """Read the live/shadow switch AND merge the regime chassis compatibility
    table, so ``config.get('regime_compatibility')`` reaches run_cutover ->
    build_forecast_bundle (the chassis socket; PKT-3). Raises on a missing brain
    switch (like config/decision_params.active.json, handler.py:90-112).

    Before PKT-3, brain.active.json carried no ``regime_compatibility`` key, so
    the key resolved to None and every symbol got ``regime_score_mult=1.0`` — the
    chassis ran INERT (Stage-1 ranked raw mu, identical to the pre-restore
    engine). Merging the standalone table here is what makes the regime tilt
    actually re-rank ``mu``."""
    p = _config_path()
    if not p.exists():
        raise BrainConfigError(
            f"Missing required brain switch at {p} — the cutover refuses to "
            "guess live/shadow mode")
    cfg = json.loads(p.read_text())
    if "mode" not in cfg:
        raise BrainConfigError(f"brain.active.json at {p} has no 'mode'")
    # An explicit non-empty regime_compatibility already in brain.active.json wins
    # (operator override); otherwise wire the standalone table.
    if not cfg.get("regime_compatibility"):
        cfg["regime_compatibility"] = _load_regime_compatibility()
    return cfg


class RegimeChassisInertError(RuntimeError):
    """engine == native_two_stage but the regime_compatibility table is
    empty/missing, so ``regime_score_mult`` would be 1.0 for every symbol (the
    chassis silently degraded to raw-mu ranking). run_cutover catches this ->
    ok=False -> incumbent retained + SNS CRITICAL (fail-loud, guarded)."""


def assert_regime_chassis_loaded(config: Optional[Mapping[str, object]]) -> None:
    """Fail-loud startup assertion (PKT-3): under the live two-stage engine the
    regime chassis MUST be loaded. An empty/missing regime_compatibility table
    collapses every ``regime_score_mult`` to 1.0 — the chassis runs INERT and
    Stage-1 ranks raw mu, indistinguishable from the pre-restore engine. That
    silent-inert failure is exactly what shipped for weeks; this gate makes it
    structurally impossible: the chassis is either loaded or the night ABORTS.
    Raises ``RegimeChassisInertError``. No-op for any engine other than
    native_two_stage (shadow / tilt_adapter legitimately carry no table)."""
    cfg = config or {}
    engine = str(cfg.get("engine") or cfg.get("genome") or "")
    if engine != "native_two_stage":
        return
    table = cfg.get("regime_compatibility")
    if not table or not isinstance(table, Mapping):
        raise RegimeChassisInertError(
            "native_two_stage engine but regime_compatibility table is "
            f"empty/missing (got {type(table).__name__}); the regime chassis "
            "would run INERT (regime_score_mult=1.0 for all symbols = raw-mu "
            "ranking). ABORTING the night — load config/regime_compatibility.json "
            "(PKT-3).")


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

    # (1b) FAIL-LOUD chassis-loaded assertion (PKT-3): the live two-stage engine
    #      must carry a non-empty regime_compatibility table or it runs inert
    #      (regime_score_mult=1.0 = raw-mu ranking). Abort-never-degrade: an inert
    #      chassis ABORTS to the incumbent (guarded) rather than shipping a
    #      dead-socket selection that looks live.
    try:
        assert_regime_chassis_loaded(config)
    except RegimeChassisInertError as e:
        return CutoverResult(ok=False, mode=mode, date=date,
                             reason=f"regime chassis INERT (ABORT): {e}",
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


def _extract_regime_label(rec: Optional[dict]) -> str:
    """Best-effort regime label from a forward-inference record (mirrors the
    live publish path: inference_output['regime']['label'])."""
    if not rec:
        return "neutral"
    reg = rec.get("regime")
    if isinstance(reg, Mapping):
        return str(reg.get("label") or reg.get("final_regime_label") or "neutral")
    if isinstance(reg, str) and reg:
        return reg
    return str(rec.get("regime_label") or "neutral")


def diagnose_regime_chassis(pending: Optional[List[str]] = None) -> dict:
    """Governed, NON-DESTRUCTIVE in-Lambda probe that the regime chassis (PKT-3)
    is LOADED and observably RE-RANKS a FRESH mu. Writes NOTHING to S3 (all work
    in /tmp). Proves, for a first-time reader:

      * ``regime_compat_loaded`` — load_brain_config now merges
        config/regime_compatibility.json into the config that reaches run_cutover.
      * ``regime_score_mult != 1.0`` for a real fraction of the universe, and the
        regime-tilted Stage-1 order (rank by mu * regime_score_mult) DIFFERS from
        the raw-mu order — the chassis socket is live, not inert.
      * the fresh mu it re-ranks (mu_sha16 / top10) so the ordering trap
        (mu changed date-over-date) is checkable against the forecast-diag probe.
      * the fail-loud startup assertion fires when the table is emptied under
        native_two_stage (assert_regime_chassis_loaded).
      * the explicit health=None rule (UNKNOWN_HEALTH_DEFAULT) is reported.
    """
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
    import pandas as pd

    SL.STATE.mkdir(parents=True, exist_ok=True)
    FI.ensure_seed_caches()
    _extend_ohlcv_from_s3(SL, FI)
    if pending is None:
        pending = [_latest_settled_trading_day()]
    run_date = pending[-1]

    # external feeds seed the panel's GDELT/CBOE columns (fail-soft, same as the
    # freshness diag / production forecaster) — build_panel reads them.
    try:
        FI.gdelt_forward()
    except Exception as e:  # noqa: BLE001
        print(f"  [REGIME-DIAG] gdelt_forward soft-fail: {type(e).__name__}: {e}")
    try:
        FI.cboe_forward()
    except Exception as e:  # noqa: BLE001
        print(f"  [REGIME-DIAG] cboe_forward soft-fail: {type(e).__name__}: {e}")

    FI.build_panel(pending)
    records = FI.run_inference(pending)
    rec = records.get(run_date) if records else None
    mu = dict(rec.get("mu", {})) if rec else {}
    inference_regime_label = _extract_regime_label(rec)

    # Resolve the regime label PRODUCTION actually uses (publish_artifacts): the
    # FUSED decisions.expert_metrics.final_regime_label from the live leaf (read
    # ONLY — non-destructive), falling back to the inference record. The raw
    # inference record commonly reads 'neutral' (no table key), so a diag keyed on
    # it would falsely show no re-rank; the fused label ('choppy', 'high_vol_panic',
    # …) is what reaches build_forecast_bundle.
    regime_label = inference_regime_label
    regime_label_source = "inference_record"
    try:
        import boto3
        s3c = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        obj = s3c.get_object(Bucket=S3_BUCKET, Key=f"daily/{run_date}/decisions.json")
        dec = json.loads(obj["Body"].read())
        fused = (dec.get("expert_metrics", {}) or {}).get("final_regime_label") \
            or dec.get("regime")
        if fused:
            regime_label = str(fused)
            regime_label_source = f"decisions.final_regime_label (daily/{run_date})"
    except Exception as e:  # noqa: BLE001 — read-only best-effort
        print(f"  [REGIME-DIAG] decisions leaf read soft-fail: {type(e).__name__}: {e}")

    mu_sha = hashlib.sha256(json.dumps(
        {k: round(float(v), 8) for k, v in sorted(mu.items())},
        sort_keys=True).encode()).hexdigest()[:16]

    # The wired config (regime_compatibility now merged) — the exact object the
    # night's run_cutover receives.
    config = load_brain_config()
    table = config.get("regime_compatibility") or {}
    regime_compat_loaded = bool(table)

    # Load the universe contract the way the engine sees it (baked config/).
    uni_path = _regime_compat_path().parent / "universe.csv"
    universe_df = pd.read_csv(uni_path)

    # Build the bundle BOTH ways — with the chassis (tilted) and without (raw mu)
    # — through the real engine adapter, and rank the eligible field each way.
    f_tilt = build_forecast_bundle(run_date, mu, None, regime_label, universe_df,
                                   health_map={}, regime_compat=table)
    f_raw = build_forecast_bundle(run_date, mu, None, regime_label, universe_df,
                                  health_map={}, regime_compat=None)

    elig = [s for s in f_tilt.mu_M1 if f_tilt.eligible.get(s)]
    mult = f_tilt.regime_score_mult
    n_nonunit = sum(1 for s in elig if abs(mult.get(s, 1.0) - 1.0) > 1e-9)
    raw_order = sorted(elig, key=lambda s: -f_raw.mu_M1[s])
    tilt_order = sorted(elig, key=lambda s: -(f_tilt.mu_M1[s] * mult.get(s, 1.0)))
    raw_top = raw_order[:10]
    tilt_top = tilt_order[:10]
    reranks = bool(raw_order != tilt_order)
    top10_reranks = bool(raw_top != tilt_top)
    # a few concrete rank moves for the reader
    moves = []
    raw_rank = {s: i for i, s in enumerate(raw_order)}
    for i, s in enumerate(tilt_order[:10]):
        moves.append({"symbol": s, "raw_rank": raw_rank.get(s), "tilt_rank": i,
                      "mult": round(mult.get(s, 1.0), 4)})

    # Stress demonstration: apply EACH regime in the table to the SAME fresh mu
    # and report which regimes move the top-10. The socket re-ranks the selection
    # when the regime differentiates; today's live regime may be mild (choppy's
    # multipliers are 0.9–1.05, so its top-10 can be stable while its full order
    # still re-ranks), but a differentiating regime (e.g. high_vol_panic, tech
    # ×0.4 / bonds ×1.3) visibly moves the top-10 of the real fresh mu.
    per_regime = []
    for reg in sorted(table.keys()) if isinstance(table, Mapping) else []:
        fr2 = build_forecast_bundle(run_date, mu, None, reg, universe_df,
                                    health_map={}, regime_compat=table)
        order2 = sorted(elig, key=lambda s: -(fr2.mu_M1[s] * fr2.regime_score_mult.get(s, 1.0)))
        top2 = order2[:10]
        per_regime.append({
            "regime": reg, "top10": top2,
            "top10_differs_from_raw_mu": bool(top2 != raw_top),
            "n_mult_nonunit": sum(1 for s in elig
                                  if abs(fr2.regime_score_mult.get(s, 1.0) - 1.0) > 1e-9),
        })

    # Fail-loud startup assertion demonstration (non-destructive).
    assertion = {}
    try:
        assert_regime_chassis_loaded(config)
        assertion["loaded_config_passes"] = True
    except RegimeChassisInertError as e:  # should NOT happen when loaded
        assertion["loaded_config_passes"] = False
        assertion["unexpected_error"] = str(e)
    empty_cfg = dict(config)
    empty_cfg["regime_compatibility"] = {}
    try:
        assert_regime_chassis_loaded(empty_cfg)
        assertion["empty_table_fires"] = False
    except RegimeChassisInertError as e:
        assertion["empty_table_fires"] = True
        assertion["empty_table_msg"] = str(e)[:220]

    from .forecast_adapter import UNKNOWN_HEALTH_DEFAULT
    out = {
        "settled_trading_day": run_date,
        "regime_label": regime_label,
        "regime_label_source": regime_label_source,
        "inference_regime_label": inference_regime_label,
        "regime_compat_loaded": regime_compat_loaded,
        "regime_table_regimes": sorted(table.keys()) if isinstance(table, Mapping) else [],
        "n_mu": len(mu), "mu_sha16": mu_sha,
        "n_eligible": len(elig),
        "n_regime_mult_nonunit": n_nonunit,
        "regime_tilt_reranks_full_order": reranks,
        "regime_tilt_reranks_top10": top10_reranks,
        "raw_mu_top10": raw_top,
        "regime_tilted_top10": tilt_top,
        "top10_rank_moves": moves,
        "per_regime_top10": per_regime,
        "startup_assertion": assertion,
        "health_none_rule": {
            "UNKNOWN_HEALTH_DEFAULT": UNKNOWN_HEALTH_DEFAULT,
            "rule": "health=None -> UNKNOWN_HEALTH_DEFAULT (explicit, counted; "
                    "identical for candidates and holdings); a reported sub-h_min "
                    "health is still cut. No silent .get(sym,1.0).",
        },
    }
    print(f"  [REGIME-DIAG] regime={regime_label} compat_loaded={regime_compat_loaded} "
          f"n_mult_nonunit={n_nonunit}/{len(elig)} reranks_top10={top10_reranks} "
          f"mu_sha={mu_sha}\n"
          f"  [REGIME-DIAG] raw_top10   ={raw_top}\n"
          f"  [REGIME-DIAG] tilted_top10={tilt_top}\n"
          f"  [REGIME-DIAG] assertion={assertion}")
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
