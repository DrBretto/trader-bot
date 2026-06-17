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
    f = build_forecast_bundle(date, mu_map, features_df, regime_label,
                              universe_df, health_map=health_map)
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


def _extend_ohlcv_from_s3(SL, FI) -> int:
    """Bring the brain's OHLCV store current: fetch the chassis's published
    daily/<D>/prices.parquet for every date newer than the seed store and splice
    them in (extend_ohlcv). Uses the Lambda execution role's default creds (NOT
    the laptop 'personal' profile that SL.S3Source assumes)."""
    import boto3
    import pandas as pd
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    spy = SL.CACHE_OHLCV / "SPY.parquet"
    last_ohlcv = None
    if spy.exists():
        df = pd.read_parquet(spy, columns=["date"])
        last_ohlcv = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")

    pg = s3.get_paginator("list_objects_v2")
    dates = set()
    for page in pg.paginate(Bucket=S3_BUCKET, Prefix="daily/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            d = cp["Prefix"].split("/")[-2]
            if len(d) == 10 and d[4] == "-":
                dates.add(d)
    gap = sorted(d for d in dates if last_ohlcv is None or d > last_ohlcv)
    n = 0
    for d in gap:
        dst = SL.CACHE_DAILY / d / "prices.parquet"
        if not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(S3_BUCKET, f"daily/{d}/prices.parquet", str(dst))
            except Exception:
                continue
        try:
            FI.extend_ohlcv(d)
            n += 1
        except Exception:
            continue
    return n


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
    _extend_ohlcv_from_s3(SL, FI)
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
