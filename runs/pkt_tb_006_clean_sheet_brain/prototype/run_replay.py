"""PKT-TB-006 — replay runner for the SYN-1 vs incumbent bake-off
(replay wiring deliverable #3).

    .venv/bin/python run_replay.py --arm incumbent|syn1 --window holdout|full --out <dir>
    .venv/bin/python run_replay.py --arm syn1 --smoke --out <dir>

Wiring adjudications (binding, identical for both arms):
  #1 E2 verdict pair  = native run_variant (seed_portfolio at its hardcoded
     2026-03-11) over the holdout window. Full-period context pair = the same
     code with replay_engine.START_PORTFOLIO_DATE monkeypatched at runtime to
     2026-02-03, applied identically for both arms and restored after.
  #2 The harness fills at raw open with NO transaction costs; the pre-registered
     cost model (src/utils/transaction_costs, seeded rng 4242) is applied as an
     identical POST-HOC overlay over each arm's executed `actions` list, in
     executed (chronological) order with ONE shared rng sequence per run.
     Both raw and cost-adjusted daily value series are emitted; cost-adjustment
     subtracts the cumulative cost cash-flow from the daily value path.

HOLDOUT GUARD: --window holdout|full refuses to run unless
PKT_TB_006_HOLDOUT_AUTHORIZED=1 (set by the orchestrator at battery time).
--smoke runs 2026-02-04 -> 2026-03-06 decision dates ONLY and is structurally
unable to touch >= 2026-03-11 (asserted on the trading-date list).

Battery knobs (close the §4.4 runner capability gaps; all manifest-recorded):
  --exec-mode linear_twin|learned|equal_trust
                                    'linear_twin' (DEFAULT) = the FROZEN SYN-1
                                    executive gate (FREEZE_SYN1.md §7.3 ladder;
                                    exec_dir/linear_twin.pt, pure numpy);
                                    'learned' = MLP seed ensemble (diagnostics);
                                    'equal_trust' = R08 bypass (tau=1/M over
                                    active members, f fixed 0.7, same rails)
  --cost-seed <int>                 R13/R14 slippage-seed override for the
                                    post-hoc cost overlay rng (default 4242)
  --sigma-source trailing21|risknet R07 sigma swap for the vol-cap sigma_hat +
                                    executive book-vol input ('trailing21' =
                                    the landed/R01 proxy convention, named)

Per-arm audit trail: every syn1 replay copies the store/nightly/<D>/
meta_decision.json + trade_intents.json it wrote into <out>/nightly_audit/<D>/
so battery arms keep their own audit artifacts instead of overwriting.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

HOLDOUT_START = "2026-03-11"
FULL_START_DIR = "2026-02-03"          # portfolio seed + trading_dates[0] (full/smoke)
SMOKE_LAST_DECISION = "2026-03-06"
COST_SEED = 4242
ENV_FLAG = "PKT_TB_006_HOLDOUT_AUTHORIZED"


# ------------------------------------------------------------------ guards
def check_holdout_authorization(window: str, smoke: bool,
                                env: Optional[dict] = None) -> None:
    """Refuse holdout/full unless the orchestrator set the env flag. Smoke never
    needs authorization (and never touches >= HOLDOUT_START by construction)."""
    env = os.environ if env is None else env
    if smoke:
        return
    if window in ("holdout", "full") and env.get(ENV_FLAG) != "1":
        raise SystemExit(
            f"REFUSED: --window {window} requires {ENV_FLAG}=1 "
            f"(orchestrator sets it at battery time; smoke runs use --smoke).")


def build_trading_dates(cache, window: str, smoke: bool) -> List[str]:
    """Trading dates = cached snapshot dirs that carry prices.parquet."""
    daily = Path(cache.cache_dir) / "daily"
    dates = sorted(d.name for d in daily.iterdir()
                   if d.is_dir() and (d / "prices.parquet").exists())
    if smoke:
        upto = [d for d in dates if d >= FULL_START_DIR]
        out = [d for d in upto if d <= SMOKE_LAST_DECISION]
        nxt = [d for d in upto if d > SMOKE_LAST_DECISION]
        if nxt:
            out.append(nxt[0])           # one extra dir prices the last decision date
        assert all(d < HOLDOUT_START for d in out), \
            "smoke trading dates touch the holdout"
        assert all(d <= SMOKE_LAST_DECISION for d in out[1:-1]), \
            "smoke decision dates exceed the smoke window"
        return out
    if window == "holdout":
        return [d for d in dates if d >= HOLDOUT_START]
    if window == "full":
        return [d for d in dates if d >= FULL_START_DIR]
    raise SystemExit(f"unknown window {window!r}")


# ------------------------------------------------------------------ cost overlay
def cost_overlay(result: Dict[str, Any], universe_df, seed: int = COST_SEED,
                 cost_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pure post-hoc transaction-cost overlay (wiring adjudication #2).

    Walks the run's executed `actions` list in its stored (chronological,
    fixed) order, applies src.utils.transaction_costs.apply_transaction_costs
    with ONE shared random.Random(seed) sequence, and returns raw +
    cost-adjusted daily value series (cumulative cost cash-flow subtracted)."""
    from src.utils.transaction_costs import apply_transaction_costs
    rng = random.Random(seed)
    sector = dict(zip(universe_df["symbol"], universe_df["sector"]))
    aclass = dict(zip(universe_df["symbol"], universe_df["asset_class"]))
    records = []
    costs_by_date: Dict[str, float] = {}
    total_traded = 0.0
    for a in result.get("actions", []):
        fill_costed, cost_bps = apply_transaction_costs(
            price=float(a["price"]), action=str(a["action"]),
            sector=sector.get(a["symbol"], "broad"),
            asset_class=aclass.get(a["symbol"], "equity"),
            rng=rng, cost_config=cost_config)
        cost_dollars = abs(fill_costed - float(a["price"])) * float(a["shares"])
        records.append({"date": a["date"], "symbol": a["symbol"],
                        "action": a["action"], "shares": float(a["shares"]),
                        "raw_fill": float(a["price"]),
                        "costed_fill": round(fill_costed, 6),
                        "cost_bps": round(cost_bps, 4),
                        "cost_dollars": round(cost_dollars, 4)})
        costs_by_date[a["date"]] = costs_by_date.get(a["date"], 0.0) + cost_dollars
        total_traded += abs(float(a.get("dollars", 0.0))
                            or float(a["shares"]) * float(a["price"]))
    dates = sorted(result["date_value_map"])
    cum = 0.0
    raw, adj = [], []
    for d in dates:
        cum += costs_by_date.get(d, 0.0)
        raw.append(float(result["date_value_map"][d]))
        adj.append(raw[-1] - cum)
    start_v = raw[0] if raw else 0.0
    return {"seed": seed, "dates": dates, "raw": raw, "cost_adjusted": adj,
            "trade_costs": records,
            "total_cost_dollars": round(cum, 4),
            "total_traded_dollars": round(total_traded, 2),
            "cost_drag_bps_of_start_nav": round(cum / start_v * 1e4, 3)
                                          if start_v else None,
            "cost_bps_of_traded": round(cum / total_traded * 1e4, 3)
                                  if total_traded else None}


# ------------------------------------------------------------------ manifest
def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()[:12]
    except Exception:
        return "unknown"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


# ------------------------------------------------------------------ audit copy
def copy_nightly_audit(nightly_dir: Path | str, out_dir: Path | str,
                       trading_dates: List[str], since_ts: float) -> int:
    """Copy the per-date meta_decision.json / trade_intents.json THIS run wrote
    (mtime >= since_ts; stale files from earlier arms are skipped) into
    <out_dir>/nightly_audit/<D>/ — every battery arm keeps its own audit trail
    instead of the shared store/nightly/<D>/ copy being overwritten."""
    import shutil
    n = 0
    for d in trading_dates:
        src = Path(nightly_dir) / d
        for name in ("meta_decision.json", "trade_intents.json"):
            f = src / name
            if f.exists() and f.stat().st_mtime >= since_ts:
                dst = Path(out_dir) / "nightly_audit" / d
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / name)
                n += 1
    return n


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["incumbent", "syn1"])
    ap.add_argument("--window", default="full", choices=["holdout", "full"])
    ap.add_argument("--smoke", action="store_true",
                    help="pre-holdout smoke: decisions 2026-02-04..2026-03-06 only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--genome", default="",
                    help="genome json path (default: B0 DEFAULT_GENOME)")
    ap.add_argument("--exec-dir", default=str(PROTO / "exec_out"))
    ap.add_argument("--nightly-dir", default=str(PROTO / "store" / "nightly"))
    ap.add_argument("--exec-mode", default="linear_twin",
                    choices=["linear_twin", "learned", "equal_trust"],
                    help="executive gate (syn1 only): linear_twin = the FROZEN "
                         "SYN-1 gate (default; FREEZE_SYN1.md); learned = MLP "
                         "seed ensemble (diagnostics); equal_trust = R08 bypass "
                         "(tau 1/M over active members, f fixed 0.7, same rails)")
    ap.add_argument("--cost-seed", type=int, default=COST_SEED,
                    help="slippage seed for the post-hoc cost overlay rng "
                         f"(R13/R14 override; default {COST_SEED})")
    ap.add_argument("--sigma-source", default="trailing21",
                    choices=["trailing21", "risknet"],
                    help="R07 sigma swap: trailing21 = trailing-vol proxy "
                         "(landed/R01 convention), risknet = E4 heads (syn1 only)")
    return ap


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    command = "python " + " ".join(sys.argv if argv is None else ["run_replay.py"] + argv)

    check_holdout_authorization(args.window, args.smoke)

    import pandas as pd
    from data_layer import DiskCachedS3Cache
    from src.utils.three_line_replay import replay_engine as RE
    from src.utils.transaction_costs import get_cost_config_snapshot

    cache = DiskCachedS3Cache(s3_client=None)          # offline, disk-cache only
    trading_dates = build_trading_dates(cache, args.window, args.smoke)
    if len(trading_dates) < 3:
        raise SystemExit(f"too few trading dates ({len(trading_dates)})")
    universe_df = pd.read_csv(REPO / "config" / "universe.csv")

    # incumbent VariantConfig for BOTH arms (engine needs it even when the
    # strategy replaces the intents)
    variant, _pre = RE.load_variant_configs(cache)

    strategy = None
    genome_dict = None
    sigma_detail = None
    if args.arm == "syn1":
        from ea import Genome
        from strategy_adapter import make_syn1_strategy, SIGMA_SOURCES
        genome = (Genome.from_json(Path(args.genome)) if args.genome
                  else Genome.b0())
        genome_dict = genome.to_dict()
        strategy = make_syn1_strategy(genome, args.exec_dir, args.nightly_dir,
                                      cache=cache, exec_mode=args.exec_mode,
                                      sigma_source=args.sigma_source)
        sigma_detail = {"mode": args.sigma_source,
                        "sigma_col": SIGMA_SOURCES[args.sigma_source][0],
                        "book_vol_key": SIGMA_SOURCES[args.sigma_source][1]}

    # adjudication #1: START_PORTFOLIO_DATE patch for full/smoke, restored after
    native_start = RE.START_PORTFOLIO_DATE
    start_used = native_start if (args.window == "holdout" and not args.smoke) \
        else FULL_START_DIR
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        RE.START_PORTFOLIO_DATE = start_used
        result = RE.run_variant(cache, variant, strategy, trading_dates,
                                universe_df)
    finally:
        RE.START_PORTFOLIO_DATE = native_start

    overlay = cost_overlay(result, universe_df, seed=args.cost_seed)

    # ---- write artifacts -------------------------------------------------------
    daily = pd.DataFrame({"date": overlay["dates"], "raw_value": overlay["raw"],
                          "cost_adjusted_value": overlay["cost_adjusted"]})
    daily.to_csv(out_dir / "daily_series.csv", index=False)
    with (out_dir / "result.json").open("w") as fh:
        json.dump({k: v for k, v in result.items() if k != "timeline"},
                  fh, indent=1, default=str)
    with (out_dir / "timeline.json").open("w") as fh:
        json.dump(result["timeline"], fh, indent=1, default=str)
    with (out_dir / "cost_overlay.json").open("w") as fh:
        json.dump(overlay, fh, indent=1)

    n_dec = len(overlay["dates"])
    n_actions = len(result.get("actions", []))
    summary = {
        "arm": args.arm, "window": "smoke" if args.smoke else args.window,
        "n_decision_dates": n_dec,
        "first_date": overlay["dates"][0] if n_dec else None,
        "last_date": overlay["dates"][-1] if n_dec else None,
        "start_value": overlay["raw"][0] if n_dec else None,
        "final_value_raw": overlay["raw"][-1] if n_dec else None,
        "final_value_cost_adjusted": overlay["cost_adjusted"][-1] if n_dec else None,
        "n_executed_actions": n_actions,
        "actions_per_decision_date": round(n_actions / n_dec, 3) if n_dec else None,
        "total_traded_dollars": overlay["total_traded_dollars"],
        "total_cost_dollars": overlay["total_cost_dollars"],
        "cost_drag_bps_of_start_nav": overlay["cost_drag_bps_of_start_nav"],
        "cost_bps_of_traded": overlay["cost_bps_of_traded"],
    }
    # per-arm audit trail: keep this run's nightly decision artifacts in --out
    n_audit = 0
    if args.arm == "syn1":
        n_audit = copy_nightly_audit(args.nightly_dir, out_dir, trading_dates,
                                     since_ts=t0 - 1.0)
    nightly_manifest = {}
    nm = Path(args.nightly_dir) / "manifest.json"
    if args.arm == "syn1" and nm.exists():
        nightly_manifest = json.loads(nm.read_text())
    from precompute_nightly import wiring_code_sha
    manifest = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "arm": args.arm, "window": "smoke" if args.smoke else args.window,
        "code_sha": {"git": _git_sha(), "prototype_wiring": wiring_code_sha()},
        "genome": genome_dict, "genome_hash": _sha(genome_dict) if genome_dict else None,
        "variant_params_hash": _sha(variant.decision_params),
        "snapshot_range": [trading_dates[0], trading_dates[-1]],
        "n_trading_dates": len(trading_dates),
        "start_portfolio_date_used": start_used,
        "native_start_portfolio_date": native_start,
        "cost_model": {"seed": args.cost_seed,
                       "version_sha": _sha(get_cost_config_snapshot())},
        "exec_mode": args.exec_mode if args.arm == "syn1" else None,
        "sigma_source": sigma_detail,
        "nightly_audit": ({"dir": "nightly_audit", "n_files": n_audit}
                          if args.arm == "syn1" else None),
        "exec_weights_dir": args.exec_dir if args.arm == "syn1" else None,
        "nightly_manifest": {k: nightly_manifest.get(k) for k in
                             ("generated", "window", "code_sha", "cast_n_seeds")}
                            if nightly_manifest else None,
        "wall_clock_s": round(time.time() - t0, 1),
        "summary": summary,
        "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1",
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"artifacts -> {out_dir} ({manifest['wall_clock_s']}s)")
    return {"summary": summary, "manifest": manifest}


if __name__ == "__main__":
    main()
