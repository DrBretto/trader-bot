"""PKT-TB-007 — replay runner for the ORB-1 vs incumbent verdict pair
(BUILD_SPEC_007 §2.1/§7; TOURNAMENT_007 §4.1 pre-registration; adapted from
the proven TB-006 run_replay.py).

  .venv/bin/python run_replay_007.py --arm incumbent|orb1 --window live --out <dir>
  .venv/bin/python run_replay_007.py --arm orb1 --window preholdout --out <dir>
  .venv/bin/python run_replay_007.py --window holdout --from-live <livedir> --out <dir>

Windows (per TOURNAMENT_007 §4.1, implemented as written):
  live       E1 primary read: full live window, dirs >= 2026-01-31 (seed) ->
             present; ~66 paired daily deltas. REQUIRES the holdout guard env.
  preholdout B0-EXPR / smoke window: decision dates 2026-02-02..2026-03-06,
             structurally unable to touch >= 2026-03-11 (asserted). No guard.
  holdout    E2 confirmation read: NOT a separate replay — the subset
             >= 2026-03-11 of an existing live run's daily series (the paired
             read shares the live run's path state). REQUIRES the guard env;
             exactly ONE read, fired by the orchestrator at bake-off.

The 2025 backfill leg is NOT run (no shim — §4.1).

Identical for BOTH arms (separate OS process per arm, §2.1 / model-cache hazard):
  - offline DiskCachedS3Cache over the TB-006 extended cache (read-only import)
  - START_PORTFOLIO_DATE monkeypatch (try/finally, sanctioned pattern)
  - the C5 harness lot monkeypatch (lot_fix_007.harness_lot_patch)
  - post-hoc cost overlay, seed 4242 (TB-006 wiring adjudication #2 carried)

Per-arm audit trail: orb1 runs keep their expression logs + the organ input
files they consumed under --out (organ_inputs_audit/).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB006_PROTO = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
for p in (str(PROTO), str(REPO), str(TB006_PROTO)):
    if p not in sys.path:
        sys.path.append(p)          # append: TB-007 + repo win name collisions

HOLDOUT_START = "2026-03-11"
LIVE_START_DIR = "2026-01-31"       # TOURNAMENT_007 §4.1: live window 2026-01-31 ->
PREHOLDOUT_LAST_DECISION = "2026-03-06"
COST_SEED = 4242
ENV_FLAG = "PKT_TB_007_HOLDOUT_AUTHORIZED"


# ------------------------------------------------------------------ guards
def check_holdout_authorization(window: str, env: Optional[dict] = None) -> None:
    """live/holdout touch >= 2026-03-11 => refuse unless the orchestrator set
    the env flag. preholdout never needs authorization (and structurally
    cannot touch the holdout)."""
    env = os.environ if env is None else env
    if window in ("live", "holdout") and env.get(ENV_FLAG) != "1":
        raise SystemExit(
            f"REFUSED: --window {window} requires {ENV_FLAG}=1 "
            f"(orchestrator sets it at bake-off; pre-holdout work uses "
            f"--window preholdout).")


def build_trading_dates(cache, window: str) -> List[str]:
    daily = Path(cache.cache_dir) / "daily"
    dates = sorted(d.name for d in daily.iterdir()
                   if d.is_dir() and (d / "prices.parquet").exists())
    live = [d for d in dates if d >= LIVE_START_DIR]
    if window == "live":
        return live
    if window == "preholdout":
        out = [d for d in live if d <= PREHOLDOUT_LAST_DECISION]
        nxt = [d for d in live if d > PREHOLDOUT_LAST_DECISION]
        if nxt:
            out.append(nxt[0])      # one extra dir prices the last decision date
        assert all(d < HOLDOUT_START for d in out), \
            "preholdout trading dates touch the holdout"
        assert all(d <= PREHOLDOUT_LAST_DECISION for d in out[1:-1]), \
            "preholdout decision dates exceed the window"
        return out
    raise SystemExit(f"unknown window {window!r}")


# ------------------------------------------------------------------ cost overlay
def cost_overlay(result: Dict[str, Any], universe_df, seed: int = COST_SEED,
                 cost_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Post-hoc transaction-cost overlay — TB-006 wiring adjudication #2
    carried verbatim: executed actions walked in stored order, ONE shared
    random.Random(seed) sequence, raw + cost-adjusted series emitted."""
    from chassis.utils.transaction_costs import apply_transaction_costs
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


# ------------------------------------------------------------------ helpers
def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=10)
        sha = r.stdout.strip()[:12]
        d = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True, timeout=10)
        return sha + ("+dirty" if d.stdout.strip() else "")
    except Exception:
        return "unknown"


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ holdout subset
def extract_holdout_subset(live_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """E2: the >= 2026-03-11 subset of an existing live run (no replay)."""
    import pandas as pd
    live_dir = Path(live_dir)
    daily = pd.read_csv(live_dir / "daily_series.csv")
    sub = daily[daily["date"] >= HOLDOUT_START].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / "daily_series.csv", index=False)
    src_manifest = json.loads((live_dir / "manifest.json").read_text())
    manifest = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "window": "holdout_subset",
        "holdout_start": HOLDOUT_START,
        "from_live_run": str(live_dir),
        "from_live_manifest_sha": _sha_obj(src_manifest),
        "n_decision_dates": int(len(sub)),
        "first_date": sub["date"].iloc[0] if len(sub) else None,
        "last_date": sub["date"].iloc[-1] if len(sub) else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return manifest


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["incumbent", "orb1"],
                    help="required for live/preholdout replays")
    ap.add_argument("--window", required=True,
                    choices=["live", "preholdout", "holdout"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-live", default="",
                    help="holdout window: path of the live run to subset")
    ap.add_argument("--genome", default="",
                    help="orb1 genome json (default: B0)")
    ap.add_argument("--nightly-dir", default=str(PROTO / "store" / "nightly_007"),
                    help="organ input files dir (organ contract v1)")
    ap.add_argument("--masks", default="",
                    help="masks json path (cross-fit interface; default: "
                         "frozen production masks)")
    ap.add_argument("--cost-seed", type=int, default=COST_SEED)
    ap.add_argument("--rllm", default="",
                    help="R-LLM falsifier arm ONLY (§4.4.9/BUILD_SPEC §2.6): "
                         "path to store/rllm_disag_007.json; T_t scaled by g")
    return ap


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    command = "python " + " ".join(sys.argv if argv is None
                                   else ["run_replay_007.py"] + argv)
    check_holdout_authorization(args.window)
    out_dir = Path(args.out)

    if args.window == "holdout":
        if not args.from_live:
            raise SystemExit("--window holdout requires --from-live <livedir> "
                             "(E2 is a subset read, not a separate replay)")
        return {"manifest": extract_holdout_subset(Path(args.from_live), out_dir)}

    if args.arm not in ("incumbent", "orb1"):
        raise SystemExit("--arm incumbent|orb1 required for replay windows")

    import pandas as pd
    from data_layer import DiskCachedS3Cache          # TB-006 import (read-only)
    from chassis.utils.three_line_replay import replay_engine as RE
    from chassis.utils.transaction_costs import get_cost_config_snapshot
    from lot_fix_007 import harness_lot_patch
    from genome_007 import Genome007

    cache = DiskCachedS3Cache(s3_client=None)         # offline, disk-cache only
    trading_dates = build_trading_dates(cache, args.window)
    if len(trading_dates) < 3:
        raise SystemExit(f"too few trading dates ({len(trading_dates)})")
    universe_df = pd.read_csv(REPO / "config" / "universe.csv")
    variant, _pre = RE.load_variant_configs(cache)

    strategy = None
    genome = None
    masks = None
    if args.arm == "orb1":
        from tilt_adapter import make_tilt_strategy, PRODUCTION_MASKS
        genome = (Genome007.from_json(Path(args.genome)) if args.genome
                  else Genome007.b0())
        masks = (json.loads(Path(args.masks).read_text()) if args.masks
                 else PRODUCTION_MASKS)
        rllm = (json.loads(Path(args.rllm).read_text())["dates"]
                if args.rllm else None)
        out_dir.mkdir(parents=True, exist_ok=True)
        strategy = make_tilt_strategy(genome, args.nightly_dir,
                                      log_dir=out_dir, cache=cache,
                                      masks=masks, rllm=rllm)

    out_dir.mkdir(parents=True, exist_ok=True)
    native_start = RE.START_PORTFOLIO_DATE
    start_used = trading_dates[0]
    try:
        RE.START_PORTFOLIO_DATE = start_used
        with harness_lot_patch(RE):                   # C5, BOTH arms identically
            result = RE.run_variant(cache, variant, strategy, trading_dates,
                                    universe_df)
    finally:
        RE.START_PORTFOLIO_DATE = native_start

    overlay = cost_overlay(result, universe_df, seed=args.cost_seed)

    # ---- artifacts -----------------------------------------------------------
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

    # ---- per-arm audit copies (organ inputs consumed by this run) -----------
    n_audit = 0
    if args.arm == "orb1":
        nd = Path(args.nightly_dir)
        for d in trading_dates:
            f = nd / f"{d}.json"
            if f.exists():
                dst = out_dir / "organ_inputs_audit"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / f.name)
                n_audit += 1

    n_dec = len(overlay["dates"])
    n_actions = len(result.get("actions", []))
    summary = {
        "arm": args.arm, "window": args.window,
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
        "sha256_timeline": sha256_file(out_dir / "timeline.json"),
        "sha256_daily_series": sha256_file(out_dir / "daily_series.csv"),
    }
    manifest = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "arm": args.arm, "window": args.window,
        "code_sha": {"git": _git_sha()},
        "genome": genome.to_dict() if genome else None,
        "genome_hash": genome.hash() if genome else None,
        "masks_hash": _sha_obj(masks) if masks else None,
        "variant_params_hash": _sha_obj(variant.decision_params),
        "snapshot_range": [trading_dates[0], trading_dates[-1]],
        "n_trading_dates": len(trading_dates),
        "start_portfolio_date_used": start_used,
        "native_start_portfolio_date": native_start,
        "lot_patch": "lot_fix_007._execute_intents_lotfix (both arms)",
        "cost_model": {"seed": args.cost_seed,
                       "version_sha": _sha_obj(get_cost_config_snapshot())},
        "nightly_dir": args.nightly_dir if args.arm == "orb1" else None,
        "rllm_conditioner": (args.rllm or None) if args.arm == "orb1" else None,
        "organ_inputs_audited": n_audit if args.arm == "orb1" else None,
        "holdout_guard_env_set": os.environ.get(ENV_FLAG) == "1",
        "wall_clock_s": round(time.time() - t0, 1),
        "summary": summary,
        "evidence_protocol": "committee/EVIDENCE_PROTOCOL.md V1",
        "pre_registration": "runs/pkt_tb_007_orthogonal_brain/TOURNAMENT_007.md §4",
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"artifacts -> {out_dir} ({manifest['wall_clock_s']}s)")
    return {"summary": summary, "manifest": manifest}


if __name__ == "__main__":
    main()
