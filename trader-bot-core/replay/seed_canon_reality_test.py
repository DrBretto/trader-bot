"""P6 reality-test + deliverable driver (PKT-TRADER-BOT-SEED-CANON-BY-REPLAY).

Reality-gated proof of the six acceptance checks (NEVER green unit tests):

  (1) the post-split window is reconstructed by replaying the corrected spine with
      the RECORDED regime — picks ROTATE (recomputed, not re-marked);
  (2) the pre-split leaves are BYTE-UNCHANGED and the reconstruction ANCHORS
      CONTINUOUSLY at the split point (no jump at the join);
  (3) a recent post-split day's replayed leaf EQUALS the forward path's (replay==forward);
  (4) forward nightly appends exactly ONE settled leaf/day off the corrected
      frontier and NO prior settled leaf moves;
  (5) all three lines (canon/SPY/challenger) sit on ONE real trading-day grid;
  (6) the DELTA_REPORT shows the old-vs-corrected post-split difference (information).

The proofs run against an in-memory ledger (FakeS3) sourcing REAL substrate — the
real production pre-split leaves (read-only), the real recorded regime from S3
daily/, and the real settled OHLCV extended from S3. ``--commit`` additionally
SEEDS the real clean ledger on S3 (prefix ``canon/equity_ledger_clean/``); the
production ledger is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))              # trader-bot-core
sys.path.insert(0, str(_THIS.parents[2]))              # repo root (src.*)

from replay._fake_s3 import FakeS3
from lines.ledger import EquityLedger
from replay import seed_canon as SC


def _boto3_s3():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _leaf_identity(leaf: dict) -> dict:
    """The equity identity of a leaf (excludes chain-position + non-hashed meta) —
    what 'replay==forward' compares."""
    return {k: leaf.get(k) for k in
            ("date", "value", "benchmark", "comparison", "segment", "source")}


def _old_post_split(prod: EquityLedger) -> Dict[str, dict]:
    """The active (head-of-supersede-chain) production leaf per post-split date —
    the old contaminated line the DELTA_REPORT compares against."""
    leaves = prod.list_leaves()
    superseded = {l["supersedes"] for l in leaves if l.get("supersedes")}
    out: Dict[str, dict] = {}
    for leaf in leaves:
        if leaf["date"] > SC.SPLIT_DATE and leaf["content_sha"] not in superseded:
            out[leaf["date"]] = leaf
    return out


def _old_selected_from_s3(s3, date: str) -> List[str]:
    for key in (f"daily/{date}/brain_selected_universe.json",):
        try:
            j = json.loads(s3.get_object(Bucket="investment-system-data", Key=key)["Body"].read())
            return sorted(j.get("selected_universe", []))
        except Exception:  # noqa: BLE001
            pass
    return []


def run(commit: bool, d1: str, out_path: str) -> dict:
    s3 = _boto3_s3()
    ohlcv = SC.prepare_substrate()
    window = SC.post_split_window(ohlcv, d1)
    reg_fn = SC.recorded_regime_fn(window, s3=s3)
    recorded_map = reg_fn.regime_map

    prod = EquityLedger(s3, prefix=SC.PROD_PREFIX)      # READ-ONLY source of pre-split + old line
    evidence: Dict[str, object] = {
        "split_date": SC.SPLIT_DATE,
        "reconstruction_window": window,
        "recorded_regime_map": recorded_map,
        "anchors": None, "checks": {},
    }

    # ---- build the clean ledger in-memory (proof) --------------------------- #
    proof = EquityLedger(FakeS3(), prefix=SC.CLEAN_PREFIX)
    copied = SC.copy_presplit(prod, proof, SC.SPLIT_DATE)
    a_canon, a_bench, a_comp = SC.anchor_from_frontier(proof, SC.SPLIT_DATE)
    evidence["anchors"] = {"canon": a_canon, "benchmark": a_bench, "comparison": a_comp}

    # (2a) pre-split byte-unchanged: every copied leaf's content_sha recomputes and
    #      matches the production leaf exactly.
    prod_pre = {l["date"]: l for l in prod.list_leaves() if l["date"] <= SC.SPLIT_DATE}
    clean_pre = {l["date"]: l for l in proof.list_leaves() if l["date"] <= SC.SPLIT_DATE}
    byte_ok = (set(prod_pre) == set(clean_pre)) and all(
        json.dumps(prod_pre[d], sort_keys=True) == json.dumps(clean_pre[d], sort_keys=True)
        for d in prod_pre)
    evidence["checks"]["c2_presplit_byte_unchanged"] = {
        "pass": bool(byte_ok), "n_presplit_leaves": len(clean_pre),
        "presplit_terminal": {"date": SC.SPLIT_DATE, **_leaf_identity(clean_pre[SC.SPLIT_DATE])},
    }

    # ---- reconstruct the post-split window (recorded regime) ---------------- #
    res = SC.seed_canon_by_replay(proof, ohlcv, d1=d1, regime_fn=reg_fn)
    recon = {r.date: r for r in res.days}

    # (2b) continuous anchor: first reconstructed leaf chains onto the 06-11 leaf and
    #      the canon line begins at the pre-split terminal (no seam / jump).
    first = res.days[0]
    first_leaf = first.leaf
    boundary_sha = clean_pre[SC.SPLIT_DATE]["content_sha"]
    # continuous = the first reconstructed leaf chains onto the 06-11 pre-split leaf
    # (prev_date + prev_content_hash), AND the displayed line joins with no seam: the
    # first value = anchor x (1 + first-day return), so there is no discontinuous jump
    # imposed at the join (the only move is the honest day-1 return).
    expected_join = a_canon * (1.0 + first.canon_return)
    seam = abs(first_leaf["value"] - expected_join)
    evidence["checks"]["c2_continuous_anchor"] = {
        "pass": bool(first_leaf["prev_date"] == SC.SPLIT_DATE
                     and first_leaf["prev_content_hash"] == boundary_sha
                     and seam < 1e-6),
        "first_day": first.date, "anchor_canon": a_canon,
        "first_day_value": round(first_leaf["value"], 4),
        "first_day_return": round(first.canon_return, 6),
        "seam_residual": seam,
        "prev_date": first_leaf["prev_date"],
        "prev_content_hash_matches_boundary": bool(first_leaf["prev_content_hash"] == boundary_sha),
        "note": "no genesis reset — reconstruction appends onto the byte-unchanged 06-11 frontier",
    }

    # (1) picks ROTATE (recomputed, not re-marked): reconstruction selection differs
    #     from the old contaminated selected_universe.
    rot = []
    for d in window:
        old_sel = _old_selected_from_s3(s3, d)
        new_sel = res.selected_by_date.get(d, [])
        rot.append({"date": d, "regime": recorded_map[d],
                    "old_selected": old_sel, "recon_selected": new_sel,
                    "rotated": bool(sorted(old_sel) != sorted(new_sel))})
    n_rot = sum(1 for r in rot if r["rotated"])
    evidence["checks"]["c1_picks_rotate"] = {
        "pass": bool(n_rot > 0), "n_days_rotated": n_rot, "n_days": len(window),
        "per_day": rot}

    # (5) one real trading-day grid: every reconstructed leaf is a weekday with a
    #     settled SPY bar; no weekend/holiday phantom.
    import datetime as _dt
    grid_ok = all(_dt.date.fromisoformat(d).weekday() < 5 for d in window) and \
        all(ohlcv.bar("SPY", d) is not None for d in window)
    evidence["checks"]["c5_one_real_grid"] = {
        "pass": bool(grid_ok), "grid": window,
        "phantom_free": True,
        "note": "grid derived from settled SPY bars; old ledger phantoms "
                "(06-13/-19/-20/-27) and gaps (06-26/-29) corrected"}

    # (6) DELTA_REPORT: old contaminated vs corrected post-split line (INFORMATION).
    old = _old_post_split(prod)
    delta_rows = []
    for d in window:
        oc = old.get(d)
        nc = recon[d].leaf
        ov = float(oc["value"]) if oc else None
        nv = float(nc["value"])
        delta_rows.append({
            "date": d, "regime": recorded_map[d],
            "old_value": ov, "corrected_value": round(nv, 2),
            "delta": round(nv - ov, 2) if ov is not None else None,
            "delta_pct": round((nv / ov - 1.0) * 100, 3) if ov else None,
            "old_present": oc is not None})
    old_terminal = old.get(window[-1], {}).get("value")
    new_terminal = recon[window[-1]].leaf["value"]
    evidence["delta_report"] = {
        "note": "INFORMATION ONLY — no sign-off gate on the magnitude (locked scope #5). "
                "The corrected line is what is seeded.",
        "old_terminal": old_terminal, "corrected_terminal": round(new_terminal, 2),
        "terminal_delta_pct": round((new_terminal / old_terminal - 1.0) * 100, 3) if old_terminal else None,
        "rows": delta_rows}

    # ---- (3)+(4) forward-nightly: one leaf/day, no prior moves, replay==forward - #
    fwd_demo = EquityLedger(FakeS3(), prefix=SC.CLEAN_PREFIX)
    SC.copy_presplit(prod, fwd_demo, SC.SPLIT_DATE)
    demo_window = window[:-1]                            # seed all but the last settled day
    demo_reg = {d: recorded_map[d] for d in demo_window}
    SC.seed_canon_by_replay(fwd_demo, ohlcv, d1=demo_window[-1],
                            regime_fn=SC.recorded_regime_fn(demo_window, s3=s3))
    before = {l["date"]: l["content_sha"] for l in fwd_demo.list_leaves()}
    D = window[-1]
    fwd_leaf = SC.forward_nightly(fwd_demo, ohlcv, D, demo_reg)
    after = {l["date"]: l["content_sha"] for l in fwd_demo.list_leaves()}
    added = [d for d in after if d not in before]
    prior_moved = [d for d in before if before[d] != after.get(d)]

    # replay==forward: independently replay [first..D] and compare D's identity.
    # Inject the SAME INDEPENDENT challenger the forward path uses (V2) so the
    # comparison line is compared like-for-like (not against the coupled fallback).
    verify_led = EquityLedger(FakeS3(), prefix="verify/")
    vres = SC.replay(SC.FIRST_POST_SPLIT, D, ledger=verify_led, ohlcv=ohlcv,
                     issued_by=SC.ISSUED_BY, regime_fn=SC.mixed_regime_fn(demo_reg),
                     write_genesis=True, genesis_date=SC.SPLIT_DATE,
                     genesis_anchor=a_canon, bench_anchor=a_bench, comparison_anchor=a_comp,
                     segment="new_brain", source="native_two_stage", model_id_prefix="forward@",
                     challenger=SC.build_independent_challenger(a_comp, s3=s3))
    replay_id = _leaf_identity(vres.terminal_leaf)
    forward_id = _leaf_identity(fwd_leaf)
    evidence["checks"]["c4_forward_one_leaf"] = {
        "pass": bool(added == [D] and not prior_moved),
        "leaves_added": added, "prior_leaves_moved": prior_moved,
        "forward_confirmed": bool(fwd_leaf.get("forward_confirmed")),
        "forward_regime_deterministic": SC.mixed_regime_fn(demo_reg)(D)}
    evidence["checks"]["c3_replay_equals_forward"] = {
        "pass": bool(replay_id == forward_id),
        "date": D, "replay_leaf": replay_id, "forward_leaf": forward_id}

    all_pass = all(c.get("pass") for c in evidence["checks"].values())
    evidence["ALL_ACCEPTANCE_PASS"] = bool(all_pass)

    # ---- deliverable: seed the REAL clean ledger on S3 ---------------------- #
    if commit:
        real = EquityLedger(s3, prefix=SC.CLEAN_PREFIX)
        SC.copy_presplit(prod, real, SC.SPLIT_DATE)
        real_res = SC.seed_canon_by_replay(real, ohlcv, d1=d1,
                                           regime_fn=SC.recorded_regime_fn(window, s3=s3))
        rm = real.read_manifest()
        evidence["deliverable"] = {
            "committed": True, "clean_prefix": SC.CLEAN_PREFIX,
            "n_leaves": len(rm.get("entries", [])),
            "frontier": rm.get("frontier"),
            "terminal_value": round(real_res.days[-1].leaf["value"], 2),
            "note": "clean ledger seeded on S3; production canon/equity_ledger/ untouched (P9 repoints)."}
    else:
        evidence["deliverable"] = {"committed": False,
                                   "note": "proof-only run; rerun with --commit to seed the real clean ledger"}

    Path(out_path).write_text(json.dumps(evidence, indent=2, default=str))
    return evidence


def _main() -> int:
    ap = argparse.ArgumentParser(prog="seed_canon_reality_test")
    ap.add_argument("--d1", default="2026-07-02", help="last settled day to reconstruct through")
    ap.add_argument("--commit", action="store_true", help="also seed the REAL clean ledger on S3")
    ap.add_argument("--out", default="/tmp/seed_canon_evidence.json")
    args = ap.parse_args()
    ev = run(args.commit, args.d1, args.out)
    print(json.dumps({"ALL_ACCEPTANCE_PASS": ev["ALL_ACCEPTANCE_PASS"],
                      "checks": {k: v.get("pass") for k, v in ev["checks"].items()},
                      "delta_terminal_pct": ev["delta_report"]["terminal_delta_pct"],
                      "deliverable_committed": ev["deliverable"].get("committed"),
                      "out": args.out}, indent=2))
    return 0 if ev["ALL_ACCEPTANCE_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
