"""PKT-TB-007 — the anchor check, L3/FM5 final form (TOURNAMENT_007 §4.4.7;
BUILD_SPEC_007 §5.3 check 3).

Surrogate vs REAL engine on the live pre-holdout leg: the same genome + the
same organ-input files through (a) the real replay harness (run_replay_007 +
tilt_adapter) and (b) the chassis surrogate's paired walk. Registered criteria:

  per-day paired-delta Pearson(dr_surr, dr_real) >= 0.8
  AND real-on-surrogate OLS slope in [0.5, 2.0]
  AND |mean(dr_surr) - mean(dr_real)| <= 2 x se_real          (mean equivalence)

Failure = stop-and-fix FINDING (never a tweak). Slope in [0.5,0.7] u [1.3,2.0]
=> the D4 magnitude caution prints (it must also print beside s_min/lambda_reg
in the champion manifest). Power limits printed (Fisher se ~ 0.23 at n~20).
Runs BEFORE the first production EA generation.

Real-engine side: the wave-1 incumbent replay (runs_battery_007/B0_EXPR/
incumbent) + an ORB-1 arm under a FIXED TEST GENOME over the pre-holdout
window (~21 decision dates; the holdout guard stays unset — pre-holdout is
structurally incapable of touching >= 2026-03-11). The wave-1 SMOKE_TILT
orb1_run1 arm (genome_smoke + nightly_007_smoke organ inputs) is reused as
that arm by default — no new replay-arm budget burned. dr_real from the
cost-adjusted daily series (shared cost seed cancels in the pair).

The registered FINAL form re-runs this exact command with the champion + B0 +
the real organ inputs once wave-2a lands:
  .venv/bin/python anchor_007.py --genome <champion.json> \
      --nightly-dir store/nightly_007 --orb-dir <fresh orb1 replay dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB006_PROTO = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
for p in (str(PROTO), str(REPO), str(TB006_PROTO)):
    if p not in sys.path:
        sys.path.append(p)

import surrogate_007 as SG                                   # noqa: E402
from genome_007 import Genome007                             # noqa: E402
import tilt_adapter as TA                                    # noqa: E402

DEFAULT_INC = PROTO / "runs_battery_007" / "B0_EXPR" / "incumbent"
DEFAULT_ORB = PROTO / "runs_battery_007" / "SMOKE_TILT" / "orb1_run1"
DEFAULT_GENOME = PROTO / "runs_battery_007" / "SMOKE_TILT" / "genome_smoke.json"
DEFAULT_NIGHTLY = PROTO / "store" / "nightly_007_smoke"
PEARSON_MIN = 0.8
SLOPE_BAND = (0.5, 2.0)
SLOPE_CAUTION = ((0.5, 0.7), (1.3, 2.0))


def daily_returns(run_dir: Path, col: str = "cost_adjusted_value") -> pd.Series:
    df = pd.read_csv(Path(run_dir) / "daily_series.csv")
    s = pd.Series(df[col].to_numpy(), index=df["date"].tolist())
    return s.pct_change().dropna()


def real_paired_deltas(inc_dir: Path, orb_dir: Path,
                       col: str = "cost_adjusted_value") -> pd.Series:
    r_inc = daily_returns(inc_dir, col)
    r_orb = daily_returns(orb_dir, col)
    common = [d for d in r_inc.index if d in set(r_orb.index)]
    assert all(d < SG.HOLDOUT_START for d in common), "anchor touches holdout"
    return (r_orb.loc[common] - r_inc.loc[common]).astype(float)


def surrogate_deltas(genome: Genome007, nightly_dir: Path,
                     packs_dir: Path = SG.STORE,
                     masks=None) -> pd.Series:
    p = Path(packs_dir) / "pack_live.pkl"
    if not p.exists():
        raise SystemExit(f"{p} missing — `surrogate_007.py --build-packs` first")
    with open(p, "rb") as fh:
        pack = pickle.load(fh)
    w = SG.walk_fold_paired(pack, genome, nightly_dir,
                            masks or TA.PRODUCTION_MASKS)
    return pd.Series(w["dr_gross"] - w["cost1"] * 1.0, index=w["dates"]), w


def anchor_stats(dr_surr: pd.Series, dr_real: pd.Series) -> dict:
    common = [d for d in dr_real.index if d in set(dr_surr.index)]
    s = dr_surr.loc[common].to_numpy(dtype=float)
    r = dr_real.loc[common].to_numpy(dtype=float)
    n = len(common)
    out = {"n_paired_deltas": n, "dates": [common[0], common[-1]] if n else None}
    if n < 3:
        out["verdict"] = "UNDETERMINABLE (too few paired deltas)"
        return out
    sd_s, sd_r = s.std(ddof=1), r.std(ddof=1)
    pearson = (float(np.corrcoef(s, r)[0, 1])
               if sd_s > 0 and sd_r > 0 else
               (1.0 if np.allclose(s, r) else 0.0))
    if sd_s > 0:
        slope = float(np.polyfit(s, r, 1)[0])      # real on surrogate
    else:
        slope = float("nan")
    se_real = float(sd_r / np.sqrt(n))
    mean_gap = float(abs(s.mean() - r.mean()))
    deg = (sd_s == 0 and sd_r == 0)                # both identically zero
    checks = {
        "pearson": round(pearson, 4),
        "pearson_pass": bool(pearson >= PEARSON_MIN) or deg,
        "slope_real_on_surr": (round(slope, 4) if np.isfinite(slope) else None),
        "slope_pass": bool(np.isfinite(slope)
                           and SLOPE_BAND[0] <= slope <= SLOPE_BAND[1]) or deg,
        "mean_surr": float(s.mean()), "mean_real": float(r.mean()),
        "mean_gap": mean_gap, "se_real_2x": 2 * se_real,
        "mean_equivalence_pass": bool(mean_gap <= 2 * se_real),
        "degenerate_both_zero": deg,
    }
    caution = (np.isfinite(slope)
               and any(lo <= slope <= hi for lo, hi in SLOPE_CAUTION))
    checks["d4_magnitude_caution"] = bool(caution)
    passed = (checks["pearson_pass"] and checks["slope_pass"]
              and checks["mean_equivalence_pass"])
    # diagnostics (never gates): rank corr + leave-one-out pearson range
    if sd_s > 0 and sd_r > 0:
        checks["spearman_diag"] = round(float(
            pd.Series(s).rank().corr(pd.Series(r).rank())), 4)
        loo = []
        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False
            if s[m].std() > 0 and r[m].std() > 0:
                loo.append(float(np.corrcoef(s[m], r[m])[0, 1]))
        checks["pearson_loo_range_diag"] = [round(min(loo), 4),
                                            round(max(loo), 4)]
    out.update(checks)
    out["verdict"] = "PASS" if passed else "FAIL (stop-and-fix finding)"
    out["power_limits"] = (f"Fisher se ~ {1/np.sqrt(max(n-3,1)):.2f} at n={n}; "
                           f"a true rho 0.5 passes rarely — pre-registered "
                           f"power limitation, printed not hidden")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inc-dir", default=str(DEFAULT_INC))
    ap.add_argument("--orb-dir", default=str(DEFAULT_ORB))
    ap.add_argument("--genome", default=str(DEFAULT_GENOME))
    ap.add_argument("--nightly-dir", default=str(DEFAULT_NIGHTLY))
    ap.add_argument("--packs-dir", default=str(SG.STORE))
    ap.add_argument("--out", default=str(SG.STORE / "anchor_007.json"))
    args = ap.parse_args(argv)

    genome = Genome007.from_json(Path(args.genome))
    orb_manifest = json.loads((Path(args.orb_dir) / "manifest.json").read_text())
    assert orb_manifest["genome_hash"] == genome.hash(), \
        ("anchor genome does not match the real-engine ORB arm's genome "
         f"({orb_manifest['genome_hash']} != {genome.hash()})")
    nightly_match = (str(Path(args.nightly_dir).resolve())
                     == str(Path(orb_manifest["nightly_dir"]).resolve()))

    dr_real = real_paired_deltas(Path(args.inc_dir), Path(args.orb_dir))
    (dr_surr, walk) = surrogate_deltas(genome, Path(args.nightly_dir),
                                       Path(args.packs_dir))
    stats = anchor_stats(dr_surr, dr_real)
    # decomposition diagnostic: the same comparison against the RAW (pre
    # cost-overlay) real series isolates the seeded cost-draw noise — both
    # arms walk independent rng sequences over different action lists, an
    # irreducible noise floor in the cost-adjusted paired deltas
    dr_real_raw = real_paired_deltas(Path(args.inc_dir), Path(args.orb_dir),
                                     col="raw_value")
    stats_raw = anchor_stats(dr_surr, dr_real_raw)
    stats["raw_value_decomposition_diag"] = {
        k: stats_raw.get(k) for k in
        ("pearson", "slope_real_on_surr", "mean_gap", "se_real_2x")}

    organ_manifest_synthetic = False
    for f in sorted(Path(args.nightly_dir).glob("*.json"))[:1]:
        organ_manifest_synthetic = bool(
            json.loads(f.read_text()).get("manifest", {}).get("synthetic"))

    doc = {
        "check": "L3/FM5 anchor — surrogate vs real engine, pre-holdout leg",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "criteria": {"pearson_min": PEARSON_MIN, "slope_band": SLOPE_BAND,
                     "mean_equivalence": "<= 2 x se_real"},
        "genome_hash": genome.hash(), "genome": genome.to_dict(),
        "real_engine": {"incumbent_dir": args.inc_dir,
                        "orb_dir": args.orb_dir,
                        "nightly_dir_matches_orb_arm": nightly_match},
        "organ_inputs_synthetic": organ_manifest_synthetic,
        "leg_note": ("fixed-test-genome certification leg; the registered "
                     "final form re-runs with champion + B0 + real organ "
                     "inputs before the first production EA generation"
                     if organ_manifest_synthetic else
                     "registered final form (champion/B0 + real organs)"),
        "surrogate_walk": {"n_days": walk["n_days"],
                           "n_active": walk["n_active"],
                           "neutral_reasons": walk["neutral_reasons"]},
        "anchor": stats,
        "surrogate_space_note": "surrogate side is (surrogate space)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, default=str))
    SG.write_manifest({"anchor_l3_fm5": stats,
                       "anchor_artifact": str(args.out)})
    print(json.dumps(doc["anchor"], indent=1, default=str))
    print(f"-> {args.out}")
    return doc


if __name__ == "__main__":
    main()
