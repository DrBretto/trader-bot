"""PKT-TB-006 — ALL E1 primary organ reads (TOURNAMENT §4.3) -> evidence/e1_reads.json.

Pooled F1-F6 with-vs-without daily-utility difference on identical dates,
HAC t (Newey-West, 10 lags), cost scenario 1.0 — e1_reads.run_e1 per organ,
with the FROZEN SYN-1 base side (FREEZE_SYN1.md): champion genome, the linear
twin executive gate (exec_out LOFO twins), trailing-21 sigma (the deployed
convention), shipping member OOFs (gbm_cond = uniform, event_head = R3-only).

Reads produced (organ -> with vs without):
  transformer   base vs ridge-in-slot (cast -> ridge_twin, exec_out_rt2)
  ensemble_*    base vs member OFF + trust renorm (genome member_gate; K17)
  risknet       risknet-sigma side vs base — DIRECTION: trailing21 is the
                DEPLOYED default, so the contrast is "RiskNet candidate MINUS
                deployed trailing21"; positive t = RiskNet would ADD
  executive     base (linear twin) vs equal-trust + fixed f=0.7 bypass (R08)
  evolution     champion genome vs B0 DEFAULT_GENOME (B2 read)
  llm           base vs _llm_neutral retrain set (exec_out_rt3, LLM_block off)
  gdelt         base vs _gdelt_ablated retrain set (exec_out_rt4, G1+G4G5 off)
  infotropy_b   base vs _uniform set (cast_uniform, uniform w_rec, exec_out_rt5)
  infotropy_a   base (R3-only screened EventHead) vs no-screen EventHead twin —
                FOLD-LEVEL ONLY per D5 (E1-only, no replay, no exec re-fit)

E2 is sign-confirmation only and comes later from the holdout replays; the
"e1_provisional_tag" here is stats.organ_verdict with E2 = None (mechanical:
t >= 2 without E2 prints indeterminate until the holdout sign lands).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))

import battery            # noqa: E402
import e1_reads as e1     # noqa: E402
import ea                 # noqa: E402
import folds as fd        # noqa: E402
import stats as st        # noqa: E402
import synth_oof as so    # noqa: E402

OOF = PROTO / "oof"
OUT = PROTO / "evidence" / "e1_reads.json"
FOLDS = [1, 2, 3, 4, 5, 6]


def log_look(component: str, decision: str, provenance: str) -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "component": component, "decision": decision,
             "provenance": provenance, "phase": "D-battery-build"}
    with (PROTO / "validation_looks.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> dict:
    t0 = time.time()
    champion = ea.Genome.from_json(PROTO / "ea" / f"genome_{fd.FITNESS_END}.json")
    world = so.world_from_panel(PROTO / "store" / "panel.npz")

    # ---- engine memoization: identical (oof_map, exec_dir, sigma, w_rec,
    # bypass) sides share one assembled walk engine -----------------------------
    orig_build = e1.build_side
    cache: dict = {}

    def memo_build(side, world_, oof_dir, fold_list):
        key = (tuple(sorted((side.oof_member_map or {}).items())),
               str(side.exec_dir), side.sigma_source, side.uniform_w_rec,
               bool(side.bypass))
        if key not in cache:
            cache[key] = orig_build(side, world_, oof_dir, fold_list)
        return cache[key]

    e1.build_side = memo_build

    def S(label, **kw):
        return e1.E1Side(label=label, genome=kw.pop("genome", champion), **kw)

    base = lambda: S("syn1_frozen_base", exec_dir=PROTO / "exec_out")  # noqa: E731

    reads_spec = [
        ("transformer", base(),
         S("ridge_in_slot_rt2", oof_member_map={"cast": "ridge_twin"},
           exec_dir=PROTO / "exec_out_rt2"),
         "CAST-Small qua transformer: identical features through the ridge twin "
         "+ RT-2 executive re-fit (R03 primary)"),
        ("ensemble_cast_drop", base(),
         S("cast_off_renorm", genome=battery.drop_genome(champion, "cast"),
           exec_dir=PROTO / "exec_out"),
         "CAST OFF, trust renormalized (R04; K17 — no retrain)"),
        ("ensemble_gbm_drop", base(),
         S("gbm_off_renorm", genome=battery.drop_genome(champion, "gbm_cond"),
           exec_dir=PROTO / "exec_out"),
         "GBM-Cond OFF, trust renormalized (R05)"),
        ("ensemble_event_drop", base(),
         S("event_off_renorm", genome=battery.drop_genome(champion, "event_head"),
           exec_dir=PROTO / "exec_out"),
         "EventHead OFF, trust renormalized (R06)"),
        ("risknet",
         S("risknet_sigma_candidate", exec_dir=PROTO / "exec_out",
           sigma_source="risknet"),
         base(),
         "DIRECTION: trailing-21 is the DEPLOYED default (FREEZE) — read = "
         "RiskNet-sigma candidate MINUS deployed trailing-21; positive t = "
         "the E4 instrument would ADD over the deployed proxy (R07)"),
        ("executive", base(),
         S("equal_trust_f0.7", bypass={"equal_trust": True, "f_fixed": 0.7}),
         "linear-twin executive (frozen gate) vs equal-trust tau=1/M + fixed "
         "f=0.7, same rails (R08); diversity-floor caveat: solo-book corr "
         "0.941 >= 0.90 -> equal-trust tie expected 'by construction' (§4.6.5)"),
        ("evolution", base(),
         S("b0_default_genome", genome=ea.Genome.b0(),
           exec_dir=PROTO / "exec_out"),
         "champion genome vs B0 DEFAULT_GENOME (R09; the EA proposal's B2 read)"),
        ("llm", base(),
         S("llm_neutral_rt3",
           oof_member_map={"gbm_cond": "gbm_cond_llm_neutral",
                           "event_head": "event_head_llm_neutral"},
           exec_dir=PROTO / "exec_out_rt3", feature_gate_off=("LLM_block",)),
         "all llm_* at neutral constants (masks 0) + RT-3 executive re-fit + "
         "LLM_block z gated (R10 primary)"),
        ("gdelt", base(),
         S("gdelt_ablated_rt4",
           oof_member_map={"gbm_cond": "gbm_cond_gdelt_ablated",
                           "event_head": "event_head_gdelt_ablated"},
           exec_dir=PROTO / "exec_out_rt4",
           feature_gate_off=("G1_themes", "G4G5_novelty_conc")),
         "G1-G5 blocks ablated everywhere + RT-4 executive re-fit (R11); "
         "champion already gates G1_themes+G3_tone off on BOTH sides"),
        ("infotropy_b", base(),
         S("uniform_weights_rt5", oof_member_map={"cast": "cast_uniform"},
           exec_dir=PROTO / "exec_out_rt5", uniform_w_rec=True),
         "Transfer-B uniform side: uniform CAST twin + uniform w_rec in the "
         "executive (RT-5). Shipping GBM is ALREADY uniform (freeze §9.2) so "
         "the contrast is exactly {CAST record-weighting, executive w_rec "
         "loss weighting, trust-tilt record weighting} (R12)"),
        ("infotropy_a", base(),
         S("event_head_noscreen",
           oof_member_map={"event_head": "event_head_noscreen"},
           exec_dir=PROTO / "exec_out"),
         "Transfer-A: R3-only screened EventHead (SHIPS per freeze; "
         "conjunctive gate DEAD) vs NO-SCREEN twin — FOLD-LEVEL ONLY per D5 "
         "(E1-only, no replay, base executive, no re-fit)"),
    ]

    reads: dict = {}
    for organ, side_w, side_wo, note in reads_spec:
        ts = time.time()
        res = e1.run_e1(side_w, side_wo, world, OOF, FOLDS)
        res["organ"] = organ
        res["note"] = note
        res["fold_level_only"] = organ == "infotropy_a"
        res["e1_provisional_tag"] = st.organ_verdict(res["hac_t"], None)
        res["mean_bp_day"] = (res["mean"] * 1e4 if np.isfinite(res["mean"])
                              else None)
        res["wall_s"] = round(time.time() - ts, 1)
        reads[organ] = res
        print(f"[{organ}] n={res['n']} mean={res['mean']:+.3e} "
              f"({res['mean'] * 1e4:+.3f} bp/d) HAC t={res['hac_t']:+.2f} "
              f"CI95=[{res['ci95'][0]:+.2e},{res['ci95'][1]:+.2e}] "
              f"MDE={res['mde']:.2e} ({res['wall_s']}s)", flush=True)

    e1.build_side = orig_build
    doc = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "spec": {
            "statistic": "pooled F1-F6 OOF daily utility difference on "
                         "identical dates, HAC t (Newey-West, 10 lags), cost "
                         "scenario 1.0 (TOURNAMENT §4.3 primary read)",
            "base_side": {
                "genome": "EA champion ea/genome_2026-02-06.json (FROZEN)",
                "executive_gate": "linear twin (FROZEN rung; LOFO LinearGate "
                                  "twins exec_out/lofo_twin_fold<f>.pt — E1 "
                                  "walks the same gate class E2 deploys)",
                "sigma_source": "trailing21 (FROZEN deployed convention)",
                "member_oofs": "shipping set: cast (record-weighted), gbm_cond "
                               "(uniform per freeze), event_head (R3-only), "
                               "risknet instrument",
            },
            "rt_exec_rungs": {t: json.loads(
                (PROTO / f"exec_out_{t}" / "ladder.json").read_text())["ships"]
                for t in ("rt2", "rt3", "rt4", "rt5")},
            "verdict_rule": "POSITIVE: E1 HAC t >= +2 AND E2 holdout mean "
                            "delta >= 0; ZERO: |t| < 1; NEGATIVE: t <= -2; "
                            "else INDETERMINATE (with CI + MDE printed). E2 "
                            "tags pending the holdout replays.",
        },
        "reads": reads,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    log_look("e1_reads_all_organs",
             "ALL §4.3 E1 primary reads computed with the FROZEN base side "
             "(champion genome, linear-twin LOFO gate, trailing-21 sigma): "
             + "; ".join(f"{k}: t={v['hac_t']:+.2f}" for k, v in reads.items())
             + ". R07 direction = risknet-candidate MINUS deployed-trailing21. "
               "infotropy_a is fold-level only (D5). E2 sign-confirmations "
               "pending holdout.",
             str(OUT))
    print(f"-> {OUT} ({doc['wall_clock_s']}s)")
    return doc


if __name__ == "__main__":
    main()
