"""PKT-TB-007 Phase D — scorecard + BAKEOFF/ATTRIBUTION assembly (mechanical).

Reads runs_battery_007/, gate artifacts, ledgers; writes evidence_007/,
../BAKEOFF_007.md, ../ATTRIBUTION_007.md, evidence_007/scorecard_007.json.
All verdicts come from the pure functions in report_evidence_007/stats —
numbers in, cells out.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import stats as st
import report_evidence_007 as RE

PROTO = Path(__file__).resolve().parent
RUNDIR = PROTO.parent                      # runs/pkt_tb_007_orthogonal_brain
RUNS = PROTO / "runs_battery_007"
EVID = PROTO / "evidence_007"
DEV = RE.DEV_LABEL

REG_DEGENERATE_CELL = ("TIES (degenerate: brain ships neutral; "
                       "evolution = instrument-failed)")
REG_TIES_CLAIM = ('registered TIES (straddling) permitted claim, quoted: '
                  '"no certifiable difference at available power; MDE '
                  '⟨printed⟩" — here the pair is DEGENERATE: paired '
                  'deltas identically zero (sha256-equal series), MDE n/a.')


def main():
    t0 = dt.datetime.now()
    EVID.mkdir(exist_ok=True)

    # ---------------- comparisons -------------------------------------------
    cmps = {}
    cmps["REG_D02_vs_D01"] = RE.write_cmp(
        "REG_D02_vs_D01", "D02_ORB_B0", "D01_INC",
        "ORB-1@B0", "incumbent",
        "**REGISTERED VERDICT PAIR** (FREEZE_ORB1 §1: B0 ships by the anchor "
        "remediation rule; hash equality asserted by run_battery_007).",
        extra_note="sha256(timeline.json), sha256(daily_series.csv) (raw AND "
        "cost-adjusted columns), sha256(cost_overlay.json) all EQUAL "
        "(runs_battery_007/battery_checks_007.json). Paired deltas are "
        "identically zero; every cell below is degenerate by construction.")
    cmps["DEV_D03_vs_D01"] = RE.write_cmp(
        "DEV_D03_vs_D01", "D03_APRIORI", "D01_INC",
        f"ORB-1@apriori {DEV}", "incumbent",
        f"**DEVIATION PAIR** {DEV} — chair adjudication 3; value-blind "
        "a-priori genome (tilt_gain 0.5, trust {M1:+1}, disp_gain 1.0, "
        "dead_zone 0.05, caps at B0 mids). NEVER replaces the registered verdict.",
        extra_note="D02 ≡ D01 bit-for-bit, so this comparison IS the "
        "D03-vs-D02 M1 utility-attribution read (roster = [M1]). Seed 4242, "
        "cost-adjusted, identical dates.")
    cmps["DEV_D04_vs_D03"] = RE.write_cmp(
        "DEV_D04_vs_D03", "D04_M2IN", "D03_APRIORI",
        f"M2-in {DEV}", f"apriori {DEV}",
        f"**CHALLENGER-IN ARM (M2)** {DEV} — trust {{M1:+1, M2:+1}}; marginal "
        "contribution of adding M2 (status OPEN, FM6) to the a-priori brain.")
    cmps["DEV_D05_vs_D03"] = RE.write_cmp(
        "DEV_D05_vs_D03", "D05_M4AIN", "D03_APRIORI",
        f"M4A-in {DEV}", f"apriori {DEV}",
        f"**CHALLENGER-IN ARM (M4-A)** {DEV} — event_damp_strength 0.5 "
        "(mid-range, value-blind), trust M4:+1; the adapter's M4 damp pathway "
        "with M4-A exceedance scores.",
        extra_note="INTERPRETATION RAIL: the damp shrinks tilt width on high "
        "p_exceed names. Because the underlying a-priori M1 tilt has a "
        "NEGATIVE point estimate on this window, any width-reduction "
        "mechanically scores positive here; this read cannot separate 'M4 "
        "skill' from 'less of a losing tilt'. Pre-registered multiplicity "
        "line applies: a single positive-looking line is NOT narratable as a "
        "discovery (TOURNAMENT §4.6.5).")
    cmps["DEV_D06_vs_D03"] = RE.write_cmp(
        "DEV_D06_vs_D03", "D06_M5IN", "D03_APRIORI",
        f"M5-in {DEV}", f"apriori {DEV}",
        f"**CHALLENGER-IN ARM (M5)** {DEV} — trust {{M1:+1, M5:+1}}, rule ON.",
        extra_note="DEGENERATE: M5 fired ZERO episodes in the live window "
        "(store/m5_signal.npz: 0 nonzero days ≥ 2026-01-31); timeline sha256 "
        "equals D03's. M5's marginal utility is unmeasurable on this window — "
        "structural, not a measured zero of the rule itself.")
    cmps["DEV_D07_vs_D03"] = RE.write_cmp(
        "DEV_D07_vs_D03", "D07_RLLM", "D03_APRIORI",
        f"R-LLM width {DEV}", f"apriori {DEV}",
        f"**R-LLM FALSIFIER ARM** (TOURNAMENT §4.4.9) {DEV} — llm_disag 252d-z "
        "→ T_t × clip(1+0.25·tanh z, 0.75, 1.25) on top of the a-priori "
        "genome. KILL pre-registered at E1 paired t < +2.0 vs the "
        "unconditioned arm; expected effect O(0.1–0.5) bp/day; expected "
        "outcome: not promoted.")
    for rid, seed in (("D08_SEED4243", 4243), ("D09_SEED4244", 4244)):
        cmps[f"DEV_{rid}_vs_D03"] = RE.write_cmp(
            f"DEV_{rid}_vs_D03", rid, "D03_APRIORI",
            f"apriori@seed{seed} {DEV}", f"apriori@4242 {DEV}",
            f"**COST-SEED SENSITIVITY** {DEV} — same config, cost seed {seed}. "
            "Raw series + timeline asserted byte-identical to D03; only the "
            "cost overlay moves.")

    # ---------------- exposure parity (deviation pair) -----------------------
    parity = RE.gross_beta_gap("D03_APRIORI", "D01_INC")

    # ---------------- verdicts (pure functions) ------------------------------
    reg_full = cmps["REG_D02_vs_D01"]["full_period"]["paired_daily"]
    reg_hold = cmps["REG_D02_vs_D01"]["holdout_only"]["paired_daily"]
    reg_degenerate = (reg_full["n"] > 0 and reg_full["sd"] == 0.0
                      and reg_full["mean"] == 0.0)
    assert reg_degenerate, "registered pair NOT degenerate — hash check should have caught this"

    dv_full = cmps["DEV_D03_vs_D01"]["full_period"]["paired_daily"]
    dv_hold = cmps["DEV_D03_vs_D01"]["holdout_only"]["paired_daily"]
    dev_cell = RE.verdict_cell_007(dv_full["t"], dv_hold["mean"],
                                   dv_hold["ci90"], dv_full["ci90"])

    def organ_read(cmp, degenerate=False, reason=""):
        p = cmp["full_period"]["paired_daily"]
        h = cmp["holdout_only"]["paired_daily"]
        v = RE.organ_verdict_007(p.get("hac_t", float("nan")), p["mean"],
                                 p["ci95"], p["mde"],
                                 degenerate=degenerate, degenerate_reason=reason)
        return {"verdict": v,
                "full": {k: p[k] for k in ("n", "mean_bp_day", "t", "hac_t",
                                           "sd_bp_day")} | {"mde_bp": p["mde"] * 1e4},
                "holdout": {k: h[k] for k in ("n", "mean_bp_day", "t", "hac_t")}}

    m1 = organ_read(cmps["DEV_D03_vs_D01"])
    m2 = organ_read(cmps["DEV_D04_vs_D03"])
    m4 = organ_read(cmps["DEV_D05_vs_D03"])
    m5 = organ_read(cmps["DEV_D06_vs_D03"], degenerate=True,
                    reason="M5 fired 0 episodes in the live window")
    rllm = cmps["DEV_D07_vs_D03"]["full_period"]["paired_daily"]
    rllm_killed = not (np.isfinite(rllm["t"]) and rllm["t"] >= 2.0)

    # ---------------- receipts ----------------------------------------------
    acc = json.loads((PROTO / "gates" / "acceptance_007.json").read_text())["gates"]
    ship = json.loads((PROTO / "ea" / "ship_decision_007.json").read_text())
    c2 = json.loads((PROTO / "designs" / "c2_gate_matrix.json").read_text())
    sig_pooled = max(abs(v["signal"]["pooled"])
                     for v in c2["directional_pairs"].values())
    m1g, m2g, m3g, m4g, m5g, m6g = (acc[k] for k in
                                    ("M1", "M2", "M3", "M4", "M5", "M6"))

    looks_battery = RE.count_lines(RE.LOOKS_LEDGER)
    arms = json.loads((RE.LEDGERS / "replay_arms.json").read_text())
    looks_reg = json.loads((RE.LEDGERS / "looks_holdout.json").read_text())

    scorecard = {
        "generated": t0.isoformat(timespec="seconds"),
        "registered_verdict": {
            "cell": REG_DEGENERATE_CELL,
            "claim": REG_TIES_CLAIM,
            "paired_full": reg_full, "paired_holdout": reg_hold,
            "hash_equality": json.loads(
                (RUNS / "battery_checks_007.json").read_text())["checks"],
        },
        "deviation_verdict": {
            "label": DEV,
            "cell": dev_cell["cell"], "claim": dev_cell["claim"],
            "paired_full": dv_full, "paired_holdout": dv_hold,
            "delta_sharpe_full": cmps["DEV_D03_vs_D01"]["full_period"]["delta"]["sharpe"],
            "delta_sharpe_holdout": cmps["DEV_D03_vs_D01"]["holdout_only"]["delta"]["sharpe"],
            "delta_return_full": cmps["DEV_D03_vs_D01"]["full_period"]["delta"]["total_return"],
            "delta_return_holdout": cmps["DEV_D03_vs_D01"]["holdout_only"]["delta"]["total_return"],
            "exposure_parity": parity,
        },
        "organs": {
            "M1": {"roster": "SHIPS (sole member)",
                   "acceptance": {"weekly_rank_ic": m1g["existence"]["weekly_rank_ic"],
                                  "ic_delta_vs_member_zero": m1g["existence"]["ic_delta_vs_member_zero"],
                                  "net_edge": m1g["materiality"]["net_edge_ratio_masked_core"],
                                  "label": "near-formality; does not count as evidence"},
                   "utility_read": {**m1, "label": DEV,
                                    "note": "D03-vs-D02 ≡ D03-vs-D01 (D02 hash-equal)"}},
            "M2": {"roster": "challenger, status OPEN (FM6)",
                   "acceptance": {"rot_rank_ic": m2g["existence"]["rot_rank_ic_16d"],
                                  "t": m2g["existence"]["t"],
                                  "power_note": "29% power at half-decay — 'not measurable yet'"},
                   "utility_read": {**m2, "label": DEV}},
            "M3": {"roster": "B-disp constant (no gene)",
                   "gate": {"existence_p": 2.054710943195078e-11,
                            "materiality_bp_day": m3g["materiality"]["conversion_bp_day"],
                            "bar": 0.2, "pass": False},
                   "note": "dispersion enters conviction as the B-disp sigmoid; "
                           "utility line pre-printed `indeterminate` (TR §1.2)"},
            "M4": {"roster": "challenger (M4-A shipped variant)",
                   "acceptance": {"auc": m4g["existence"]["pooled_auc_A"],
                                  "delta_vs_vol_control": m4g["existence"]["delta_vs_control"],
                                  "ab_look": "M4-A (dAUC(B−A) −.020 < bar +.011; 1 ledger look)"},
                   "utility_read": {**m4, "label": DEV,
                                    "caution": "damping a losing tilt mechanically scores "
                                               "positive; cannot separate M4 skill from "
                                               "less-of-a-losing-tilt; multiplicity §4.6.5"}},
            "M5": {"roster": "DISABLED (12/19 sign tally, p=.18, tally printed)",
                   "utility_read": {**m5, "label": DEV}},
            "M6": {"line": "forecast-altitude PASS — pooled OOF 1d rank-IC "
                           f"{m6g['forecast_falsifier']['pooled_1d_rank_ic']:.4f} "
                           "(bar 0.02), sign-positive 6/6 folds; structurally "
                           "unresolvable at this T_max — forecast-altitude "
                           "verdict only (no expression channel, no gene, no "
                           "replay arm)"},
            "LLM": {"line": "retired from the decision path with receipts (46 "
                            "pre-registered screens, 0 BH-FDR(10%) survivors); "
                            "monitoring emission stays ($0.14/mo); 900-day "
                            "re-test trigger live",
                    "falsifier": {"label": DEV,
                                  "kill_bar": "E1 paired t < +2.0",
                                  "e1_t": rllm["t"], "e1_hac_t": rllm["hac_t"],
                                  "mean_bp_day": rllm["mean_bp_day"],
                                  "mde_bp_day": rllm["mde"] * 1e4,
                                  "n": rllm["n"],
                                  "expected_zero": "pre-registered expected effect "
                                                   "O(0.1–0.5) bp/day; expected outcome: "
                                                   "not promoted",
                                  "killed": rllm_killed}},
            "evolution": {"line": "instrument-failed: anchor Pearson 0.282 < 0.5 "
                                  "at representative scale ⇒ B0 ships outright "
                                  "(chair adjudication 2, pre-stated); EA "
                                  "production sequence NOT RUN (no certified "
                                  "fitness instrument); ea_cycles 0/3",
                          "ship_decision": ship["decision"]},
            "C2": {"line": f"PASS on the binding signal-space leg (max pooled "
                           f"|rho| {sig_pooled:.3f} ≤ .70); registered G-book "
                           "instrument mechanically inapplicable (saturates on "
                           "the market factor) — pre-registration defect, chair "
                           "adjudication 1; remediation ladder NOT fired; "
                           "market-residualized book diagnostic .498 ≤ .70 printed"},
        },
        "ledgers": {
            "holdout_looks_battery_jsonl": looks_battery,
            "registered_e2_reads": looks_reg,
            "replay_arms_used": len(arms), "replay_cap": 18,
            "retrains_used": RE.count_lines(RE.LEDGERS / "retrains.json") and
                             len(json.loads((RE.LEDGERS / "retrains.json").read_text())),
            "retrain_cap": 9,
            "ea_cycles": "0/3",
            "perm_desc": "CUT — reported not-run (chair adjudication 3); the "
                         "pre-committed forfeit sentence applies: selection "
                         "intelligence (channel 1a) is forfeited on this window "
                         "because it cannot be certified, not because the "
                         "question is settled",
        },
        "multiplicity": {
            "pre_registered": "with 5 per-organ utility arms read one-sided at "
                              "t ≥ +2: P(≥1 spurious positive) ≈ 11%; across all "
                              "~12–14 arms ≈ 25–30% (TOURNAMENT §4.6.5)",
            "realized_deviation_comparisons": 7,
            "note": "no deviation read is averaged with, or substituted for, "
                    "the registered verdict",
        },
        "boundary": {
            "window": [dv_full["first_date"], dv_full["last_date"]],
            "n_full": dv_full["n"], "n_holdout": dv_hold["n"],
            "disclosed_exclusions": "Monday/cadence gaps; 2026-03-30..31; "
                                    "2026-05-11..20 outage hole (7 td, inside "
                                    "the holdout) — identical in both arms",
        },
    }
    (EVID / "scorecard_007.json").write_text(
        json.dumps(scorecard, indent=1, default=float))

    # ---------------- final line --------------------------------------------
    fl_reg = (f"BRAIN+CHASSIS vs INCUMBENT: {REG_DEGENERATE_CELL} by +0.00% "
              f"dReturn, +0.00 dSharpe on holdout (paired deltas identically 0, "
              f"n={reg_hold['n']}; sha256-equal series)")
    fl_dev = (f"DEVIATION READ {DEV}: {dev_cell['cell']} — full n={dv_full['n']} "
              f"mean {dv_full['mean_bp_day']:+.2f} bp/day t={dv_full['t']:+.2f} "
              f"(HAC {dv_full['hac_t']:+.2f}), MDE {dv_full['mde']*1e4:.1f} bp/day; "
              f"holdout n={dv_hold['n']} mean {dv_hold['mean_bp_day']:+.2f} bp/day "
              f"t={dv_hold['t']:+.2f}, dSharpe "
              f"{scorecard['deviation_verdict']['delta_sharpe_holdout']:+.2f}, "
              f"dReturn {scorecard['deviation_verdict']['delta_return_holdout']*100:+.2f}%")
    fl = (f"{fl_reg}; {fl_dev}; "
          f"ORTHOGONALITY: PASS (signal-space max pooled |rho| = "
          f"{sig_pooled:.3f} <= .70; G-book instrument mechanically "
          f"inapplicable — adjudicated defect); "
          f"ORGANS: M1:{m1['verdict']['tag'].replace(' ', '-')}(dev-read; ships, "
          f"acceptance near-formality) "
          f"M2:{m2['verdict']['tag'].replace(' ', '-')}(challenger, OPEN) "
          f"M3:B-disp-constant(materiality FAIL .011 bp/day) "
          f"M4:{m4['verdict']['tag'].replace(' ', '-')}(challenger M4-A; "
          f"+{m4['full']['mean_bp_day']:.2f} bp/day, HAC t "
          f"{m4['full']['hac_t']:+.2f} < +2, width-of-losing-tilt caution) "
          f"M5:disabled(12/19 p=.18; challenger arm degenerate — 0 live episodes) "
          f"M6:forecast-PASS(IC .0224, 6/6) LLM:retired(falsifier killed: "
          f"t={rllm['t']:+.2f} < +2.0, MDE {rllm['mde']*1e4:.2f} bp/day) "
          f"EA:instrument-failed(anchor .282 < .5; B0 shipped; 0/3 cycles); "
          f"COST: $9.50/mo absolute, +$0.21/mo marginal — reductions: PERM-DESC "
          f"cut (not run), EA production not run, $0 new Bedrock. ")
    fl += CAVEAT()
    (EVID / "final_line_007.txt").write_text(fl + "\n")
    print(fl)
    print()
    print(f"evidence written -> {EVID}")
    return scorecard, fl, cmps


def CAVEAT():
    return RE.CAVEAT_SENTENCE


if __name__ == "__main__":
    main()
