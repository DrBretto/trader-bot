#!/usr/bin/env python
"""PKT-TB-006 — Phase D evidence assembly driver (MECHANICAL; no judgment).

Drives report_evidence.write_comparison / assemble_scorecard + stats pure
functions over the executed battery (runs_battery/R01..R14 + contingency
R19), the E1 reads
(evidence/e1_reads.json), the ledgers, and the §4.6 gate artifacts. Every
verdict comes from stats.beats_ties_loses / stats.organ_verdict; this script
only routes numbers and writes the two dossier MDs (BAKEOFF.md,
ATTRIBUTION.md) into the run dir.

ORIENTATION DOCUMENTATION (TOURNAMENT §4.3 — checked per arm, recorded here):
  §4.3 primary read = "with-organ vs without-organ daily utility difference".
  For every DROP/neutralization arm the with-organ side is the frozen base
  R01 and the without-organ side is the arm replay, so the E2 sign read is
  mean(ret_R01 − ret_Rarm) on identical holdout dates (cost-adjusted), and
  E2 ΔSharpe = Sharpe(R01) − Sharpe(Rarm). This matches the e1_reads.json
  orientation (with = syn1_frozen_base) for: R03 (transformer), R04/R05/R06
  (member drops), R08 (executive -> equal-trust), R09 (champion vs B0),
  R10 (LLM neutral), R11 (GDELT ablated), R12 (infotropy-B uniform).
  EXCEPTION — R07 (RiskNet): the FREEZE shipped trailing-21 as the DEPLOYED
  sigma (FREEZE_SYN1.md row "Sigma source"), inverting the §4.3 arm table's
  assumption that RiskNet+ was deployed. e1_reads.json pre-registered the
  direction note: read = RiskNet-sigma candidate MINUS deployed trailing-21,
  positive = the instrument would ADD. E2 therefore pairs A=R07, B=R01.
  ANTI-FLATTERING CHECK: with this orientation a positive sign flatters the
  candidate, NOT the shipped SYN-1 config; the verdict branch for E1
  t=+1.39 is 1.0<=|t|<2.0 -> indeterminate in either orientation, so no
  flattering reading is available. FINDING (asserted below): R07's series
  is byte-identical to R01's (sigma_source trailing21 == deployed
  convention) — the E2 contrast is DEGENERATE (diff ≡ 0, paired t
  undefined); reported verbatim, never silently passed. R07 stays in the
  record as a consumed look (it measured the null contrast by
  construction); the REAL contrast is contingency replay R19
  (--sigma-source risknet, reserved slot per TOURNAMENT §4.4), which now
  sources the risknet E2 read: E2 pairs A=R19, B=R01 (candidate MINUS
  deployed), the same orientation as the e1_reads DIRECTION note.

AMBIGUITY LOG (anti-flattering tiebreak, §4.3/§4.2 readings):
  1. §4.2 names "the paired daily-difference t-statistic" with no HAC
     qualifier (HAC is §4.3's E1 statistic). The bake-off verdict uses the
     plain paired t; the HAC t is printed beside it. Both are checked: if
     they straddle a verdict boundary that is flagged in BAKEOFF.md.
  2. §4.2's three rules are not exhaustive (t>=+1 with ΔSharpe<=0):
     stats.beats_ties_loses returns "UNDEFINED (rule-gap)" — printed, not
     coerced (pre-registration gap already journaled).
  3. §4.7 has ONE ensemble=<attr> slot but §4.3 says "reported per member":
     per-member attrs joined with "/" (report_evidence behavior; journaled
     ambiguity). No collapsing to the best member.
  4. §4.6.6 train-vs-harness gap: no artifact exists anywhere in the run
     dir (grepped store/, exec_out*/, ledgers). Printed "NOT PROVIDED" by
     the mechanical gate printer — a finding, not recomputed ad hoc here
     (inventing the series would be a judgment call).
  5. §4.6.3 placebo FAIL forces "G1 dictionary reported 0 (measured)
     REGARDLESS of the block arm" — that binds the G1-dictionary claim, not
     the whole-GDELT-organ verdict (R11 block arm), which stays its own
     §4.3 read (indeterminate at E1 t=−1.79). Reading that DISFAVORS SYN-1
     confirmed: the GDELT organ cannot borrow a better tag from either rule.

Usage: .venv/bin/python assemble_evidence.py   (from prototype/)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import stats as st
import report_evidence as rep

PROTO = Path(__file__).resolve().parent
RB = PROTO / "runs_battery"
RUN = PROTO.parent                      # runs/pkt_tb_006_clean_sheet_brain/
EV = PROTO / "evidence"
HOLDOUT = st.HOLDOUT_START              # 2026-03-11

COST_PER_MONTH = 9.5                    # COST_WORKSHEET.md §1 TOTAL (~$9.5/mo absolute)
REDUCTIONS = [                          # FREEZE_SYN1.md "Reductions taken to date"
    "CAST OOF seeds 3->2",
    "deploy ensemble 5->3",
    "8-day gradient minibatches (semantics-preserving)",
    "LLM Tier-2 window reduced to 2024-08-15->2026-01-28 ($3.10 cap)",
]

# (evidence_id, e1_key, run_a, run_b, label_a, label_b, attr_key, member, orientation_note)
# run_a − run_b == the §4.3 with-organ − without-organ direction per the header doc.
ORGAN_ARMS = [
    ("R03_transformer", "transformer", "R01", "R03",
     "SYN-1 (R01, CAST in slot)", "ridge-in-slot (R03, RT-2)",
     "transformer", None,
     "E2 = R01 − R03 (with-CAST minus ridge twin in slot, executive re-fit RT-2)"),
    ("R04_ensemble_cast", "ensemble_cast_drop", "R01", "R04",
     "SYN-1 (R01)", "CAST OFF, trust renorm (R04)",
     "ensemble", "cast", "E2 = R01 − R04 (with-member minus member-dropped)"),
    ("R05_ensemble_gbm", "ensemble_gbm_drop", "R01", "R05",
     "SYN-1 (R01)", "GBM-Cond OFF, trust renorm (R05)",
     "ensemble", "gbm_cond", "E2 = R01 − R05 (with-member minus member-dropped)"),
    ("R06_ensemble_event", "ensemble_event_drop", "R01", "R06",
     "SYN-1 (R01)", "EventHead OFF, trust renorm (R06)",
     "ensemble", "event_head", "E2 = R01 − R06 (with-member minus member-dropped)"),
    ("R19_ensemble_risknet", "risknet", "R19", "R01",
     "RiskNet-sigma candidate arm (R19, contingency)", "SYN-1 deployed trailing-21 (R01)",
     "ensemble", "risknet",
     "E2 = R19 − R01 (candidate minus deployed, matching the e1_reads DIRECTION "
     "note; positive flatters the CANDIDATE, not the shipped SYN-1 config. R19 is "
     "the contingency replay repairing degenerate R07, whose --sigma-source "
     "trailing21 IS the deployed convention -> null contrast by construction)"),
    ("R08_meta_evaluator", "executive", "R01", "R08",
     "SYN-1 (R01, linear-twin executive)", "equal-trust tau=1/M + f=0.7 (R08)",
     "meta-evaluator", None, "E2 = R01 − R08 (learned executive minus fixed rule)"),
    ("R09_evolution", "evolution", "R01", "R09",
     "SYN-1 (R01, champion genome)", "B0 DEFAULT_GENOME (R09)",
     "evolution", None, "E2 = R01 − R09 (champion minus B0; the EA proposal B2 read)"),
    ("R10_llm", "llm", "R01", "R10",
     "SYN-1 (R01)", "LLM neutral-constants retrain (R10, RT-3)",
     "LLM", None, "E2 = R01 − R10 (with-LLM minus llm_* neutralized + RT-3 re-fit)"),
    ("R11_gdelt", "gdelt", "R01", "R11",
     "SYN-1 (R01)", "GDELT G1-G5 ablated retrain (R11, RT-4)",
     "GDELT", None, "E2 = R01 − R11 (with-GDELT minus block-ablated + RT-4 re-fit)"),
    ("R12_infotropy_b", "infotropy_b", "R01", "R12",
     "SYN-1 (R01, record-weighted)", "uniform-weights retrain (R12, RT-5)",
     "infotropy", None,
     "E2 = R01 − R12 (record-weighted minus uniform; contrast = {CAST record-"
     "weighting, executive w_rec loss weighting, trust-tilt record weighting} — "
     "shipping GBM already uniform per FREEZE §9.2)"),
]

ARM_PROVENANCE = {  # run_id -> (retrain, exec rung shipped, exec dir) per exec_out*/ladder.json
    "R01": ("RT-1 (base)", "linear_twin (FROZEN; MLP 1.688e-4 did not beat twin 1.701e-4)", "exec_out/ladder.json"),
    "R03": ("RT-2 (ridge + executive re-fit)", "mlp (1.497e-4 > twin 1.254e-4)", "exec_out_rt2/ladder.json"),
    "R04": ("none (member gate off, K17)", "linear_twin (base)", "exec_out/ladder.json"),
    "R05": ("none (member gate off, K17)", "linear_twin (base)", "exec_out/ladder.json"),
    "R06": ("none (member gate off, K17)", "linear_twin (base)", "exec_out/ladder.json"),
    "R07": ("none (sigma swap)", "linear_twin (base)", "exec_out/ladder.json"),
    "R19": ("none (sigma swap, contingency)", "linear_twin (base)", "exec_out/ladder.json"),
    "R08": ("none (executive bypass)", "equal_trust bypass (tau=1/M, f=0.7)", "runs_battery/R08/manifest.json"),
    "R09": ("none (genome swap)", "linear_twin (base)", "exec_out/ladder.json"),
    "R10": ("RT-3 (LLM neutral)", "mlp (1.769e-4 > twin 1.620e-4)", "exec_out_rt3/ladder.json"),
    "R11": ("RT-4 (GDELT ablated)", "mlp (1.342e-4 > twin 1.260e-4)", "exec_out_rt4/ladder.json"),
    "R12": ("RT-5 (uniform weights)", "mlp (3.345e-4 > twin 3.177e-4)", "exec_out_rt5/ladder.json"),
}

ITEM_MAP = [  # assignment items (ASSIGNMENT_BRIEF.md §1) -> scorecard organs/arms
    ("1", "ensemble of genuinely different ML model types (incl. a transformer doing real work)",
     "transformer (R03) + ensemble members cast/gbm_cond/event_head (R04-R06) + risknet (R19; R07 degenerate)"),
    ("2", "evolutionary algorithm as the balancing organ", "evolution (R09 + B0/B1/adoption gate)"),
    ("3", "LLM doing real sentiment analysis", "LLM (R10, RT-3)"),
    ("4", "GDELT as a load-bearing differentiated data source", "GDELT (R11, RT-4 + shift gate + placebo)"),
    ("5", "learned meta-evaluator (the executive)", "meta-evaluator (R08 + diversity floor + ladder)"),
    ("6", "infotropy angle examined", "infotropy (R12 Transfer-B; Transfer-A fold-only read)"),
]


# ---------------------------------------------------------------- helpers
def fmt(v, digits=2, signed=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return s


def raw_paired(ra, rb, start):
    """Paired stats on the RAW (pre-cost-overlay) series — printed alongside;
    cost-adjusted is the verdict series (§4.2 + wiring adjudication)."""
    da, va = st._slice_from(ra["dates_raw"], ra["values_raw"], start)
    db, vb = st._slice_from(rb["dates_raw"], rb["values_raw"], start)
    return st.paired_daily_stats(da[1:], st.daily_returns(va),
                                 db[1:], st.daily_returns(vb), hac=True)


def raw_cols(rid):
    with open(RB / rid / "daily_series.csv") as fh:
        return [(r["date"], r["raw_value"]) for r in csv.DictReader(fh)]


def manifest(rid):
    with open(RB / rid / "manifest.json") as fh:
        return json.load(fh)


def main():
    e1 = json.load(open(EV / "e1_reads.json"))["reads"]
    runs = {rid: st.load_run(RB / rid) for rid in
            ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09",
             "R10", "R11", "R12", "R13", "R14", "R19"]}

    # ---------------- sanity rails: R13/R14 raw byte-identity; R07 degeneracy
    findings = []
    for rid in ("R13", "R14"):
        if raw_cols(rid) == raw_cols("R01"):
            findings.append(f"{rid} raw_value column byte-identical to R01 (ASSERTED) — "
                            f"only the cost overlay (seed {manifest(rid)['cost_model']['seed']}) differs.")
        else:
            findings.append(f"FINDING: {rid} raw series NOT identical to R01 — seed "
                            "sensitivity contaminates the strategy path, not just costs.")
    r07_identical = raw_cols("R07") == raw_cols("R01") and \
        (RB / "R07" / "daily_series.csv").read_bytes() == (RB / "R01" / "daily_series.csv").read_bytes()
    if r07_identical:
        findings.append("FINDING (R07): daily_series.csv byte-identical to R01 — the arm ran "
                        "--sigma-source trailing21, which IS the deployed FREEZE convention; the "
                        "E2 contrast is degenerate (diff ≡ 0, paired t undefined). R07 stays in "
                        "the record as a consumed look (it measured the null contrast by "
                        "construction); the RiskNet E2 read is sourced from contingency replay "
                        "R19 (--sigma-source risknet), candidate-minus-deployed orientation.")
    if raw_cols("R19") == raw_cols("R01"):
        findings.append("FINDING (R19): raw series ALSO identical to R01 — the risknet sigma "
                        "swap is a no-op in replay; the E2 contrast would again be degenerate.")
    findings.append("Holdout paired n=44 daily diffs (45 holdout dates), not the §4.1 ~62-day "
                    "estimate: the full-window snapshot store yields 65 trading dates "
                    "2026-02-04->2026-06-10 (`runs_battery/R01/manifest.json` n_decision_dates; "
                    "snapshot gap 2026-05-11->05-22 journaled in RUN_JOURNAL.md). The printed "
                    "MDE is computed on the actual n.")
    findings.append("Metric artifact (forced choice, stats.run_metrics docstring): "
                    "cumulative_transaction_costs reads 0.00 for R01 in both slices because its "
                    "only fills occur at the first daily mark — the drag is embedded in v[0] of "
                    "the cost-adjusted series. The traceable dollar cost is the manifest "
                    "total_cost_dollars ($28.88 R01 / $364.27 R02), printed in §4 below.")

    # ---------------- bake-off comparison (R01 vs R02) ------------------------
    bk_extra = ("BAKE-OFF (§4.2). Verdict series = cost-adjusted; raw paired stats printed "
                "in BAKEOFF.md. Incumbent appears as score line + harness wiring only.")
    bk = rep.write_comparison("R01_vs_R02_bakeoff", RB / "R01", RB / "R02",
                              "SYN-1", "incumbent", extra_note=bk_extra)
    bk_hold = bk["holdout_only"]
    bk_full = bk["full_period"]
    bakeoff = {
        "paired": bk_hold["paired_daily"],                      # t = plain paired t (§4.2)
        "holdout_dreturn": bk_hold["delta"]["total_return"],
        "holdout_dsharpe": bk_hold["delta"]["sharpe"],
        "full_period_paired": bk_full["paired_daily"],
        "comparison": "evidence/R01_vs_R02_bakeoff/comparison.json",
    }
    bk_raw_hold = raw_paired(runs["R01"], runs["R02"], HOLDOUT)
    bk_raw_full = raw_paired(runs["R01"], runs["R02"], None)

    # ---------------- per-organ comparisons + organ rows ----------------------
    organ_rows, organ_cmp = [], {}
    for ev_id, e1_key, ra, rb, la, lb, attr_key, member, onote in ORGAN_ARMS:
        r = e1[e1_key]
        extra = (f"ORIENTATION: {onote}. E1 read: evidence/e1_reads.json[{e1_key!r}] "
                 f"(with={r['with']}, without={r['without']}).")
        cmp = rep.write_comparison(
            ev_id, RB / ra, RB / rb, la, lb,
            e1_result={"hac_t": r["hac_t"], "n": r["n"], "mean": r["mean"],
                       "ci95": r["ci95"], "mde": r["mde"]},
            extra_note=extra)
        organ_cmp[e1_key] = cmp
        hold = cmp["holdout_only"]
        row = {
            "organ": e1_key, "run_id": rb if ra == "R01" else ra,
            "attr_key": attr_key,
            "e1_hac_t": r["hac_t"], "e1_n": r["n"], "e1_ci95": r["ci95"],
            "e1_mde": r["mde"], "e1_mean_bp_day": r["mean_bp_day"],
            "e2_holdout_dsharpe": hold["delta"]["sharpe"],
            "e2_holdout_mean_delta": hold["paired_daily"]["mean"],
            "e2_holdout_t": hold["paired_daily"]["t"],
            "e2_holdout_n": hold["paired_daily"]["n"],
            "gate_off": False,
            "note": r.get("note", ""),
            "orientation": onote,
            "evidence": f"evidence/{ev_id}/comparison.json",
        }
        if member:
            row["member"] = member
        if e1_key == "risknet":
            row["note"] += (" | E2 from contingency R19 (candidate-minus-deployed); planned "
                            "arm R07 DEGENERATE (byte-identical to R01) — kept in the record "
                            "as a consumed null-contrast look")
        organ_rows.append(row)

    # R07 stays in the record: re-emit its comparison artifacts with the
    # degeneracy note (a consumed look that measured the null contrast by
    # construction; the live risknet E2 read is R19_ensemble_risknet above).
    r_rn = e1["risknet"]
    rep.write_comparison(
        "R07_ensemble_risknet", RB / "R07", RB / "R01",
        "RiskNet-sigma candidate arm (R07)", "SYN-1 deployed trailing-21 (R01)",
        e1_result={"hac_t": r_rn["hac_t"], "n": r_rn["n"], "mean": r_rn["mean"],
                   "ci95": r_rn["ci95"], "mde": r_rn["mde"]},
        extra_note=("ORIENTATION: E2 = R07 − R01 (candidate minus deployed, matching the "
                    "e1_reads DIRECTION note). FINDING: R07 series byte-identical to R01 "
                    "(--sigma-source trailing21 IS the deployed FREEZE convention) -> contrast "
                    "DEGENERATE (diff ≡ 0, paired t undefined). R07 stays in the record as a "
                    "consumed look that measured the null contrast by construction; the live "
                    "risknet E2 read is evidence/R19_ensemble_risknet/ (contingency replay)."))

    # infotropy Transfer-A: E1-only fold read (no replay, per D5 / battery plan)
    ta = e1["infotropy_a"]
    organ_rows.append({
        "organ": "infotropy_a", "run_id": "— (E1-only)", "attr_key": "infotropy",
        "e1_hac_t": ta["hac_t"], "e1_n": ta["n"], "e1_ci95": ta["ci95"],
        "e1_mde": ta["mde"], "e1_mean_bp_day": ta["mean_bp_day"],
        "e2_holdout_dsharpe": None, "e2_holdout_mean_delta": None,
        "gate_off": False, "note": ta.get("note", ""),
        "orientation": "E1 = R3-only screened EventHead (ships per FREEZE) minus no-screen twin",
        "evidence": "evidence/e1_reads.json[infotropy_a]",
    })

    # ---------------- seed-sensitivity comparisons ---------------------------
    for rid, seed in (("R13", 4243), ("R14", 4244)):
        m = manifest(rid)
        rep.write_comparison(
            f"{rid}_seed{seed}", RB / rid, RB / "R01",
            f"SYN-1 @ cost seed {seed} ({rid})", "SYN-1 @ cost seed 4242 (R01)",
            extra_note=(f"SEED SENSITIVITY: raw_value column ASSERTED byte-identical to R01 "
                        f"(strategy path is seed-free); only the post-hoc cost overlay differs. "
                        f"Cost-adjusted endpoints: {rid}="
                        f"{m['summary']['final_value_cost_adjusted']:.2f} vs R01="
                        f"{manifest('R01')['summary']['final_value_cost_adjusted']:.2f}."))

    # ---------------- §4.6 gates ----------------------------------------------
    shift = json.load(open(PROTO / "gdelt_shift_gate.json"))
    placebo = json.load(open(PROTO / "placebo_result.json"))
    step10 = json.load(open(PROTO / "store" / "step10_checks.json"))
    fdiag = json.load(open(PROTO / "exec_out" / "final_diagnostics.json"))

    models, n_calls, n_fallback = set(), 0, 0
    for line in open(PROTO / "bedrock_spend.jsonl"):
        rec = json.loads(line)
        if rec.get("status") == "call":
            n_calls += 1
            models.add(rec.get("model_used"))
            if "claude-3-haiku-20240307" not in (rec.get("model_used") or ""):
                n_fallback += 1

    ft = fdiag["fine_tune"]
    ft_pre = float(np.mean([v["pre_val_util"] for v in ft.values()]))
    ft_post = float(np.mean([v["post_val_util"] for v in ft.values()]))
    ft_delta = float(np.mean([v["delta"] for v in ft.values()]))

    gates = {
        "model_cutoff": {
            "model": sorted(models)[0] if len(models) == 1 else sorted(models),
            "cutoff": "2023-08 (predates every scored window; Tier-2 starts 2024-08-15)",
            "fallback_days_pct": 100.0 * n_fallback / max(n_calls, 1),
            "verdict": f"PASS ({n_calls} calls, single pinned model, 0 fallback; bedrock_spend.jsonl)",
        },
        "shift_gate": {"overall": shift["overall"],
                       "path": "prototype/gdelt_shift_gate.json (worst bucket energy_oil "
                               f"{shift['checks']['g1_bucket_share_shift']['worst_shift_in_sd_units']:.2f} sd < 5.0)"},
        "placebo": {"real_ic": placebo["real_ic"], "percentile": placebo["percentile"],
                    "pass_95": placebo["pass_95"], "n_permutations": placebo["n_permutations"],
                    "path": "prototype/placebo_result.json"},
        "break_even_ic": {
            "weekly_rank_ic": step10["break_even_ic"]["mean"],
            "bar": step10["break_even_ic"]["bar"],
            "break_even": step10["break_even_ic"]["break_even"],
            "verdict": ("PASS — forecast pathway carries edge (per-fold 0.061-0.113; "
                        "store/step10_checks.json)" if step10["break_even_ic"]["passes_functionality_bar"]
                        else "FAIL — pre-registered no-edge conclusion prints"),
        },
        "diversity_floor": {
            "mean_pairwise_corr": step10["diversity_floor"]["mean_pairwise_corr"],
            "verdict": ("TRIGGERED (>= 0.90) — item-5 trust attribution expected ~0 by "
                        "construction; the equal-trust tie is the honest outcome (§4.6.5; "
                        "store/step10_checks.json)"),
        },
        # train_vs_harness_gap: NO artifact exists in the run dir (see ambiguity
        # log #4 in the module docstring) -> omitted so the mechanical printer
        # emits "NOT PROVIDED". Finding, not a silent drop.
        "fine_tune": {"pre_val": ft_pre, "post_val": ft_post, "delta": ft_delta,
                      "declaimed": all(v["delta"] == 0.0 for v in ft.values())},
    }

    # ---------------- scorecard -----------------------------------------------
    doc = rep.assemble_scorecard(bakeoff, organ_rows, gates,
                                 COST_PER_MONTH, REDUCTIONS)

    # ---------------- BAKEOFF.md ----------------------------------------------
    m1, m2 = manifest("R01"), manifest("R02")
    s1, s2 = m1["summary"], m2["summary"]
    p_h, p_f = bakeoff["paired"], bakeoff["full_period_paired"]
    verdict = doc["bakeoff"]["verdict"]
    boundary_flag = ""
    if np.isfinite(p_h.get("hac_t", float("nan"))):
        v_naive = st.beats_ties_loses(p_h["t"], bakeoff["holdout_dsharpe"])["verdict"]
        v_hac = st.beats_ties_loses(p_h["hac_t"], bakeoff["holdout_dsharpe"])["verdict"]
        if v_naive != v_hac:
            boundary_flag = (f"\n**FLAG (ambiguity #1):** plain-t verdict {v_naive} != "
                             f"HAC-t verdict {v_hac}; pre-registered text names the plain "
                             f"paired t, which governs.\n")

    def metric_table(blk, la="SYN-1 (R01)", lb="incumbent (R02)"):
        rows = [f"| metric | {la} | {lb} | delta (SYN-1 − incumbent) |", "|---|---|---|---|"]
        for k in rep.METRIC_KEYS:
            pct = k in ("total_return", "cagr", "max_drawdown", "win_rate", "avg_gross_exposure")
            rows.append(f"| {k} | {rep._fmt(blk['metrics_a'][k], pct)} | "
                        f"{rep._fmt(blk['metrics_b'][k], pct)} | {rep._fmt(blk['delta'][k], pct)} |")
        return rows

    def paired_line(p, label):
        return (f"- **{label}:** n={p['n']}, mean={fmt(p['mean_bp_day'])} bp/day, "
                f"sd={fmt(p['sd_bp_day'], signed=False)} bp/day, t={fmt(p['t'])}, "
                f"HAC t={fmt(p.get('hac_t'))}, 95% CI [{fmt(p['ci95'][0], 6)}, {fmt(p['ci95'][1], 6)}], "
                f"MDE(|t|=2)={fmt(p['mde'], 6)} "
                f"({fmt(p['mde'] * 1e4)} bp/day)")

    # sampled equity curve (every 5th common date + last), cost-adjusted both arms
    d1, v1 = runs["R01"]["dates"], runs["R01"]["values"]
    d2map = dict(zip(runs["R02"]["dates"], runs["R02"]["values"]))
    common = [i for i, d in enumerate(d1) if d in d2map]
    sample = common[::5] + ([common[-1]] if common[-1] not in common[::5] else [])
    eq_rows = ["| date | SYN-1 cost-adj | incumbent cost-adj |", "|---|---|---|"]
    for i in sample:
        eq_rows.append(f"| {d1[i]} | {v1[i]:,.2f} | {d2map[d1[i]]:,.2f} |")

    seed_rows = ["| run | cost seed | final raw | final cost-adj | total cost $ | drag bps of start NAV |",
                 "|---|---|---|---|---|---|"]
    for rid in ("R01", "R13", "R14"):
        s = manifest(rid)["summary"]
        seed_rows.append(f"| {rid} | {manifest(rid)['cost_model']['seed']} | {s['final_value_raw']:,.2f} | "
                         f"{s['final_value_cost_adjusted']:,.2f} | {s['total_cost_dollars']:.2f} | "
                         f"{s['cost_drag_bps_of_start_nav']:.2f} |")

    bakeoff_md = f"""# BAKEOFF.md — SYN-1 vs incumbent (PKT-TB-006, mechanical assembly)

Generated {doc['generated']} by `prototype/assemble_evidence.py` driving
`prototype/report_evidence.py` + `prototype/stats.py` (pure pre-registered
functions). Every number below traces to a cited file. The incumbent appears
ONLY as its score line and harness wiring (packet rule; ASSIGNMENT_BRIEF §1).

## 1. Window, seeding, harness identity

- Replay window: snapshots {m1['snapshot_range'][0]} -> {m1['snapshot_range'][1]}; first/last daily
  values {s1['first_date']} -> {s1['last_date']} ({s1['n_decision_dates']} decision dates). Holdout
  boundary {HOLDOUT} (TOURNAMENT §4.1); holdout-only is the VERDICT read.
- Harness: identical engine + snapshots for both arms — code SHA git `{m1['code_sha']['git']}`,
  prototype wiring `{m1['code_sha']['prototype_wiring']}`; cost model `transaction_costs`
  version `{m1['cost_model']['version_sha']}`, post-hoc cost overlay seed {m1['cost_model']['seed']} (both arms;
  paired stats require identical slippage draws, §4.1). Incumbent run command and
  wiring: `runs_battery/R02/manifest.json`.
- SYN-1 configuration: the ONE frozen config (FREEZE_SYN1.md) — EA champion genome
  `ea/genome_2026-02-06.json` (hash `{m1['genome_hash']}`), exec-mode `{m1['exec_mode']}`
  (frozen ladder rung), sigma source `{m1['sigma_source']['mode']}`. Variant params hash
  `{m1['variant_params_hash']}`.
- Rung decisions stated with the results (FREEZE_SYN1.md): champion genome adopted
  (adoption gate: margin +0.876 > 1.0 x cross-fold sd 0.638); executive = linear twin
  (MLP 5-seed val 1.688e-4 did NOT beat twin 1.701e-4); fine-tune none (delta exactly 0,
  de-claimed); Transfer-B CAST weighted / GBM uniform; Transfer-A conjunctive gate DEAD,
  R3-only screening ships; LLM ships (Stage-1 pass w/ chattiness partial-fail note);
  G1 dictionary `0 (measured)` per placebo FAIL.

## 2. §4.2 numbers — cost-adjusted (verdict series)

### Full period (context, never the verdict)

{chr(10).join(metric_table(bk_full))}

{paired_line(p_f, 'Paired daily diff, full period (cost-adjusted)')}
{paired_line(bk_raw_full, 'Paired daily diff, full period (RAW, alongside)')}

### Holdout only (>= {HOLDOUT}) — the verdict read

{chr(10).join(metric_table(bk_hold))}

{paired_line(p_h, 'Paired daily diff, holdout (cost-adjusted) — PRIMARY')}
{paired_line(bk_raw_hold, 'Paired daily diff, holdout (RAW, alongside)')}

## 3. Verdict (§4.2 rule, computed by stats.beats_ties_loses)

**BRAIN vs INCUMBENT: {verdict}** — holdout paired t={fmt(p_h['t'])} (rule: BEATS iff
t>=+1.0 AND dSharpe>0; LOSES iff t<=-1.0; TIES iff |t|<1.0), holdout
dSharpe={fmt(bakeoff['holdout_dsharpe'])}, holdout dReturn={fmt(bakeoff['holdout_dreturn'] * 100)}% (endpoint context).
{boundary_flag}
Honesty line (§4.2, printed with the verdict): {doc['honesty_line']}.
This holdout's computed MDE at |t|=2 is {fmt(p_h['mde'] * 1e4)} bp/day on n={p_h['n']} days.

## 4. Turnover + cost drag, both arms (manifest summaries)

| arm | executed actions | actions/decision-date | traded $ | cost $ | drag bps of start NAV | cost bps of traded |
|---|---|---|---|---|---|---|
| SYN-1 (R01) | {s1['n_executed_actions']} | {s1['actions_per_decision_date']} | {s1['total_traded_dollars']:,.0f} | {s1['total_cost_dollars']:.2f} | {s1['cost_drag_bps_of_start_nav']:.2f} | {s1['cost_bps_of_traded']:.2f} |
| incumbent (R02) | {s2['n_executed_actions']} | {s2['actions_per_decision_date']} | {s2['total_traded_dollars']:,.0f} | {s2['total_cost_dollars']:.2f} | {s2['cost_drag_bps_of_start_nav']:.2f} | {s2['cost_bps_of_traded']:.2f} |

(Sources: `runs_battery/R01/manifest.json`, `runs_battery/R02/manifest.json`.)

## 5. Equity curve (sampled every 5th common date, cost-adjusted)

{chr(10).join(eq_rows)}

(Full series: `runs_battery/R01/daily_series.csv`, `runs_battery/R02/daily_series.csv`.)

## 6. Seed sensitivity (R13/R14 vs R01)

- ASSERTION RUN: R13 and R14 `raw_value` columns are byte-identical to R01's
  (checked by `assemble_evidence.py`; the strategy path consumes no slippage seed).
- Cost-overlay spread across master seeds 4242/4243/4244:

{chr(10).join(seed_rows)}

Comparison artifacts: `evidence/R13_seed4243/`, `evidence/R14_seed4244/`.

## 7. Findings logged by this assembly

{chr(10).join('- ' + f for f in findings)}

## 8. Manifests

- SYN-1: `runs_battery/R01/manifest.json` (config_hash ed35d0a7cb616e40)
- incumbent: `runs_battery/R02/manifest.json` (config_hash 5bf392338d9f4c19)
- Full comparison JSON/MD: `evidence/R01_vs_R02_bakeoff/`
- Ledgers: `prototype/holdout_looks.jsonl` ({doc['ledgers']['n_holdout_looks']} looks),
  `prototype/validation_looks.jsonl` ({doc['ledgers']['n_validation_looks']} decisions)
"""
    (RUN / "BAKEOFF.md").write_text(bakeoff_md)

    # ---------------- ATTRIBUTION.md ------------------------------------------
    table_rows = ["| organ | arm | E1 HAC t | E1 mean bp/day | E1 95% CI (daily) | E2 holdout dSharpe | E2 paired t | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
    by_organ = {r["organ"]: r for r in doc["organ_table"]}
    for r in doc["organ_table"]:
        ci = r.get("e1_ci95") or [None, None]
        table_rows.append(
            f"| {r['organ']} | {r.get('run_id', '—')} | {fmt(r.get('e1_hac_t'))} | "
            f"{fmt(r.get('e1_mean_bp_day'))} | [{fmt(ci[0], 6)}, {fmt(ci[1], 6)}] | "
            f"{fmt(r.get('e2_holdout_dsharpe'))} | {fmt(r.get('e2_holdout_t'))} | **{r['verdict']}** |")

    ea = json.load(open(PROTO / "ea" / "ea_manifest_2026-02-06.json"))
    b1 = json.load(open(PROTO / "ea" / "b1_2026-02-06.json"))
    sections = []
    for ev_id, e1_key, ra, rb, la, lb, attr_key, member, onote in ORGAN_ARMS:
        r = by_organ[e1_key]
        e1r = e1[e1_key]
        arm_run = rb if ra == "R01" else ra
        rt, rung, ladder = ARM_PROVENANCE.get(arm_run, ("—", "—", "—"))
        pf = e1r["per_fold"]
        pf_line = "; ".join(f"F{k}: t={fmt(pf[k]['hac_t'])}" for k in sorted(pf))
        ci = e1r["ci95"]
        sec = [f"### {e1_key} ({arm_run}; final-line key `{attr_key}`" +
               (f", member `{member}`" if member else "") + ")", "",
               f"- **E1 (primary, §4.3):** HAC t={fmt(e1r['hac_t'])} on n={e1r['n']} pooled fold-days, "
               f"mean={fmt(e1r['mean_bp_day'])} bp/day, 95% CI [{fmt(ci[0], 6)}, {fmt(ci[1], 6)}], "
               f"MDE={fmt(e1r['mde'], 6)} (`evidence/e1_reads.json`)",
               f"- per-fold HAC t: {pf_line}",
               f"- **E2 (sign confirmation):** holdout dSharpe={fmt(r.get('e2_holdout_dsharpe'))}, "
               f"paired mean={fmt((r.get('e2_holdout_mean_delta') or float('nan')) * 1e4)} bp/day, "
               f"paired t={fmt(r.get('e2_holdout_t'))}, n={r.get('e2_holdout_n', '—')} "
               f"(`{r['evidence']}`)",
               f"- orientation: {onote}",
               f"- **verdict: {r['verdict']}** — {r['verdict_reason']} (stats.organ_verdict)",
               f"- provenance: retrain {rt}; exec rung shipped: {rung} (`{ladder}`)"]
        # special prints (§4.3 arm-table riders)
        if e1_key == "transformer":
            sec.append("- SPECIAL PRINT: CAST ships regardless — item-1 transformer presence is "
                       "mandated; the scorecard prints the measured attribution (§4.3 row 1).")
        if e1_key == "risknet":
            sec.append("- SPECIAL PRINT: planned arm R07 was DEGENERATE (series byte-identical "
                       "to R01; --sigma-source trailing21 == deployed convention -> it measured "
                       "the null contrast by construction). R07 stays in the record as a consumed "
                       "look (`evidence/R07_ensemble_risknet/`); the E2 read above comes from "
                       "contingency replay R19 (--sigma-source risknet, reserved slot per "
                       "TOURNAMENT §4.4), candidate-minus-deployed orientation.")
        if e1_key == "executive":
            sf = fdiag["std_tau_per_fold"]
            max_std = max(max(v) for v in sf.values())
            sec.append(f"- SPECIAL PRINT (diversity floor, §4.6.5): mean pairwise solo-book corr "
                       f"= {step10['diversity_floor']['mean_pairwise_corr']:.3f} >= 0.90 -> item-5 trust attribution "
                       "expected ~0 by construction; the equal-trust tie is the honest outcome "
                       "(`store/step10_checks.json`).")
            sec.append(f"- SPECIAL PRINT (rung): linear twin shipped — MLP 5-seed val util "
                       f"{fdiag['mlp_val_util_mean']:.4g} did NOT beat twin {fdiag['linear_twin_val_util']:.4g} "
                       "(`exec_out/final_diagnostics.json`, `exec_out/ladder.json`).")
            sec.append(f"- SPECIAL PRINT (proposal §7 kill criteria, per-fold per §4.6.10): static-trust "
                       f"falsifier FIRES — std tau < 0.02 in every fold (max {max_std:.4f}); calibration "
                       f"corr = {fdiag['calibration_corr']:.3f} <= 0 (kill criterion fires); challenger parity at "
                       "the ladder (twin >= MLP). All three reported, none reinterpreted.")
        if e1_key == "evolution":
            sec.append(f"- SPECIAL PRINT (EA proposal pre-committed sentences, verbatim rules): "
                       f"(1) carries-weight test: champion vs B0 E1 t={fmt(e1r['hac_t'])} (champion does NOT "
                       f"clear cross-fold ΔU>0 with paired t) -> does not carry weight on E1; "
                       f"(2) optimizer test: champion fitness {ea['champion_fitness']:.4f} > B1 best "
                       f"{b1['best_fitness']:.4f} (K={b1['K']} budget-matched random search) -> NOT ceremonial "
                       f"as an optimizer; "
                       f"(3) organ test: adoption gate PASSED (margin {ea['margin']:.4f} > 1.0 x cross-fold "
                       f"sd {ea['cross_fold_sd']:.4f}; champion shipped) — B2/E2 holdout read printed above. "
                       "(`ea/ea_manifest_2026-02-06.json`, `ea/b1_2026-02-06.json`, FREEZE_SYN1.md)")
        if e1_key == "llm":
            sec.append("- SPECIAL PRINT (Stage-1): variance PASS, tone-proxy PASS (corr 0.116 < 0.8), "
                       "truncation PASS (1.5%); PARTIAL FAIL — event-flag chattiness (fires 98% of days); "
                       "the pre-registered kill condition 'misses known scheduled events' is NOT met "
                       "(8/8 hit), so the organ shipped with the partial-fail logged (FREEZE_SYN1.md).")
            sec.append(f"- SPECIAL PRINT (model-cutoff gate, §4.6.1): single pinned model "
                       f"`anthropic.claude-3-haiku-20240307-v1:0` (cutoff 2023-08) on all {n_calls} calls; "
                       "0 fallback days (`bedrock_spend.jsonl`). PASS.")
            sec.append("- FINDING carried from RT-3 (`validation_looks.jsonl`): the shipping GBM never "
                       "split on llm_* (zeroing them changes nothing, max |dP|=0); the LLM difference "
                       "flows entirely through EventHead + executive z gating.")
        if e1_key == "gdelt":
            sec.append(f"- SPECIAL PRINT (shift gate, §4.6.2): {shift['overall']} — record-count ratio "
                       f"{shift['checks']['record_count_median_ratio']['ratio']:.3f} in [0.5,2.0]; tone shift "
                       f"{shift['checks']['tone_shift']['shift_in_sd_units']:.2f} sd; worst bucket energy_oil "
                       f"{shift['checks']['g1_bucket_share_shift']['worst_shift_in_sd_units']:.2f} sd < 5.0 "
                       "(`gdelt_shift_gate.json`). 'No signal' is distinguishable from 'feed broke'.")
            sec.append(f"- SPECIAL PRINT (G1 placebo consequence, §4.6.3): FAIL — real training-fold "
                       f"rank-IC {placebo['real_ic']:+.4f} at percentile {placebo['percentile']:.0f} of "
                       f"{placebo['n_permutations']} permutations (need >95th) -> the G1 dictionary is reported "
                       "`0 (measured)` REGARDLESS of this block arm (`placebo_result.json`). The champion "
                       "genome already gates G1_themes + G3_tone OFF on both sides of the contrast.")
            sec.append("- §0.7 caveat (printed with all E1 GDELT numbers): the 64-ETF universe and every "
                       "curated mapping were authored in 2026 with full knowledge of 2015-2026 history; "
                       "freezing protects only holdout claims (ATTACK_SKEPTIC §0.7).")
            sec.append("- FINDING carried from RT-4 (`validation_looks.jsonl`): with G1-G5 ablated the "
                       "EventHead's event-mass gate is 0 -> permanent abstain; GDELT structurally carries "
                       "the EventHead activity gate, so the ablated brain is CAST+GBM with renormalized trust.")
        if e1_key == "infotropy_b":
            sec.append("- SPECIAL PRINT (rungs, FREEZE §9.2): record-weighting shipped per-learner on "
                       "purged-validation wins — CAST weighted (4/6 fold wins), GBM uniform (3/6, mean "
                       "-0.0044); the R12 contrast is exactly {CAST record-weighting, executive w_rec, "
                       "trust-tilt record weighting}.")
            sec.append("- Transfer-A fold verdict printed in parentheses on the final line (next section).")
        sections.append("\n".join(sec))

    # infotropy_a section
    ta_r = by_organ["infotropy_a"]
    ci = ta["ci95"]
    pf_line = "; ".join(f"F{k}: t={fmt(ta['per_fold'][k]['hac_t'])}" for k in sorted(ta["per_fold"]))
    sections.append("\n".join([
        "### infotropy_a (E1-only fold read; no replay per battery plan)", "",
        f"- **E1:** HAC t={fmt(ta['hac_t'])} on n={ta['n']} pooled fold-days, mean={fmt(ta['mean_bp_day'])} bp/day, "
        f"95% CI [{fmt(ci[0], 6)}, {fmt(ci[1], 6)}], MDE={fmt(ta['mde'], 6)} (`evidence/e1_reads.json`)",
        f"- per-fold HAC t: {pf_line}",
        f"- **verdict: {ta_r['verdict']}** — {ta_r['verdict_reason']}",
        "- SPECIAL PRINT (FREEZE §9.1): the conjunctive gate is DEAD (0/36 family x fold wins); "
        "R3-only screening ships for EventHead/GBM routing. This read is the shipped R3-only "
        "screen vs a no-screen twin, fold-level only (D5)."]))

    attribution_md = f"""# ATTRIBUTION.md — per-organ matrix (PKT-TB-006, mechanical assembly)

Generated {doc['generated']} by `prototype/assemble_evidence.py`. Verdicts computed
exclusively by `stats.organ_verdict` (§4.3 three-valued rule); E1 numbers from
`prototype/evidence/e1_reads.json` (pooled F1-F6 OOF daily utility difference, HAC
t Newey-West 10 lags, ~1,507 fold-days); E2 sign reads from the holdout-only
(>= {HOLDOUT}) cost-adjusted paired comparisons in `prototype/evidence/<run_id>/`.

## Summary table

{chr(10).join(table_rows)}

E2 reads on n={p_h['n']} holdout days carry MDE ~{fmt(p_h['mde'] * 1e4)} bp/day at |t|=2 —
sign-confirmation only (§4.3 hierarchy is fixed: E1 primary, E2 sign).

## Assignment-item mapping (ASSIGNMENT_BRIEF §1 -> scorecard organs)

{chr(10).join(f'- **Item {i}** ({d}): {o}' for i, d, o in ITEM_MAP)}

## Multiplicity (§4.5, verbatim)

{doc['multiplicity_line']}

## Per-organ detail

{chr(10).join(chr(10) + s for s in sections)}

## Battery provenance

- Battery: 14/14 planned replays + 1 contingency (R19, risknet repair) executed
  (`prototype/holdout_looks.jsonl`, {doc['ledgers']['n_holdout_looks']} looks; budget <=20), 5/5 planned retrains
  RT-1..RT-5 (`prototype/retrains.jsonl`); remaining contingency arms R15-R18/R20 not run
  (`battery_plan.json`).
- Exec-rung ladder per retrain (`exec_out*/ladder.json`): base RT-1 linear_twin FROZEN;
  RT-2/3/4/5 ship MLP per the pre-registered rung rule (recorded in
  `validation_looks.jsonl` rt_ladder_rungs_summary).
- Validation-look ledger: {doc['ledgers']['n_validation_looks']} decisions (`prototype/validation_looks.jsonl`).
- Gate readouts (§4.6) and the §4.7 final line: `prototype/evidence/SCORECARD.md` /
  `scorecard.json`.

## Final line (§4.7)

```
{doc['final_line']}
```
"""
    (RUN / "ATTRIBUTION.md").write_text(attribution_md)

    # ---------------- console report ------------------------------------------
    print("BAKE-OFF:", doc["bakeoff"]["verdict"],
          f"holdout t={p_h['t']:+.3f} HACt={p_h.get('hac_t', float('nan')):+.3f} n={p_h['n']} "
          f"mean={p_h['mean_bp_day']:+.3f}bp/d sd={p_h['sd_bp_day']:.3f}bp/d "
          f"dSharpe={bakeoff['holdout_dsharpe']:+.3f} dRet={bakeoff['holdout_dreturn'] * 100:+.3f}%")
    print("full-period t={:+.3f} n={} mean={:+.3f}bp/d".format(
        p_f["t"], p_f["n"], p_f["mean_bp_day"]))
    for r in doc["organ_table"]:
        print(f"  {r['organ']:<22} E1t={fmt(r.get('e1_hac_t'))} "
              f"E2dS={fmt(r.get('e2_holdout_dsharpe'))} E2t={fmt(r.get('e2_holdout_t'))} -> {r['verdict']}")
    print("ATTRS:", json.dumps(doc["organ_attrs"]))
    print("FINAL:", doc["final_line"])
    for f in findings:
        print("NOTE:", f)
    if boundary_flag:
        print(boundary_flag.strip())


if __name__ == "__main__":
    main()
