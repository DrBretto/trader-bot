"""PKT-TB-006 — evidence reporting (EVIDENCE_PROTOCOL; TOURNAMENT §4.2/§4.3/§4.5–§4.7).

Two jobs, both MECHANICAL (no judgment enters at report time):

1. ``write_comparison`` — one JSON + one MD per E1/E2 comparison into
   ``evidence/<run_id>/``: §4.2 metric set for both runs, FULL-PERIOD AND
   HOLDOUT-ONLY, deltas, paired daily-difference stats on identical dates
   (the required form — endpoint deltas never headline), run manifests.

2. ``assemble_scorecard`` — takes all arm results and emits:
   - the §4.3 per-organ verdict table (verdicts from stats.organ_verdict only),
   - the §4.5 multiplicity line VERBATIM with <N_holdout>/<N_validation>
     filled from the ledgers,
   - the §4.6 gate readouts (model-cutoff, shift gate, placebo, break-even IC,
     diversity floor, train-vs-harness gap, fine-tune delta),
   - the §4.7 final line string.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import stats as st

PROTO = Path(__file__).resolve().parent
EVIDENCE_ROOT = PROTO / "evidence"
HOLDOUT_LEDGER = PROTO / "holdout_looks.jsonl"
VALIDATION_LEDGER = PROTO / "validation_looks.jsonl"
HOLDOUT_START = st.HOLDOUT_START

METRIC_KEYS = ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate",
               "realized_round_trips", "cumulative_transaction_costs",
               "avg_gross_exposure")

# §4.5, verbatim (only the two ledger counts are filled at print time)
MULTIPLICITY_LINE = (
    '"7 scorecard organs were each given one pre-registered arm. At the E2 '
    "holdout sign bar (t >= 1) the per-arm false-positive rate under the null "
    "is ~ 16%; expected false 'carries weight' sign-confirmations across 7 "
    "organs ~ 1.1; family-wise P(>=1 false positive) ~ 70%. That is why E2 is "
    "sign-confirmation only. At the E1 primary bar (HAC |t| >= 2 on ~1,500 "
    "pooled fold-days) the per-arm FPR is ~ 5%; expected false positives "
    "across 7 organs ~ 0.35. The holdout MDE is ~ 5 bp/day (dSharpe_ann ~ 4 "
    "at t=2); no claimed effect is that large, and no holdout number below it "
    "is treated as certified. This battery consumed {n_holdout} holdout looks "
    'and {n_validation} validation-fold decisions (ledgers below)."'
)

FINAL_LINE_ORGANS = ("transformer", "ensemble", "evolution", "LLM", "GDELT",
                     "meta-evaluator", "infotropy")


# ============================ per-comparison ===================================
def compare_runs(run_a_dir: Path, run_b_dir: Path, label_a: str, label_b: str,
                 holdout_start: str = HOLDOUT_START) -> dict:
    """Full §4.2/EVIDENCE_PROTOCOL comparison block: A vs B (delta = A − B)."""
    ra, rb = st.load_run(run_a_dir), st.load_run(run_b_dir)
    out = {"a": label_a, "b": label_b,
           "run_dir_a": str(run_a_dir), "run_dir_b": str(run_b_dir),
           "holdout_start": holdout_start,
           "manifest_a": ra["manifest"], "manifest_b": rb["manifest"]}
    for period, start in (("full_period", None), ("holdout_only", holdout_start)):
        ma = st.run_metrics(ra, start)
        mb = st.run_metrics(rb, start)
        da, va = st._slice_from(ra["dates"], ra["values"], start)
        db, vb = st._slice_from(rb["dates"], rb["values"], start)
        paired = st.paired_daily_stats(da[1:], st.daily_returns(va),
                                       db[1:], st.daily_returns(vb), hac=True)
        out[period] = {
            "metrics_a": ma, "metrics_b": mb,
            "delta": {k: (ma[k] - mb[k]) if isinstance(ma[k], (int, float)) else None
                      for k in METRIC_KEYS},
            "paired_daily": paired,
        }
    return out


def _fmt(v, pct=False, digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if pct:
        return f"{v * 100:+.2f}%"
    return f"{v:+.{digits}f}" if isinstance(v, float) else str(v)


def comparison_md(cmp: dict, extra_note: str = "") -> str:
    lines = [f"# Evidence comparison — {cmp['a']} vs {cmp['b']}",
             "",
             f"Holdout boundary: {cmp['holdout_start']}. Delta = {cmp['a']} − {cmp['b']}.",
             "Paired daily-difference stats on identical dates are the required",
             "form; endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).",
             ""]
    for period, title in (("full_period", "Full period"),
                          ("holdout_only", "Holdout only")):
        blk = cmp[period]
        p = blk["paired_daily"]
        lines += [f"## {title}", "",
                  f"| metric | {cmp['a']} | {cmp['b']} | delta |", "|---|---|---|---|"]
        for k in METRIC_KEYS:
            pct = k in ("total_return", "cagr", "max_drawdown", "win_rate",
                        "avg_gross_exposure")
            lines.append(f"| {k} | {_fmt(blk['metrics_a'][k], pct)} | "
                         f"{_fmt(blk['metrics_b'][k], pct)} | "
                         f"{_fmt(blk['delta'][k], pct)} |")
        lines += ["",
                  f"Paired daily diff: n={p['n']}, mean={_fmt(p['mean_bp_day'], digits=2)} bp/day, "
                  f"sd={_fmt(p['sd_bp_day'], digits=2)} bp/day, t={_fmt(p['t'], digits=2)}, "
                  f"HAC t={_fmt(p.get('hac_t'), digits=2)}, "
                  f"95% CI [{_fmt(p['ci95'][0])}, {_fmt(p['ci95'][1])}], "
                  f"MDE(|t|=2)={_fmt(p['mde'])}",
                  ""]
    lines += ["## Manifests", "",
              f"- {cmp['a']}: `{json.dumps(cmp['manifest_a'], sort_keys=True)}`",
              f"- {cmp['b']}: `{json.dumps(cmp['manifest_b'], sort_keys=True)}`",
              "",
              "Forced metric choices: round trips = executed SELL/REDUCE count; "
              "cumulative costs = raw-minus-costadj value drag; gross exposure "
              "from timeline ending_value/ending_cash (stats.run_metrics docstring)."]
    if extra_note:
        lines += ["", f"NOTE: {extra_note}"]
    return "\n".join(lines) + "\n"


def write_comparison(run_id: str, run_a_dir: Path, run_b_dir: Path,
                     label_a: str, label_b: str,
                     out_root: Path = EVIDENCE_ROOT,
                     e1_result: dict | None = None,
                     extra_note: str = "") -> dict:
    """One JSON + one MD per comparison into evidence/<run_id>/ (EVIDENCE_PROTOCOL)."""
    cmp = compare_runs(run_a_dir, run_b_dir, label_a, label_b)
    cmp["run_id"] = run_id
    cmp["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    if e1_result is not None:
        cmp["e1_primary"] = e1_result
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w") as fh:
        json.dump(cmp, fh, indent=1, default=float)
    md = comparison_md(cmp, extra_note)
    if e1_result is not None:
        md += (f"\n## E1 primary read\n\nHAC t={_fmt(e1_result.get('hac_t'), digits=2)} "
               f"on n={e1_result.get('n')} pooled fold-days, "
               f"mean={_fmt(e1_result.get('mean'))}, "
               f"CI95=[{_fmt((e1_result.get('ci95') or [None, None])[0])}, "
               f"{_fmt((e1_result.get('ci95') or [None, None])[1])}], "
               f"MDE={_fmt(e1_result.get('mde'))}\n")
    with open(out_dir / "comparison.md", "w") as fh:
        fh.write(md)
    return cmp


# ============================ scorecard assembly ===============================
def _count_lines(path: Path) -> int:
    if not Path(path).exists():
        return 0
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())


def gate_readouts(gates: dict) -> list[str]:
    """§4.6 gate lines, mechanical. ``gates`` keys (any may be absent ->
    'NOT PROVIDED' is printed, never silently dropped):
      model_cutoff {model, cutoff, fallback_days_pct, verdict}
      shift_gate {overall, path}            (gdelt_shift_gate.json content)
      placebo {real_ic, percentile, pass_95, path}
      break_even_ic {weekly_rank_ic, bar, break_even, verdict}
      diversity_floor {mean_pairwise_corr, threshold, verdict}
      train_vs_harness_gap {gap_pct_of_mean_abs_utility, limit_pct, verdict}
      fine_tune {pre_val, post_val, delta, declaimed}
    """
    lines = []
    g = gates.get("model_cutoff")
    lines.append("1. LLM model-cutoff: " + (
        f"model={g['model']}, cutoff={g['cutoff']}, fallback-days="
        f"{g.get('fallback_days_pct', 0)}% -> {g['verdict']}" if g else "NOT PROVIDED"))
    g = gates.get("shift_gate")
    lines.append("2. GDELT distribution-shift gate: " + (
        f"{g['overall']} ({g.get('path', 'gdelt_shift_gate.json')})" if g else "NOT PROVIDED"))
    g = gates.get("placebo")
    lines.append("3. Permuted-dictionary placebo: " + (
        f"real training-fold rank-IC={g['real_ic']:+.4f} at percentile "
        f"{g['percentile']:.1f} of {g.get('n_permutations', 50)} placebos -> "
        f"{'PASS (>95th)' if g['pass_95'] else 'FAIL -> G1 dictionary reported 0 (measured)'}"
        if g else "NOT PROVIDED"))
    g = gates.get("break_even_ic")
    lines.append("4. Break-even IC: " + (
        f"purged-validation weekly rank IC={g['weekly_rank_ic']:+.4f} vs "
        f"functionality bar {g.get('bar', 0.02)} (cost break-even "
        f"{g.get('break_even', 0.006)}) -> {g['verdict']}" if g else "NOT PROVIDED"))
    g = gates.get("diversity_floor")
    lines.append("5. Diversity floor: " + (
        f"mean pairwise solo-book corr={g['mean_pairwise_corr']:+.3f} "
        f"(threshold 0.90) -> {g['verdict']}" if g else "NOT PROVIDED"))
    g = gates.get("train_vs_harness_gap")
    lines.append("6. Train-vs-harness gap: " + (
        f"{g['gap_pct_of_mean_abs_utility']:.1f}% of mean |daily utility| "
        f"(limit 25%) -> {g['verdict']}" if g else "NOT PROVIDED"))
    g = gates.get("fine_tune")
    lines.append("7. Fine-tune delta: " + (
        f"pre={g['pre_val']:+.6f} post={g['post_val']:+.6f} delta={g['delta']:+.6f}"
        f"{' — de-claimed (~0)' if g.get('declaimed') else ''}" if g else "NOT PROVIDED"))
    return lines


def final_line(bakeoff: dict, organ_attrs: dict, cost_per_month: float,
               reductions: list[str]) -> str:
    """§4.7 mapping, mechanical. ``bakeoff`` needs verdict, holdout dreturn /
    dsharpe endpoint deltas, paired {t, n, mean_bp_day, sd_bp_day}.
    ``organ_attrs`` maps each FINAL_LINE_ORGANS key to its printed attr string."""
    p = bakeoff["paired"]
    paired_str = (f"paired daily t={p['t']:+.2f}, n={p['n']}, "
                  f"mean={p['mean_bp_day']:+.2f} bp/day, sd={p['sd_bp_day']:.2f} bp/day")
    score = " ".join(f"{k}={organ_attrs.get(k, 'n/a (missing)')}"
                     for k in FINAL_LINE_ORGANS)
    red = ", ".join(reductions) if reductions else "none"
    return (f"BRAIN vs INCUMBENT: {bakeoff['verdict']} by "
            f"{bakeoff['holdout_dreturn'] * 100:+.2f}%, "
            f"dSharpe {bakeoff['holdout_dsharpe']:+.2f} on holdout ({paired_str}); "
            f"ASSIGNMENT SCORECARD: {score}; "
            f"COST: ${cost_per_month:.2f}/mo — {red}")


def assemble_scorecard(bakeoff: dict, organs: list[dict], gates: dict,
                       cost_per_month: float, reductions: list[str],
                       out_dir: Path = EVIDENCE_ROOT,
                       holdout_ledger: Path = HOLDOUT_LEDGER,
                       validation_ledger: Path = VALIDATION_LEDGER) -> dict:
    """The full §4.3/§4.5/§4.6/§4.7 scorecard. ``organs`` entries:
        {organ, run_id, e1_hac_t, e1_n, e1_ci95, e1_mde,
         e2_holdout_dsharpe, e2_holdout_mean_delta, gate_off (bool),
         attr_key (one of FINAL_LINE_ORGANS), note (optional)}
    Verdicts are computed HERE via stats.organ_verdict — callers pass numbers,
    never verdicts."""
    n_holdout = _count_lines(holdout_ledger)
    n_validation = _count_lines(validation_ledger)
    table = []
    attrs: dict[str, list[str]] = {}
    for o in organs:
        v = st.organ_verdict(o.get("e1_hac_t"), o.get("e2_holdout_mean_delta"),
                             gate_off=bool(o.get("gate_off")))
        attr = st.scorecard_attr(o.get("e2_holdout_dsharpe"), v["tag"])
        row = {**o, "verdict": v["tag"], "verdict_reason": v["reason"], "attr": attr}
        table.append(row)
        attrs.setdefault(o["attr_key"], []).append(
            attr if o["attr_key"] != "ensemble" or "member" not in o
            else f"{o['member']}:{attr.replace(' ', '')}")
    organ_attrs = {k: "/".join(v) for k, v in attrs.items()}
    # infotropy: Transfer B arm with Transfer A fold verdict in parentheses,
    # or `no-transfer` when both are zero (§4.7)
    tb = next((r for r in table if r.get("organ") == "infotropy_b"), None)
    ta = next((r for r in table if r.get("organ") == "infotropy_a"), None)
    if tb is not None:
        zeroes = ("0 (measured)", "0 (gate-honest)")
        if ta is not None and tb["verdict"] in zeroes and ta["verdict"] in zeroes:
            organ_attrs["infotropy"] = "no-transfer"
        elif ta is not None:
            organ_attrs["infotropy"] = f"{tb['attr']} (A: {ta['verdict']})"
        else:
            organ_attrs["infotropy"] = tb["attr"]
    verdict = st.beats_ties_loses(bakeoff["paired"]["t"], bakeoff["holdout_dsharpe"])
    bk = {**bakeoff, "verdict": verdict["verdict"], "rule_gap": verdict["rule_gap"]}
    mult = MULTIPLICITY_LINE.format(n_holdout=n_holdout, n_validation=n_validation)
    fl = final_line(bk, organ_attrs, cost_per_month, reductions)
    doc = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "bakeoff": bk,
        "honesty_line": ("at t=+-1 the per-comparison false-call probability under "
                         "the null ~ 16% one-sided; holdout MDE at t=2 ~ 5 bp/day "
                         "~ dSharpe_ann ~ 4.0 — the bake-off verdict is a sign-grade "
                         "read; full-period paired stats are context, never the verdict"),
        "organ_table": table,
        "organ_attrs": organ_attrs,
        "multiplicity_line": mult,
        "ledgers": {"n_holdout_looks": n_holdout,
                    "n_validation_looks": n_validation,
                    "holdout_ledger": str(holdout_ledger),
                    "validation_ledger": str(validation_ledger)},
        "gate_readouts": gate_readouts(gates),
        "cost_per_month": cost_per_month,
        "reductions": reductions,
        "final_line": fl,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scorecard.json", "w") as fh:
        json.dump(doc, fh, indent=1, default=float)
    md = ["# PKT-TB-006 assignment scorecard (mechanical assembly)", "",
          f"Generated {doc['generated']}.",
          "", "## Bake-off (§4.2)",
          "",
          f"**{bk['verdict']}** — holdout paired t={bakeoff['paired']['t']:+.2f}, "
          f"n={bakeoff['paired']['n']}, dSharpe={bakeoff['holdout_dsharpe']:+.2f}, "
          f"dReturn={bakeoff['holdout_dreturn'] * 100:+.2f}%"
          + (" — **RULE GAP** (see stats.beats_ties_loses)" if bk["rule_gap"] else ""),
          "", doc["honesty_line"], "",
          "## Organ verdicts (§4.3)", "",
          "| organ | arm | E1 HAC t | E1 n | E2 holdout dSharpe | verdict | reason |",
          "|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r.get('organ')} | {r.get('run_id', '—')} | "
                  f"{_fmt(r.get('e1_hac_t'), digits=2)} | {r.get('e1_n', '—')} | "
                  f"{_fmt(r.get('e2_holdout_dsharpe'), digits=2)} | {r['verdict']} | "
                  f"{r['verdict_reason']} |")
    md += ["", "## Multiplicity (§4.5, verbatim)", "", mult, "",
           "## Evidence gates (§4.6)", ""]
    md += [f"- {line}" for line in doc["gate_readouts"]]
    md += ["", "## Final line (§4.7)", "", "```", fl, "```", ""]
    with open(out_dir / "SCORECARD.md", "w") as fh:
        fh.write("\n".join(md))
    return doc
