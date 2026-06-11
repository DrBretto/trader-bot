"""PKT-TB-007 Phase D — evidence assembly (TOURNAMENT_007 §4.2/§4.5–§4.7;
TB-006 report_evidence conventions adapted; everything verdict-shaped is a
pure function of numbers — no judgment at report time).

Emits:
  evidence_007/<cmp_id>/comparison.json + comparison.md   (per comparison)
  evidence_007/scorecard_007.json                          (machine record)
  ../BAKEOFF_007.md  ../ATTRIBUTION_007.md                 (run-level reports)

HONESTY RAILS (instruction-level, enforced structurally):
  - the REGISTERED verdict is the degenerate B0 cell and headlines as such;
  - every deviation number carries DEV_LABEL in the same sentence/object;
  - no averaging across the label; the deviation read is subordinated.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import stats as st

PROTO = Path(__file__).resolve().parent
RUNS = PROTO / "runs_battery_007"
EVID = PROTO / "evidence_007"
HOLDOUT_START = st.HOLDOUT_START
DEV_LABEL = "(deviation: a-priori genome, instrument-failed EA)"
LOOKS_LEDGER = PROTO / "holdout_looks_007.jsonl"
LEDGERS = PROTO / "ledgers"

METRIC_KEYS = ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate",
               "realized_round_trips", "cumulative_transaction_costs",
               "avg_gross_exposure")

CAVEAT_SENTENCE = (
    "Caveats (§4.1, attached to every read): the operator's repaired-timeline "
    "caveat (the live record was repaired at points; only paired same-harness "
    "comparisons are admissible); the 2026-04-01 signals-rebuild epoch covers "
    "live-era dirs through ~2026-03-31; L7 deployed-MLP training-data end "
    "2026-01-14 < 2026-03-11 holdout boundary (paired-cancellation: both arms "
    "share the model); Monday/cadence gaps, 2026-03-30..31 and the "
    "2026-05-11..20 outage hole (7 td, inside the holdout) are excluded "
    "identically in both arms.")


# ====================== §4.2 verdict cells (pure) ==============================
def ci90_of(diff: np.ndarray) -> list[float]:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return [float("nan"), float("nan")]
    se = st.hac_se(d)
    return [float(d.mean() - 1.645 * se), float(d.mean() + 1.645 * se)]


def verdict_cell_007(t1: float, e2_mean: float, e2_ci90: list[float],
                     full_ci90: list[float]) -> dict:
    """TOURNAMENT_007 §4.2 claim table, mechanical. t1 = E1 full-window paired
    daily t (naive, the registered statistic; HAC printed beside everywhere).
    X = −(lower bound of full-window CI90 of mean Δ, bp/day) rounded to 0.1."""
    lo90 = e2_ci90[0] if e2_ci90 else float("nan")
    hi90 = e2_ci90[1] if e2_ci90 else float("nan")
    if not np.isfinite(t1):
        return {"cell": "DEGENERATE", "claim": "paired t not computable"}
    if t1 <= -2.0 or (np.isfinite(hi90) and hi90 < 0):
        return {"cell": "LOSES",
                "claim": "loses to incumbent at matched exposure (+ the numbers)"}
    if t1 >= 2.0:
        if np.isfinite(e2_mean) and e2_mean >= 0:
            return {"cell": "BEATS",
                    "claim": "beats incumbent at matched exposure, live engine, "
                             "holdout-confirmed"}
        x = -(full_ci90[0] * 1e4)
        return {"cell": f"TIES (positive point estimate; certified "
                        f"not-worse-than −{max(x, 0):.1f} bp/day)",
                "claim": "downgrade cell: E1>=+2 but E2<0 — files as bounded-TIES "
                         "with the E1/E2 disagreement printed in the same "
                         "sentence; the word 'beats' is banned"}
    if t1 > 0:
        x = -(full_ci90[0] * 1e4)
        sub = (" one-sided-significant positive point estimate at 90%, below "
               "the BEATS bar." if t1 >= 1.645 else "")
        return {"cell": f"TIES (positive point estimate; certified "
                        f"not-worse-than −{max(x, 0):.1f} bp/day)",
                "claim": "no certifiable difference; loss bounded at "
                         f"{max(x, 0):.1f} bp/day (90%); point estimates favor "
                         "the brain." + sub}
    return {"cell": "TIES (straddling)",
            "claim": "no certifiable difference at available power; MDE printed"}


# ====================== §4.5 three-valued organ verdict (pure) =================
def organ_verdict_007(hac_t: float, mean: float, ci95: list[float],
                      mde: float, degenerate: bool = False,
                      degenerate_reason: str = "") -> dict:
    """TOURNAMENT_007 §4.5: `positive` (t >= +2), `zero (measured)`
    (|effect| CI inside ±MDE), else `indeterminate at available power`."""
    if degenerate:
        return {"tag": "zero (measured)",
                "reason": f"degenerate: {degenerate_reason} — paired deltas "
                          "identically 0 (CI = {0} ⊂ ±MDE trivially)"}
    if np.isfinite(hac_t) and hac_t >= 2.0:
        return {"tag": "positive", "reason": f"E1 HAC t={hac_t:+.2f} >= +2.0"}
    if (np.isfinite(ci95[0]) and np.isfinite(ci95[1]) and np.isfinite(mde)
            and ci95[0] >= -mde and ci95[1] <= mde):
        return {"tag": "zero (measured)",
                "reason": f"95% CI [{ci95[0]*1e4:+.2f},{ci95[1]*1e4:+.2f}] "
                          f"bp/day inside ±MDE {mde*1e4:.2f}"}
    return {"tag": "indeterminate at available power",
            "reason": f"HAC t={hac_t:+.2f}, CI not inside ±MDE "
                      f"({mde*1e4:.2f} bp/day)"}


# ====================== comparison machinery ===================================
def paired_block(ra, rb, start):
    da, va = st._slice_from(ra["dates"], ra["values"], start)
    db, vb = st._slice_from(rb["dates"], rb["values"], start)
    dts, a, b = st.align_paired(da[1:], st.daily_returns(va),
                                db[1:], st.daily_returns(vb))
    diff = a - b
    p = st.paired_daily_stats(da[1:], st.daily_returns(va),
                              db[1:], st.daily_returns(vb), hac=True)
    p["ci90"] = ci90_of(diff)
    return p, diff


def compare(a_dir, b_dir, label_a, label_b):
    ra, rb = st.load_run(RUNS / a_dir), st.load_run(RUNS / b_dir)
    out = {"a": label_a, "b": label_b, "run_dir_a": str(RUNS / a_dir),
           "run_dir_b": str(RUNS / b_dir), "holdout_start": HOLDOUT_START,
           "manifest_a": ra["manifest"].get("summary"),
           "manifest_b": rb["manifest"].get("summary"),
           "genome_a": ra["manifest"].get("genome_hash"),
           "genome_b": rb["manifest"].get("genome_hash")}
    for period, start in (("full_period", None), ("holdout_only", HOLDOUT_START)):
        ma, mb = st.run_metrics(ra, start), st.run_metrics(rb, start)
        p, _ = paired_block(ra, rb, start)
        out[period] = {"metrics_a": ma, "metrics_b": mb,
                       "delta": {k: (ma[k] - mb[k])
                                 if isinstance(ma[k], (int, float)) else None
                                 for k in METRIC_KEYS},
                       "paired_daily": p}
    return out


def gross_beta_gap(a_dir, b_dir):
    """§4.1 exposure-parity diagnostics: realized daily gross gap + rolling
    21d β gap (each arm's cost-adj daily returns OLS on SPY returns),
    disclosure trigger |Δβ| > 0.05 sustained 5 days."""
    import risk_stats_007 as RS
    ra, rb = st.load_run(RUNS / a_dir), st.load_run(RUNS / b_dir)

    def gross(r):
        return {row["date"]: (row["ending_value"] - row.get("ending_cash", 0.0))
                / row["ending_value"]
                for row in r["result"].get("timeline", []) if row.get("ending_value")}
    ga, gb = gross(ra), gross(rb)
    common = sorted(set(ga) & set(gb))
    gaps = np.array([ga[d] - gb[d] for d in common])

    rs = RS.RiskStats()
    spy = rs._returns("SPY")
    def beta_series(r):
        dts = r["dates"][1:]
        rets = pd.Series(st.daily_returns(r["values"]), index=dts)
        s = spy.reindex(dts)
        roll_cov = rets.rolling(21).cov(s)
        roll_var = s.rolling(21).var()
        return roll_cov / roll_var
    bgap = (beta_series(ra) - beta_series(rb)).dropna()
    sustained = bool((bgap.abs() > 0.05).rolling(5).sum().ge(5).any()) \
        if len(bgap) >= 5 else False
    return {"gross_gap_mean": float(np.mean(np.abs(gaps))) if len(gaps) else None,
            "gross_gap_max": float(np.max(np.abs(gaps))) if len(gaps) else None,
            "beta_gap_21d_mean": float(bgap.abs().mean()) if len(bgap) else None,
            "beta_gap_21d_max": float(bgap.abs().max()) if len(bgap) else None,
            "beta_gap_trigger_0p05_5d": sustained}


def _fmt(v, pct=False, digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if pct:
        return f"{v * 100:+.2f}%"
    return f"{v:+.{digits}f}" if isinstance(v, float) else str(v)


def comparison_md(cmp, label_line, extra_note=""):
    L = [f"# Evidence comparison — {cmp['a']} vs {cmp['b']}", "",
         label_line, "",
         f"Holdout boundary: {cmp['holdout_start']}. Delta = {cmp['a']} − {cmp['b']}.",
         "Paired daily-difference stats on identical dates are the required form;",
         "endpoint deltas are context, never the verdict (EVIDENCE_PROTOCOL §2).", ""]
    for period, title in (("full_period", "Full period"),
                          ("holdout_only", "Holdout only (≥ 2026-03-11)")):
        blk = cmp[period]
        p = blk["paired_daily"]
        L += [f"## {title}", "",
              f"| metric | {cmp['a']} | {cmp['b']} | delta |", "|---|---|---|---|"]
        for k in METRIC_KEYS:
            pct = k in ("total_return", "cagr", "max_drawdown", "win_rate",
                        "avg_gross_exposure")
            L.append(f"| {k} | {_fmt(blk['metrics_a'][k], pct)} | "
                     f"{_fmt(blk['metrics_b'][k], pct)} | "
                     f"{_fmt(blk['delta'][k], pct)} |")
        ci90 = p.get("ci90", [float("nan")] * 2)
        L += ["", f"Paired daily diff (cost-adjusted): n={p['n']}, "
              f"mean={_fmt(p['mean_bp_day'], digits=3)} bp/day, "
              f"sd={_fmt(p['sd_bp_day'], digits=2)} bp/day, t={_fmt(p['t'], digits=3)}, "
              f"HAC t={_fmt(p.get('hac_t'), digits=3)}, "
              f"95% CI [{_fmt(p['ci95'][0])}, {_fmt(p['ci95'][1])}], "
              f"90% CI [{_fmt(ci90[0])}, {_fmt(ci90[1])}], "
              f"**per-arm MDE(|t|=2) = {_fmt(p['mde'] * 1e4 if np.isfinite(p['mde']) else float('nan'), digits=2)} bp/day**", ""]
    L += ["## Manifests (summaries)", "",
          f"- {cmp['a']} (genome {cmp.get('genome_a')}): "
          f"`{json.dumps(cmp['manifest_a'], sort_keys=True)}`",
          f"- {cmp['b']} (genome {cmp.get('genome_b')}): "
          f"`{json.dumps(cmp['manifest_b'], sort_keys=True)}`", "",
          "Forced metric choices carried from TB-006 (stats.run_metrics "
          "docstring): round trips = executed SELL/REDUCE count; cumulative "
          "costs = raw-minus-costadj value drag; gross exposure from timeline.",
          "", CAVEAT_SENTENCE]
    if extra_note:
        L += ["", f"NOTE: {extra_note}"]
    return "\n".join(L) + "\n"


def write_cmp(cmp_id, a_dir, b_dir, label_a, label_b, label_line, extra_note=""):
    cmp = compare(a_dir, b_dir, label_a, label_b)
    cmp["cmp_id"] = cmp_id
    cmp["label"] = label_line
    cmp["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    d = EVID / cmp_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "comparison.json").write_text(json.dumps(cmp, indent=1, default=float))
    (d / "comparison.md").write_text(comparison_md(cmp, label_line, extra_note))
    return cmp


def count_lines(p: Path) -> int:
    return sum(1 for ln in p.read_text().splitlines() if ln.strip()) \
        if p.exists() else 0
