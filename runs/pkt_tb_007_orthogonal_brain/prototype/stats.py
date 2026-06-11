"""PKT-TB-006 — attribution-battery statistics (TOURNAMENT §4.2/§4.3, pure functions).

Everything verdict-shaped in here is a PURE function of numbers so that report
assembly is mechanical (no judgment at report time — §4.3/§4.7). The numeric
rules are transcribed from TOURNAMENT.md §4.2 (BEATS/TIES/LOSES) and §4.3
(three-valued organ verdicts) verbatim; tests/test_battery.py holds the truth
table.

Run-dir metric extraction codes against the replay-runner contract. TWO shapes
are accepted (CONTRACT DIVERGENCE, reported in the build handback — the packet
documented shape A; the landed run_replay.py emits shape B):
  A) daily_raw.csv + daily_costadj.csv (date,value); result.json includes
     timeline; manifest.json.
  B) daily_series.csv (date,raw_value,cost_adjusted_value); result.json
     WITHOUT timeline (timeline.json beside it); manifest.json.

PRE-REGISTRATION GAP (finding, not reinterpretation): §4.2's three rules are
not exhaustive — holdout paired t >= +1.0 with holdout ΔSharpe <= 0 satisfies
none of BEATS/TIES/LOSES. ``beats_ties_loses`` returns verdict
"UNDEFINED (rule-gap)" with ``rule_gap: True`` in that cell instead of silently
coercing; the scorecard prints it as such.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

TRADING_DAYS = 252
HAC_LAGS = 10                 # Newey-West lags (§4.3, fixed)
HOLDOUT_START = "2026-03-11"  # TOURNAMENT §4.1


# ============================ series plumbing ==================================
def daily_returns(values: np.ndarray) -> np.ndarray:
    """Simple daily returns from a value series; r[i] = v[i+1]/v[i] - 1 is
    indexed to the LATER date (the day the return realizes)."""
    v = np.asarray(values, dtype=np.float64)
    return v[1:] / v[:-1] - 1.0


def align_paired(dates_a, vals_a, dates_b, vals_b):
    """Intersect two (dates, values) daily series on identical dates.

    Returns (dates, a, b). Paired stats are only admissible on identical dates
    (EVIDENCE_PROTOCOL §2); non-overlap is returned for the manifest, never
    silently padded.
    """
    da = {d: float(v) for d, v in zip(dates_a, vals_a)}
    db = {d: float(v) for d, v in zip(dates_b, vals_b)}
    common = sorted(set(da) & set(db))
    a = np.array([da[d] for d in common])
    b = np.array([db[d] for d in common])
    return np.array(common), a, b


# ============================ paired stats =====================================
def naive_t(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 1e-300 else float("nan")


def hac_se(diff: np.ndarray, lags: int = HAC_LAGS) -> float:
    """Newey-West (Bartlett kernel) standard error of the MEAN of ``diff``."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan")
    e = d - d.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    s = max(s, 0.0)
    return float(np.sqrt(s / n))


def hac_tstat(diff: np.ndarray, lags: int = HAC_LAGS) -> float:
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    se = hac_se(d, lags)
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float(d.mean() / se)


def paired_daily_stats(dates_a, rets_a, dates_b, rets_b, hac: bool = False) -> dict:
    """Mean/sd/t of (a − b) daily-return differences on identical dates."""
    dates, a, b = align_paired(dates_a, rets_a, dates_b, rets_b)
    diff = a - b
    n = len(diff)
    mean = float(diff.mean()) if n else float("nan")
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    out = {
        "n": n,
        "mean": mean,
        "sd": sd,
        "t": naive_t(diff),
        "mean_bp_day": mean * 1e4 if n else float("nan"),
        "sd_bp_day": sd * 1e4 if n > 1 else float("nan"),
        "first_date": str(dates[0]) if n else None,
        "last_date": str(dates[-1]) if n else None,
    }
    if hac:
        out["hac_t"] = hac_tstat(diff)
        out["hac_se"] = hac_se(diff)
    out["ci95"] = ci95(diff)
    out["mde"] = mde(diff)
    return out


def ci95(diff: np.ndarray, hac: bool = True) -> list[float]:
    """95% CI of the mean daily difference (HAC se by default)."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return [float("nan"), float("nan")]
    se = hac_se(d) if hac else d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]


def mde(diff: np.ndarray, t_crit: float = 2.0, hac: bool = True) -> float:
    """Minimum detectable mean daily effect at |t| = ``t_crit`` for THIS series'
    noise level — the computed "underpowered" label of §4.3."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan")
    se = hac_se(d) if hac else d.std(ddof=1) / np.sqrt(len(d))
    return float(t_crit * se)


# ============================ point metrics ====================================
def sharpe(rets: np.ndarray) -> float:
    r = np.asarray(rets, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) <= 1e-300:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=1))


def cagr(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or v[0] <= 0:
        return float("nan")
    n_rets = len(v) - 1
    return float((v[-1] / v[0]) ** (TRADING_DAYS / n_rets) - 1.0)


def max_drawdown(values: np.ndarray) -> float:
    """Max peak-to-trough drawdown as a POSITIVE fraction."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float("nan")
    peak = np.maximum.accumulate(v)
    return float(np.max(1.0 - v / peak))


def win_rate(rets: np.ndarray) -> float:
    r = np.asarray(rets, dtype=np.float64)
    r = r[np.isfinite(r)]
    return float((r > 0).mean()) if len(r) else float("nan")


# ============================ run-dir extraction ===============================
def _read_daily_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    dates, vals = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            dates.append(row["date"])
            vals.append(float(row["value"]))
    return np.asarray(dates), np.asarray(vals, dtype=np.float64)


def _read_daily_series_csv(path: Path):
    """Shape-B daily_series.csv: date,raw_value,cost_adjusted_value."""
    dates, raw, adj = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            dates.append(row["date"])
            raw.append(float(row["raw_value"]))
            adj.append(float(row["cost_adjusted_value"]))
    return (np.asarray(dates), np.asarray(raw, dtype=np.float64),
            np.asarray(adj, dtype=np.float64))


def load_run(run_dir: Path) -> dict:
    """Load one replay run dir; accepts contract shape A or B (module docstring)."""
    run_dir = Path(run_dir)
    if (run_dir / "daily_series.csv").exists():            # shape B (landed runner)
        dc, vr, vc = _read_daily_series_csv(run_dir / "daily_series.csv")
        dr = dc
    else:                                                  # shape A (documented)
        dr, vr = _read_daily_csv(run_dir / "daily_raw.csv")
        dc, vc = _read_daily_csv(run_dir / "daily_costadj.csv")
    with open(run_dir / "result.json") as fh:
        result = json.load(fh)
    if "timeline" not in result and (run_dir / "timeline.json").exists():
        with open(run_dir / "timeline.json") as fh:
            result["timeline"] = json.load(fh)
    manifest = {}
    mp = run_dir / "manifest.json"
    if mp.exists():
        with open(mp) as fh:
            manifest = json.load(fh)
    return {"run_dir": str(run_dir), "dates_raw": dr, "values_raw": vr,
            "dates": dc, "values": vc, "result": result, "manifest": manifest}


def _slice_from(dates: np.ndarray, values: np.ndarray, start: str | None):
    if start is None:
        return dates, values
    keep = dates >= start
    return dates[keep], values[keep]


def run_metrics(run: dict, start: str | None = None) -> dict:
    """§4.2 reported metric set for one run over [start, end] (None = full).

    Forced choices (logged here, printed in every comparison MD):
    - realized round trips = count of executed SELL/REDUCE actions in
      result.json (exit events; the harness has no lot-matching ledger).
    - cumulative transaction costs = final raw value − final cost-adjusted
      value over the slice (the cost model's total dollar drag).
    - average gross exposure = mean of (ending_value − ending_cash)/ending_value
      over timeline rows in the slice.
    """
    dates, values = _slice_from(run["dates"], run["values"], start)
    dr, vr = _slice_from(run["dates_raw"], run["values_raw"], start)
    rets = daily_returns(values)
    actions = run["result"].get("actions", [])
    timeline = run["result"].get("timeline", [])
    in_slice = (lambda d: True) if start is None else (lambda d: d >= start)
    exits = sum(1 for a in actions
                if a.get("action") in ("SELL", "REDUCE") and in_slice(a.get("date", "")))
    gross = [(row["ending_value"] - row.get("ending_cash", 0.0)) / row["ending_value"]
             for row in timeline
             if in_slice(row.get("date", "")) and row.get("ending_value")]
    cum_cost = float("nan")
    if len(vr) and len(values):
        cum_cost = float((vr[-1] - vr[0]) - (values[-1] - values[0]))
    return {
        "window_start": str(dates[0]) if len(dates) else None,
        "window_end": str(dates[-1]) if len(dates) else None,
        "n_days": int(len(dates)),
        "total_return": float(values[-1] / values[0] - 1.0) if len(values) > 1 else float("nan"),
        "cagr": cagr(values),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(values),
        "win_rate": win_rate(rets),
        "realized_round_trips": exits,
        "cumulative_transaction_costs": cum_cost,
        "avg_gross_exposure": float(np.mean(gross)) if gross else float("nan"),
    }


# ============================ §4.2 verdict rule ================================
def beats_ties_loses(holdout_paired_t: float, holdout_dsharpe: float) -> dict:
    """TOURNAMENT §4.2, verbatim:
      BEATS iff t >= +1.0 AND ΔSharpe > 0
      LOSES iff t <= −1.0
      TIES  iff |t| < 1.0 (regardless of endpoint sign)
    The t >= +1.0 & ΔSharpe <= 0 cell is uncovered by the pre-registration —
    returned as "UNDEFINED (rule-gap)" with rule_gap=True (reported, not coerced).
    """
    t, ds = float(holdout_paired_t), float(holdout_dsharpe)
    if not np.isfinite(t):
        return {"verdict": "UNDEFINED (rule-gap)", "rule_gap": True,
                "reason": "paired t not computable"}
    if t >= 1.0 and ds > 0:
        return {"verdict": "BEATS", "rule_gap": False}
    if t <= -1.0:
        return {"verdict": "LOSES", "rule_gap": False}
    if abs(t) < 1.0:
        return {"verdict": "TIES", "rule_gap": False}
    return {"verdict": "UNDEFINED (rule-gap)", "rule_gap": True,
            "reason": f"t={t:+.3f} >= +1.0 but holdout dSharpe={ds:+.3f} <= 0 "
                      f"satisfies none of BEATS/TIES/LOSES as pre-registered"}


# ============================ §4.3 organ verdict ===============================
VERDICT_TAGS = ("positive", "0 (measured)", "0 (gate-honest)", "negative",
                "indeterminate")


def organ_verdict(e1_hac_t: float | None, e2_holdout_mean_delta: float | None,
                  gate_off: bool = False) -> dict:
    """TOURNAMENT §4.3 three-valued rule, mechanical:

      gate-off (EA gated the organ off)        -> "0 (gate-honest)"
      E1 HAC t >= +2.0 AND E2 mean Δ >= 0      -> "positive"
      E1 t <= −2.0                             -> "negative" (organ hurts)
      |E1 t| < 1.0                             -> "0 (measured)"
      else (1.0 <= |t| < 2.0, or t >= 2.0 with
            E2 sign disagreement, or missing)  -> "indeterminate"
    """
    if gate_off:
        return {"tag": "0 (gate-honest)", "reason": "EA gated the organ off; gate state reported"}
    if e1_hac_t is None or not np.isfinite(e1_hac_t):
        return {"tag": "indeterminate", "reason": "E1 HAC t unavailable"}
    t = float(e1_hac_t)
    if t >= 2.0:
        if e2_holdout_mean_delta is not None and np.isfinite(e2_holdout_mean_delta) \
                and float(e2_holdout_mean_delta) >= 0:
            return {"tag": "positive", "reason": f"E1 t={t:+.2f} >= +2.0, E2 sign confirms"}
        return {"tag": "indeterminate",
                "reason": f"E1 t={t:+.2f} >= +2.0 but E2 holdout sign disagrees/unavailable"}
    if t <= -2.0:
        return {"tag": "negative", "reason": f"E1 t={t:+.2f} <= -2.0 — the organ hurts"}
    if abs(t) < 1.0:
        return {"tag": "0 (measured)", "reason": f"E1 |t|={abs(t):.2f} < 1.0"}
    return {"tag": "indeterminate", "reason": f"1.0 <= |E1 t|={abs(t):.2f} < 2.0"}


def scorecard_attr(e2_holdout_dsharpe: float | None, verdict_tag: str) -> str:
    """§4.3 print format: `organ=<E2 holdout ΔSharpe> (<tag>)`; gate-honest and
    measured-zero print 0 as the value per the §4.7 examples."""
    if verdict_tag in ("0 (measured)", "0 (gate-honest)"):
        return f"0 ({verdict_tag.split('(')[1]}".rstrip()  # "0 (measured)" / "0 (gate-honest)"
    if e2_holdout_dsharpe is None or not np.isfinite(e2_holdout_dsharpe):
        return f"n/a ({verdict_tag})"
    return f"{e2_holdout_dsharpe:+.2f} ({verdict_tag})"
