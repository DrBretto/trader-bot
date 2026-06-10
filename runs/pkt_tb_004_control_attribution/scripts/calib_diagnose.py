#!/usr/bin/env python
"""calib_diagnose.py — PKT-TB-004 Model Calibration Specialist.

Diagnosis of the published regime_confidence series:
  - verify stored cosine disagreement against recomputation from stored probs
  - verify published regime_confidence == 1 - disagreement on no-override days
  - distribution + time series of confidence, per-model max-prob, label agreement
  - model-version (retraining boundary) breaks
  - failure-story classification: flat models vs confident-contradiction vs
    broken measure (cosine on near-one-hot orthogonal vectors)
  - label persistence by confidence bin; accuracy/ECE vs baseline-rule proxy

Outputs runs/pkt_tb_004_control_attribution/calib_diagnosis_stats.json (numbers
backing every claim in REGIME_CONFIDENCE_DIAGNOSIS.md).
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.dirname(__file__), "..")
HIST = os.path.join(D, "regime_confidence_history.json")
OUT = os.path.join(D, "calib_diagnosis_stats.json")
HOLDOUT_START = "2026-03-11"

data = json.load(open(HIST))
LABELS = data["labels_order"]
rows = [r for r in data["rows"] if r.get("gru_probs")]
rows.sort(key=lambda r: r["date"])

stats = {"n_ensemble_days": len(rows),
         "ensemble_span": [rows[0]["date"], rows[-1]["date"]],
         "holdout_start": HOLDOUT_START}

# C1. verify stored disagreement vs recomputed
diffs = [abs(r["disagreement_stored"] - r["disagreement_recomputed"])
         for r in rows if r.get("disagreement_stored") is not None]
stats["C1_disagreement_recompute_max_abs_diff"] = max(diffs)

# C2. published confidence vs 1 - disagreement (no-override days)
checks = []
for r in rows:
    pub = r.get("regime_confidence_published")
    if pub is None:
        continue
    if r.get("override_reason"):
        continue
    checks.append(abs(pub - (1.0 - r["disagreement_stored"])))
stats["C2_published_eq_1_minus_disagreement"] = {
    "n_no_override_days_checked": len(checks),
    "max_abs_diff": max(checks) if checks else None}
stats["C2_override_days"] = [
    {"date": r["date"], "override": r["override_reason"],
     "published_conf": r.get("regime_confidence_published")}
    for r in rows if r.get("override_reason")]

# C3. distributions
def pct(x, q):
    return float(np.percentile(x, q))

conf = np.array([1.0 - r["disagreement_stored"] for r in rows])
gmax = np.array([r["gru_maxp"] for r in rows])
tmax = np.array([r["tr_maxp"] for r in rows])
emax = np.array([r["ens_maxp"] for r in rows])
agree = np.array([bool(r["labels_agree"]) for r in rows])
jsdv = np.array([r["jsd"] for r in rows]) / math.log(2)  # normalized to [0,1]

stats["C3_distributions"] = {
    "confidence_1_minus_cosine_disagreement": {
        "mean": float(conf.mean()), "median": pct(conf, 50),
        "p10": pct(conf, 10), "p25": pct(conf, 25), "p75": pct(conf, 75),
        "min": float(conf.min()), "max": float(conf.max()),
        "n_below_0.5": int((conf < 0.5).sum()),
        "n_below_0.2": int((conf < 0.2).sum()),
        "n_below_0.1": int((conf < 0.1).sum()),
        "n_above_0.9": int((conf > 0.9).sum())},
    "gru_max_prob": {"mean": float(gmax.mean()), "median": pct(gmax, 50),
                     "p10": pct(gmax, 10), "min": float(gmax.min()),
                     "n_above_0.7": int((gmax > 0.7).sum()),
                     "n_below_0.4": int((gmax < 0.4).sum())},
    "transformer_max_prob": {"mean": float(tmax.mean()), "median": pct(tmax, 50),
                             "p10": pct(tmax, 10), "min": float(tmax.min()),
                             "n_above_0.7": int((tmax > 0.7).sum()),
                             "n_below_0.4": int((tmax < 0.4).sum())},
    "ensemble_max_prob": {"mean": float(emax.mean()), "median": pct(emax, 50),
                          "p10": pct(emax, 10), "min": float(emax.min())},
    "label_agreement_rate": float(agree.mean()),
    "n_label_disagree_days": int((~agree).sum()),
    "jsd_normalized": {"mean": float(jsdv.mean()), "median": pct(jsdv, 50),
                       "max": float(jsdv.max())},
    "corr_cosine_conf_vs_ens_maxp": float(np.corrcoef(conf, emax)[0, 1]),
}

# C4. story classification on label-disagree days
disagree_days = [r for r in rows if not r["labels_agree"]]
story = []
for r in disagree_days:
    story.append({
        "date": r["date"], "gru": r["gru_label"], "tr": r["tr_label"],
        "gru_maxp": round(r["gru_maxp"], 3), "tr_maxp": round(r["tr_maxp"], 3),
        "conf_published_proxy": round(1.0 - r["disagreement_stored"], 4),
        "jsd_norm": round(r["jsd"] / math.log(2), 3),
        "ens_label": r["ens_label"], "ens_maxp": round(r["ens_maxp"], 3),
        "final_label": r.get("final_regime_label"),
        "psm": r.get("position_size_modifier"),
        "throttle": r.get("risk_throttle_factor")})
stats["C4_label_disagree_days"] = story
both_confident = [s for s in story if s["gru_maxp"] > 0.6 and s["tr_maxp"] > 0.6]
stats["C4_summary"] = {
    "n_disagree_days": len(story),
    "n_both_models_maxp_gt_0.6": len(both_confident),
    "n_either_model_maxp_lt_0.4": len([s for s in story
                                       if s["gru_maxp"] < 0.4 or s["tr_maxp"] < 0.4]),
    "mean_gru_maxp_on_disagree_days": float(np.mean([s["gru_maxp"] for s in story])) if story else None,
    "mean_tr_maxp_on_disagree_days": float(np.mean([s["tr_maxp"] for s in story])) if story else None,
    "mean_conf_on_disagree_days": float(np.mean([s["conf_published_proxy"] for s in story])) if story else None,
    "mean_conf_on_agree_days": float(conf[agree].mean()),
}

# C5. agreement-day confidence: cosine vs maxp decoupling
stats["C5_measure_behavior"] = {
    "conf_on_agree_days": {"mean": float(conf[agree].mean()),
                           "min": float(conf[agree].min())},
    "conf_on_disagree_days": {"mean": float(conf[~agree].mean()),
                              "max": float(conf[~agree].max())},
    "note": "cosine confidence is effectively a binary same-argmax indicator for sharp models"}

# C6. model version breaks
by_ver = defaultdict(list)
for r in rows:
    by_ver[r.get("model_version") or "unknown"].append(r)
ver_stats = {}
for v, rs in sorted(by_ver.items()):
    c = np.array([1.0 - r["disagreement_stored"] for r in rs])
    a = np.array([bool(r["labels_agree"]) for r in rs])
    g = np.array([r["gru_maxp"] for r in rs])
    t = np.array([r["tr_maxp"] for r in rs])
    ver_stats[v] = {
        "n_days": len(rs), "span": [rs[0]["date"], rs[-1]["date"]],
        "mean_conf": float(c.mean()), "agree_rate": float(a.mean()),
        "mean_gru_maxp": float(g.mean()), "mean_tr_maxp": float(t.mean()),
        "gru_label_counts": dict(zip(*np.unique([r["gru_label"] for r in rs], return_counts=True))),
        "tr_label_counts": dict(zip(*np.unique([r["tr_label"] for r in rs], return_counts=True))),
        "ens_label_counts": dict(zip(*np.unique([r["ens_label"] for r in rs], return_counts=True)))}
    ver_stats[v]["gru_label_counts"] = {k: int(x) for k, x in ver_stats[v]["gru_label_counts"].items()}
    ver_stats[v]["tr_label_counts"] = {k: int(x) for k, x in ver_stats[v]["tr_label_counts"].items()}
    ver_stats[v]["ens_label_counts"] = {k: int(x) for k, x in ver_stats[v]["ens_label_counts"].items()}
stats["C6_by_model_version"] = ver_stats

# C7. label persistence by ensemble max-prob bin (k=1, 5 trading days ahead)
def persistence(k):
    out = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        if i + k >= len(rows):
            continue
        b = "high(>=0.6)" if r["ens_maxp"] >= 0.6 else ("mid(0.4-0.6)" if r["ens_maxp"] >= 0.4 else "low(<0.4)")
        out[b][1] += 1
        if rows[i + k]["ens_label"] == r["ens_label"]:
            out[b][0] += 1
    return {b: {"hold_rate": v[0] / v[1], "n": v[1]} for b, v in out.items()}

stats["C7_label_persistence"] = {"k1": persistence(1), "k5": persistence(5)}

# C8. accuracy & calibration vs baseline-rule proxy
def ece_acc(rs, get_probs, nbins=5):
    recs = []
    for r in rs:
        gt = r.get("baseline_rule_label")
        if not gt:
            continue
        p = get_probs(r)
        idx = int(np.argmax(p))
        recs.append((max(p), LABELS[idx] == gt, LABELS[idx], gt,
                     p[LABELS.index(gt)]))
    if not recs:
        return None
    maxp = np.array([x[0] for x in recs])
    correct = np.array([x[1] for x in recs])
    nll = -np.mean([math.log(max(x[4], 1e-12)) for x in recs])
    bins = np.linspace(0.2, 1.0, nbins + 1)
    ece = 0.0
    bin_detail = []
    for i in range(nbins):
        m = (maxp >= bins[i]) & (maxp < bins[i + 1] + (1e-9 if i == nbins - 1 else 0))
        if m.sum() == 0:
            continue
        gap = abs(maxp[m].mean() - correct[m].mean())
        ece += (m.sum() / len(recs)) * gap
        bin_detail.append({"bin": f"[{bins[i]:.2f},{bins[i+1]:.2f})",
                           "n": int(m.sum()), "mean_conf": float(maxp[m].mean()),
                           "accuracy": float(correct[m].mean())})
    return {"n": len(recs), "accuracy": float(correct.mean()),
            "mean_maxp": float(maxp.mean()), "ece": float(ece), "nll": float(nll),
            "bins": bin_detail}

for name, fn in [("gru", lambda r: r["gru_probs"]),
                 ("transformer", lambda r: r["tr_probs"]),
                 ("ensemble", lambda r: r["ens_probs"])]:
    stats[f"C8_calibration_vs_baseline_rule_{name}"] = {
        "full": ece_acc(rows, fn),
        "preholdout": ece_acc([r for r in rows if r["date"] < HOLDOUT_START], fn),
        "holdout": ece_acc([r for r in rows if r["date"] >= HOLDOUT_START], fn)}

# baseline-rule label distribution
gt_counts = defaultdict(int)
for r in rows:
    gt_counts[r.get("baseline_rule_label", "missing")] += 1
stats["C8_baseline_rule_label_counts"] = dict(gt_counts)

# C9. published confidence series (decisions.json) summary incl. the 4.3% days
pub = [(r["date"], r["regime_confidence_published"], r.get("final_regime_label"),
        r.get("override_reason"))
       for r in rows if r.get("regime_confidence_published") is not None]
pubv = np.array([x[1] for x in pub])
stats["C9_published_confidence"] = {
    "n": len(pub), "mean": float(pubv.mean()), "median": float(np.median(pubv)),
    "n_below_0.1": int((pubv < 0.1).sum()),
    "n_below_0.5": int((pubv < 0.5).sum()),
    "days_below_0.1": [{"date": d, "conf": round(c, 4), "final": f, "ovr": o}
                       for d, c, f, o in pub if c < 0.1]}

with open(OUT, "w") as f:
    json.dump(stats, f, indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
