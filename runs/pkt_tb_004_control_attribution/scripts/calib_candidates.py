#!/usr/bin/env python
"""calib_candidates.py — PKT-TB-004 Model Calibration Specialist.

Offline (diagnostic, non-replay) evaluation of three calibration candidates
against the stored 84-day ensemble history:

  C-TEMP     temperature scaling per model, T fitted on PRE-HOLDOUT days only
             (dates < 2026-03-11) by NLL against the baseline-rule pseudo-label
             (the models' own training target).
  C-COLLAPSE 5->3 class collapse: {calm_uptrend,risk_on_trend}->risk_on,
             {choppy}->neutral, {risk_off_trend,high_vol_panic}->risk_off.
             Hard-override inputs (panic_prob, trend prob) stay raw 5-class.
  C-MEASURE  disagreement measure swap: cosine -> normalized Jensen-Shannon
             divergence (proper divergence on the simplex). Labels unchanged.

For each candidate the full fusion chain (src/signals/regime_fusion.decide_regime_v3,
imported read-only) is recomputed with stored expert inputs, and per-day diffs in
final_regime_label / position_size_modifier / risk_throttle / effective exposure /
cash-reserve tier are counted vs the SAME machinery run on uncalibrated inputs
(baseline replay), after validating that baseline replay reproduces the published
decisions.json values.

No production code is modified. Output: calib_candidate_eval.json.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = "/Users/drbretto/Desktop/Projects/trader-bot"
sys.path.insert(0, ROOT)
from src.signals.regime_fusion import decide_regime_v3  # noqa: E402  (read-only import)

D = os.path.join(ROOT, "runs", "pkt_tb_004_control_attribution")
HIST = os.path.join(D, "regime_confidence_history.json")
OUT = os.path.join(D, "calib_candidate_eval.json")
HOLDOUT_START = "2026-03-11"

data = json.load(open(HIST))
LABELS = data["labels_order"]
L = {l: i for i, l in enumerate(LABELS)}
rows = sorted([r for r in data["rows"] if r.get("gru_probs")], key=lambda r: r["date"])

# active production fusion params (config/decision_params.active.json regime_fusion)
ACTIVE_FUSION = {"fragility_relax_in_risk_on": True,
                 "fragility_relax_confidence": 0.8,
                 "fragility_relax_regimes": ["risk_on_trend", "calm_uptrend"]}
CASH_RESERVE = {"calm_uptrend": 0.1, "risk_on_trend": 0.1, "choppy": 0.2,
                "risk_off_trend": 0.4, "high_vol_panic": 0.4}
REGIME_ADJ = {"calm_uptrend": 1.10, "risk_on_trend": 1.10, "choppy": 0.90,
              "risk_off_trend": 0.80, "high_vol_panic": 0.50}


def cosine_disagreement(g, t):
    dot = float(np.dot(g, t))
    n = float(np.linalg.norm(g) * np.linalg.norm(t))
    return max(0.0, min(1.0, 1.0 - dot / n)) if n > 0 else 0.0


def jsd_norm(g, t):
    g, t = np.asarray(g), np.asarray(t)
    m = (g + t) / 2

    def kl(p, q):
        mask = p > 1e-12
        return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))

    return (0.5 * kl(g, m) + 0.5 * kl(t, m)) / math.log(2)


def position_multiplier(confidence, disagreement, threshold=0.3):
    """Verbatim logic of EnsembleRegimeModel._compute_position_multiplier."""
    conf_mult = 0.5 + 0.5 * confidence
    if disagreement > threshold:
        penalty = (disagreement - threshold) / (1 - threshold)
        conf_mult *= (1 - 0.5 * penalty)
    return max(0.5, min(1.0, conf_mult))


def temp_scale(p, T):
    logp = np.log(np.maximum(np.asarray(p), 1e-12)) / T
    e = np.exp(logp - logp.max())
    return e / e.sum()


def run_fusion(ens_label, probs5, disagreement, multiplier, r, fusion_params):
    """Recompute decide_regime_v3 with stored expert inputs for row r."""
    return decide_regime_v3(
        ensemble_regime_label=ens_label,
        trend_risk_on_prob=probs5[L["risk_on_trend"]],
        panic_prob=probs5[L["high_vol_panic"]],
        ensemble_disagreement=disagreement,
        ensemble_multiplier=multiplier,
        macro_credit_score=r.get("macro_credit_score", 0.0),
        vol_uncertainty_score=r.get("vol_uncertainty_score", 0.5),
        vol_regime_label=r.get("vol_regime_label", "calm"),
        fragility_score=r.get("fragility_score", 0.5),
        entropy_score=0.5,
        entropy_shift_flag=bool(r.get("entropy_shift_flag", False)),
        params=fusion_params,
    )


fusion_rows = [r for r in rows if r.get("regime_confidence_published") is not None]
out = {"n_ensemble_days": len(rows), "n_fusion_days": len(fusion_rows),
       "holdout_start": HOLDOUT_START}

# fragility-relax went live with the 2026-04-30 recalibration
# (config/decision_params.recalibrated_2026_04_30.json) — era-aware params.
RELAX_LIVE_DATE = "2026-04-30"


def era_params(date):
    return ACTIVE_FUSION if date >= RELAX_LIVE_DATE else None


# ---------- baseline replay validation ----------
for tag, fp_fn in [("default_params", lambda d: None),
                   ("active_params", lambda d: ACTIVE_FUSION),
                   ("era_params", era_params)]:
    match_label = match_psm = match_thr = 0
    mism = []
    for r in fusion_rows:
        f = run_fusion(r["ens_label"], r["ens_probs"], r["disagreement_stored"],
                       r["position_size_multiplier_stored"], r, fp_fn(r["date"]))
        ok_l = f["final_regime_label"] == r["final_regime_label"]
        ok_p = abs(f["position_size_modifier"] - r["position_size_modifier"]) < 1e-6
        ok_t = abs(f["risk_throttle_factor"] - r["risk_throttle_factor"]) < 1e-6
        match_label += ok_l
        match_psm += ok_p
        match_thr += ok_t
        if not (ok_l and ok_p and ok_t):
            mism.append({"date": r["date"],
                         "replay": [f["final_regime_label"],
                                    round(f["position_size_modifier"], 4),
                                    round(f["risk_throttle_factor"], 4)],
                         "published": [r["final_regime_label"],
                                       round(r["position_size_modifier"], 4),
                                       round(r["risk_throttle_factor"], 4)]})
    out[f"V_baseline_replay_{tag}"] = {
        "label_match": f"{match_label}/{len(fusion_rows)}",
        "psm_match": f"{match_psm}/{len(fusion_rows)}",
        "throttle_match": f"{match_thr}/{len(fusion_rows)}",
        "mismatches": mism[:12]}

# era-aware params chosen after validation (see V_baseline_replay_era_params)

# ---------- C-TEMP: fit temperatures on pre-holdout only ----------
pre = [r for r in rows if r["date"] < HOLDOUT_START and r.get("baseline_rule_label")]


def fit_T(key):
    def nll(logT):
        T = math.exp(logT)
        tot = 0.0
        for r in pre:
            p = temp_scale(r[key], T)
            tot -= math.log(max(p[L[r["baseline_rule_label"]]], 1e-12))
        return tot / len(pre)

    res = minimize_scalar(nll, bounds=(math.log(0.2), math.log(50)), method="bounded")
    return math.exp(res.x), res.fun, nll(0.0)


T_gru, nll_gru_after, nll_gru_before = fit_T("gru_probs")
T_tr, nll_tr_after, nll_tr_before = fit_T("tr_probs")
out["C_TEMP_fit"] = {
    "fit_window": [pre[0]["date"], pre[-1]["date"]], "n_fit_days": len(pre),
    "T_gru": round(T_gru, 4), "T_transformer": round(T_tr, 4),
    "preholdout_nll_gru_before_after": [round(nll_gru_before, 4), round(nll_gru_after, 4)],
    "preholdout_nll_tr_before_after": [round(nll_tr_before, 4), round(nll_tr_after, 4)]}

# holdout NLL/ECE before vs after (per model + ensemble)
hold = [r for r in rows if r["date"] >= HOLDOUT_START and r.get("baseline_rule_label")]


def eval_set(rs, transform):
    nll, correct, maxps = [], [], []
    for r in rs:
        g, t = transform(r)
        e = (np.asarray(g) + np.asarray(t)) / 2
        e = e / e.sum()
        gt = L[r["baseline_rule_label"]]
        nll.append(-math.log(max(e[gt], 1e-12)))
        correct.append(int(np.argmax(e)) == gt)
        maxps.append(float(e.max()))
    return {"nll": round(float(np.mean(nll)), 4),
            "acc": round(float(np.mean(correct)), 4),
            "mean_maxp": round(float(np.mean(maxps)), 4),
            "conf_acc_gap": round(float(np.mean(maxps) - np.mean(correct)), 4)}


ident = lambda r: (r["gru_probs"], r["tr_probs"])
scaled = lambda r: (temp_scale(r["gru_probs"], T_gru), temp_scale(r["tr_probs"], T_tr))
out["C_TEMP_holdout_ensemble_eval"] = {"before": eval_set(hold, ident),
                                       "after": eval_set(hold, scaled)}
out["C_TEMP_preholdout_ensemble_eval"] = {"before": eval_set(pre, ident),
                                          "after": eval_set(pre, scaled)}

# ---------- per-candidate downstream chain ----------
COLLAPSE = {"calm_uptrend": "risk_on", "risk_on_trend": "risk_on",
            "choppy": "neutral", "risk_off_trend": "risk_off",
            "high_vol_panic": "risk_off"}
BACK = {"risk_on": "risk_on_trend", "neutral": "choppy", "risk_off": "risk_off_trend"}
G3 = ["risk_on", "neutral", "risk_off"]


def candidate_chain(r, which):
    """Return (ens_label, probs5_for_fusion, disagreement, multiplier)."""
    g = np.asarray(r["gru_probs"])
    t = np.asarray(r["tr_probs"])
    if which == "TEMP":
        g, t = temp_scale(g, T_gru), temp_scale(t, T_tr)
        e = (g + t) / 2
        e = e / e.sum()
        label = LABELS[int(np.argmax(e))]
        dis = cosine_disagreement(g, t)
        mult = position_multiplier(float(e.max()), dis)
        return label, e.tolist(), dis, mult
    if which == "COLLAPSE":
        g3 = np.array([g[L["calm_uptrend"]] + g[L["risk_on_trend"]], g[L["choppy"]],
                       g[L["risk_off_trend"]] + g[L["high_vol_panic"]]])
        t3 = np.array([t[L["calm_uptrend"]] + t[L["risk_on_trend"]], t[L["choppy"]],
                       t[L["risk_off_trend"]] + t[L["high_vol_panic"]]])
        e3 = (g3 + t3) / 2
        label = BACK[G3[int(np.argmax(e3))]]
        dis = cosine_disagreement(g3, t3)
        mult = position_multiplier(float(e3.max()), dis)
        # hard-override inputs stay raw 5-class
        return label, r["ens_probs"], dis, mult
    if which == "MEASURE":
        dis = jsd_norm(g, t)
        e = np.asarray(r["ens_probs"])
        mult = position_multiplier(float(e.max()), dis)
        return r["ens_label"], r["ens_probs"], dis, mult
    raise ValueError(which)


results = {}
for cand in ["TEMP", "COLLAPSE", "MEASURE"]:
    diffs = []
    n_label = n_psm = n_thr = 0
    d_psm, d_eff, d_cash = [], [], []
    hold_d_eff = []
    label_flips = {}
    ens_label_changes = 0
    for r in fusion_rows:
        fp = era_params(r["date"])
        base = run_fusion(r["ens_label"], r["ens_probs"], r["disagreement_stored"],
                          r["position_size_multiplier_stored"], r, fp)
        lab, p5, dis, mult = candidate_chain(r, cand)
        new = run_fusion(lab, p5, dis, mult, r, fp)
        if lab != r["ens_label"]:
            ens_label_changes += 1
        dl = new["final_regime_label"] != base["final_regime_label"]
        dp = abs(new["position_size_modifier"] - base["position_size_modifier"]) > 1e-6
        dt = abs(new["risk_throttle_factor"] - base["risk_throttle_factor"]) > 1e-6
        n_label += dl
        n_psm += dp
        n_thr += dt
        dpsm = new["position_size_modifier"] - base["position_size_modifier"]
        deff = (new["effective_exposure_multiplier"] - base["effective_exposure_multiplier"])
        dcash = (CASH_RESERVE.get(new["final_regime_label"], 0.1)
                 - CASH_RESERVE.get(base["final_regime_label"], 0.1))
        d_psm.append(dpsm)
        d_eff.append(deff)
        d_cash.append(dcash)
        if r["date"] >= HOLDOUT_START:
            hold_d_eff.append(deff)
        if dl:
            k = f'{base["final_regime_label"]}->{new["final_regime_label"]}'
            label_flips[k] = label_flips.get(k, 0) + 1
        if dl or dp or dt:
            diffs.append({"date": r["date"],
                          "base": [base["final_regime_label"],
                                   round(base["position_size_modifier"], 3),
                                   round(base["risk_throttle_factor"], 3)],
                          "cand": [new["final_regime_label"],
                                   round(new["position_size_modifier"], 3),
                                   round(new["risk_throttle_factor"], 3)],
                          "d_eff_exposure": round(deff, 3)})
    results[cand] = {
        "n_days_evaluated": len(fusion_rows),
        "ens_label_changes": ens_label_changes,
        "final_label_diff_days": n_label,
        "psm_diff_days": n_psm,
        "throttle_diff_days": n_thr,
        "label_flip_kinds": label_flips,
        "mean_d_psm": round(float(np.mean(d_psm)), 4),
        "mean_d_eff_exposure": round(float(np.mean(d_eff)), 4),
        "mean_d_eff_exposure_holdout": round(float(np.mean(hold_d_eff)), 4) if hold_d_eff else None,
        "mean_d_cash_reserve_tier": round(float(np.mean(d_cash)), 4),
        "diff_days": diffs}
out["candidates"] = results

# measure-divergence stats (cosine vs JSD vs maxp confidence)
cosc = np.array([1 - r["disagreement_stored"] for r in rows])
jsdc = np.array([1 - r["jsd"] / math.log(2) for r in rows])
mxp = np.array([r["ens_maxp"] for r in rows])
out["measure_divergence"] = {
    "mean_abs_cosconf_minus_jsdconf": round(float(np.abs(cosc - jsdc).mean()), 4),
    "max_abs_cosconf_minus_jsdconf": round(float(np.abs(cosc - jsdc).max()), 4),
    "corr_cos_jsd": round(float(np.corrcoef(cosc, jsdc)[0, 1]), 4),
    "corr_cos_maxp": round(float(np.corrcoef(cosc, mxp)[0, 1]), 4),
    "corr_jsd_maxp": round(float(np.corrcoef(jsdc, mxp)[0, 1]), 4)}

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=1))
for c, v in results.items():
    print("=" * 10, c)
    print(json.dumps({k: x for k, x in v.items() if k != "diff_days"}, indent=1))
    print("first diff days:", json.dumps(v["diff_days"][:6], indent=1))
