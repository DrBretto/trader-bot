#!/usr/bin/env python
"""calib_pull_history.py — PKT-TB-004 Model Calibration Specialist.

Pulls the full regime-confidence history from S3 stored artifacts:
  - daily/<date>/inference.json  -> regime block (gru/transformer/ensemble probs,
    stored confidence, stored disagreement, position_size_multiplier, model version)
  - daily/<date>/decisions.json  -> expert_metrics (final_regime_label,
    regime_confidence as published, psm, throttle, expert inputs) + ensemble_metrics

Recomputes the cosine disagreement from stored prob vectors to verify the stored
value, and computes diagnostic quantities (max probs, label agreement, entropy, JSD).

Writes runs/pkt_tb_004_control_attribution/regime_confidence_history.json
Read-only against S3. No production code touched.
"""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError

BUCKET = "investment-system-data"
OUT = os.path.join(os.path.dirname(__file__), "..", "regime_confidence_history.json")
LABELS = ["calm_uptrend", "risk_on_trend", "risk_off_trend", "choppy", "high_vol_panic"]

s3 = boto3.client("s3")


def list_dates():
    paginator = s3.get_paginator("list_objects_v2")
    dates = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="daily/", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            d = p["Prefix"].split("/")[1]
            if len(d) == 10:
                dates.append(d)
    return sorted(dates)


def get_json(key):
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return json.loads(body)
    except (ClientError, json.JSONDecodeError):
        return None


def vec(probs):
    if not isinstance(probs, dict) or not probs:
        return None
    return [float(probs.get(l, 0.0)) for l in LABELS]


def cosine_disagreement(g, t):
    dot = sum(a * b for a, b in zip(g, t))
    ng = math.sqrt(sum(a * a for a in g))
    nt = math.sqrt(sum(b * b for b in t))
    if ng <= 0 or nt <= 0:
        return None
    return 1.0 - dot / (ng * nt)


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 1e-12)


def jsd(g, t):
    """Jensen-Shannon divergence, natural log, in [0, ln2]."""
    m = [(a + b) / 2 for a, b in zip(g, t)]

    def kl(p, q):
        return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 1e-12 and b > 1e-12)

    return 0.5 * kl(g, m) + 0.5 * kl(t, m)


def pull_one(date):
    inf = get_json(f"daily/{date}/inference.json")
    dec = get_json(f"daily/{date}/decisions.json")
    row = {"date": date}

    if inf:
        r = inf.get("regime", {}) or {}
        gru = (r.get("gru_prediction") or {})
        tr = (r.get("transformer_prediction") or {})
        gv, tv = vec(gru.get("probs")), vec(tr.get("probs"))
        row.update({
            "inference_date_field": inf.get("date"),
            "model_version": (inf.get("model_versions") or {}).get("regime"),
            "ens_label": r.get("label"),
            "ens_probs": vec(r.get("probs")),
            "ens_confidence_stored": r.get("confidence"),
            "disagreement_stored": r.get("disagreement"),
            "position_size_multiplier_stored": r.get("position_size_multiplier"),
            "gru_label": gru.get("label"),
            "gru_probs": gv,
            "tr_label": tr.get("label"),
            "tr_probs": tv,
        })
        if gv and tv:
            row["disagreement_recomputed"] = cosine_disagreement(gv, tv)
            row["jsd"] = jsd(gv, tv)
            row["gru_maxp"] = max(gv)
            row["tr_maxp"] = max(tv)
            row["gru_entropy"] = entropy(gv)
            row["tr_entropy"] = entropy(tv)
            row["labels_agree"] = (gru.get("label") == tr.get("label"))
            ev = row.get("ens_probs")
            if ev:
                row["ens_maxp"] = max(ev)
                row["ens_entropy"] = entropy(ev)

    if dec:
        em = dec.get("expert_metrics") or {}
        row.update({
            "final_regime_label": em.get("final_regime_label"),
            "regime_confidence_published": em.get("regime_confidence"),
            "position_size_modifier": em.get("position_size_modifier"),
            "risk_throttle_factor": em.get("risk_throttle_factor"),
            "override_reason": em.get("override_reason"),
            "effective_exposure_multiplier": em.get("effective_exposure_multiplier"),
            "macro_credit_score": em.get("macro_credit_score"),
            "vol_uncertainty_score": em.get("vol_uncertainty_score"),
            "vol_regime_label": em.get("vol_regime_label"),
            "fragility_score": em.get("fragility_score"),
            "entropy_shift_flag": em.get("entropy_shift_flag"),
        })
    return row


def main():
    dates = list_dates()
    print(f"{len(dates)} daily prefixes: {dates[0]} .. {dates[-1]}")
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(pull_one, dates))
    rows = [r for r in rows if len(r) > 1]
    with open(OUT, "w") as f:
        json.dump({"labels_order": LABELS, "rows": rows}, f, indent=1)
    n_inf = sum(1 for r in rows if r.get("gru_probs"))
    n_dec = sum(1 for r in rows if r.get("regime_confidence_published") is not None)
    print(f"wrote {len(rows)} rows ({n_inf} with inference probs, {n_dec} with decisions expert_metrics) -> {OUT}")


if __name__ == "__main__":
    main()
