#!/usr/bin/env python
"""calib_add_groundtruth.py — PKT-TB-004 Model Calibration Specialist.

Adds a ground-truth-proxy regime label per ensemble day by applying the SAME
pseudo-label rule the models were trained on (training/utils/metrics.py::
compute_regime_labels_from_baseline) to the stored daily context.parquet
snapshot. This is the most defensible proxy available: the models were trained
to reproduce this rule, so calibration against it measures fit to the training
target, not to an externally-imposed regime definition. Limits are stated in
the diagnosis doc.

Merges `baseline_rule_label` + the rule inputs into regime_confidence_history.json.
Read-only against S3.
"""
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
from botocore.exceptions import ClientError

BUCKET = "investment-system-data"
HIST = os.path.join(os.path.dirname(__file__), "..", "regime_confidence_history.json")

s3 = boto3.client("s3")


def baseline_rule(spy_21d_ret, spy_21d_vol, credit_stress, vixy_21d_ret):
    """Verbatim port of training/utils/metrics.py::compute_regime_labels_from_baseline."""
    vol_p85 = 0.14
    vol_p50 = 0.10
    vol_p40 = 0.08
    if (vixy_21d_ret > 0.10 or spy_21d_vol > vol_p85) and spy_21d_ret < -0.02:
        return 'high_vol_panic'
    elif spy_21d_ret < -0.01 or credit_stress < -0.01:
        return 'risk_off_trend'
    elif spy_21d_ret > 0.03 and spy_21d_vol < vol_p40:
        return 'calm_uptrend'
    elif abs(spy_21d_ret) < 0.01 and spy_21d_vol > vol_p50:
        return 'choppy'
    else:
        return 'risk_on_trend'


def pull_context(date):
    try:
        body = s3.get_object(Bucket=BUCKET, Key=f"daily/{date}/context.parquet")["Body"].read()
    except ClientError:
        return date, None
    df = pd.read_parquet(io.BytesIO(body))
    if len(df) == 0:
        return date, None
    row = df.iloc[-1]
    ctx = {
        "spy_return_21d": float(row.get("spy_return_21d", 0) or 0),
        "spy_vol_21d": float(row.get("spy_vol_21d", 0) or 0),
        "credit_spread_proxy": float(row.get("credit_spread_proxy", 0) or 0),
        "vixy_return_21d": float(row.get("vixy_return_21d", 0) or 0),
        "context_date": str(row.get("date", "")),
    }
    ctx["baseline_rule_label"] = baseline_rule(
        ctx["spy_return_21d"], ctx["spy_vol_21d"],
        ctx["credit_spread_proxy"], ctx["vixy_return_21d"],
    )
    return date, ctx


def main():
    data = json.load(open(HIST))
    rows = data["rows"]
    ens_dates = [r["date"] for r in rows if r.get("gru_probs")]
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = dict(ex.map(pull_context, ens_dates))
    n = 0
    for r in rows:
        ctx = results.get(r["date"])
        if ctx:
            r.update(ctx)
            n += 1
    with open(HIST, "w") as f:
        json.dump(data, f, indent=1)
    print(f"added baseline_rule_label to {n}/{len(ens_dates)} ensemble days")


if __name__ == "__main__":
    main()
