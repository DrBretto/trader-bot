"""Aggregate member-training results for the PKT-TB-006 build report (read-only)."""
import json
from pathlib import Path

import numpy as np

PROTO = Path(__file__).resolve().parent


def load(m):
    p = PROTO / "models_out" / f"metrics_{m}.json"
    return json.loads(p.read_text()) if p.exists() else None


def row(d, key):
    return [None if d is None or f"F{f}" not in d or d[f"F{f}"].get(key) is None
            or (isinstance(d[f"F{f}"][key], float) and np.isnan(d[f"F{f}"][key]))
            else round(d[f"F{f}"][key], 4) for f in range(1, 7)]


def mean(vals):
    v = [x for x in vals if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def main():
    out = {}
    for m in ["cast", "cast_uniform", "ridge_twin", "gbm_cond",
              "gbm_cond_uniform", "event_head", "event_head_r3only"]:
        d = load(m)
        if d is None:
            out[m] = "MISSING"
            continue
        out[m] = {"purged_spearman": row(d, "purged_spearman"),
                  "weekly_rank_ic": row(d, "weekly_rank_ic"),
                  "mean_spearman": mean(row(d, "purged_spearman")),
                  "mean_weekly_ic": mean(row(d, "weekly_rank_ic")),
                  "wall_s": row(d, "wall_s")}
        if m.startswith("cast"):
            out[m]["gauss_nll"] = row(d, "gauss_nll")
            out[m]["sigma_abs_err_corr"] = row(d, "sigma_abs_err_corr")
            out[m]["mean_c1"] = mean(row(d, "mean_c"))
        if m.startswith("gbm"):
            out[m]["auc"] = row(d, "auc")
            out[m]["best_iter"] = row(d, "best_iter")
    r = load("risknet")
    if r:
        out["risknet"] = {k: row(r, k) for k in
                          ["qlike_vol5", "qlike_book", "mse_beta"]}
    # deltas
    def delta(a, b, key="purged_spearman"):
        if isinstance(out.get(a), dict) and isinstance(out.get(b), dict):
            pa, pb = out[a][key], out[b][key]
            return [None if x is None or y is None else round(x - y, 4)
                    for x, y in zip(pa, pb)]
        return None
    out["DELTA_cast_minus_ridge_spearman"] = delta("cast", "ridge_twin")
    out["DELTA_cast_minus_ridge_weekly_ic"] = delta("cast", "ridge_twin",
                                                    "weekly_rank_ic")
    out["DELTA_wrec_cast_minus_uniform"] = delta("cast", "cast_uniform")
    out["DELTA_wrec_gbm_minus_uniform"] = delta("gbm_cond", "gbm_cond_uniform")
    out["FUNCTIONALITY_BAR"] = {"weekly_rank_ic_bar": 0.02,
                                "cast_mean_weekly_ic":
                                    out.get("cast", {}).get("mean_weekly_ic")
                                    if isinstance(out.get("cast"), dict) else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
