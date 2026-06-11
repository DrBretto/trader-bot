"""PKT-TB-007 — nightly organ-input precompute (BUILD_SPEC_007 §6.2).

Writes store/nightly_007/<D>.json (schema organ_inputs_007.v1, frozen in
tilt_adapter.py) for every panel date in the live window
2026-01-31..2026-06-09, from the DEPLOY-grade organ artifacts (trained with
per-horizon pre-holdout cutoffs; scoring dates >= 2026-03-11 is E1/E2
inference, never training/selection).

Emission policy (ledgered): mu/q are emitted for ALL directional organs
(M1, M2, M5) plus disp_z (M3) and p_exceed (M4) regardless of gate outcome;
the manifest carries the roster and per-organ gate state, and the
genome/arm config decides what is listened to (challenger arms need
failed organs' outputs; the adapter's neutral path covers absent organs).

q convention (spec: "rank-z dispersion of mu scaled to [0,1]"): trailing-252d
percentile rank of the day's cross-sectional sd of raw mu over the support
set — point-in-time, deterministic; interpretation ledgered. M5's q =
episode magnitude (mag x decay), per spec.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
sys.path.insert(0, str(PROTO))

import folds as F                                   # noqa: E402
import organs_007 as G                              # noqa: E402

LIVE_START = "2026-01-31"
LIVE_END = "2026-06-09"
MODELS = PROTO / "models_out_007"
OUT = PROTO / "store" / "nightly_007"
SUPPORT = G.SUPPORT


def trailing_pct(series: np.ndarray, window: int = 252, minp: int = 64):
    s = pd.Series(series)
    return s.rolling(window + 1, min_periods=minp).apply(
        lambda w: float((w.iloc[-1] > w.iloc[:-1]).mean()), raw=False) \
        .to_numpy()


def main():
    t0 = time.time()
    import torch
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    p6 = {k: z6[k] for k in ["dates", "symbols", "X", "scalars", "T",
                             "T_cols", "B", "B_cols", "symbol_mask",
                             "gdelt_available", "X_features"]}
    z7 = np.load(PROTO / "store" / "panel_007.npz", allow_pickle=False)
    p7 = {k: z7[k] for k in z7.files}
    dates = np.asarray(p6["dates"]).astype(str)
    symbols = [str(s) for s in p6["symbols"]]
    sup_j = np.array([symbols.index(s) for s in SUPPORT])
    sm = p6["symbol_mask"].astype(bool)
    N = len(dates)
    live = np.nonzero((dates >= LIVE_START) & (dates <= LIVE_END))[0]

    # ---------------- M1 deploy inference (2-seed mean) --------------------
    xf = [str(c) for c in p6["X_features"]]
    cols10 = [xf.index(c) for c in G.M1_XCOLS]
    X10 = np.ascontiguousarray(p6["X"][:, :, :, cols10])
    sec_ids, cls_ids = G.sector_class_ids(REPO / "config" / "universe.csv")
    sec_t = torch.from_numpy(sec_ids)
    cls_t = torch.from_numpy(cls_ids)
    mus = []
    for seed in F.CAST_OOF_SEEDS_007:
        model = G.build_cast_xs(seed)
        model.load_state_dict(torch.load(MODELS / "m1_cast" / f"seed_{seed}.pt",
                                         weights_only=True))
        model.eval()
        mu = np.zeros((N, len(symbols)), dtype=np.float64)
        with torch.no_grad():
            for c0 in range(0, N, 64):
                sl = slice(c0, min(c0 + 64, N))
                m, _, _ = model(torch.from_numpy(X10[sl]),
                                torch.from_numpy(p6["scalars"][sl]),
                                sec_t, cls_t, torch.from_numpy(sm[sl]))
                mu[sl] = m.numpy()
        mus.append(mu)
    mu_m1 = np.mean(np.stack(mus), axis=0)
    q_m1 = trailing_pct(np.nanstd(mu_m1[:, sup_j], axis=1))

    # ---------------- M2 deploy inference ----------------------------------
    with (MODELS / "m2_rot" / "model.pkl").open("rb") as fh:
        m2 = pickle.load(fh)
    Dall, _, _, _ = G.m2_design(p6, p7, np.arange(N), m2["sym_is_credit"])
    P2 = np.full((N, len(symbols)), np.nan)
    flat = Dall.reshape(N * len(symbols), -1)
    vm = sm.reshape(-1)                     # predict wherever the symbol exists
    P2.reshape(-1)[vm] = m2["clf"].predict(flat[vm])
    q_m2 = trailing_pct(np.nanstd(P2[:, sup_j], axis=1))

    # ---------------- M3 deploy forecast + disp_z ---------------------------
    z3 = np.load(MODELS / "m3_disp" / "coefs.npz")
    X_full, _, _, _ = G.m3_design(
        {k: z6[k] for k in ["T", "T_cols"]}, p7)
    fc = G.ols_predict(z3["coef_full"], np.nan_to_num(X_full, nan=0.0))
    fz = pd.Series(fc)
    mz_ = fz.shift(1).rolling(252, min_periods=60).mean()
    sz_ = fz.shift(1).rolling(252, min_periods=60).std()
    disp_z = ((fz - mz_) / sz_).to_numpy()

    # ---------------- M4 deploy (shipped variant per gates) -----------------
    gates = json.loads((PROTO / "gates" / "acceptance_007.json").read_text())
    ships = gates["gates"]["M4"]["ab_decision"]["ships"]    # "M4-A"/"M4-B"
    member4 = "m4_evt_a" if ships == "M4-A" else "m4_evt_b"
    with (MODELS / member4 / "model.pkl").open("rb") as fh:
        m4 = pickle.load(fh)
    designs = G.m4_design(
        {k: p6[k] for k in ["B", "B_cols", "gdelt_available"]}, p7, dates)
    X4 = designs["A" if ships == "M4-A" else "B"]
    p4 = G.m4_predict(m4["fitted"], X4, np.arange(N))       # [N,27]
    bucket_sym = m4["bucket_sym"]
    # per-symbol exceedance prob = mean over buckets containing the symbol
    p_sym = np.zeros((N, len(symbols)))
    cnt = np.zeros(len(symbols))
    for k in range(bucket_sym.shape[0]):
        p_sym[:, bucket_sym[k]] += p4[:, k][:, None]
        cnt[bucket_sym[k]] += 1
    cnt[cnt == 0] = 1
    p_sym /= cnt[None, :]

    # ---------------- M5 rule ------------------------------------------------
    m5sig = np.load(PROTO / "store" / "m5_signal.npz")
    mu_m5 = m5sig["mu"].astype(np.float64)
    mag_m5 = m5sig["mag"].astype(np.float64)

    roster = gates["roster"]
    OUT.mkdir(parents=True, exist_ok=True)
    man_common = {
        "schema": "organ_inputs_007.v1",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "code_sha": G.code_sha(), "seeds": {"m1": F.CAST_OOF_SEEDS_007,
                                            "master": F.MASTER_SEED_007},
        "roster": roster, "m4_variant_shipped": ships,
        "emission_policy": "all organs emitted; genome/arm config decides "
                           "what is listened to (ledgered)",
        "deploy_cutoffs": {"m1": "2026-03-03", "m2": "~2026-02-06 (h=21)",
                           "m3": "2026-03-03", "m4": "2026-03-03",
                           "m5": "rule (no training)"},
    }
    n_written = 0
    for i in live:
        d = str(dates[i])
        organs = {}
        organs["M1"] = {
            "mu": {s: round(float(mu_m1[i, j]), 6)
                   for s, j in zip(SUPPORT, sup_j) if sm[i, j]},
            "q": round(float(np.nan_to_num(q_m1[i], nan=0.0)), 4)}
        organs["M2"] = {
            "mu": {s: round(float(P2[i, j]), 6)
                   for s, j in zip(SUPPORT, sup_j)
                   if sm[i, j] and np.isfinite(P2[i, j])},
            "q": round(float(np.nan_to_num(q_m2[i], nan=0.0)), 4)}
        if np.any(mu_m5[i] != 0):
            organs["M5"] = {
                "mu": {s: round(float(mu_m5[i, j]), 6)
                       for s, j in zip(SUPPORT, sup_j) if mu_m5[i, j] != 0},
                "q": round(float(min(mag_m5[i], 1.0)), 4)}
        doc = {"date": d, "schema_version": "organ_inputs_007.v1",
               "organs": organs,
               "disp_z": (round(float(disp_z[i]), 6)
                          if np.isfinite(disp_z[i]) else None),
               "p_exceed": {s: round(float(p_sym[i, j]), 4)
                            for s, j in zip(SUPPORT, sup_j) if sm[i, j]},
               "manifest": man_common}
        (OUT / f"{d}.json").write_text(json.dumps(doc, indent=1))
        n_written += 1
    print(f"nightly_007: {n_written} dates written "
          f"({dates[live[0]]}..{dates[live[-1]]}) in {time.time()-t0:.0f}s")
    with (PROTO / "validation_looks_007.jsonl").open("a") as f:
        f.write(json.dumps({
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "phase": "C-wave2a", "kind": "build_artifact",
            "component": "precompute_nightly_007",
            "decision": f"{n_written} organ_inputs_007.v1 files written for "
                        f"the live window; emission policy = all organs + "
                        f"roster in manifest",
            "provenance": "BUILD_SPEC_007 §6.2; scoring >= 2026-03-11 is "
                          "inference for E1/E2 replay, no training/selection "
                          "touched the holdout"}) + "\n")


def write_fold_packs():
    """OOF-true fold packs store/nightly_007_folds/<D>.json (EA contract,
    ea_007.py header). Every mu/forecast/p for date D comes from the OOF
    matrix of the fold containing D (model trained strictly pre-fold,
    walk-forward) — never from deploy artifacts. q/disp_z trailing windows
    run over the chronologically concatenated OOF series (all values OOF;
    window may span a fold boundary — point-in-time safe)."""
    t0 = time.time()
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    dates = np.asarray(z6["dates"]).astype(str)
    symbols = [str(s) for s in z6["symbols"]]
    sm = z6["symbol_mask"].astype(bool)
    sup_j = np.array([symbols.index(s) for s in SUPPORT])
    didx = {d: i for i, d in enumerate(dates)}
    out_dir = PROTO / "store" / "nightly_007_folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_concat(member, key):
        rows, vals = [], []
        for f in range(1, 7):
            z = np.load(PROTO / "oof" / f"fold_{f}_{member}.npz",
                        allow_pickle=True)
            idx = np.array([didx[str(d)] for d in
                            np.asarray(z["dates"]).astype(str)])
            rows.append(idx)
            vals.append(np.asarray(z[key], dtype=np.float64))
        rows = np.concatenate(rows)
        vals = np.concatenate(vals)
        o = np.argsort(rows)
        return rows[o], vals[o]

    r1, mu1 = load_concat("m1_cast", "mu")
    r2, mu2 = load_concat("m2_rot", "mu")
    r3, fc3 = load_concat("m3_disp", "forecast")
    if not (np.array_equal(r1, r2) and np.array_equal(r1, r3)):
        raise AssertionError("OOF fold-day indices differ across members")

    gates = json.loads((PROTO / "gates" / "acceptance_007.json").read_text())
    ships = gates["gates"]["M4"]["ab_decision"]["ships"]
    member4 = "m4_evt_a" if ships == "M4-A" else "m4_evt_b"
    r4, p4 = load_concat(member4, "p")
    if not np.array_equal(r1, r4):
        raise AssertionError("M4 OOF fold-day indices differ")
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    buckets = [str(b) for b in z6["buckets"]]
    p_sym = np.zeros((len(r4), len(symbols)))
    cnt = np.zeros(len(symbols))
    for k, b in enumerate(buckets):
        for s in bucket_map[b]:
            if s in symbols:
                j = symbols.index(s)
                p_sym[:, j] += p4[:, k]
                cnt[j] += 1
    cnt[cnt == 0] = 1
    p_sym /= cnt[None, :]

    q1 = trailing_pct(np.nanstd(mu1[:, sup_j], axis=1))
    q2 = trailing_pct(np.nanstd(mu2[:, sup_j], axis=1))
    fz = pd.Series(fc3)
    disp_z = ((fz - fz.shift(1).rolling(252, min_periods=60).mean())
              / fz.shift(1).rolling(252, min_periods=60).std()).to_numpy()

    m5sig = np.load(PROTO / "store" / "m5_signal.npz")
    mu_m5 = m5sig["mu"].astype(np.float64)
    mag_m5 = m5sig["mag"].astype(np.float64)

    roster_doc = json.loads((PROTO / "store" / "roster_007.json").read_text())
    man_common = {
        "schema": "organ_inputs_007.v1", "oof_true": True,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "code_sha": G.code_sha(),
        "seeds": {"m1": F.CAST_OOF_SEEDS_007, "master": F.MASTER_SEED_007},
        "roster": roster_doc["roster"], "m4_variant_shipped": ships,
        "emission_policy": "all organs emitted; genome/arm config decides "
                           "what is listened to (ledgered)",
        "provenance": "walk-forward OOF matrices (oof/fold_<f>_<member>.npz);"
                      " q/disp_z trailing windows over the concatenated OOF "
                      "series",
    }
    n_written = 0
    for k, i in enumerate(r1):
        d = str(dates[i])
        organs = {}
        organs["M1"] = {
            "mu": {s: round(float(mu1[k, j]), 6)
                   for s, j in zip(SUPPORT, sup_j)
                   if sm[i, j] and np.isfinite(mu1[k, j])},
            "q": round(float(np.nan_to_num(q1[k], nan=0.0)), 4)}
        organs["M2"] = {
            "mu": {s: round(float(mu2[k, j]), 6)
                   for s, j in zip(SUPPORT, sup_j)
                   if sm[i, j] and np.isfinite(mu2[k, j])},
            "q": round(float(np.nan_to_num(q2[k], nan=0.0)), 4)}
        if np.any(mu_m5[i] != 0):
            organs["M5"] = {
                "mu": {s: round(float(mu_m5[i, j]), 6)
                       for s, j in zip(SUPPORT, sup_j) if mu_m5[i, j] != 0},
                "q": round(float(min(mag_m5[i], 1.0)), 4)}
        doc = {"date": d, "schema_version": "organ_inputs_007.v1",
               "organs": organs,
               "disp_z": (round(float(disp_z[k]), 6)
                          if np.isfinite(disp_z[k]) else None),
               "p_exceed": {s: round(float(p_sym[k, j]), 4)
                            for s, j in zip(SUPPORT, sup_j) if sm[i, j]},
               "manifest": man_common}
        (out_dir / f"{d}.json").write_text(json.dumps(doc, indent=1))
        n_written += 1
    print(f"nightly_007_folds: {n_written} OOF-true dates written "
          f"({dates[r1[0]]}..{dates[r1[-1]]}) in {time.time()-t0:.0f}s")
    with (PROTO / "validation_looks_007.jsonl").open("a") as f:
        f.write(json.dumps({
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "phase": "C-wave2a", "kind": "build_artifact",
            "component": "precompute_nightly_007 --fold-packs",
            "decision": f"{n_written} OOF-true organ_inputs_007.v1 files "
                        f"written to store/nightly_007_folds/ "
                        f"(m4 variant {ships}); q = trailing-252-OOF-day "
                        f"percentile of support-set mu sd (deploy "
                        f"convention mirrored)",
            "provenance": "BUILD_SPEC_007 §6.2 + ea_007.py wave-2a "
                          "contract; all values walk-forward OOF, "
                          "fitness data ends 2026-02-06"}) + "\n")


if __name__ == "__main__":
    if "--fold-packs" in sys.argv:
        write_fold_packs()
    else:
        main()
