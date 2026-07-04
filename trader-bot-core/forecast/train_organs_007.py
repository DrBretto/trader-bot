"""PKT-TB-007 — organ training driver (BUILD_SPEC_007 §1/§5.1, build step 5).

Usage:
  .venv/bin/python train_organs_007.py --members m1,ridge,m2,m3,m4,m5,m6 --deploy

Outputs (TB-006 OOF conventions, walk_forward manifests):
  oof/fold_<f>_<member>.npz      member in {m1_cast, ridge_twin_007, m2_rot,
                                 m3_disp, m4_evt_a, m4_evt_b, m4_control,
                                 m5_posz, m6_gru}
  models_out_007/<member>/       deploy artifacts + manifest.json
  models_out_007/metrics_<member>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
sys.path.insert(0, str(PROTO))

import folds as F            # noqa: E402
import organs_007 as G       # noqa: E402

OOF_DIR = PROTO / "oof"
MODELS_DIR = PROTO / "models_out_007"
LEDGER = PROTO / "validation_looks_007.jsonl"


def log_look(component: str, decision: str, provenance: str,
             kind: str = "build_decision") -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "phase": "C-wave2a", "kind": kind, "component": component,
             "decision": decision, "provenance": provenance}
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_panels():
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    keys = ["dates", "symbols", "buckets", "X", "scalars", "T", "B",
            "T_cols", "B_cols", "symbol_mask", "y5_raw", "y5_rank", "y1_z",
            "w_rec", "open_px", "close_px", "gdelt_available"]
    p6 = {k: z6[k] for k in keys}
    z7 = np.load(PROTO / "store" / "panel_007.npz", allow_pickle=False)
    p7 = {k: z7[k] for k in z7.files}
    return p6, p7


def manifest(member, train_window_end, seeds, params, wall, command,
             extra=None) -> str:
    m = {"member": member, "model_version": f"{member}-{train_window_end}",
         "train_window_end": str(train_window_end), "walk_forward": True,
         "seeds": list(seeds), "code_sha": G.code_sha(),
         "params_hash": G.params_hash(params), "params": params,
         "wall_clock_s": round(wall, 1), "command": command,
         "data_range": "panel 2015-02-18..2026-06-10 (training masked per "
                       "folds.py; holdout >= 2026-03-11 untouched)",
         "generated": dt.datetime.now().isoformat(timespec="seconds")}
    if extra:
        m.update(extra)
    return json.dumps(m, default=float)


def _write_metrics(member, metrics):
    MODELS_DIR.mkdir(exist_ok=True)
    p = MODELS_DIR / f"metrics_{member}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(metrics)
    p.write_text(json.dumps(old, indent=1, default=float))


def daily_spearman(mu, y, sm, val_idx):
    from scipy.stats import spearmanr
    sp = []
    for j, i in enumerate(val_idx):
        m = sm[i] & np.isfinite(y[i]) & np.isfinite(mu[j])
        if m.sum() >= 8:
            sp.append(spearmanr(mu[j][m], y[i][m]).statistic)
    return float(np.nanmean(sp)) if sp else np.nan


def windowed_rank_ic(mu, y, sm, val_idx, step):
    from scipy.stats import spearmanr
    ics = []
    for j in range(0, len(val_idx), step):
        i = val_idx[j]
        m = sm[i] & np.isfinite(y[i]) & np.isfinite(mu[j])
        if m.sum() >= 8:
            ics.append(spearmanr(mu[j][m], y[i][m]).statistic)
    return (float(np.nanmean(ics)) if ics else np.nan), len(ics)


def seed_agreement(mus, sm, val_idx):
    from scipy.stats import spearmanr
    n = len(val_idx)
    c = np.full(n, np.nan)
    for j, i in enumerate(val_idx):
        m = sm[i]
        if m.sum() < 8:
            continue
        cors = []
        for a in range(len(mus)):
            for b in range(a + 1, len(mus)):
                cors.append(spearmanr(mus[a][j][m], mus[b][j][m]).statistic)
        c[j] = float(np.nanmean(cors)) if cors else np.nan
    return c


# =========================== M1 ==============================================

def run_m1(p6, p7, dates, fold_list, deploy, command):
    import torch
    member = "m1_cast"
    xf = [str(c) for c in np.load(TB6 / "store" / "panel.npz",
                                  allow_pickle=False)["X_features"]]
    cols10 = [xf.index(c) for c in G.M1_XCOLS]
    X10 = np.ascontiguousarray(p6["X"][:, :, :, cols10])
    sec_ids, cls_ids = G.sector_class_ids(REPO / "config" / "universe.csv")
    seeds = F.CAST_OOF_SEEDS_007
    sm = p6["symbol_mask"]
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val_007(dates, f, F.EMBARGO_TD_007)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        mus, sgs, eps_ = [], [], []
        for seed in seeds:
            sd, vs, be, vmu, vsg, _ = G.train_cast_xs_one(
                p6, X10, train_idx, val_idx, seed, sec_ids, cls_ids)
            mus.append(vmu)
            sgs.append(vsg)
            eps_.append(be)
            print(f"[m1 F{f} seed {seed}] best_epoch {be} spear {vs:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        mu = np.nanmean(np.stack(mus), axis=0)
        sg = np.nanmean(np.stack(sgs), axis=0)
        c = seed_agreement(mus, sm, val_idx)
        wall = time.time() - t0
        params = dict(x_cols=G.M1_XCOLS, max_epochs=60, patience=10,
                      days_per_step=8, lr=1e-3, wd=1e-3)
        model = G.build_cast_xs(seeds[0])
        pc = sum(p.numel() for p in model.parameters())
        man = manifest(member, str(dates[train_idx][-1]), seeds, params, wall,
                       command, extra={"fold": f, "best_epochs": eps_,
                                       "param_count": pc})
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=dates[val_idx], mu=mu, sigma=sg, c=c,
                 manifest=np.array(man))
        wic, nw = windowed_rank_ic(mu, p6["y5_raw"], sm, val_idx, 5)
        metrics[f"F{f}"] = {
            "purged_spearman": daily_spearman(mu, p6["y5_raw"], sm, val_idx),
            "weekly_rank_ic": wic, "n_weeks": nw, "best_epochs": eps_,
            "param_count": pc, "wall_s": round(wall, 1)}
        print(f"[m1 F{f}] spear {metrics[f'F{f}']['purged_spearman']:.4f} "
              f"weekly_ic {wic:.4f} wall {wall:.0f}s", flush=True)
    _write_metrics(member, metrics)
    if deploy:
        eps_all = [e for f in range(1, 7)
                   if (OOF_DIR / f"fold_{f}_{member}.npz").exists()
                   for e in
                   json.loads(str(np.load(OOF_DIR / f"fold_{f}_{member}.npz",
                                          allow_pickle=True)["manifest"]))
                   ["best_epochs"]]
        ep = max(5, int(np.median(eps_all)))
        dm = F.deploy_train_mask_007(dates, 5)
        train_idx = np.nonzero(dm)[0]
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for seed in seeds:
            sd, *_ = G.train_cast_xs_one(p6, X10, train_idx,
                                         np.array([], dtype=int), seed,
                                         sec_ids, cls_ids, max_epochs=ep,
                                         patience=ep)
            torch.save(sd, out / f"seed_{seed}.pt")
            print(f"[m1 deploy seed {seed}] done ({time.time()-t0:.0f}s)",
                  flush=True)
        (out / "manifest.json").write_text(manifest(
            member, str(dates[train_idx][-1]), seeds,
            dict(epochs=ep, x_cols=G.M1_XCOLS), time.time() - t0, command,
            extra={"deploy": True, "fixed_epochs_provenance":
                   "median best_epoch across OOF folds"}))
    return metrics


def run_ridge(p6, p7, dates, fold_list, deploy, command):
    member = "ridge_twin_007"
    xf = [str(c) for c in np.load(TB6 / "store" / "panel.npz",
                                  allow_pickle=False)["X_features"]]
    cols10 = [xf.index(c) for c in G.M1_XCOLS]
    X10 = np.ascontiguousarray(p6["X"][:, :, :, cols10])
    sm = p6["symbol_mask"]
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val_007(dates, f, F.EMBARGO_TD_007)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        val_mu, lam, coef = G.train_ridge_twin(p6, X10, train_idx, val_idx)
        wall = time.time() - t0
        man = manifest(member, str(dates[train_idx][-1]), [0],
                       {"lambda": lam, "grid": G.RIDGE_LAMBDAS}, wall,
                       command, extra={"fold": f})
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=dates[val_idx], mu=val_mu,
                 sigma=np.ones_like(val_mu), c=np.ones(len(val_idx)),
                 manifest=np.array(man))
        wic, nw = windowed_rank_ic(val_mu, p6["y5_raw"], sm, val_idx, 5)
        metrics[f"F{f}"] = {"purged_spearman": daily_spearman(
            val_mu, p6["y5_raw"], sm, val_idx),
            "weekly_rank_ic": wic, "lambda": lam, "wall_s": round(wall, 1)}
        print(f"[ridge F{f}] spear {metrics[f'F{f}']['purged_spearman']:.4f} "
              f"weekly_ic {wic:.4f} lam {lam:g}", flush=True)
    _write_metrics(member, metrics)
    return metrics


# =========================== M2 ==============================================

def run_m2(p6, p7, dates, fold_list, deploy, command):
    member = "m2_rot"
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    symbols = [str(s) for s in p6["symbols"]]
    buckets = [str(b) for b in p6["buckets"]]
    sym_bucket = {s: b for b, ss in bucket_map.items() for s in ss}
    bucket_ids = np.array([buckets.index(sym_bucket.get(s, "us_broad"))
                           for s in symbols])
    sym_is_credit = np.array([1.0 if sym_bucket.get(s) == "credit" else 0.0
                              for s in symbols])
    sm = p6["symbol_mask"]
    seed = F.MASTER_SEED_007
    metrics = {}
    best_iters = []
    for f in fold_list:
        tm, vm = F.fold_train_val_007(dates, f, F.EMBARGO_TD_007_M2)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        Dt, yt, mt, mono = G.m2_design(p6, p7, train_idx, sym_is_credit)
        Dv, yv, mv, _ = G.m2_design(p6, p7, val_idx, sym_is_credit)
        n_tr = len(train_idx)
        cut = int(n_tr * 0.85)
        day_is_inner_va = np.zeros(n_tr, dtype=bool)
        day_is_inner_va[cut:] = True
        day_is_inner_tr = np.zeros(n_tr, dtype=bool)
        day_is_inner_tr[:max(1, cut - G.M2_EMBARGO_INNER)] = True
        S = Dt.shape[1]
        rows_tr = np.repeat(day_is_inner_tr, S)
        rows_va = np.repeat(day_is_inner_va, S)
        clf, best_iter = G.fit_m2(Dt.reshape(-1, Dt.shape[2]),
                                  yt.reshape(-1), mt.reshape(-1), mono,
                                  (rows_tr, rows_va), seed)
        best_iters.append(best_iter)
        P, mu, c = G.m2_outputs(clf, Dv, mv)
        sigma = G.rolling_bucket_residual_sd(
            np.nan_to_num(mu, nan=0.0),
            np.nan_to_num(p7["y_rot_rank"][val_idx].astype(np.float64),
                          nan=0.0), bucket_ids, h_lag=22)
        wall = time.time() - t0
        man = manifest(member, str(dates[train_idx][-1]), [seed],
                       dict(G.M2_PARAMS, embargo_td=26), wall, command,
                       extra={"fold": f, "best_iter": best_iter,
                              "n_leaf_values_max":
                                  G.M2_PARAMS["max_iter"]
                                  * G.M2_PARAMS["max_leaf_nodes"]})
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=dates[val_idx], mu=mu, sigma=sigma, c=c,
                 manifest=np.array(man))
        ric, nwin = windowed_rank_ic(mu, p7["y_rot_raw"], sm, val_idx, 16)
        metrics[f"F{f}"] = {
            "rot_rank_ic_16d": ric, "n_windows": nwin,
            "purged_spearman_rot": daily_spearman(
                mu, p7["y_rot_raw"], sm, val_idx),
            "best_iter": best_iter, "wall_s": round(wall, 1)}
        print(f"[m2 F{f}] rot_ic {ric:.4f} ({nwin} win) iters {best_iter} "
              f"wall {wall:.0f}s", flush=True)
    _write_metrics(member, metrics)
    if deploy:
        import pickle
        dm = F.deploy_train_mask_007(dates, 21)
        train_idx = np.nonzero(dm)[0]
        t0 = time.time()
        Dt, yt, mt, mono = G.m2_design(p6, p7, train_idx, sym_is_credit)
        n_iter = int(np.median(best_iters)) if best_iters else 60
        clf, _ = G.fit_m2(Dt.reshape(-1, Dt.shape[2]), yt.reshape(-1),
                          mt.reshape(-1), mono, (None, None), seed,
                          fixed_iter=n_iter)
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        with (out / "model.pkl").open("wb") as fh:
            pickle.dump({"clf": clf, "bucket_ids": bucket_ids,
                         "sym_is_credit": sym_is_credit}, fh)
        (out / "manifest.json").write_text(manifest(
            member, str(dates[train_idx][-1]), [seed],
            dict(G.M2_PARAMS, n_iter=n_iter, embargo_td=26),
            time.time() - t0, command, extra={"deploy": True}))
    return metrics


# =========================== M3 ==============================================

def run_m3(p6, p7, dates, fold_list, deploy, command):
    member = "m3_disp"
    X_full, lags, y, ok = G.m3_design(p6, p7)
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val_007(dates, f, F.EMBARGO_TD_007)
        tr = np.nonzero(tm & ok)[0]
        va = np.nonzero(vm)[0]
        t0 = time.time()
        coef_full = G.ols_fit(X_full[tr], y[tr])
        coef_twin = G.ols_fit(lags[tr], y[tr])
        fc = G.ols_predict(coef_full, X_full[va])
        ft = G.ols_predict(coef_twin, np.nan_to_num(lags[va], nan=0.0))
        wall = time.time() - t0
        man = manifest(member, str(dates[tr][-1]), [0],
                       {"n_predictors": 13, "twin": "lags-only HAR (3)"},
                       wall, command, extra={"fold": f})
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=dates[va], forecast=fc, twin_forecast=ft,
                 y=y[va], manifest=np.array(man))
        vmask = np.isfinite(y[va])
        r2f = 1 - np.nanvar(y[va][vmask] - fc[vmask]) / np.nanvar(y[va][vmask])
        r2t = 1 - np.nanvar(y[va][vmask] - ft[vmask]) / np.nanvar(y[va][vmask])
        metrics[f"F{f}"] = {"oof_r2_full": float(r2f),
                            "oof_r2_lags_twin": float(r2t),
                            "wall_s": round(wall, 1)}
        print(f"[m3 F{f}] R2 full {r2f:.3f} twin {r2t:.3f}", flush=True)
    _write_metrics(member, metrics)
    if deploy:
        dm = F.deploy_train_mask_007(dates, 5)
        tr = np.nonzero(dm & ok)[0]
        coef_full = G.ols_fit(X_full[tr], y[tr])
        coef_twin = G.ols_fit(lags[tr], y[tr])
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "coefs.npz", coef_full=coef_full, coef_twin=coef_twin)
        (out / "manifest.json").write_text(manifest(
            member, str(dates[tr][-1]), [0], {"n_predictors": 13},
            0.0, command, extra={"deploy": True}))
    return metrics


# =========================== M4 ==============================================

def run_m4(p6, p7, dates, fold_list, deploy, command):
    designs = G.m4_design(p6, p7, dates)
    y = designs["y"]
    seed = F.MASTER_SEED_007
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    symbols = [str(s) for s in p6["symbols"]]
    buckets = [str(b) for b in p6["buckets"]]
    bucket_sym = np.zeros((len(buckets), len(symbols)), dtype=bool)
    for k, b in enumerate(buckets):
        for s in bucket_map[b]:
            if s in symbols:
                bucket_sym[k, symbols.index(s)] = True
    from sklearn.metrics import roc_auc_score
    all_metrics = {}
    for variant, member in [("A", "m4_evt_a"), ("B", "m4_evt_b"),
                            ("CTL", "m4_control")]:
        X = designs[variant]
        metrics = {}
        for f in fold_list:
            tm, vm = F.fold_train_val_007(dates, f, F.EMBARGO_TD_007)
            tr, va = np.nonzero(tm)[0], np.nonzero(vm)[0]
            t0 = time.time()
            fitted = G.fit_m4(X, y, tr, seed)
            p = G.m4_predict(fitted, X, va)
            mu_sym = G.m4_mu_sym(p, bucket_sym)
            wall = time.time() - t0
            man = manifest(member, str(dates[tr][-1]), [seed],
                           {"l1_ratio": fitted["l1_ratio"], "C": fitted["C"],
                            "grid": G.M4_GRID, "variant": variant}, wall,
                           command, extra={"fold": f})
            np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                     dates=dates[va], p=p, mu=mu_sym,
                     sigma=np.ones_like(mu_sym), c=np.ones(len(va)),
                     manifest=np.array(man))
            yv = y[va].reshape(-1)
            pv = p.reshape(-1)
            okv = np.isfinite(yv)
            auc = float(roc_auc_score(yv[okv], pv[okv])) \
                if len(np.unique(yv[okv])) == 2 else np.nan
            metrics[f"F{f}"] = {"auc": auc, "l1_ratio": fitted["l1_ratio"],
                                "C": fitted["C"],
                                "n_nonzero_coefs": int(np.sum(np.abs(
                                    fitted["model"].coef_) > 1e-9)),
                                "wall_s": round(wall, 1)}
            print(f"[m4_{variant} F{f}] auc {auc:.4f} "
                  f"(l1 {fitted['l1_ratio']}, C {fitted['C']}) "
                  f"wall {wall:.0f}s", flush=True)
        _write_metrics(member, metrics)
        all_metrics[variant] = metrics
        if deploy and variant != "CTL":
            import pickle
            dm = F.deploy_train_mask_007(dates, 5)
            tr = np.nonzero(dm)[0]
            fitted = G.fit_m4(X, y, tr, seed)
            out = MODELS_DIR / member
            out.mkdir(parents=True, exist_ok=True)
            with (out / "model.pkl").open("wb") as fh:
                pickle.dump({"fitted": fitted, "variant": variant,
                             "bucket_sym": bucket_sym}, fh)
            (out / "manifest.json").write_text(manifest(
                member, str(dates[tr][-1]), [seed],
                {"l1_ratio": fitted["l1_ratio"], "C": fitted["C"],
                 "variant": variant}, 0.0, command, extra={"deploy": True}))
    return all_metrics


# =========================== M5 ==============================================

def run_m5(p6, p7, dates, fold_list, command):
    member = "m5_posz"
    symbols = [str(s) for s in p6["symbols"]]
    t0 = time.time()
    zmap = G.m5_daily_z(dates)
    mu, episodes, mag = G.m5_signal(dates, symbols, zmap)
    wall = time.time() - t0
    np.savez(PROTO / "store" / "m5_signal.npz", dates=p6["dates"], mu=mu,
             mag=mag, z_es=zmap["es"]["z"], z_ust10y=zmap["ust10y"]["z"],
             z_vix=zmap["vix"]["z"])
    eps_out = [{**e, "start_date": str(dates[e["start"]]),
                "end_date": str(dates[e["end"]])} for e in episodes]
    (PROTO / "store" / "m5_episodes.json").write_text(
        json.dumps({"constants": {"z_window_weeks": 156, "threshold": 2.0,
                                  "decay_td": 21, "fresh_days": 10},
                    "episodes": eps_out}, indent=1))
    params = {"z_window_weeks": 156, "threshold": 2.0, "decay_td": 21,
              "sleeves": {k: v[0] for k, v in G.M5_SLEEVES.items()}}
    for f in fold_list:
        vm = F.fold_date_mask(dates, f)
        va = np.nonzero(vm)[0]
        man = manifest(member, "rule (0 trained params)", [0], params, wall,
                       command, extra={"fold": f})
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=dates[va], mu=mu[va], sigma=np.ones_like(mu[va]),
                 c=mag[va], manifest=np.array(man))
    n_active = int(np.sum(np.any(mu != 0, axis=1)))
    print(f"[m5] {len(episodes)} episodes; {n_active} active days "
          f"of {len(dates)}", flush=True)
    _write_metrics(member, {"n_episodes": len(episodes),
                            "n_active_days": n_active})
    return episodes


# =========================== M6 ==============================================

def run_m6(p6, p7, dates, fold_list, deploy, command):
    import torch
    member = "m6_gru"
    m6f = p7["m6_feat"].astype(np.float32)
    sm = p6["symbol_mask"]
    metrics = {}
    best_eps = []
    for f in fold_list:
        tm, vm = F.fold_train_val_007(dates, f, F.EMBARGO_TD_007)
        tr, va = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        sd, mse, be, vmu, n_par = G.train_m6(p6, m6f, tr, va,
                                             seed=F.MASTER_SEED_007)
        best_eps.append(be)
        wall = time.time() - t0
        man = manifest(member, str(dates[tr][-1]), [F.MASTER_SEED_007],
                       {"hidden": 24, "seq": 10, "param_count": n_par},
                       wall, command, extra={"fold": f, "best_epoch": be,
                                             "val_mse": mse})
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=dates[va], mu=vmu, sigma=np.ones_like(vmu),
                 c=np.ones(len(va)), manifest=np.array(man))
        ic = daily_spearman(vmu, p7["r1f_raw"], sm, va)
        metrics[f"F{f}"] = {"daily_rank_ic_1d": ic, "val_mse": mse,
                            "best_epoch": be, "param_count": n_par,
                            "wall_s": round(wall, 1)}
        print(f"[m6 F{f}] 1d_ic {ic:.4f} mse {mse:.4f} epoch {be} "
              f"wall {wall:.0f}s", flush=True)
    _write_metrics(member, metrics)
    if deploy:
        ep = max(3, int(np.median(best_eps)))
        dm = F.deploy_train_mask_007(dates, 1)
        tr = np.nonzero(dm)[0]
        t0 = time.time()
        sd, *_ , n_par = G.train_m6(p6, m6f, tr, np.array([], dtype=int),
                                    seed=F.MASTER_SEED_007, max_epochs=ep,
                                    patience=ep)
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        torch.save(sd, out / "model.pt")
        (out / "manifest.json").write_text(manifest(
            member, str(dates[tr][-1]), [F.MASTER_SEED_007],
            {"hidden": 24, "seq": 10, "epochs": ep, "param_count": n_par},
            time.time() - t0, command, extra={"deploy": True}))
    return metrics


# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="m1,ridge,m2,m3,m4,m5,m6")
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    command = "python " + " ".join(sys.argv)
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    OOF_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    p6, p7 = load_panels()
    dates = np.asarray(p6["dates"]).astype(str)
    fold_list = [int(x) for x in args.folds.split(",") if x]
    members = [m for m in args.members.split(",") if m]
    t0 = time.time()
    if "m5" in members:
        run_m5(p6, p7, dates, fold_list, command)
    if "m3" in members:
        run_m3(p6, p7, dates, fold_list, args.deploy, command)
    if "m2" in members:
        run_m2(p6, p7, dates, fold_list, args.deploy, command)
    if "m4" in members:
        run_m4(p6, p7, dates, fold_list, args.deploy, command)
    if "m6" in members:
        run_m6(p6, p7, dates, fold_list, args.deploy, command)
    if "ridge" in members:
        run_ridge(p6, p7, dates, fold_list, args.deploy, command)
    if "m1" in members:
        run_m1(p6, p7, dates, fold_list, args.deploy, command)
    print(f"TOTAL wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
