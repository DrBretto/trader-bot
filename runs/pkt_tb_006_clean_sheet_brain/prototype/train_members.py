"""PKT-TB-006 — member training driver (BUILD_SPEC §5/§10; deliverable 4).

Usage (orchestrator):
  # final, text-free members (now):
  .venv/bin/python train_members.py --screen
  .venv/bin/python train_members.py --members cast,ridge --final
  .venv/bin/python train_members.py --members cast --uniform-twin
  # smoke on the pre-LLM store (now), final after the LLM merge:
  .venv/bin/python train_members.py --members gbm,event,risk
  .venv/bin/python train_members.py --members gbm,event,risk --final   # post-merge

Outputs: oof/fold_<f>_<member>.npz (mu[D,64], sigma[D,64], c[D], dates,
manifest JSON string w/ walk_forward: true — books.py contract) and
models_out/<member>/ deploy artifacts + manifest.json.
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
sys.path.insert(0, str(PROTO))

import folds as F          # noqa: E402
import infotropy as IT     # noqa: E402
import members as M        # noqa: E402

OOF_DIR = PROTO / "oof"
MODELS_DIR = PROTO / "models_out"
SCREEN_PATH = PROTO / "store" / "infotropy_a_screen.json"


def log_look(component: str, decision: str, provenance: str) -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "component": component, "decision": decision,
             "provenance": provenance, "phase": "C-build"}
    with (PROTO / "validation_looks.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_panel() -> dict:
    z = np.load(PROTO / "store" / "panel.npz", allow_pickle=False)
    keys = ["dates", "symbols", "buckets", "X", "scalars", "T", "B", "Z",
            "T_cols", "B_cols", "Z_cols", "symbol_mask", "y5_raw", "y5_rank",
            "y1_z", "y5_bucket", "fwd_vol5", "fwd_beta63", "fwd_book_vol5",
            "w_rec_score", "w_rec", "close_px"]
    return {k: z[k] for k in keys}


def manifest(member: str, train_window_end: str, seeds, params: dict,
             wall: float, command: str, smoke: bool, extra: dict | None = None) -> str:
    m = {"member": member, "model_version": f"{member}-{train_window_end}"
                                            f"{'-smoke' if smoke else ''}",
         "train_window_end": train_window_end, "walk_forward": True,
         "seeds": list(seeds), "code_sha": M.code_sha(),
         "params_hash": M.params_hash(params), "params": params,
         "wall_clock_s": round(wall, 1), "command": command, "smoke": smoke,
         "generated": dt.datetime.now().isoformat(timespec="seconds")}
    if extra:
        m.update(extra)
    return json.dumps(m)


def weekly_rank_ic(mu, panel, val_idx) -> float:
    """Non-overlapping weekly rank IC: every 5th fold day, Spearman(mu, y5_raw)."""
    from scipy.stats import spearmanr
    y = panel["y5_raw"]
    sm = panel["symbol_mask"]
    ics = []
    for j in range(0, len(val_idx), 5):
        i = val_idx[j]
        m = sm[i] & np.isfinite(y[i]) & np.isfinite(mu[j])
        if m.sum() >= 8:
            ics.append(spearmanr(mu[j][m], y[i][m]).statistic)
    return float(np.nanmean(ics)) if ics else np.nan


def daily_spearman(mu, panel, val_idx) -> float:
    from scipy.stats import spearmanr
    y = panel["y5_raw"]
    sm = panel["symbol_mask"]
    sp = []
    for j, i in enumerate(val_idx):
        m = sm[i] & np.isfinite(y[i]) & np.isfinite(mu[j])
        if m.sum() >= 8:
            sp.append(spearmanr(mu[j][m], y[i][m]).statistic)
    return float(np.nanmean(sp)) if sp else np.nan


def seed_agreement(mus: list[np.ndarray], panel, val_idx) -> np.ndarray:
    """c per day = mean pairwise Spearman of the seeds' mu."""
    from scipy.stats import spearmanr
    n = len(val_idx)
    c = np.full(n, np.nan)
    sm = panel["symbol_mask"]
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


# ===========================================================================
# CAST
# ===========================================================================

def run_cast_oof(panel, dates, fold_list, seeds, uniform, days_per_step,
                 max_epochs, patience, command, smoke) -> dict:
    member = "cast_uniform" if uniform else "cast"
    sec_ids, cls_ids = M.sector_class_ids(PROTO.parents[2] / "config" / "universe.csv")
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val(dates, f)
        train_idx = np.nonzero(tm)[0]
        val_idx = np.nonzero(vm)[0]
        t0 = time.time()
        mus, sgs, eps_, nll = [], [], [], []
        for seed in seeds:
            sd, vs, be, vmu, vsg, hist = M.train_cast_one(
                panel, train_idx, val_idx, seed, sec_ids, cls_ids,
                uniform_w=uniform, max_epochs=max_epochs, patience=patience,
                days_per_step=days_per_step)
            mus.append(vmu)
            sgs.append(vsg)
            eps_.append(be)
            print(f"[{member} F{f} seed {seed}] best_epoch {be} "
                  f"val_spear {vs:.4f} ({time.time()-t0:.0f}s)", flush=True)
        mu = np.nanmean(np.stack(mus), axis=0)
        sg = np.nanmean(np.stack(sgs), axis=0)
        c = seed_agreement(mus, panel, val_idx)
        # GaussNLL diagnostics on the fold
        y = panel["y5_rank"][val_idx]
        m = panel["symbol_mask"][val_idx] & np.isfinite(y)
        var = np.maximum(sg, 1e-3) ** 2
        nll_v = float(np.mean(0.5 * (np.log(var[m]) + (y[m] - mu[m]) ** 2 / var[m])))
        sig_err_corr = float(np.corrcoef(sg[m], np.abs(y[m] - mu[m]))[0, 1])
        wall = time.time() - t0
        params = dict(max_epochs=max_epochs, patience=patience,
                      days_per_step=days_per_step, lr=1e-3, wd=1e-3,
                      uniform_w=uniform)
        man = manifest(member, str(np.asarray(dates)[train_idx][-1]), seeds,
                       params, wall, command, smoke,
                       extra={"fold": f, "best_epochs": eps_,
                              "param_count": None})
        OOF_DIR.mkdir(exist_ok=True)
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=np.asarray(dates)[val_idx], mu=mu, sigma=sg, c=c,
                 manifest=np.array(man))
        metrics[f"F{f}"] = {
            "purged_spearman": daily_spearman(mu, panel, val_idx),
            "weekly_rank_ic": weekly_rank_ic(mu, panel, val_idx),
            "gauss_nll": nll_v, "sigma_abs_err_corr": sig_err_corr,
            "best_epochs": eps_, "wall_s": round(wall, 1),
            "mean_c": float(np.nanmean(c))}
        print(f"[{member} F{f}] spear {metrics[f'F{f}']['purged_spearman']:.4f} "
              f"weekly_ic {metrics[f'F{f}']['weekly_rank_ic']:.4f} "
              f"nll {nll_v:.3f} wall {wall:.0f}s", flush=True)
    _write_metrics(member, metrics)
    return metrics


def run_cast_deploy(panel, dates, seeds, epochs, days_per_step, command, smoke):
    import torch
    member = "cast"
    sec_ids, cls_ids = M.sector_class_ids(PROTO.parents[2] / "config" / "universe.csv")
    dm = F.deploy_train_mask(dates)
    train_idx = np.nonzero(dm)[0]
    out = MODELS_DIR / member
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pc = None
    for seed in seeds:
        sd, _, _, _, _, _ = M.train_cast_one(
            panel, train_idx, np.array([], dtype=int), seed, sec_ids, cls_ids,
            max_epochs=epochs, patience=epochs, days_per_step=days_per_step)
        torch.save(sd, out / f"seed_{seed}.pt")
        model = M.build_cast(seed)
        pc = M.cast_param_count(model)
        print(f"[cast deploy seed {seed}] done ({time.time()-t0:.0f}s)", flush=True)
    params = dict(epochs=epochs, days_per_step=days_per_step, lr=1e-3, wd=1e-3)
    man = manifest(member, F.FINE_TUNE_CUTOFF, seeds, params,
                   time.time() - t0, command, smoke,
                   extra={"deploy": True, "param_count": pc,
                          "fixed_epochs_provenance":
                              "median best_epoch across OOF folds (deploy window "
                              "has no purged validation by construction)"})
    (out / "manifest.json").write_text(man)


# ===========================================================================
# Ridge twin
# ===========================================================================

def run_ridge(panel, dates, fold_list, deploy, command, smoke):
    member = "ridge_twin"
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val(dates, f)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        val_mu, lam, coef = M.train_ridge_twin(panel, train_idx, val_idx)
        wall = time.time() - t0
        man = manifest(member, str(np.asarray(dates)[train_idx][-1]), [0],
                       {"lambda": lam, "grid": M.RIDGE_LAMBDAS}, wall, command,
                       smoke, extra={"fold": f})
        OOF_DIR.mkdir(exist_ok=True)
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=np.asarray(dates)[val_idx], mu=val_mu,
                 sigma=np.ones_like(val_mu), c=np.ones(len(val_idx)),
                 manifest=np.array(man))
        metrics[f"F{f}"] = {"purged_spearman": daily_spearman(val_mu, panel, val_idx),
                            "weekly_rank_ic": weekly_rank_ic(val_mu, panel, val_idx),
                            "lambda": lam, "wall_s": round(wall, 1)}
        print(f"[ridge F{f}] spear {metrics[f'F{f}']['purged_spearman']:.4f} "
              f"weekly_ic {metrics[f'F{f}']['weekly_rank_ic']:.4f} "
              f"lam {lam:g} wall {wall:.0f}s", flush=True)
    if deploy:
        dm = F.deploy_train_mask(dates)
        train_idx = np.nonzero(dm)[0]
        t0 = time.time()
        _, lam, coef = M.train_ridge_twin(panel, train_idx, np.array([], dtype=int))
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "coef.npz", coef=coef, lam=lam)
        (out / "manifest.json").write_text(
            manifest(member, F.FINE_TUNE_CUTOFF, [0], {"lambda": lam},
                     time.time() - t0, command, smoke, extra={"deploy": True}))
    _write_metrics(member, metrics)
    return metrics


# ===========================================================================
# GBM-Cond
# ===========================================================================

def run_gbm(panel, dates, fold_list, screen, uniform, deploy, command, smoke,
            variant="conjunctive"):
    member = "gbm_cond_uniform" if uniform else "gbm_cond"
    u = pd.read_csv(PROTO.parents[2] / "config" / "universe.csv")
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    symbols = [str(s) for s in panel["symbols"]]
    buckets = [str(b) for b in panel["buckets"]]
    sym_bucket = {s: b for b, ss in bucket_map.items() for s in ss}
    bucket_ids = np.array([buckets.index(sym_bucket.get(s, "us_broad"))
                           for s in symbols])
    sym_is_credit = np.array([1.0 if sym_bucket.get(s) == "credit" else 0.0
                              for s in symbols])
    sym_is_equity = np.array([1.0 if u.set_index("symbol").loc[s, "asset_class"]
                              == "equity" else 0.0 for s in symbols])
    seed = F.component_master_seed(member, F.FINE_TUNE_CUTOFF) % (2 ** 31)
    metrics = {}
    oof_P, oof_y = [], []          # walk-forward isotonic chain
    best_iters = []
    for f in fold_list:
        tm, vm = F.fold_train_val(dates, f)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        Dt, yt, mt, wt, mono = M.gbm_design(panel, train_idx, sym_is_credit,
                                            sym_is_equity, screen, f, variant)
        Dv, yv, mv, _, _ = M.gbm_design(panel, val_idx, sym_is_credit,
                                        sym_is_equity, screen, f, variant)
        clf, best_iter = M.fit_gbm(Dt.reshape(-1, Dt.shape[2]),
                                   yt.reshape(-1), mt.reshape(-1),
                                   wt.reshape(-1), mono,
                                   val_tuple=(Dv.reshape(-1, Dv.shape[2]),
                                              yv.reshape(-1), mv.reshape(-1)),
                                   seed=seed, uniform_w=uniform)
        best_iters.append(best_iter)
        iso = None
        if oof_P:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.concatenate(oof_P), np.concatenate(oof_y))
        P, mu2, c2 = M.gbm_outputs(clf, Dv, mv, panel, val_idx, iso=iso)
        sigma2 = M.rolling_bucket_residual_sd(
            np.nan_to_num(mu2, nan=0.0), np.nan_to_num(panel["y5_rank"][val_idx],
                                                       nan=0.0), bucket_ids)
        pm = np.isfinite(P) & mv
        oof_P.append(P[pm])
        oof_y.append(yv[pm])
        wall = time.time() - t0
        man = manifest(member, str(np.asarray(dates)[train_idx][-1]), [seed],
                       dict(M.GBM_PARAMS, uniform_w=uniform, variant=variant),
                       wall, command, smoke,
                       extra={"fold": f, "best_iter": best_iter,
                              "isotonic": iso is not None})
        OOF_DIR.mkdir(exist_ok=True)
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=np.asarray(dates)[val_idx],
                 mu=np.nan_to_num(mu2, nan=0.0), sigma=sigma2,
                 c=np.nan_to_num(c2, nan=0.0), manifest=np.array(man))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yv[pm], P[pm])) if pm.sum() > 100 else np.nan
        metrics[f"F{f}"] = {"purged_spearman": daily_spearman(mu2, panel, val_idx),
                            "weekly_rank_ic": weekly_rank_ic(mu2, panel, val_idx),
                            "auc": auc, "best_iter": best_iter,
                            "mean_c2": float(np.nanmean(c2)),
                            "wall_s": round(wall, 1)}
        print(f"[{member} F{f}] spear {metrics[f'F{f}']['purged_spearman']:.4f} "
              f"auc {auc:.4f} iters {best_iter} wall {wall:.0f}s", flush=True)
    if deploy:
        import pickle
        dm = F.deploy_train_mask(dates)
        train_idx = np.nonzero(dm)[0]
        t0 = time.time()
        Dt, yt, mt, wt, mono = M.gbm_design(panel, train_idx, sym_is_credit,
                                            sym_is_equity, screen, 6, variant)
        n_iter = int(np.median(best_iters)) if best_iters else M.GBM_PARAMS["max_iter"]
        params = dict(M.GBM_PARAMS)
        params["max_iter"] = n_iter
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(monotonic_cst=list(mono),
                                             early_stopping=False,
                                             random_state=seed, **params)
        flat = Dt.reshape(-1, Dt.shape[2])
        fm = mt.reshape(-1)
        sw = np.ones(fm.sum()) if uniform else wt.reshape(-1)[fm]
        clf.fit(flat[fm], yt.reshape(-1)[fm], sample_weight=sw)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.concatenate(oof_P), np.concatenate(oof_y))
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        with (out / "model.pkl").open("wb") as fh:
            pickle.dump({"clf": clf, "iso": iso, "bucket_ids": bucket_ids,
                         "sym_is_credit": sym_is_credit,
                         "sym_is_equity": sym_is_equity}, fh)
        (out / "manifest.json").write_text(
            manifest(member, F.FINE_TUNE_CUTOFF, [seed],
                     dict(params, variant=variant), time.time() - t0, command,
                     smoke, extra={"deploy": True, "n_iter": n_iter}))
    _write_metrics(member, metrics)
    return metrics


# ===========================================================================
# EventHead / RiskNet
# ===========================================================================

def run_event(panel, dates, fold_list, screen, deploy, command, smoke,
              variant="conjunctive"):
    member = "event_head" if variant == "conjunctive" else "event_head_r3only"
    b_cols = [str(c) for c in panel["B_cols"]]
    buckets = [str(b) for b in panel["buckets"]]
    symbols = [str(s) for s in panel["symbols"]]
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    br = IT.bucket_daily_returns(panel, bucket_map, symbols, buckets)
    fams = [f for f in IT.FAMILIES if f != "LLM_event_flags"]
    inten_all = np.mean([IT.family_intensity(panel["B"], b_cols, f)
                         for f in fams], axis=0)
    rt_w = IT.runtime_r2_downweight(inten_all, br)
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val(dates, f)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        res = M.train_event_head(panel, train_idx, val_idx, screen, f,
                                 variant=variant, rt_weights=rt_w)
        wall = time.time() - t0
        man = manifest(member, str(np.asarray(dates)[train_idx][-1]), [0],
                       {"alphas": M.EVENT_ALPHAS, "l1_ratio": M.EVENT_L1_RATIO,
                        "variant": variant, "event_weight_cap": 1.0,
                        "abstain_threshold": 0.10}, wall, command, smoke,
                       extra={"fold": f, "keep_cols": res["keep_cols"],
                              "n_pass_families": res["n_pass_families"]})
        OOF_DIR.mkdir(exist_ok=True)
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=np.asarray(dates)[val_idx], mu=res["mu3"],
                 sigma=res["sigma3"], c=res["c3"], manifest=np.array(man))
        abst = float(np.mean(np.all(res["mu3"] == 0, axis=1)))
        metrics[f"F{f}"] = {"purged_spearman": daily_spearman(res["mu3"], panel, val_idx),
                            "weekly_rank_ic": weekly_rank_ic(res["mu3"], panel, val_idx),
                            "abstain_frac": abst,
                            "n_pass_families": res["n_pass_families"],
                            "wall_s": round(wall, 1)}
        print(f"[{member} F{f}] spear {metrics[f'F{f}']['purged_spearman']:.4f} "
              f"abstain {abst:.2f} pass_fams {res['n_pass_families']} "
              f"wall {wall:.0f}s", flush=True)
    if deploy:
        import pickle
        dm = F.deploy_train_mask(dates)
        train_idx = np.nonzero(dm)[0]
        t0 = time.time()
        res = M.train_event_head(panel, train_idx, np.array([], dtype=int),
                                 screen, 6, variant=variant, rt_weights=rt_w)
        out = MODELS_DIR / member
        out.mkdir(parents=True, exist_ok=True)
        with (out / "model.pkl").open("wb") as fh:
            pickle.dump({"models": res["models"], "keep_cols": res["keep_cols"]}, fh)
        (out / "manifest.json").write_text(
            manifest(member, F.FINE_TUNE_CUTOFF, [0], {"variant": variant},
                     time.time() - t0, command, smoke, extra={"deploy": True}))
    _write_metrics(member, metrics)
    return metrics


def run_risk(panel, dates, fold_list, deploy, command, smoke):
    member = "risknet"
    t0 = time.time()
    RF = M.risk_design(panel)
    print(f"[risknet] design built ({time.time()-t0:.0f}s)", flush=True)
    metrics = {}
    for f in fold_list:
        tm, vm = F.fold_train_val(dates, f)
        train_idx, val_idx = np.nonzero(tm)[0], np.nonzero(vm)[0]
        t0 = time.time()
        out = M.train_risknet(panel, RF, train_idx, val_idx)
        wall = time.time() - t0
        # validation QLIKE on the fold
        act = panel["fwd_vol5"][val_idx]
        m = np.isfinite(act)
        q = M.qlike(act[m], out["sigma_hat"][m]) if m.any() else np.nan
        actb = panel["fwd_book_vol5"][val_idx]
        mb = np.isfinite(actb)
        qb = M.qlike(actb[mb], out["book_vol_hat"][mb]) if mb.any() else np.nan
        be = panel["fwd_beta63"][val_idx]
        me = np.isfinite(be)
        mse_b = float(np.mean((out["beta_hat"][me] - be[me]) ** 2)) if me.any() else np.nan
        man = manifest(member, str(np.asarray(dates)[train_idx][-1]), [0],
                       {"lambdas": M.RISK_LAMBDAS,
                        "features": M.RISK_FEATURES}, wall, command, smoke,
                       extra={"fold": f, "n_coefs": out["n_coefs"]})
        OOF_DIR.mkdir(exist_ok=True)
        np.savez(OOF_DIR / f"fold_{f}_{member}.npz",
                 dates=np.asarray(dates)[val_idx],
                 sigma_hat=out["sigma_hat"], beta_hat=out["beta_hat"],
                 book_vol_hat=out["book_vol_hat"], manifest=np.array(man))
        metrics[f"F{f}"] = {"qlike_vol5": q, "qlike_book": qb,
                            "mse_beta": mse_b, "n_coefs": out["n_coefs"],
                            "wall_s": round(wall, 1)}
        print(f"[risknet F{f}] qlike {q:.4f} book {qb:.4f} beta_mse {mse_b:.4f} "
              f"wall {wall:.0f}s", flush=True)
    if deploy:
        dm = F.deploy_train_mask(dates)
        train_idx = np.nonzero(dm)[0]
        t0 = time.time()
        out = M.train_risknet(panel, RF, train_idx, np.array([], dtype=int))
        outdir = MODELS_DIR / member
        outdir.mkdir(parents=True, exist_ok=True)
        arrs = {}
        for k, (hd, lam, _) in out["coefs"].items():
            arrs[f"{k}_coef"] = hd["coef"]
            arrs[f"{k}_mu"] = hd["mu"]
            arrs[f"{k}_sd"] = hd["sd"]
            arrs[f"{k}_logspace"] = np.array(hd["logspace"])
            arrs[f"{k}_lambda"] = np.array(lam)
        np.savez(outdir / "coefs.npz", **arrs)
        (outdir / "manifest.json").write_text(
            manifest(member, F.FINE_TUNE_CUTOFF, [0],
                     {"features": M.RISK_FEATURES}, time.time() - t0, command,
                     smoke, extra={"deploy": True, "n_coefs": out["n_coefs"]}))
    _write_metrics(member, metrics)
    return metrics


def _write_metrics(member: str, metrics: dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    p = MODELS_DIR / f"metrics_{member}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(metrics)
    p.write_text(json.dumps(old, indent=1, default=float))


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="")
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--oof-seeds", type=int, default=2)      # shrink rung 1 taken
    ap.add_argument("--deploy-seeds", type=int, default=3)   # shrink rung 2 taken
    ap.add_argument("--uniform-twin", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--deploy-only", action="store_true")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--r3-only-twin", action="store_true")
    ap.add_argument("--days-per-step", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--deploy-epochs", type=int, default=0)  # 0 => median OOF
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    command = "python " + " ".join(sys.argv)

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    panel = load_panel()
    dates = panel["dates"]
    meta = json.loads((PROTO / "store" / "panel_meta.json").read_text())
    llm_merged = bool(meta.get("llm_merged", False))
    members = [m for m in args.members.split(",") if m]
    fold_list = [int(x) for x in args.folds.split(",") if x]

    if args.final and not llm_merged and any(m in members for m in
                                             ["gbm", "event", "risk"]):
        raise SystemExit("--final for gbm/event/risk requires the LLM merge "
                         "(store llm_merged=false). Re-run after "
                         "store/llm_features.parquet lands + feature_store rebuild.")
    text_free = bool(members) and set(members) <= {"cast", "ridge"}
    smoke = not (args.final and (text_free or llm_merged))

    if args.screen:
        t0 = time.time()
        res = IT.run_screen(panel, dates, out_path=SCREEN_PATH,
                            llm_pending=not llm_merged)
        print(f"infotropy A screen -> {SCREEN_PATH} ({time.time()-t0:.0f}s)")
        for fam, folds_res in res["families"].items():
            line = " ".join(f"F{f}:{'P' if folds_res[f'F{f}'].get('pass') else '-'}"
                            for f in range(1, 7))
            print(f"  {fam:20s} {line}")

    screen = json.loads(SCREEN_PATH.read_text()) if SCREEN_PATH.exists() else None

    oof_seeds = F.OOF_SEEDS[:args.oof_seeds]
    deploy_seeds = F.DEPLOY_SEEDS[:args.deploy_seeds]

    if "cast" in members:
        if not args.deploy_only:
            run_cast_oof(panel, dates, fold_list, oof_seeds, args.uniform_twin,
                         args.days_per_step, args.max_epochs, args.patience,
                         command, smoke)
        if (args.deploy or args.deploy_only) and not args.uniform_twin:
            ep = args.deploy_epochs
            if ep <= 0:                     # median best_epoch across OOF folds
                eps = []
                for f in range(1, 7):
                    p = OOF_DIR / f"fold_{f}_cast.npz"
                    if p.exists():
                        man = json.loads(str(np.load(p)["manifest"]))
                        eps += man.get("best_epochs", [])
                ep = max(5, int(np.median(eps))) if eps else 20
            run_cast_deploy(panel, dates, deploy_seeds, ep,
                            args.days_per_step, command, smoke)
    if "ridge" in members:
        run_ridge(panel, dates, fold_list, args.deploy, command, smoke)
    if "gbm" in members:
        if screen is None:
            raise SystemExit("run --screen first (Transfer-A routing needed)")
        run_gbm(panel, dates, fold_list, screen, args.uniform_twin,
                args.deploy, command, smoke)
    if "event" in members:
        if screen is None:
            raise SystemExit("run --screen first")
        run_event(panel, dates, fold_list, screen, args.deploy, command, smoke,
                  variant="r3_only" if args.r3_only_twin else "conjunctive")
    if "risk" in members:
        run_risk(panel, dates, fold_list, args.deploy, command, smoke)


if __name__ == "__main__":
    main()
