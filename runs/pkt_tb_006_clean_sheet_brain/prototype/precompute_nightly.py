"""PKT-TB-006 — nightly precompute driver (replay wiring deliverable #1).

For every decision date D in the replay window (default 2026-02-02 -> 2026-06-09)
runs the CURRENT deploy members (models_out/: CAST seed ensemble, GBM-Cond,
EventHead, RiskNet+) on the feature store's tensors for D and writes, per date:

    store/nightly/<D>/expert_opinions.parquet   (member, symbol, mu, sigma, c, manifest cols)
    store/nightly/<D>/solo_books.parquet        (member, symbol, w)
    store/nightly/<D>/risknet.parquet           (symbol, sigma_hat, beta_hat, book_vol_hat,
                                                 sigma_hat_proxy, book_vol_hat_proxy)
    store/nightly/<D>/exec_inputs.json          (r[3,4] lagged ledger stats, z[24], c[3],
                                                 agree[3], g[3], book_vol_hat_*)

plus once per run:

    store/nightly/ledger.parquet   chronological trust ledger (warm + window)
    store/nightly/ledger_meta.json u_std / warm standardization stats / provenance
    store/nightly/manifest.json    window, member manifests, code sha, command, wall clock

LEDGER WARM-UP (documented contract):
  The trust ledger threads chronologically from PRE-WINDOW history. It is warmed
  with the FOLD-6 OOF member outputs (oof/fold_6_<member>.npz — walk-forward,
  validation dates 2025-02-02..2026-01-30) restricted to dates >= 2025-08-04
  (the fine-tune window start), i.e. ~6 months of out-of-fold solo-book utility
  history immediately preceding the window. From 2026-02-02 onward the ledger
  continues on the DEPLOY members' outputs. u_realized per books.realized_u_series
  (reference sizing f=0.7, vol cap 0.10, half-spread cost table); EWMA stats are
  standardized with WARM-PERIOD mean/sd only (training standardizes per fold;
  using full-series stats here would leak future scale into the window — forced
  choice, logged in validation_looks.jsonl).

SIGMA_HAT CONVENTION (finding, logged): the executive (executive.py main) and the
EA fitness walk consume the trailing-vol PROXY sigma_hat / book_vol_hat from
synth_oof.world_from_panel — NOT the E4 RiskNet heads. To keep the deployed
executive's inputs distributionally identical to its training inputs, the
exec_inputs bundle and the ledger use the proxy; the E4 outputs are still written
to risknet.parquet per BUILD_SPEC §12 (both columns carried, both auditable).

LLM/GDELT features come from the store panel (store/panel.npz). The driver is
idempotent — one command, overwrites store/nightly/ — so it is simply re-run
after the final LLM merge + member retrain.

VARIANT STORES (battery contract, TOURNAMENT §4.4 / battery.py): --variant
writes store/nightly_<tag>/ with the retrain-arm member sources substituted:
  rt2_ridge_slot     CAST slot <- ridge twin deploy (models_out/ridge_twin),
                     ledger warmed with fold-6 ridge_twin OOF
  rt3_llm_neutral    GBM/EventHead <- models_out/*_llm_neutral on the
                     LLM-neutralized panel (llm_* columns + masks at 0)
  rt4_gdelt_ablated  GBM/EventHead <- models_out/*_gdelt_ablated on the
                     G1-G5-ablated panel
  rt5_uniform        CAST slot <- models_out/cast_uniform deploy seeds;
                     ledger w_rec_raw forced to 1.0 (uniform record weights —
                     the replay-side expression of the executive's uniform
                     w_rec; the adapter's genome-eps tilt then weights all
                     days equally)
The exec_inputs z vector is NEVER masked here — arm genomes gate z at replay
time (feature_gate); member opinions/books/ledger carry the substitution.

Usage:
    .venv/bin/python precompute_nightly.py [--start 2026-02-02] [--end 2026-06-09]
                                           [--variant rt2_ridge_slot|...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import books as bk            # noqa: E402
import executive as ex        # noqa: E402
import folds as fd            # noqa: E402
import infotropy as IT        # noqa: E402
import members as M           # noqa: E402
from train_members import load_panel  # noqa: E402

NIGHTLY = PROTO / "store" / "nightly"
MODELS = PROTO / "models_out"
OOF = PROTO / "oof"
SCREEN_PATH = PROTO / "store" / "infotropy_a_screen.json"

WINDOW_START = "2026-02-02"      # first post-OOF panel date (decisions start 2026-02-04)
WINDOW_END = "2026-06-09"        # last decision date priced by the 2026-06-10 snapshot
WARM_START = "2025-08-04"        # ledger warm-up start (fine-tune window start)
WARM_END = "2026-02-01"          # warm uses fold-6 OOF dates strictly before the window
EVENT_ABSTAIN = 0.10             # EventHead deploy abstain (registered constant)
EVENT_WEIGHT_CAP_DEPLOY = 1.0    # genome event_weight_cap applies at blend time, not here

# ---- retrain-arm variant stores (battery.py declared dirs) -------------------
VARIANTS = {
    "base": {},
    "rt2_ridge_slot": {"cast_source": "ridge_twin",
                       "warm_map": {"cast": "ridge_twin"}},
    "rt3_llm_neutral": {"neutralize": "llm",
                        "gbm_dir": "gbm_cond_llm_neutral",
                        "event_dir": "event_head_llm_neutral",
                        "warm_map": {"gbm_cond": "gbm_cond_llm_neutral",
                                     "event_head": "event_head_llm_neutral"}},
    "rt4_gdelt_ablated": {"neutralize": "gdelt",
                          "gbm_dir": "gbm_cond_gdelt_ablated",
                          "event_dir": "event_head_gdelt_ablated",
                          "warm_map": {"gbm_cond": "gbm_cond_gdelt_ablated",
                                       "event_head": "event_head_gdelt_ablated"}},
    "rt5_uniform": {"cast_source": "cast_uniform",
                    "warm_map": {"cast": "cast_uniform"},
                    "uniform_w_rec": True},
}


def log_look(component: str, decision: str, provenance: str) -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "component": component, "decision": decision,
             "provenance": provenance, "phase": "C-wire"}
    with (PROTO / "validation_looks.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def wiring_code_sha() -> str:
    h = hashlib.sha256()
    for f in ["precompute_nightly.py", "strategy_adapter.py", "run_replay.py",
              "books.py", "executive.py", "members.py", "folds.py"]:
        p = PROTO / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- proxy world
def proxy_vol_world(panel: dict) -> dict:
    """Trailing-vol proxy sigma_hat + fwd1, exactly synth_oof.world_from_panel
    (the executive's training convention)."""
    close = np.asarray(panel["close_px"], dtype=np.float64)
    n, s = close.shape
    fwd1 = np.full((n, s), np.nan)
    fwd1[:-1] = close[1:] / close[:-1] - 1.0
    fwd1 = np.where(np.isfinite(fwd1), fwd1, 0.0)
    r_hist = np.zeros_like(close)
    r_hist[1:] = np.where(close[:-1] > 0, close[1:] / close[:-1] - 1.0, 0.0)
    r_hist = np.where(np.isfinite(r_hist), r_hist, 0.0)
    vol = np.full((n, s), 0.012)
    for d in range(1, n):
        vol[d] = 0.95 * vol[d - 1] + 0.05 * np.abs(r_hist[d])
    sigma_hat = vol * np.sqrt(252) * 1.25
    return {"fwd1": fwd1, "sigma_hat": sigma_hat}


# ---------------------------------------------------------------- member inference
def infer_cast(panel: dict, idxs: np.ndarray, model_dir: str = "cast") -> dict:
    import torch
    seeds_files = sorted((MODELS / model_dir).glob("seed_*.pt"))
    if not seeds_files:
        raise FileNotFoundError(f"models_out/{model_dir}/seed_*.pt missing")
    sec_ids, cls_ids = M.sector_class_ids(REPO / "config" / "universe.csv")
    sec_t = torch.from_numpy(sec_ids)
    cls_t = torch.from_numpy(cls_ids)
    X = panel["X"]
    SC = panel["scalars"]
    smask = panel["symbol_mask"]
    mus, sgs = [], []
    for pf in seeds_files:
        seed = int(pf.stem.split("_")[1])
        model = M.build_cast(seed)
        model.load_state_dict(torch.load(pf, weights_only=True))
        model.eval()
        mu = np.zeros((len(idxs), X.shape[1]), dtype=np.float64)
        sg = np.zeros_like(mu)
        with torch.no_grad():
            for c0 in range(0, len(idxs), 64):
                ii = idxs[c0:c0 + 64]
                m, s, _ = model(torch.from_numpy(X[ii]), torch.from_numpy(SC[ii]),
                                sec_t, cls_t, torch.from_numpy(smask[ii]))
                mu[c0:c0 + len(ii)] = m.numpy()
                sg[c0:c0 + len(ii)] = s.numpy()
        mus.append(mu)
        sgs.append(sg)
    mu = np.nanmean(np.stack(mus), axis=0)
    sigma = np.nanmean(np.stack(sgs), axis=0)
    # c = seed agreement: mean pairwise Spearman of the seeds' mu per day
    from scipy.stats import spearmanr
    c = np.full(len(idxs), np.nan)
    for j, i in enumerate(idxs):
        m = smask[i]
        if m.sum() < 8:
            continue
        cors = []
        for a in range(len(mus)):
            for b in range(a + 1, len(mus)):
                cors.append(spearmanr(mus[a][j][m], mus[b][j][m]).statistic)
        c[j] = float(np.nanmean(cors)) if cors else np.nan
    return {"mu": mu, "sigma": np.maximum(sigma, 1e-2), "c": np.nan_to_num(c),
            "n_seeds": len(seeds_files),
            "manifest": json.loads((MODELS / model_dir / "manifest.json").read_text())}


def infer_ridge(panel: dict, idxs: np.ndarray) -> dict:
    """Ridge-twin deploy inference for the RT-2 cast slot: mu = design @ coef;
    sigma/c = 1 (the OOF twin convention, members.run_ridge)."""
    z = np.load(MODELS / "ridge_twin" / "coef.npz", allow_pickle=False)
    coef = np.asarray(z["coef"], dtype=np.float64)
    A, _, _ = M._ridge_design(panel, idxs)
    mu = np.einsum("nsd,d->ns", A.astype(np.float64), coef)
    n = len(idxs)
    return {"mu": mu, "sigma": np.ones_like(mu), "c": np.ones(n),
            "n_seeds": 1,
            "manifest": json.loads((MODELS / "ridge_twin" / "manifest.json").read_text())}


def infer_gbm(panel: dict, idxs: np.ndarray, screen: dict,
              model_dir: str = "gbm_cond") -> dict:
    import pickle
    with (MODELS / model_dir / "model.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    clf, iso = blob["clf"], blob["iso"]
    bucket_ids = blob["bucket_ids"]
    # routing variant comes from the DEPLOY manifest (registered §9.1 outcome
    # ships r3_only; hardwired "conjunctive" broke the design width — fixed)
    man = json.loads((MODELS / model_dir / "manifest.json").read_text())
    variant = man.get("params", {}).get("variant", "conjunctive")
    D, y, valid, w, mono = M.gbm_design(panel, idxs, blob["sym_is_credit"],
                                        blob["sym_is_equity"], screen, 6,
                                        variant)
    P, mu2, c2 = M.gbm_outputs(clf, D, valid, panel, idxs, iso=iso)
    return {"mu_raw": np.nan_to_num(mu2, nan=0.0), "c": np.nan_to_num(c2),
            "bucket_ids": bucket_ids, "manifest": man}


def gbm_sigma_chronological(panel: dict, dates: np.ndarray, mu_window: np.ndarray,
                            window_idxs: np.ndarray, bucket_ids: np.ndarray,
                            warm_oof: dict) -> np.ndarray:
    """sigma2 = trailing per-bucket residual sd over a chronological series that
    is WARMED by the fold-6 OOF mu (walk-forward) and continues on the deploy mu.
    Returns sigma rows aligned to window_idxs."""
    warm_idx = warm_oof["gidx"]
    mu_cat = np.concatenate([warm_oof["mu"]["gbm_cond"], mu_window], axis=0)
    y_cat = np.nan_to_num(panel["y5_rank"][np.concatenate([warm_idx, window_idxs])],
                          nan=0.0)
    sig = M.rolling_bucket_residual_sd(np.nan_to_num(mu_cat, nan=0.0), y_cat,
                                       bucket_ids)
    return sig[len(warm_idx):]


def infer_event(panel: dict, idxs: np.ndarray, screen: dict,
                model_dir: str = "event_head") -> dict:
    import pickle
    with (MODELS / model_dir / "model.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    models, keep_cols = blob["models"], blob["keep_cols"]
    b_cols = [str(c) for c in panel["B_cols"]]
    buckets = [str(b) for b in panel["buckets"]]
    symbols = [str(s) for s in panel["symbols"]]
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    kidx = [b_cols.index(c) for c in keep_cols]
    B = panel["B"]
    n = len(idxs)
    val_mu_b = np.zeros((n, len(buckets)))
    for k in range(len(buckets)):
        if models.get(k) is None:
            continue
        en, mu_, sd_, _a = models[k]
        Xk = np.nan_to_num(B[idxs][:, k, :][:, kidx].astype(np.float64), nan=0.0)
        val_mu_b[:, k] = en.predict((Xk - mu_) / sd_)
    # gated event mass percentile (c3): full-panel history so the 252d window is warm
    ev_man = json.loads((MODELS / model_dir / "manifest.json").read_text())
    ev_variant = ev_man.get("params", {}).get("variant", "conjunctive")
    passing = [f for f in IT.FAMILIES if f != "LLM_event_flags"
               and IT.screen_pass(screen, f, 6, ev_variant)]
    if passing:
        mass = np.mean([IT.family_intensity(B, b_cols, f).mean(axis=1)
                        for f in passing], axis=0)
    else:
        mass = np.zeros(B.shape[0])
    mz = pd.Series(mass)
    c3_full = mz.rolling(253, min_periods=64).apply(
        lambda w: float((w[-1] > w[:-1]).mean()), raw=True).to_numpy()
    c3 = np.nan_to_num(c3_full[idxs], nan=0.0)
    # bucket -> symbol map
    S = len(symbols)
    sym_of_bucket = np.full((len(buckets), S), False)
    for k, b in enumerate(buckets):
        for s in bucket_map[b]:
            if s in symbols:
                sym_of_bucket[k, symbols.index(s)] = True
    mu3 = np.zeros((n, S))
    cnt = np.zeros(S)
    for k in range(len(buckets)):
        mu3[:, sym_of_bucket[k]] += val_mu_b[:, k][:, None] * EVENT_WEIGHT_CAP_DEPLOY
        cnt[sym_of_bucket[k]] += 1
    cnt[cnt == 0] = 1
    mu3 /= cnt[None, :]
    mu3[c3 < EVENT_ABSTAIN] = 0.0
    # sigma3: per-bucket residual-sd proxy over the deploy training window
    dm = fd.deploy_train_mask(np.asarray(panel["dates"]).astype(str))
    resid_sd = np.nanstd(panel["y5_bucket"][np.nonzero(dm)[0]], axis=0)
    sigma3 = np.full((n, S), 1.0)
    for k in range(len(buckets)):
        v = resid_sd[k] if np.isfinite(resid_sd[k]) else 1.0
        sigma3[:, sym_of_bucket[k]] = max(v / max(np.nanmean(resid_sd), 1e-9), 0.25)
    return {"mu": mu3, "sigma": sigma3, "c": c3, "n_pass_families": len(passing),
            "manifest": ev_man}


def infer_risknet(panel: dict, idxs: np.ndarray) -> dict:
    z = np.load(MODELS / "risknet" / "coefs.npz", allow_pickle=False)

    def head(name):
        return {"coef": z[f"{name}_coef"], "mu": z[f"{name}_mu"],
                "sd": z[f"{name}_sd"], "logspace": bool(z[f"{name}_logspace"])}

    RF = M.risk_design(panel)
    Xv = RF[idxs].reshape(-1, RF.shape[2])
    sigma_hat = M._predict_head(head("sigma_hat"), Xv).reshape(len(idxs), -1)
    beta_hat = M._predict_head(head("beta_hat"), Xv).reshape(len(idxs), -1)
    bX = RF[idxs].mean(axis=1)
    book_vol_hat = M._predict_head(head("book_vol_hat"), bX)
    return {"sigma_hat": sigma_hat, "beta_hat": beta_hat,
            "book_vol_hat": book_vol_hat,
            "manifest": json.loads((MODELS / "risknet" / "manifest.json").read_text())}


# ---------------------------------------------------------------- warm OOF
def load_warm_oof(panel_dates: np.ndarray, name_map: dict | None = None) -> dict:
    """Fold-6 OOF member outputs restricted to [WARM_START, WARM_END].
    name_map redirects a slot to a variant OOF file (e.g. cast -> ridge_twin)
    so variant ledgers warm on the SAME member lineage they deploy."""
    nm = name_map or {}
    oofs = {m: bk.load_oof(OOF, 6, nm.get(m, m)) for m in bk.MEMBERS}
    d0 = oofs[bk.MEMBERS[0]]["dates"]
    for m in bk.MEMBERS:
        assert np.array_equal(oofs[m]["dates"], d0), "fold-6 OOF date mismatch"
    keep = (d0 >= WARM_START) & (d0 <= WARM_END)
    dates = d0[keep]
    gidx = np.searchsorted(panel_dates, dates)
    assert np.array_equal(panel_dates[gidx], dates), "warm OOF dates not in panel"
    out = {"dates": dates, "gidx": gidx, "mu": {}, "sigma": {}, "c": {},
           "manifests": {m: oofs[m]["manifest"] for m in bk.MEMBERS}}
    for m in bk.MEMBERS:
        out["mu"][m] = oofs[m]["mu"][keep]
        out["sigma"][m] = oofs[m]["sigma"][keep]
        out["c"][m] = oofs[m]["c"][keep]
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=WINDOW_START)
    ap.add_argument("--end", default=WINDOW_END)
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS),
                    help="retrain-arm variant store -> store/nightly_<tag>/ "
                         "(base -> store/nightly/)")
    args = ap.parse_args()
    t0 = time.time()
    command = "python " + " ".join(sys.argv)
    vc = VARIANTS[args.variant]
    out_root = NIGHTLY if args.variant == "base" \
        else PROTO / "store" / f"nightly_{args.variant}"

    panel = load_panel()
    dates = np.asarray(panel["dates"]).astype(str)
    symbols = [str(s) for s in panel["symbols"]]
    screen = json.loads(SCREEN_PATH.read_text())
    world = proxy_vol_world(panel)
    hs = bk.load_half_spread_bps()

    # variant neutralization: member INFERENCE sees the neutralized panel
    # (mirrors the retrained members' training inputs); world prices / proxy
    # sigma / Z are untouched (z gating is the genome's job at replay time)
    panel_inf = panel
    if vc.get("neutralize"):
        from retrain_arms import neutralize_panel
        panel_inf = neutralize_panel(panel, vc["neutralize"])

    window_mask = (dates >= args.start) & (dates <= args.end)
    widx = np.nonzero(window_mask)[0]
    wdates = dates[widx]
    print(f"window {wdates[0]}..{wdates[-1]} ({len(wdates)} panel dates) "
          f"variant={args.variant}")

    # ---- deploy member inference over the window (vectorized) -----------------
    cast_source = vc.get("cast_source", "cast")
    print(f"CAST slot ({cast_source})...", flush=True)
    if cast_source == "ridge_twin":
        cast = infer_ridge(panel_inf, widx)
    else:
        cast = infer_cast(panel_inf, widx, model_dir=cast_source)
    print("GBM-Cond...", flush=True)
    gbm = infer_gbm(panel_inf, widx, screen, model_dir=vc.get("gbm_dir", "gbm_cond"))
    print("EventHead...", flush=True)
    event = infer_event(panel_inf, widx, screen,
                        model_dir=vc.get("event_dir", "event_head"))
    print("RiskNet+...", flush=True)
    risk = infer_risknet(panel, widx)

    # ---- warm history (fold-6 OOF) + chronological ledger ---------------------
    print("ledger (warm fold-6 OOF + window deploy)...", flush=True)
    warm = load_warm_oof(dates, name_map=vc.get("warm_map"))
    gbm_sigma = gbm_sigma_chronological(panel, dates, gbm["mu_raw"], widx,
                                        gbm["bucket_ids"], warm)
    member_out = {
        "cast": {"mu": cast["mu"], "sigma": cast["sigma"], "c": cast["c"]},
        "gbm_cond": {"mu": gbm["mu_raw"], "sigma": gbm_sigma, "c": gbm["c"]},
        "event_head": {"mu": event["mu"], "sigma": event["sigma"], "c": event["c"]},
    }

    cat_gidx = np.concatenate([warm["gidx"], widx])
    cat_dates = dates[cat_gidx]
    n_warm = len(warm["gidx"])
    books_cat, u_by = {}, {}
    rets_seq = world["fwd1"][cat_gidx]
    sh_seq = world["sigma_hat"][cat_gidx]
    for m in bk.MEMBERS:
        mu_cat = np.concatenate([warm["mu"][m], member_out[m]["mu"]], axis=0)
        sg_cat = np.concatenate([warm["sigma"][m], member_out[m]["sigma"]], axis=0)
        books_cat[m] = bk.solo_books(mu_cat, sg_cat)
        u_by[m] = bk.realized_u_series(books_cat[m], rets_seq, sh_seq, hs)

    # raw ledger on one pseudo-fold, then warm-stat standardization
    ledger = bk.build_trust_ledger(cat_dates, u_by, np.zeros(len(cat_dates)))
    warm_stats = {}
    r_stats = {}
    for m in bk.MEMBERS:
        st = ledger[m]
        ws = {}
        for key in ("ewma21_raw", "ewma63_raw"):
            v = st[key][:n_warm]
            v = v[np.isfinite(v)]
            ws[key] = {"mean": float(np.mean(v)) if len(v) else 0.0,
                       "sd": float(np.std(v)) if len(v) and np.std(v) > 1e-12 else 1.0}
        warm_stats[m] = ws
        e21z = (st["ewma21_raw"] - ws["ewma21_raw"]["mean"]) / ws["ewma21_raw"]["sd"]
        e63z = (st["ewma63_raw"] - ws["ewma63_raw"]["mean"]) / ws["ewma63_raw"]["sd"]
        r_stats[m] = np.nan_to_num(
            np.stack([e21z, e63z, st["hit_rate"], st["cf_drawdown"]], axis=1))
    u_warm = np.concatenate([u_by[m][:n_warm] for m in bk.MEMBERS])
    u_std = float(max(np.nanstd(u_warm), 1e-4))
    w_rec_raw_cat = np.nan_to_num(panel["w_rec_score"][cat_gidx], nan=0.0).mean(axis=1)
    if vc.get("uniform_w_rec"):
        # rt5: uniform record weights in the replay tilt (clip(eps+1,...)=1)
        w_rec_raw_cat = np.ones_like(w_rec_raw_cat)

    # ---- per-date artifacts ----------------------------------------------------
    out_root.mkdir(parents=True, exist_ok=True)
    rows_ledger = []
    for m in bk.MEMBERS:
        st = ledger[m]
        rows_ledger.append(pd.DataFrame({
            "date": cat_dates, "member": m, "u_realized": u_by[m],
            "w_rec_raw": w_rec_raw_cat,
            "ewma21_z": r_stats[m][:, 0], "ewma63_z": r_stats[m][:, 1],
            "hit_rate": np.nan_to_num(st["hit_rate"]),
            "cf_drawdown": np.nan_to_num(st["cf_drawdown"]),
            "segment": np.where(np.arange(len(cat_dates)) < n_warm,
                                "warm_oof_f6", "window_deploy"),
        }))
    pd.concat(rows_ledger, ignore_index=True).to_parquet(
        out_root / "ledger.parquet", index=False)
    (out_root / "ledger_meta.json").write_text(json.dumps({
        "u_std": u_std, "warm_stats": warm_stats, "lag_rule": "D-h-1 (h=5)",
        "warm_source": "oof/fold_6_<member>.npz (walk-forward OOF), dates "
                       f"{WARM_START}..{WARM_END} (n={n_warm})",
        "window_source": "deploy members (models_out/) over the panel window",
        "u_reference": "books.realized_u_series f=0.7, vol cap 0.10, half-spread table",
        "sigma_hat_source": "trailing-vol proxy (synth_oof.world_from_panel "
                            "convention; matches executive training inputs)",
    }, indent=1))

    eq_book = np.ones(len(symbols)) / len(symbols)
    member_manifests = {"cast": cast["manifest"], "gbm_cond": gbm["manifest"],
                        "event_head": event["manifest"], "risknet": risk["manifest"]}
    n_written = 0
    for j, d in enumerate(wdates):
        out = out_root / str(d)
        out.mkdir(parents=True, exist_ok=True)
        gi = n_warm + j                              # index in concatenated series
        # expert_opinions
        op_rows = []
        for m in bk.MEMBERS:
            man = member_manifests["cast" if m == "cast" else
                                   ("gbm_cond" if m == "gbm_cond" else "event_head")]
            op_rows.append(pd.DataFrame({
                "member": m, "symbol": symbols,
                "mu": member_out[m]["mu"][j], "sigma": member_out[m]["sigma"][j],
                "c": float(member_out[m]["c"][j]),
                "model_version": man["model_version"],
                "train_window_end": man["train_window_end"],
                "walk_forward": bool(man["walk_forward"]),
                "code_sha": man["code_sha"],
            }))
        pd.concat(op_rows, ignore_index=True).to_parquet(
            out / "expert_opinions.parquet", index=False)
        # solo_books
        bk_rows = [pd.DataFrame({"member": m, "symbol": symbols,
                                 "w": books_cat[m][gi]}) for m in bk.MEMBERS]
        pd.concat(bk_rows, ignore_index=True).to_parquet(
            out / "solo_books.parquet", index=False)
        # risknet (E4 heads + the proxy convention the executive trained on)
        pd.DataFrame({
            "symbol": symbols,
            "sigma_hat": risk["sigma_hat"][j], "beta_hat": risk["beta_hat"][j],
            "book_vol_hat": float(risk["book_vol_hat"][j]),
            "sigma_hat_proxy": world["sigma_hat"][widx[j]],
            "book_vol_hat_proxy": float(world["sigma_hat"][widx[j]] @ eq_book),
        }).to_parquet(out / "risknet.parquet", index=False)
        # executive-input bundle
        wb = np.stack([books_cat[m][gi] for m in bk.MEMBERS])
        agree, g = ex._book_stats(wb)
        (out / "exec_inputs.json").write_text(json.dumps({
            "date": str(d),
            "r": [r_stats[m][gi].tolist() for m in bk.MEMBERS],
            "z": np.nan_to_num(np.asarray(panel["Z"][widx[j]],
                                          dtype=np.float64)).tolist(),
            "c": [float(member_out[m]["c"][j]) for m in bk.MEMBERS],
            "agree": agree.tolist(), "g": g.tolist(),
            "book_vol_hat_proxy": float(world["sigma_hat"][widx[j]] @ eq_book),
            "book_vol_hat_e4": float(risk["book_vol_hat"][j]),
            "members": list(bk.MEMBERS),
        }, indent=1))
        n_written += 1

    meta = json.loads((PROTO / "store" / "panel_meta.json").read_text())
    (out_root / "manifest.json").write_text(json.dumps({
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "variant": args.variant, "variant_config": vc,
        "window": [str(wdates[0]), str(wdates[-1])], "n_dates": n_written,
        "panel": {"generated": meta["generated"], "llm_merged": meta.get("llm_merged"),
                  "n_dates": meta["n_dates"]},
        "member_manifests": member_manifests,
        "warm": {"source": "oof/fold_6_<member>.npz", "start": WARM_START,
                 "end": WARM_END, "n": n_warm},
        "event_head": {"n_pass_families": event["n_pass_families"],
                       "abstain_threshold": EVENT_ABSTAIN},
        "cast_n_seeds": cast["n_seeds"],
        "code_sha": wiring_code_sha(), "command": command,
        "wall_clock_s": round(time.time() - t0, 1),
    }, indent=1))
    log_look("precompute_nightly",
             "ledger EWMAs standardized with warm-period (fold-6 OOF) stats; "
             "exec inputs use trailing-vol proxy sigma_hat/book_vol_hat "
             "(executive training convention), E4 heads carried alongside",
             "wiring adjudication — training/deploy input consistency")
    print(f"wrote {n_written} nightly dirs -> {out_root} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
