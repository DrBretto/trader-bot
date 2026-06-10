"""PKT-TB-006 — BUILD_SPEC §15 step 10: floor baselines, diversity, break-even IC.

All reads are PRE-HOLDOUT validation only (exec data ends at FITNESS_END
2026-02-06; folds.assert_no_holdout guards upstream). Three checks:

  1. §5.5 floor baselines: SYN-1 (shipping genome through the E1 walk) vs
     (a) P2-slow linear floor  w ∝ softplus(W·x) on {ret21, ret63, vol63, dd63,
         rs21, rs63, breadth},
     (b) P2-fast linear floor on {ret1, ret5, vol21, dd21, rs21, vixy_beta},
     (c) equal-weight risk-targeted book (1/64, vol-capped at 0.10).
     Floors fit per fold by ridge on y5_raw over the embargoed expanding train
     window (walk-forward); all books vol-targeted via the same sigma_hat
     proxy; same half-spread cost table; U_f = min over cost x{1.0,1.5} of
     sqrt(252)*Sharpe - 0.5*MaxDD/0.10 (the EA fitness shape).
     FORCED CHOICES (logged): floor features = the panel's per-symbol trailing-z
     X columns at the last window step (dd63/rs63 etc. exist only there);
     breadth = br_pct_above_200d; vixy_beta = trailing-252d beta of close
     returns to VIXY (days < D, min 126 obs).
  2. §4.6.5 diversity floor: mean pairwise correlation of member solo-book
     daily returns on validation folds (>= 0.90 triggers the pre-registered
     statement).
  3. §4.6.4 break-even IC: mu-stack purged weekly rank IC vs the 0.02 bar
     (stack = equal-weight mean of per-day cross-sectional ranks of member mu;
     forced choice, logged).

Output: store/step10_checks.json + console table.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))

import books as bk        # noqa: E402
import e1_reads as e1     # noqa: E402
import ea                 # noqa: E402
import executive as ex    # noqa: E402
import folds as fd        # noqa: E402
import members as M       # noqa: E402
import synth_oof as so    # noqa: E402

GENOME_PATH = PROTO / "ea" / "genome_2026-02-06.json"
X_IDX = {"return_1d": 0, "return_5d": 1, "return_21d": 2, "return_63d": 3,
         "vol_21d": 4, "vol_63d": 5, "drawdown_21d": 6, "drawdown_63d": 7,
         "rel_strength_21d": 8, "rel_strength_63d": 9}
SLOW = ["return_21d", "return_63d", "vol_63d", "drawdown_63d",
        "rel_strength_21d", "rel_strength_63d"]          # + breadth
FAST = ["return_1d", "return_5d", "vol_21d", "drawdown_21d",
        "rel_strength_21d"]                              # + vixy_beta


def util_from_r(r: np.ndarray) -> float:
    sd = r.std()
    shrp = np.sqrt(252) * r.mean() / sd if sd > 1e-12 else 0.0
    eq = np.cumsum(np.log1p(np.clip(r, -0.99, None)))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    maxdd = float(np.max(np.concatenate([[0.0], peak - eq])))
    return float(shrp - 0.5 * (1.0 - np.exp(-maxdd)) / 0.10)


def book_walk(w_seq: np.ndarray, fwd1: np.ndarray, hs: np.ndarray,
              cost_scale: float) -> np.ndarray:
    """Daily returns of a weight sequence with turnover costs (no rails)."""
    n = len(w_seq)
    r = np.zeros(n)
    w_prev = np.zeros(w_seq.shape[1])
    for i in range(n):
        traded = w_seq[i] - w_prev
        cost = np.abs(traded) @ hs * cost_scale / 1e4
        r[i] = float(w_seq[i] @ fwd1[i]) - cost
        w_prev = w_seq[i]
    return r


def vol_target(w: np.ndarray, sigma_hat: np.ndarray, cap: float = 0.10) -> np.ndarray:
    """Scale each day's book so the linear vol bound <= cap."""
    out = w.copy()
    for i in range(len(w)):
        v = w[i] @ sigma_hat[i]
        if v > cap > 0:
            out[i] *= cap / v
    return out


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def main():
    t0 = time.time()
    z = np.load(PROTO / "store" / "panel.npz")
    panel_dates = z["dates"].astype(str)
    symbols = [str(s) for s in z["symbols"]]
    world = so.world_from_panel(PROTO / "store" / "panel.npz")
    fold_list = [1, 2, 3, 4, 5, 6]
    data = ex.assemble_exec_data(world, PROTO / "oof", fold_list,
                                 end_date=fd.FITNESS_END)
    lofo = {}
    for f in fold_list:
        m = ex.Executive()
        m.load_state_dict(torch.load(PROTO / "exec_out" / f"lofo_fold{f}.pt"))
        m.eval()
        lofo[f] = m
    engine = e1.E1WalkEngine(data, lofo, fold_list)
    genome = ea.Genome.from_json(GENOME_PATH)
    hs = data.half_spread

    # panel row index for each exec-data day
    gidx = np.searchsorted(panel_dates, data.dates)
    assert np.array_equal(panel_dates[gidx], data.dates)

    # ---- floor-policy feature matrices ------------------------------------
    Xlast = z["X"][:, :, -1, :]                       # [N,S,14] trailing-z at D
    T = z["T"]
    t_cols = [str(c) for c in z["T_cols"]]
    breadth = T[:, :, t_cols.index("br_pct_above_200d")]
    # vixy beta: trailing 252d beta of close returns to VIXY, days < D
    px = z["close_px"]
    rets = np.diff(np.log(px), axis=0)
    rets = np.vstack([np.full((1, px.shape[1]), np.nan), rets])
    vix_i = symbols.index("VIXY")
    import pandas as pd
    rdf = pd.DataFrame(rets)
    v = rdf[vix_i]
    cov = rdf.shift(1).rolling(252, min_periods=126).cov(v.shift(1))
    var = v.shift(1).rolling(252, min_periods=126).var()
    vixy_beta = (cov.div(var, axis=0)).to_numpy()

    feats = {
        "p2_slow": np.concatenate(
            [Xlast[:, :, [X_IDX[c] for c in SLOW]], breadth[:, :, None]], axis=2),
        "p2_fast": np.concatenate(
            [Xlast[:, :, [X_IDX[c] for c in FAST]], vixy_beta[:, :, None]], axis=2),
    }
    y = z["y5_raw"]
    sm = z["symbol_mask"].astype(bool)

    # ---- per-fold walk ------------------------------------------------------
    from sklearn.linear_model import Ridge
    policies = ["syn1", "p2_slow", "p2_fast", "equal_weight_rt"]
    per_fold: dict = {p: {} for p in policies}
    pooled_r = {p: [] for p in policies}
    div_per_fold = {}
    ic_per_fold = {}
    for f in fold_list:
        idx = engine.fold_idx[f]                     # exec-data rows for fold f
        gi = gidx[idx]
        fwd1 = data.rets5[idx, 0, :]
        sh = data.sigma_hat[idx]

        # SYN-1 with the shipping genome (cost scenarios via the E1 walk)
        us = []
        for sc in ea.COST_SCENARIOS:
            _, r_s = engine.fold_daily_returns(genome, f, cost_scale=sc)
            us.append(util_from_r(r_s))
            if sc == 1.0:
                r_syn = r_s
        per_fold["syn1"][f"F{f}"] = float(min(us))
        pooled_r["syn1"].append(r_syn)

        # floor policies
        trm = fd.train_mask_for_fold(panel_dates, f)
        tr_rows = np.where(trm)[0]
        for name, FT in feats.items():
            Xtr = FT[tr_rows].reshape(-1, FT.shape[2])
            ytr = y[tr_rows].reshape(-1)
            mtr = (sm[tr_rows].reshape(-1) & np.isfinite(ytr)
                   & np.all(np.isfinite(Xtr), axis=1))
            mu_f = Xtr[mtr].mean(0)
            sd_f = Xtr[mtr].std(0) + 1e-9
            model = Ridge(alpha=1.0)
            model.fit((Xtr[mtr] - mu_f) / sd_f, ytr[mtr])
            Xv = FT[gi]
            n, S, K = Xv.shape
            pred = model.predict(
                ((Xv.reshape(-1, K)) - mu_f) / sd_f).reshape(n, S)
            pred[~(sm[gi] & np.all(np.isfinite(Xv), axis=2))] = -np.inf
            w = softplus(pred / (np.nanstd(pred[np.isfinite(pred)]) + 1e-12))
            w[~np.isfinite(w)] = 0.0
            gs = w.sum(axis=1, keepdims=True)
            w = np.divide(w, gs, out=np.zeros_like(w), where=gs > 1e-12)
            w = np.minimum(w, 0.10)                  # the same w_cap rail
            w = vol_target(w, sh)
            us = [util_from_r(book_walk(w, np.nan_to_num(fwd1), hs, sc))
                  for sc in ea.COST_SCENARIOS]
            per_fold[name][f"F{f}"] = float(min(us))
            pooled_r[name].append(book_walk(w, np.nan_to_num(fwd1), hs, 1.0))

        # equal-weight risk-targeted
        active = sm[gi].astype(float)
        w_eq = active / np.maximum(active.sum(axis=1, keepdims=True), 1)
        w_eq = vol_target(w_eq, sh)
        us = [util_from_r(book_walk(w_eq, np.nan_to_num(fwd1), hs, sc))
              for sc in ea.COST_SCENARIOS]
        per_fold["equal_weight_rt"][f"F{f}"] = float(min(us))
        pooled_r["equal_weight_rt"].append(book_walk(w_eq, np.nan_to_num(fwd1), hs, 1.0))

        # ---- diversity: member solo-book daily returns ----------------------
        r_m = np.einsum("nms,ns->nm", data.books[idx], np.nan_to_num(fwd1))
        cors = []
        for a in range(3):
            for b in range(a + 1, 3):
                sa, sb = r_m[:, a], r_m[:, b]
                if sa.std() > 1e-12 and sb.std() > 1e-12:
                    cors.append(float(np.corrcoef(sa, sb)[0, 1]))
        div_per_fold[f"F{f}"] = {"pairwise": cors, "mean": float(np.mean(cors))}

        # ---- break-even IC: mu-stack weekly rank IC --------------------------
        from scipy.stats import spearmanr, rankdata
        mu = data.mu[idx]                            # [n,3,64]
        ranks = np.zeros_like(mu)
        for mi in range(3):
            for i in range(mu.shape[0]):
                ranks[i, mi] = rankdata(mu[i, mi])
        stack = ranks.mean(axis=1)
        ics = []
        for j in range(0, len(idx), 5):
            yy = y[gi[j]]
            mm = sm[gi[j]] & np.isfinite(yy)
            if mm.sum() >= 8:
                ics.append(spearmanr(stack[j][mm], yy[mm]).statistic)
        ic_per_fold[f"F{f}"] = float(np.nanmean(ics))

    means = {p: float(np.mean(list(per_fold[p].values()))) for p in policies}
    pooled = {p: {"mean_daily": float(np.concatenate(pooled_r[p]).mean()),
                  "sharpe": float(np.sqrt(252) * np.concatenate(pooled_r[p]).mean()
                                  / np.concatenate(pooled_r[p]).std()),
                  "util": util_from_r(np.concatenate(pooled_r[p]))}
              for p in policies}
    div_mean = float(np.mean([d["mean"] for d in div_per_fold.values()]))
    ic_mean = float(np.mean(list(ic_per_fold.values())))

    res = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "walk_forward": True, "code_sha": M.code_sha(),
        "command": "python " + " ".join(sys.argv),
        "genome": "ea/genome_2026-02-06.json (shipped champion)",
        "window": [str(sorted(data.dates)[0]), str(sorted(data.dates)[-1])],
        "floor_check": {"per_fold_util_min_cost_scenarios": per_fold,
                        "fold_mean": means, "pooled_cost1": pooled,
                        "syn1_beats_p2_slow": means["syn1"] > means["p2_slow"],
                        "syn1_beats_p2_fast": means["syn1"] > means["p2_fast"],
                        "syn1_beats_equal_weight": means["syn1"] > means["equal_weight_rt"]},
        "diversity_floor": {"per_fold": div_per_fold, "mean_pairwise_corr": div_mean,
                            "threshold": 0.90, "triggered": div_mean >= 0.90},
        "break_even_ic": {"per_fold_weekly_rank_ic": ic_per_fold,
                          "mean": ic_mean, "bar": 0.02, "break_even": 0.006,
                          "passes_functionality_bar": ic_mean >= 0.02},
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (PROTO / "store" / "step10_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["floor_check"]["fold_mean"], indent=1))
    print("diversity mean pairwise corr:", round(div_mean, 4),
          "| triggered(>=0.90):", div_mean >= 0.90)
    print("mu-stack weekly rank IC per fold:",
          {k: round(v, 4) for k, v in ic_per_fold.items()})
    print("mu-stack weekly rank IC mean:", round(ic_mean, 4), "| bar 0.02 ->",
          "PASS" if ic_mean >= 0.02 else "FAIL")


if __name__ == "__main__":
    main()
