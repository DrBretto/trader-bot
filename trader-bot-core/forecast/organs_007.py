"""PKT-TB-007 — ORB-1 organ architectures + training functions (BUILD_SPEC_007 §1).

M1 CAST-XS    TB-006 CAST-Small shrink-rung config on the 10-col fast block
              (NO 63d stats — the TB-006 deploy used all 14 X cols, so the
              partition CHANGED and M1 is retrained; ledgered). 2 OOF seeds
              {4242, 4243}; anti-memorization spine unchanged (w_rec, rank +
              aux head, early stop on fold val).
Ridge twin    referee on the same 10-col partition (reported, not gated).
M2 ROT-GBM    HistGB regressor on y_rot_rank; slow+macro partition; embargo
              26 td; <=600 leaf values; purged early stop (truncation against
              an embargoed inner tail, NOT the OOF fold).
M3 DISP-HAR   OLS, log link, 13 predictors + intercept; lags-only twin.
M4 EVT-NET    pooled (day,bucket) logistic elastic net; A = GDELT-only
              mask-interaction + vol control; B = A + llm_* masked
              pre-2024-08-15; control = vol-control columns only.
M5 POS-Z      rule, 0 trained params; 156-week COT z from cache/cot
              (publication-keyed; the panel's 52w z is NOT the spec's z).
M6 GAP-GRU    GRU(8,24,1)+linear head ~2.5k params, seq 10, MSE on y1_z,
              seed 4242, early stop on fold val. Forecast altitude only.

All functions honor folds.py geometry (007 extensions), never touch
>= 2026-03-11, and return everything the driver needs for OOF npz + manifests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"

SUPPORT = ["ITA", "SOXX", "XRT", "TLT", "AGG", "MUB", "FXE", "USO", "FXI",
           "RSP", "IYR", "SHY", "KRE"]

# ---------------------------------------------------------------- shared utils

def code_sha() -> str:
    h = hashlib.sha256()
    for f in ["folds.py", "organs_007.py", "train_organs_007.py",
              "make_targets_007.py"]:
        p = PROTO / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


# ======================= M1 CAST-XS (10-col fast block) =======================

M1_XCOLS = ["return_1d", "return_5d", "return_21d", "vol_21d", "drawdown_21d",
            "rel_strength_21d", "range_pct", "volume_z_21d", "gap_open_pct",
            "dist_from_52w_high"]

# sector/class vocabularies — FROZEN TB-006 members.py mappings, carried verbatim
SECTOR13 = ["us_broad", "tech", "financials", "energy", "defensives",
            "cyclicals", "real_estate", "intl_dev", "em", "rates",
            "credit", "metals_commod", "vol_fx"]
_SECTOR_MAP = {
    "broad": "us_broad", "style_growth": "us_broad", "style_value": "us_broad",
    "factor_momentum": "us_broad", "factor_quality": "us_broad",
    "factor_value": "us_broad", "factor_minvol": "us_broad",
    "factor_dividend": "us_broad", "factor_dividend_growth": "us_broad",
    "theme_innovation": "us_broad",
    "sector_tech": "tech", "industry_semis": "tech", "sector_comm": "tech",
    "sector_financials": "financials", "industry_regional_banks": "financials",
    "sector_energy": "energy", "oil": "energy", "natural_gas": "energy",
    "sector_cons_staples": "defensives", "sector_utilities": "defensives",
    "sector_healthcare": "defensives", "industry_biotech": "defensives",
    "sector_cons_disc": "cyclicals", "industry_retail": "cyclicals",
    "sector_industrials": "cyclicals", "industry_aerospace_defense": "cyclicals",
    "industry_transport": "cyclicals", "sector_materials": "cyclicals",
    "sector_reit": "real_estate",
    "international_dev": "intl_dev", "region_europe": "intl_dev",
    "country_japan": "intl_dev", "global": "intl_dev",
    "international_em": "em", "country_china": "em", "country_india": "em",
    "country_brazil": "em",
    "treas_short": "rates", "treas_intermediate": "rates", "treas_long": "rates",
    "treas_tips": "rates", "aggregate": "rates", "muni": "rates",
    "credit_high_yield": "credit", "credit_investment_grade": "credit",
    "gold": "metals_commod", "silver": "metals_commod",
    "broad_commodities": "metals_commod",
    "volatility": "vol_fx", "usd": "vol_fx", "eur": "vol_fx",
}
CLASS6 = ["equity_us", "equity_intl", "bond", "commodity", "fx", "vol"]
_INTL_SECTORS = {"international_dev", "international_em", "country_china",
                 "country_india", "country_brazil", "country_japan",
                 "region_europe", "global", "eur"}


def sector_class_ids(universe_csv: Path):
    u = pd.read_csv(universe_csv)
    sec, cls = [], []
    for _, r in u.iterrows():
        sec.append(SECTOR13.index(_SECTOR_MAP[r["sector"]]))
        if r["asset_class"] == "equity":
            cls.append(CLASS6.index(
                "equity_intl" if r["sector"] in _INTL_SECTORS else "equity_us"))
        else:
            cls.append(CLASS6.index({"bond": "bond", "commodity": "commodity",
                                     "fx": "fx", "vol": "vol"}[r["asset_class"]]))
    return np.array(sec, dtype=np.int64), np.array(cls, dtype=np.int64)


def build_cast_xs(seed: int, n_feat: int = 10):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class CASTXS(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_feat, 32, batch_first=True)
            self.sector_emb = nn.Embedding(13, 8)
            self.class_emb = nn.Embedding(6, 4)
            self.token = nn.Linear(48, 32)
            self.enc = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, dropout=0.20,
                batch_first=True, norm_first=True)
            self.mu_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                         nn.Linear(16, 1))
            self.sigma_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                            nn.Linear(16, 1))
            self.aux_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                          nn.Linear(16, 1))

        def forward(self, x, scalars, sec_ids, cls_ids, mask):
            Bd, S = x.shape[0], x.shape[1]
            h = self.gru(x.reshape(Bd * S, 63, n_feat))[1][0].reshape(Bd, S, 32)
            emb = torch.cat([self.sector_emb(sec_ids),
                             self.class_emb(cls_ids)], -1)
            tok = self.token(torch.cat([h, emb.unsqueeze(0).expand(Bd, -1, -1),
                                        scalars], -1))
            out = self.enc(tok, src_key_padding_mask=~mask)
            mu = self.mu_head(out).squeeze(-1)
            sigma = torch.nn.functional.softplus(
                self.sigma_head(out).squeeze(-1)) + 1e-2
            aux = self.aux_head(out).squeeze(-1)
            return mu, sigma, aux

    return CASTXS()


def soft_spearman(mu, y_rank, mask, tau: float = 0.1):
    import torch
    cors = []
    for b in range(mu.shape[0]):
        m = mask[b]
        if int(m.sum()) < 8:
            continue
        x = mu[b][m]
        y = y_rank[b][m]
        sr = torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / tau).sum(dim=1)
        yr = torch.argsort(torch.argsort(y)).float()
        sx = sr - sr.mean()
        sy = yr - yr.mean()
        c = (sx * sy).sum() / (sx.norm() * sy.norm() + 1e-8)
        cors.append(c)
    if not cors:
        import torch as _t
        return _t.tensor(0.0)
    import torch
    return torch.stack(cors).mean()


def cast_loss(mu, sigma, aux, y5_rank, y1_z, w_rec, mask):
    import torch
    m = mask & torch.isfinite(y5_rank)
    if int(m.sum()) == 0:
        return None
    hub = torch.nn.functional.smooth_l1_loss(mu[m], y5_rank[m],
                                             reduction="none")
    w = torch.nan_to_num(w_rec[m], nan=0.0)
    l_hub = (hub * w).mean()
    l_sp = 1.0 - soft_spearman(mu, y5_rank, m)
    var = sigma[m] ** 2
    l_nll = (0.5 * (torch.log(var)
                    + (y5_rank[m] - mu[m].detach()) ** 2 / var)).mean()
    ma = mask & torch.isfinite(y1_z)
    l_aux = ((aux[ma] - y1_z[ma]) ** 2).mean() if int(ma.sum()) else mu.sum() * 0
    return l_hub + 0.25 * l_sp + l_nll + 0.3 * l_aux


def train_cast_xs_one(panel, X10, train_idx, val_idx, seed, sec_ids, cls_ids,
                      max_epochs=60, patience=10, lr=1e-3, days_per_step=8):
    """One CAST-XS run on the 10-col block. Mirrors TB-006 train_cast_one."""
    import torch
    torch.manual_seed(seed)
    model = build_cast_xs(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    SC = panel["scalars"]
    y5 = panel["y5_rank"].astype(np.float32)
    y1 = panel["y1_z"].astype(np.float32)
    wr = panel["w_rec"].astype(np.float32)
    smask = panel["symbol_mask"]
    sec_t = torch.from_numpy(sec_ids)
    cls_t = torch.from_numpy(cls_ids)

    def day_batch(idxs):
        return (torch.from_numpy(X10[idxs]), torch.from_numpy(SC[idxs]),
                torch.from_numpy(y5[idxs]), torch.from_numpy(y1[idxs]),
                torch.from_numpy(wr[idxs]), torch.from_numpy(smask[idxs]))

    @torch.no_grad()
    def val_eval():
        from scipy.stats import spearmanr
        model.eval()
        mus = np.zeros((len(val_idx), X10.shape[1]), dtype=np.float32)
        sgs = np.zeros_like(mus)
        sp = []
        for c0 in range(0, len(val_idx), 64):
            ii = val_idx[c0:c0 + 64]
            xb, sb, yb, _, _, mb = day_batch(ii)
            mu, sg, _ = model(xb, sb, sec_t, cls_t, mb)
            mus[c0:c0 + len(ii)] = mu.numpy()
            sgs[c0:c0 + len(ii)] = sg.numpy()
        y_raw = panel["y5_raw"]
        for j, i in enumerate(val_idx):
            m = smask[i] & np.isfinite(y_raw[i])
            if m.sum() >= 8:
                sp.append(spearmanr(mus[j][m], y_raw[i][m]).statistic)
        model.train()
        return (float(np.nanmean(sp)) if sp else np.nan), mus, sgs

    rng = np.random.RandomState(seed)
    best = (-np.inf, None, 0, None, None)
    bad = 0
    hist = []
    for epoch in range(max_epochs):
        order = train_idx.copy()
        rng.shuffle(order)
        for c0 in range(0, len(order), days_per_step):
            ii = order[c0:c0 + days_per_step]
            xb, sb, yb, y1b, wb, mb = day_batch(ii)
            if not bool(mb.any()):
                continue
            loss = cast_loss(*model(xb, sb, sec_t, cls_t, mb), yb, y1b, wb, mb)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
        vs, vmu, vsg = (val_eval() if len(val_idx) else (np.nan, None, None))
        hist.append(vs)
        if len(val_idx) == 0:
            best = (np.nan, {k: v.clone() for k, v in model.state_dict().items()},
                    epoch, None, None)
            continue
        if vs > best[0]:
            best = (vs, {k: v.clone() for k, v in model.state_dict().items()},
                    epoch, vmu, vsg)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[1], best[0], best[2], best[3], best[4], hist


# ---------------- ridge twin on the 10-col partition --------------------------

RIDGE_LAMBDAS = [1.0, 10.0, 100.0, 1e3, 1e4]


def _ridge_design(panel, X10, idxs):
    X = X10[idxs]
    SC = panel["scalars"][idxs]
    n, S = X.shape[0], X.shape[1]
    A = np.concatenate([X.reshape(n, S, -1), SC], axis=2)
    y = panel["y5_rank"][idxs]
    m = panel["symbol_mask"][idxs] & np.isfinite(y)
    return A, y.astype(np.float64), m


def _gram_accumulate(A, y, m, chunk: int = 64):
    d = A.shape[2]
    G = np.zeros((d, d))
    b = np.zeros(d)
    for c0 in range(0, A.shape[0], chunk):
        Aa = A[c0:c0 + chunk].astype(np.float64)
        yy = y[c0:c0 + chunk]
        mm = m[c0:c0 + chunk]
        rows = Aa[mm]
        G += rows.T @ rows
        b += rows.T @ yy[mm]
    return G, b


def train_ridge_twin(panel, X10, train_idx, val_idx, internal_frac=0.8):
    from scipy.stats import spearmanr
    cut = int(len(train_idx) * internal_frac)
    A_tr, y_tr, m_tr = _ridge_design(panel, X10, train_idx[:cut])
    G, b = _gram_accumulate(A_tr, y_tr, m_tr)
    A_iv, y_iv, m_iv = _ridge_design(panel, X10, train_idx[cut:])
    d = G.shape[0]
    best = (None, -np.inf)
    for lam in RIDGE_LAMBDAS:
        coef = np.linalg.solve(G + lam * np.eye(d), b)
        sp = []
        for j in range(A_iv.shape[0]):
            m = m_iv[j]
            if m.sum() >= 8:
                sp.append(spearmanr(A_iv[j, m].astype(np.float64) @ coef,
                                    y_iv[j][m]).statistic)
        s = float(np.nanmean(sp)) if sp else -np.inf
        if s > best[1]:
            best = (lam, s)
    lam = best[0] if best[0] is not None else 100.0
    G2, b2 = _gram_accumulate(A_iv, y_iv, m_iv)
    coef = np.linalg.solve(G + G2 + lam * np.eye(d), b + b2)
    val_mu = None
    if len(val_idx):
        A_v, _, _ = _ridge_design(panel, X10, val_idx)
        val_mu = np.einsum("nsd,d->ns", A_v.astype(np.float64), coef)
    return val_mu, lam, coef


# =========================== M2 ROT-GBM =======================================

M2_MACRO_TCOLS = ["ctx_rate_2y", "ctx_rate_10y", "ctx_yield_slope",
                  "ctx_credit_spread", "ctx_risk_off", "ctx_spy_ret_21d",
                  "ctx_spy_vol_21d", "fred_nfci", "fred_stlfsi4",
                  "fred_hy_oas", "fred_hy_oas_d5", "fred_icsa_z",
                  "fred_t10yie_d21", "fred_dfii10", "fred_dfii10_d63",
                  "br_adv_dec", "br_pct_above_50d", "br_pct_above_200d",
                  "br_rsp_spy_21"]
M2_SLOW_COLS = ["return_63d", "vol_63d", "drawdown_63d", "rel_strength_63d"]
M2_PARAMS = dict(max_iter=60, max_leaf_nodes=10, max_depth=4,
                 learning_rate=0.05, l2_regularization=1.0)
M2_ES_GRID = list(range(10, 61, 5))
M2_EMBARGO_INNER = 27          # 26 td + 1, inner early-stop gap


def m2_design(panel6, p7, idxs, sym_is_credit):
    """[n,S,24] design rows for fold/deploy indices.

    4 slow + 19 macro + hy_oas_d5 x is_credit (monotone -1). The TB-006
    cor3m_z x is_equity interaction is NOT carried: s1_cor3m_z is vol-surface,
    excluded from M2's partition (forced choice, ledgered)."""
    T = panel6["T"][idxs]
    t_cols = [str(c) for c in panel6["T_cols"]]
    slow = p7["m2_slow"][idxs].astype(np.float64)
    macro = np.stack([T[:, :, t_cols.index(c)].astype(np.float64)
                      for c in M2_MACRO_TCOLS], axis=2)
    hy = T[:, :, t_cols.index("fred_hy_oas_d5")].astype(np.float64)
    inter = (hy * sym_is_credit[None, :])[:, :, None]
    D = np.concatenate([slow, macro, inter], axis=2)
    mono = np.zeros(D.shape[2], dtype=np.int8)
    mono[-1] = -1
    y = p7["y_rot_rank"][idxs].astype(np.float64)
    valid = panel6["symbol_mask"][idxs] & np.isfinite(y)
    return np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), y, valid, mono


def fit_m2(Dt, yt, mt, mono, inner_split, seed: int, fixed_iter: Optional[int] = None):
    """HistGB regressor with purged early stop by predictor truncation.

    inner_split: (train_day_rows, val_day_rows) boolean masks over rows —
    chronological tail of the TRAINING window, embargoed 27 td, never the OOF
    fold."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    clf = HistGradientBoostingRegressor(
        monotonic_cst=list(mono), early_stopping=False, random_state=seed,
        **M2_PARAMS)
    if fixed_iter is not None:
        params = dict(M2_PARAMS)
        params["max_iter"] = int(fixed_iter)
        clf = HistGradientBoostingRegressor(
            monotonic_cst=list(mono), early_stopping=False, random_state=seed,
            **params)
        fit_rows = mt
        clf.fit(Dt[fit_rows], yt[fit_rows])
        return clf, int(fixed_iter)
    inner_tr, inner_va = inner_split
    clf.fit(Dt[mt & inner_tr], yt[mt & inner_tr])
    best_iter = M2_PARAMS["max_iter"]
    Xv, yv = Dt[mt & inner_va], yt[mt & inner_va]
    if len(yv) > 200:
        try:
            preds = clf._predictors
            base = clf._baseline_prediction
            Xb = clf._bin_mapper.transform(np.ascontiguousarray(Xv))
            raw_iter = np.full(len(Xv), float(np.ravel(base)[0]))
            losses = []
            stage = {}
            for i, pred_it in enumerate(preds, start=1):
                raw_iter = raw_iter + pred_it[0].predict_binned(
                    Xb, clf._bin_mapper.missing_values_bin_idx_, n_threads=1)
                if i in M2_ES_GRID:
                    stage[i] = raw_iter.copy()
            for i in M2_ES_GRID:
                losses.append((float(np.mean((stage[i] - yv) ** 2)), i))
            best_iter = min(losses)[1]
        except Exception:
            best_iter = M2_PARAMS["max_iter"]
    # refit on the FULL training window at the chosen iteration count
    params = dict(M2_PARAMS)
    params["max_iter"] = int(best_iter)
    from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
    clf2 = HGBR(monotonic_cst=list(mono), early_stopping=False,
                random_state=seed, **params)
    clf2.fit(Dt[mt], yt[mt])
    return clf2, int(best_iter)


def m2_outputs(clf, D, valid):
    """raw preds -> per-day cross-sectional z (mu); c = per-day raw pred sd."""
    n, S, _ = D.shape
    P = np.full((n, S), np.nan)
    flat = D.reshape(n * S, -1)
    vm = valid.reshape(-1)
    if vm.any():
        P.reshape(-1)[vm] = clf.predict(flat[vm])
    mu = np.full_like(P, 0.0)
    c = np.zeros(n)
    for i in range(n):
        m = np.isfinite(P[i])
        if m.sum() >= 2:
            sd = P[i][m].std()
            mu[i][m] = (P[i][m] - P[i][m].mean()) / (sd if sd > 1e-9 else 1.0)
            c[i] = sd
    return P, mu, c


def rolling_bucket_residual_sd(mu, y_real, bucket_ids, h_lag: int,
                               window: int = 63, floor: float = 0.25):
    n, S = mu.shape
    resid = y_real - mu
    sig = np.full((n, S), 1.0)
    K = int(bucket_ids.max()) + 1
    for k in range(K):
        js = np.nonzero(bucket_ids == k)[0]
        if not len(js):
            continue
        r = pd.Series(np.nanmean(resid[:, js], axis=1))
        s = r.shift(h_lag).rolling(window, min_periods=21).std()
        sig[:, js] = np.maximum(np.nan_to_num(s.to_numpy(), nan=1.0),
                                floor)[:, None]
    return sig


# =========================== M3 DISP-HAR ======================================

M3_TCOLS = ["s1_cor3m_z", "ctx_vvix_z", "ctx_skew_z", "ctx_vix_term_slope",
            "s1_vxn_minus_vix", "s1_curvature", "br_dispersion"]


def m3_design(panel6, p7):
    """[N,13] full design + [N,3] lags-only; y = y_disp (log)."""
    T = panel6["T"]
    t_cols = [str(c) for c in panel6["T_cols"]]
    lags = p7["disp_lags"].astype(np.float64)                  # [N,3] log-space
    vols = np.stack([T[:, 0, t_cols.index(c)].astype(np.float64)
                     for c in M3_TCOLS], axis=1)               # broadcast cols
    cal = p7["cal_dummies"].astype(np.float64)
    X_full = np.concatenate([lags, vols, cal], axis=1)
    y = p7["y_disp"].astype(np.float64)
    ok = np.all(np.isfinite(X_full), axis=1) & np.isfinite(y)
    return np.nan_to_num(X_full, nan=0.0), lags, y, ok


def ols_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def ols_predict(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


# =========================== M4 EVT-NET =======================================

M4_GDELT_BCOLS = ["g1_z", "g3_tone", "g3_tone_dispersion", "g2_goldstein",
                  "g2_conflict", "g4_burst_z", "g4_novelty", "g5_hhi_loc",
                  "g5_hhi_org"]
M4_LLM_BCOLS = ["llm_sent_x", "llm_conf_x", "llm_sal_x", "llm_sent_d1_x"] + \
               [f"llm_event_{i:02d}_x" for i in range(12)]
M4_GRID = [(l1, C) for l1 in (0.5, 0.8, 0.95) for C in (0.03, 0.1, 0.3, 1.0)]
M4_LLM_MASK_START = "2024-08-15"


def m4_design(panel6, p7, dates):
    """Returns dict of pooled designs: A [N,K,12], B [N,K,29], CTL [N,K,2],
    y [N,K]."""
    B = panel6["B"].astype(np.float64)             # [N,27,26]
    b_cols = [str(c) for c in panel6["B_cols"]]
    gd_avail = panel6["gdelt_available"].astype(np.float64)    # [N]
    N, K, _ = B.shape
    g = np.stack([B[:, :, b_cols.index(c)] for c in M4_GDELT_BCOLS], axis=2)
    g = g * gd_avail[:, None, None]                # mask-interaction form
    avail = np.broadcast_to(gd_avail[:, None, None], (N, K, 1))
    ctl = np.nan_to_num(p7["evt_ctl"].astype(np.float64), nan=0.0)
    XA = np.concatenate([g, avail, ctl], axis=2)   # 9 + 1 + 2 = 12
    llm = np.stack([B[:, :, b_cols.index(c)] for c in M4_LLM_BCOLS], axis=2)
    llm_avail = B[:, :, b_cols.index("llm_available")][:, :, None]
    post = (np.asarray(dates).astype(str) >= M4_LLM_MASK_START)
    llm = llm * post[:, None, None]                # masked pre-2024-08-15
    XB = np.concatenate([XA, llm, llm_avail], axis=2)          # 12 + 16 + 1
    y = p7["y_evt"].astype(np.float64)
    return {"A": np.nan_to_num(XA, nan=0.0), "B": np.nan_to_num(XB, nan=0.0),
            "CTL": ctl, "y": y}


def fit_m4(X, y, train_idx, seed: int, inner_frac: float = 0.8,
           gap: int = 6, fixed: Optional[tuple] = None):
    """Pooled (day,bucket) logistic elastic net; hyperparams by inner
    chronological CV (within the training window only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def pool(idxs):
        Xi = X[idxs].reshape(-1, X.shape[2])
        yi = y[idxs].reshape(-1)
        ok = np.isfinite(yi)
        return Xi[ok], yi[ok]

    cut = int(len(train_idx) * inner_frac)
    tr, va = train_idx[:max(1, cut - gap)], train_idx[cut:]
    Xt, yt = pool(tr)
    Xv, yv = pool(va)
    mu_, sd_ = Xt.mean(0), Xt.std(0)
    sd_ = np.where(sd_ < 1e-12, 1.0, sd_)
    if fixed is None:
        best = (None, -np.inf)
        for l1, C in M4_GRID:
            lr = LogisticRegression(penalty="elasticnet", solver="saga",
                                    l1_ratio=l1, C=C, max_iter=2000,
                                    random_state=seed)
            lr.fit((Xt - mu_) / sd_, yt)
            if len(np.unique(yv)) == 2:
                s = float(roc_auc_score(yv, lr.predict_proba(
                    (Xv - mu_) / sd_)[:, 1]))
            else:
                s = 0.0
            if s > best[1]:
                best = ((l1, C), s)
        l1, C = best[0] if best[0] else (0.8, 0.1)
    else:
        l1, C = fixed
    Xa, ya = pool(train_idx)
    mu_, sd_ = Xa.mean(0), Xa.std(0)
    sd_ = np.where(sd_ < 1e-12, 1.0, sd_)
    lr = LogisticRegression(penalty="elasticnet", solver="saga",
                            l1_ratio=l1, C=C, max_iter=3000, random_state=seed)
    lr.fit((Xa - mu_) / sd_, ya)
    return {"model": lr, "mu": mu_, "sd": sd_, "l1_ratio": l1, "C": C}


def m4_predict(fitted, X, idxs):
    Xi = X[idxs].reshape(-1, X.shape[2])
    p = fitted["model"].predict_proba(
        (Xi - fitted["mu"]) / fitted["sd"])[:, 1]
    return p.reshape(len(idxs), X.shape[1])


def m4_mu_sym(p, bucket_sym):
    """negated exceedance prob -> symbol space, per-day standardized."""
    n, K = p.shape
    S = bucket_sym.shape[1]
    mu = np.zeros((n, S))
    cnt = np.zeros(S)
    for k in range(K):
        mu[:, bucket_sym[k]] += -p[:, k][:, None]
        cnt[bucket_sym[k]] += 1
    cnt[cnt == 0] = 1
    mu /= cnt[None, :]
    out = np.zeros_like(mu)
    for i in range(n):
        sd = mu[i].std()
        out[i] = (mu[i] - mu[i].mean()) / (sd if sd > 1e-9 else 1.0)
    return out


# =========================== M5 POS-Z =========================================

M5_Z_WINDOW = 156          # weekly obs
M5_Z_MINP = 78
M5_THRESH = 2.0
M5_DECAY_TD = 21
M5_SLEEVES = {
    "es": (["ITA", "SOXX", "XRT", "RSP", "KRE", "IYR", "FXI"], 1.0),
    "ust10y": (["TLT", "AGG", "MUB", "SHY"], 1.0),
    "vix": (["ITA", "SOXX", "XRT", "RSP", "KRE", "IYR", "FXI"], 0.5),
}


def m5_daily_z(dates: np.ndarray) -> dict:
    """156-week trailing z of lev_net_pct_oi per complex, publication-keyed,
    mapped to panel dates (visible_from <= D). The panel's cot_*_z (52w) is
    NOT used — spec says 156w."""
    dts = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates).astype(str)))
    out = {}
    for slug in ["es", "ust10y", "vix"]:
        df = pd.read_parquet(TB6 / "cache" / "cot" / f"{slug}.parquet")
        df = df.sort_values("report_date")
        s = df["lev_net_pct_oi"].astype(float)
        m = s.shift(1).rolling(M5_Z_WINDOW, min_periods=M5_Z_MINP).mean()
        sd = s.shift(1).rolling(M5_Z_WINDOW, min_periods=M5_Z_MINP).std()
        z = (s - m) / sd.replace(0.0, np.nan)
        vf = pd.to_datetime(df["visible_from"])
        j = pd.merge_asof(pd.DataFrame({"D": dts}),
                          pd.DataFrame({"vf": vf, "z": z.to_numpy(),
                                        "pub": pd.to_datetime(df["publication"])})
                          .sort_values("vf"),
                          left_on="D", right_on="vf", direction="backward")
        age = (j["D"] - j["pub"].dt.normalize()).dt.days
        fresh = (age <= 10).to_numpy()             # COT_FRESH_DAYS=10 (TB-006)
        zd = j["z"].to_numpy()
        out[slug] = {"z": zd, "fresh": fresh}
    return out


def m5_signal(dates, symbols, zmap):
    """mu [N,S] + episode list. Episode: starts when |z|>2 with fresh data;
    active while |z|>2; on exit decays linearly over 21 td (sign/magnitude
    frozen at the last active value). mu = -sign(z)*min(|z|-2,1)*decay*sleeve_w."""
    N = len(dates)
    S = len(symbols)
    sym_ix = {s: j for j, s in enumerate(symbols)}
    mu = np.zeros((N, S))
    episodes = []
    mag_series = np.zeros(N)
    for slug, (sleeve, w) in M5_SLEEVES.items():
        z = zmap[slug]["z"]
        fresh = zmap[slug]["fresh"]
        state = None                                # dict while episode live
        for k in range(N):
            zk = z[k]
            active = np.isfinite(zk) and abs(zk) > M5_THRESH and fresh[k]
            if active:
                if state is None:
                    state = {"start": k, "sign": np.sign(zk), "mag": 0.0,
                             "decay_left": None}
                if np.sign(zk) != state["sign"]:    # sign flip = new episode
                    state["end"] = k - 1
                    episodes.append({**state, "complex": slug})
                    state = {"start": k, "sign": np.sign(zk), "mag": 0.0,
                             "decay_left": None}
                state["mag"] = min(abs(zk) - M5_THRESH, 1.0)
                state["decay_left"] = None
                d = 1.0
            elif state is not None:
                if state["decay_left"] is None:
                    state["decay_left"] = M5_DECAY_TD
                state["decay_left"] -= 1
                if state["decay_left"] <= 0:
                    state["end"] = k
                    episodes.append({**state, "complex": slug})
                    state = None
                    continue
                d = state["decay_left"] / M5_DECAY_TD
            else:
                continue
            val = -state["sign"] * state["mag"] * d * w
            for s in sleeve:
                mu[k, sym_ix[s]] += val
            mag_series[k] = max(mag_series[k], state["mag"] * d)
        if state is not None:
            state["end"] = N - 1
            episodes.append({**state, "complex": slug})
    eps = [{"complex": e["complex"], "start": int(e["start"]),
            "end": int(e["end"]), "sign": float(e["sign"]),
            "mag": float(e["mag"])} for e in episodes]
    return mu, eps, mag_series


# =========================== M6 GAP-GRU =======================================

M6_SEQ = 10


def build_m6(seed: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class GapGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(8, 24, batch_first=True)
            self.head = nn.Linear(24, 1)

        def forward(self, x):                       # x [B,10,8]
            h = self.gru(x)[1][0]
            return self.head(h).squeeze(-1)

    return GapGRU()


def m6_sequences(m6_feat, idxs):
    """[n,S,10,8] sequences ending at row k-1 for each decision row k."""
    n = len(idxs)
    S = m6_feat.shape[1]
    out = np.zeros((n, S, M6_SEQ, 8), dtype=np.float32)
    for j, k in enumerate(idxs):
        lo = k - M6_SEQ
        if lo < 0:
            continue
        out[j] = m6_feat[lo:k].transpose(1, 0, 2)
    return out


def train_m6(panel6, m6_feat, train_idx, val_idx, seed=4242, max_epochs=40,
             patience=5, lr=1e-3, days_per_step=16):
    import torch
    torch.manual_seed(seed)
    model = build_m6(seed)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y1 = panel6["y1_z"].astype(np.float32)
    smask = panel6["symbol_mask"]
    train_idx = train_idx[train_idx >= M6_SEQ]

    def eval_val():
        model.eval()
        mus = np.zeros((len(val_idx), m6_feat.shape[1]), dtype=np.float32)
        with torch.no_grad():
            for c0 in range(0, len(val_idx), 64):
                ii = val_idx[c0:c0 + 64]
                xb = torch.from_numpy(m6_sequences(m6_feat, ii)
                                      .reshape(-1, M6_SEQ, 8))
                mus[c0:c0 + len(ii)] = model(xb).numpy() \
                    .reshape(len(ii), -1)
        model.train()
        m = smask[val_idx] & np.isfinite(y1[val_idx])
        if m.sum() == 0:
            return np.nan, mus
        mse = float(np.mean((mus[m] - y1[val_idx][m]) ** 2))
        return mse, mus

    rng = np.random.RandomState(seed)
    best = (np.inf, None, 0, None)
    bad = 0
    for epoch in range(max_epochs):
        order = train_idx.copy()
        rng.shuffle(order)
        for c0 in range(0, len(order), days_per_step):
            ii = order[c0:c0 + days_per_step]
            xb = torch.from_numpy(m6_sequences(m6_feat, ii)
                                  .reshape(-1, M6_SEQ, 8))
            yb = torch.from_numpy(y1[ii].reshape(-1))
            mb = torch.from_numpy(smask[ii].reshape(-1)) & torch.isfinite(yb)
            if int(mb.sum()) == 0:
                continue
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred[mb], yb[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if len(val_idx):
            mse, vmu = eval_val()
            if mse < best[0]:
                best = (mse, {k: v.clone() for k, v in
                              model.state_dict().items()}, epoch, vmu)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            best = (np.nan, {k: v.clone() for k, v in
                             model.state_dict().items()}, epoch, None)
    return best[1], best[0], best[2], best[3], n_par
