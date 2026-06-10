"""PKT-TB-006 — ensemble members (BUILD_SPEC §5, exact architectures).

E1 CAST-Small   ≈16k transformer (GRU 14→32, token Linear 48→32 with
                sector_emb 8 / class_emb 4, 1 pre-LN layer d32/4h/FFN64/do0.20,
                mu/sigma/aux heads; NO symbol identity).
Ridge twin      cross-sectional ridge on flattened X(882)+4 scalars, λ fold-CV.
E2 GBM-Cond     HistGradientBoostingClassifier 150 iters / 15 leaves, monotone
                constraints via signed interaction columns, isotonic on OOF.
E3 EventHead    per-bucket elastic net over B with Transfer-A family routing,
                event_weight_cap, abstain-on-quiet.
E4 RiskNet+     three ridge heads (QLIKE validation for vol heads).

All training honors folds.py geometry; every artifact carries a
walk_forward: true manifest (the executive refuses to train without it).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent

# ---- sector/class vocabularies for CAST embeddings (forced choice, logged):
# spec registers "13 sectors from universe.csv" / "6 classes" but universe.csv
# carries 51 sector strings and 5 asset classes; the 13-group / 6-class
# mappings below are authored once here and FROZEN.
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


def sector_class_ids(universe_csv: Path) -> tuple[np.ndarray, np.ndarray]:
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


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ["folds.py", "infotropy.py", "members.py", "train_members.py"]:
        p = PROTO / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


# ===========================================================================
# E1 — CAST-Small
# ===========================================================================

def build_cast(seed: int):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class CASTSmall(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(14, 32, batch_first=True)
            self.sector_emb = nn.Embedding(13, 8)
            self.class_emb = nn.Embedding(6, 4)
            self.token = nn.Linear(48, 32)
            self.enc = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, dropout=0.20,
                batch_first=True, norm_first=True)
            self.mu_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            self.sigma_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            self.aux_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x, scalars, sec_ids, cls_ids, mask):
            # x [B,S,63,14], scalars [B,S,4], mask [B,S] bool (True = valid)
            Bd, S = x.shape[0], x.shape[1]
            h = self.gru(x.reshape(Bd * S, 63, 14))[1][0].reshape(Bd, S, 32)
            emb = torch.cat([self.sector_emb(sec_ids), self.class_emb(cls_ids)], -1)
            tok = self.token(torch.cat([h, emb.unsqueeze(0).expand(Bd, -1, -1),
                                        scalars], -1))
            out = self.enc(tok, src_key_padding_mask=~mask)
            mu = self.mu_head(out).squeeze(-1)
            sigma = torch.nn.functional.softplus(self.sigma_head(out).squeeze(-1)) + 1e-2
            aux = self.aux_head(out).squeeze(-1)
            return mu, sigma, aux

    return CASTSmall()


def cast_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def soft_spearman(mu, y_rank, mask, tau: float = 0.1):
    """Differentiable per-day Spearman surrogate via pairwise-sigmoid soft ranks."""
    import torch
    cors = []
    for b in range(mu.shape[0]):
        m = mask[b]
        if int(m.sum()) < 8:
            continue
        x = mu[b][m]
        y = y_rank[b][m]
        # ascending soft rank: sr[a] = Σ_b sigmoid((x[a] − x[b]) / tau)
        sr = torch.sigmoid((x.unsqueeze(1) - x.unsqueeze(0)) / tau).sum(dim=1)
        yr = torch.argsort(torch.argsort(y)).float()
        sx = sr - sr.mean()
        sy = yr - yr.mean()
        c = (sx * sy).sum() / (sx.norm() * sy.norm() + 1e-8)
        cors.append(c)
    if not cors:
        import torch as _t
        return _t.tensor(0.0)
    return torch.stack(cors).mean()


def cast_loss(mu, sigma, aux, y5_rank, y1_z, w_rec, mask):
    """Huber(mu,y)·w_rec + 0.25·(1−softSpearman) + GaussNLL(sigma) + 0.3·MSE(aux).
    GaussNLL trains sigma against a detached mu (spec names sigma only;
    forced choice logged)."""
    import torch
    m = mask & torch.isfinite(y5_rank)
    if int(m.sum()) == 0:
        return None
    hub = torch.nn.functional.smooth_l1_loss(mu[m], y5_rank[m], reduction="none")
    w = torch.nan_to_num(w_rec[m], nan=0.0)
    l_hub = (hub * w).mean()
    l_sp = 1.0 - soft_spearman(mu, y5_rank, m)
    var = sigma[m] ** 2
    l_nll = (0.5 * (torch.log(var) + (y5_rank[m] - mu[m].detach()) ** 2 / var)).mean()
    ma = mask & torch.isfinite(y1_z)
    l_aux = ((aux[ma] - y1_z[ma]) ** 2).mean() if int(ma.sum()) else mu.sum() * 0
    return l_hub + 0.25 * l_sp + l_nll + 0.3 * l_aux


def train_cast_one(panel, train_idx, val_idx, seed: int, sec_ids, cls_ids,
                   uniform_w: bool = False, max_epochs: int = 60,
                   patience: int = 10, lr: float = 1e-3, days_per_step: int = 8,
                   verbose: bool = False):
    """One CAST run. Returns (state_dict, best_val_spearman, best_epoch,
    val_mu [n_val,64], val_sigma, history)."""
    import torch
    torch.manual_seed(seed)
    model = build_cast(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    X = panel["X"]; SC = panel["scalars"]
    y5 = panel["y5_rank"].astype(np.float32)
    y1 = panel["y1_z"].astype(np.float32)
    wr = (np.ones_like(panel["w_rec"], dtype=np.float32) if uniform_w
          else panel["w_rec"].astype(np.float32))
    smask = panel["symbol_mask"]
    sec_t = torch.from_numpy(sec_ids); cls_t = torch.from_numpy(cls_ids)

    def day_batch(idxs):
        return (torch.from_numpy(X[idxs]), torch.from_numpy(SC[idxs]),
                torch.from_numpy(y5[idxs]), torch.from_numpy(y1[idxs]),
                torch.from_numpy(wr[idxs]), torch.from_numpy(smask[idxs]))

    @torch.no_grad()
    def val_eval():
        from scipy.stats import spearmanr
        model.eval()
        mus = np.zeros((len(val_idx), X.shape[1]), dtype=np.float32)
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
        if verbose:
            print(f"  seed {seed} epoch {epoch} val_spear {vs:.4f}")
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


# ===========================================================================
# Ridge twin (qua-transformer referee)
# ===========================================================================

RIDGE_LAMBDAS = [1.0, 10.0, 100.0, 1e3, 1e4]


def _ridge_design(panel, idxs):
    """Pooled (day,symbol) rows: flattened X (882) + 4 scalars; y = y5_rank."""
    X = panel["X"][idxs]                      # [n,S,63,14]
    SC = panel["scalars"][idxs]
    n, S = X.shape[0], X.shape[1]
    A = np.concatenate([X.reshape(n, S, -1), SC], axis=2)   # [n,S,886]
    y = panel["y5_rank"][idxs]
    m = panel["symbol_mask"][idxs] & np.isfinite(y)
    return A, y.astype(np.float64), m


def _gram_accumulate(A, y, m, chunk: int = 64):
    d = A.shape[2]
    G = np.zeros((d, d))
    b = np.zeros(d)
    n = 0
    for c0 in range(0, A.shape[0], chunk):
        Aa = A[c0:c0 + chunk].astype(np.float64)
        yy = y[c0:c0 + chunk]
        mm = m[c0:c0 + chunk]
        rows = Aa[mm]
        G += rows.T @ rows
        b += rows.T @ yy[mm]
        n += int(mm.sum())
    return G, b, n


def train_ridge_twin(panel, train_idx, val_idx, internal_frac: float = 0.8):
    """Fold-internal λ CV (chronological 80/20 inside the training window),
    refit on the full window, predict the fold. Returns (val_mu, lam, coef)."""
    cut = int(len(train_idx) * internal_frac)
    A_tr, y_tr, m_tr = _ridge_design(panel, train_idx[:cut])
    G, b, _ = _gram_accumulate(A_tr, y_tr, m_tr)
    A_iv, y_iv, m_iv = _ridge_design(panel, train_idx[cut:])
    d = G.shape[0]
    best = (None, -np.inf)
    from scipy.stats import spearmanr
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
    A2, y2, m2 = _ridge_design(panel, train_idx[cut:])
    G2, b2, _ = _gram_accumulate(A2, y2, m2)
    coef = np.linalg.solve(G + G2 + lam * np.eye(d), b + b2)
    if len(val_idx):
        A_v, _, _ = _ridge_design(panel, val_idx)
        val_mu = np.einsum("nsd,d->ns", A_v.astype(np.float64), coef)
    else:
        val_mu = None
    return val_mu, lam, coef


# ===========================================================================
# E2 — GBM-Cond
# ===========================================================================

GBM_PARAMS = dict(max_iter=150, max_leaf_nodes=15, max_depth=6,
                  learning_rate=0.05, l2_regularization=1.0)
# monotone constraints via signed interaction columns (forced choice, logged):
# a single global model cannot carry bucket-conditional constraints, so two
# derived columns are appended and constrained −1: hy_oas_d5 × 1{credit bucket}
# and cor3m_z × 1{equity class}; the base columns stay unconstrained.


def gbm_design(panel, idxs, sym_is_credit, sym_is_equity, screen=None,
               fold: Optional[int] = None, variant: str = "conjunctive"):
    T = panel["T"][idxs]                       # [n,S,56]
    t_cols = [str(c) for c in panel["T_cols"]]
    n, S, _ = T.shape
    hy = T[:, :, t_cols.index("fred_hy_oas_d5")]
    cz = T[:, :, t_cols.index("s1_cor3m_z")]
    extra = np.stack([hy * sym_is_credit[None, :], cz * sym_is_equity[None, :]],
                     axis=2)
    cols = list(range(len(t_cols)))
    if screen is not None and fold is not None:
        # Transfer-A family routing for the GDELT-fed T columns
        import infotropy as IT
        fam_cols = {"G1_themes": ["gd_g1_share", "gd_g1_z"],
                    "G3_tone": ["gd_g3_tone"],
                    "G4_novelty_burst": ["gd_g4_burst_z", "gd_novelty",
                                         "gd_novelty_21"],
                    "G5_concentration": ["gd_hhi_loc", "gd_hhi_org"]}
        drop = []
        for fam, cs in fam_cols.items():
            if not IT.screen_pass(screen, fam, fold, variant):
                drop += [t_cols.index(c) for c in cs]
        cols = [c for c in cols if c not in drop]
    D = np.concatenate([T[:, :, cols], extra], axis=2)
    mono = np.zeros(D.shape[2], dtype=np.int8)
    mono[-2] = -1
    mono[-1] = -1
    y = (panel["y5_raw"][idxs] > 0).astype(np.float64)
    valid = panel["symbol_mask"][idxs] & np.isfinite(panel["y5_raw"][idxs])
    w = np.nan_to_num(panel["w_rec"][idxs], nan=0.0)
    return D, y, valid, w, mono


def fit_gbm(D, y, valid, w, mono, val_tuple=None, seed: int = 0,
            uniform_w: bool = False):
    """Fit HGBC; manual purged early stopping by predictor truncation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xf = D[valid]
    yf = y[valid]
    wf = np.ones_like(yf) if uniform_w else w[valid]
    clf = HistGradientBoostingClassifier(
        monotonic_cst=list(mono), early_stopping=False, random_state=seed,
        **GBM_PARAMS)
    clf.fit(Xf, yf, sample_weight=wf)
    best_iter = GBM_PARAMS["max_iter"]
    if val_tuple is not None:
        Dv, yv, mv = val_tuple
        Xv, yvv = Dv[mv], yv[mv]
        from sklearn.metrics import log_loss
        # staged purged-validation log-loss via predictor truncation
        try:
            losses = []
            grid = list(range(10, GBM_PARAMS["max_iter"] + 1, 10))
            preds = clf._predictors
            base = clf._baseline_prediction
            from scipy.special import expit
            Xb = clf._bin_mapper.transform(np.ascontiguousarray(Xv))
            raw_iter = np.full(len(Xv), float(np.ravel(base)[0]))
            stage_raw = {}
            for i, pred_it in enumerate(preds, start=1):
                raw_iter = raw_iter + pred_it[0].predict_binned(
                    Xb, clf._bin_mapper.missing_values_bin_idx_,
                    n_threads=1)
                if i in grid:
                    stage_raw[i] = raw_iter.copy()
            for i in grid:
                p = np.clip(expit(stage_raw[i]), 1e-6, 1 - 1e-6)
                losses.append((log_loss(yvv, p), i))
            best_iter = min(losses)[1]
            clf._predictors = clf._predictors[:best_iter]
        except Exception:
            best_iter = GBM_PARAMS["max_iter"]      # fallback: no truncation
    return clf, int(best_iter)


def gbm_outputs(clf, D, valid, panel, idxs, iso=None):
    """Raw P -> probit z -> per-day cross-sectional z = mu2 (monotone-invariant,
    so calibration cannot collapse the rank signal); c2 = CALIBRATED mean
    |P-0.5| (isotonic fit on prior-fold OOF). Forced choice logged: applying
    isotonic BEFORE the probit flattened mu2 to a constant on weak folds."""
    from scipy.stats import norm
    n, S, _ = D.shape
    P = np.full((n, S), np.nan)
    flat = D.reshape(n * S, -1)
    vm = valid.reshape(-1)
    if vm.any():
        P.reshape(-1)[vm] = clf.predict_proba(flat[vm])[:, 1]
    P_cal = P.copy()
    if iso is not None:
        m = np.isfinite(P)
        P_cal[m] = iso.predict(P[m])
    Pc = np.clip(P, 1e-4, 1 - 1e-4)
    pz = norm.ppf(Pc)
    mu2 = np.full_like(pz, np.nan)
    for i in range(n):
        m = np.isfinite(pz[i])
        if m.sum() >= 2:
            sd = pz[i][m].std()
            mu2[i][m] = (pz[i][m] - pz[i][m].mean()) / (sd if sd > 1e-9 else 1.0)
    c2 = np.array([np.nanmean(np.abs(P_cal[i] - 0.5)) if np.isfinite(P_cal[i]).any()
                   else np.nan for i in range(n)])
    return P, mu2, c2


def rolling_bucket_residual_sd(mu, y_real, bucket_ids, window: int = 63,
                               floor: float = 0.25) -> np.ndarray:
    """sigma per (day,symbol): trailing per-bucket sd of residual (y − mu),
    past-only (shift 1, h-lag is inside the trailing window by construction)."""
    n, S = mu.shape
    resid = y_real - mu
    sig = np.full((n, S), 1.0)
    K = int(bucket_ids.max()) + 1
    for k in range(K):
        js = np.nonzero(bucket_ids == k)[0]
        if not len(js):
            continue
        r = pd.Series(np.nanmean(resid[:, js], axis=1))
        s = r.shift(H_LAG).rolling(window, min_periods=21).std()
        sig[:, js] = np.maximum(np.nan_to_num(s.to_numpy(), nan=1.0), floor)[:, None]
    return sig


H_LAG = 6  # residual at t needs y realized through t+5; trailing stats lag h+1


# ===========================================================================
# E3 — EventHead
# ===========================================================================

EVENT_ALPHAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
EVENT_L1_RATIO = 0.5


def train_event_head(panel, train_idx, val_idx, screen, fold,
                     variant: str = "conjunctive", event_weight_cap: float = 1.0,
                     abstain_threshold: float = 0.10, rt_weights=None):
    """Per-bucket elastic nets over routed B columns. Returns dict with
    val-day mu3[n_val,64], sigma3, c3 and the per-bucket models."""
    from sklearn.linear_model import ElasticNet
    from scipy.stats import spearmanr
    import infotropy as IT

    b_cols = [str(c) for c in panel["B_cols"]]
    buckets = [str(b) for b in panel["buckets"]]
    symbols = [str(s) for s in panel["symbols"]]
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    B = panel["B"]; y5b = panel["y5_bucket"]

    # Transfer-A routing: failing families' columns leave the design matrix
    keep_cols = []
    for c in b_cols:
        fam = next((f for f, cs in IT.FAMILIES.items() if c in cs), None)
        if fam is None:
            keep_cols.append(c)                      # masks / llm sent block
        elif IT.screen_pass(screen, fam, fold, variant):
            keep_cols.append(c)
    kidx = [b_cols.index(c) for c in keep_cols]

    # event mass (gated): mean family intensity over PASSING GDELT families
    passing = [f for f in IT.FAMILIES
               if f != "LLM_event_flags" and IT.screen_pass(screen, f, fold, variant)]
    if passing:
        mass = np.mean([IT.family_intensity(B, b_cols, f).mean(axis=1)
                        for f in passing], axis=0)
    else:
        mass = np.zeros(B.shape[0])

    cut = int(len(train_idx) * 0.8)
    models = {}
    val_mu_b = np.zeros((len(val_idx), len(buckets)))
    for k in range(len(buckets)):
        Xk = B[:, k, kidx].astype(np.float64)
        yk = y5b[:, k]
        sw = None if rt_weights is None else rt_weights[:, k]
        tr, iv = train_idx[:cut], train_idx[cut:]
        ok_tr = np.isfinite(yk[tr]) & np.all(np.isfinite(Xk[tr]), axis=1)
        ok_iv = np.isfinite(yk[iv]) & np.all(np.isfinite(Xk[iv]), axis=1)
        if ok_tr.sum() < 100 or np.allclose(Xk[tr][ok_tr].std(0), 0):
            models[k] = None
            continue
        mu_, sd_ = Xk[tr][ok_tr].mean(0), Xk[tr][ok_tr].std(0)
        sd_[sd_ < 1e-12] = 1.0
        best = (None, -np.inf)
        for a in EVENT_ALPHAS:
            en = ElasticNet(alpha=a, l1_ratio=EVENT_L1_RATIO, max_iter=2000)
            en.fit((Xk[tr][ok_tr] - mu_) / sd_, yk[tr][ok_tr],
                   sample_weight=None if sw is None else sw[tr][ok_tr])
            if ok_iv.sum() >= 20:
                p = en.predict((Xk[iv][ok_iv] - mu_) / sd_)
                s = spearmanr(p, yk[iv][ok_iv]).statistic
                s = -np.inf if not np.isfinite(s) else s
            else:
                s = 0.0
            if s > best[1]:
                best = (a, s)
        a = best[0] if best[0] is not None else EVENT_ALPHAS[2]
        ok_all = np.isfinite(yk[train_idx]) & np.all(np.isfinite(Xk[train_idx]), axis=1)
        en = ElasticNet(alpha=a, l1_ratio=EVENT_L1_RATIO, max_iter=2000)
        en.fit((Xk[train_idx][ok_all] - mu_) / sd_, yk[train_idx][ok_all],
               sample_weight=None if sw is None else sw[train_idx][ok_all])
        models[k] = (en, mu_, sd_, a)
        if len(val_idx):
            val_mu_b[:, k] = en.predict((Xk[val_idx] - mu_) / sd_)

    # c3 = gated event mass as trailing-252d percentile rank (past-only); abstain
    mz = pd.Series(mass)
    c3_full = mz.rolling(253, min_periods=64).apply(
        lambda w: float((w[-1] > w[:-1]).mean()), raw=True).to_numpy()
    c3 = np.nan_to_num(c3_full[val_idx], nan=0.0) if len(val_idx) else np.array([])

    # map buckets -> symbols
    S = len(symbols)
    sym_of_bucket = np.full((len(buckets), S), False)
    for k, b in enumerate(buckets):
        for s in bucket_map[b]:
            if s in symbols:
                sym_of_bucket[k, symbols.index(s)] = True
    mu3 = np.zeros((len(val_idx), S))
    cnt = np.zeros(S)
    for k in range(len(buckets)):
        mu3[:, sym_of_bucket[k]] += val_mu_b[:, k][:, None] * event_weight_cap
        cnt[sym_of_bucket[k]] += 1
    cnt[cnt == 0] = 1
    mu3 /= cnt[None, :]
    quiet = c3 < abstain_threshold
    mu3[quiet] = 0.0                                   # abstain-on-quiet

    # sigma3: rolling bucket residual sd (training residuals, past-only proxy)
    sigma3 = np.full((len(val_idx), S), 1.0)
    resid_sd = np.nanstd(y5b[train_idx], axis=0)
    for k in range(len(buckets)):
        v = resid_sd[k] if np.isfinite(resid_sd[k]) else 1.0
        sigma3[:, sym_of_bucket[k]] = max(v / max(np.nanmean(resid_sd), 1e-9), 0.25)
    return {"mu3": mu3, "sigma3": sigma3, "c3": c3, "models": models,
            "keep_cols": keep_cols, "val_mu_bucket": val_mu_b,
            "n_pass_families": len(passing)}


# ===========================================================================
# E4 — RiskNet+
# ===========================================================================

RISK_FEATURES = ["vol_21d", "vol_63d", "vol_5d", "range_pct", "vix_level",
                 "s1_slope", "s1_curvature", "cor3m_z", "vvix_z",
                 "g4_novelty", "g5_hhi", "llm_geopol_risk"]
RISK_LAMBDAS = [0.1, 1.0, 10.0, 100.0]


def risk_design(panel, proto: Path = PROTO) -> np.ndarray:
    """[N,S,12] RiskNet+ features (§5.4 list). vol_5d / range_pct recomputed
    from cache OHLCV (store keeps X z-scored only); VIX level from cache/cboe;
    all other columns read from T/Z. Inputs ≤ D−1 close throughout."""
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(panel["dates"])))
    symbols = [str(s) for s in panel["symbols"]]
    t_cols = [str(c) for c in panel["T_cols"]]
    z_cols = [str(c) for c in panel["Z_cols"]]
    T, Z = panel["T"], panel["Z"]
    N, S = T.shape[0], T.shape[1]
    F = np.zeros((N, S, len(RISK_FEATURES)), dtype=np.float64)

    def tcol(name):
        return T[:, :, t_cols.index(name)].astype(np.float64)

    F[:, :, 0] = tcol("vol_21d")
    F[:, :, 1] = tcol("vol_63d")
    # vol_5d + range_pct from cache OHLCV at D-1
    spy = pd.read_parquet(proto / "cache" / "ohlcv" / "SPY.parquet")
    cal = pd.DatetimeIndex(pd.to_datetime(spy["date"]).dt.normalize()).unique().sort_values()
    pos = cal.searchsorted(dates)                  # date positions in calendar
    for j, sym in enumerate(symbols):
        df = pd.read_parquet(proto / "cache" / "ohlcv" / f"{sym}.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").reindex(cal)
        r1 = df["adj_close"].pct_change()
        v5 = (r1.rolling(5).std() * np.sqrt(252)).to_numpy()
        rp = ((df["high"] - df["low"]) / df["close"]).to_numpy()
        F[:, j, 2] = np.nan_to_num(v5[pos - 1], nan=0.0)
        F[:, j, 3] = np.nan_to_num(rp[pos - 1], nan=0.0)
    vix = pd.read_parquet(proto / "cache" / "cboe" / "VIX.parquet")
    vix["visible_from"] = pd.to_datetime(vix["date"]) + pd.Timedelta(days=1)
    vj = pd.merge_asof(pd.DataFrame({"D": dates}), vix.sort_values("visible_from"),
                       left_on="D", right_on="visible_from", direction="backward")
    F[:, :, 4] = np.nan_to_num(vj["close"].to_numpy(), nan=0.0)[:, None]
    F[:, :, 5] = tcol("ctx_vix_term_slope")
    F[:, :, 6] = tcol("s1_curvature")
    F[:, :, 7] = tcol("s1_cor3m_z")
    F[:, :, 8] = tcol("ctx_vvix_z")
    F[:, :, 9] = tcol("gd_novelty")
    F[:, :, 10] = 0.5 * (tcol("gd_hhi_loc") + tcol("gd_hhi_org"))
    F[:, :, 11] = Z[:, z_cols.index("z_llm_geopol_risk")].astype(np.float64)[:, None]
    return np.nan_to_num(F, nan=0.0)


def qlike(actual_vol, pred_vol):
    a2 = np.maximum(actual_vol, 1e-4) ** 2
    p2 = np.maximum(pred_vol, 1e-4) ** 2
    r = a2 / p2
    return float(np.mean(r - np.log(r) - 1.0))


def _fit_ridge_head(Xtr, ytr, Xiv, yiv, metric: str):
    """Ridge with training-stat standardization. Vol heads ('qlike') fit
    log(vol) and predict exp(·) — always-positive vol, no clamp explosions
    (forced choice, logged: raw-space ridge produced ≤0 vol predictions and
    unbounded QLIKE on calm folds)."""
    mu_, sd_ = Xtr.mean(0), Xtr.std(0)
    sd_ = np.where(sd_ < 1e-12, 1.0, sd_)
    Xs = (Xtr - mu_) / sd_
    Xv = (Xiv - mu_) / sd_
    logspace = metric == "qlike"
    yt = np.log(np.maximum(ytr, 1e-4)) if logspace else ytr
    d = Xs.shape[1]
    X1 = np.column_stack([np.ones(len(Xs)), Xs])
    G = X1.T @ X1
    b = X1.T @ yt
    best = (None, np.inf)
    for lam in RISK_LAMBDAS:
        pen = lam * np.eye(d + 1)
        pen[0, 0] = 0.0
        coef = np.linalg.solve(G + pen, b)
        p = np.column_stack([np.ones(len(Xv)), Xv]) @ coef
        loss = qlike(yiv, np.exp(np.clip(p, -10, 3))) if logspace \
            else float(np.mean((p - yiv) ** 2))
        if loss < best[1]:
            best = (lam, loss)
    lam = best[0]
    pen = lam * np.eye(d + 1)
    pen[0, 0] = 0.0
    coef = np.linalg.solve(G + pen, b)
    return {"coef": coef, "mu": mu_, "sd": sd_, "logspace": logspace}, lam, best[1]


def _predict_head(head, X):
    Xs = (X - head["mu"]) / head["sd"]
    p = np.column_stack([np.ones(len(Xs)), Xs]) @ head["coef"]
    return np.exp(np.clip(p, -10, 3)) if head["logspace"] else p


def train_risknet(panel, RF, train_idx, val_idx):
    """Three ridge heads: sigma_hat (QLIKE), beta_hat (MSE), book_vol_hat (QLIKE)."""
    sm = panel["symbol_mask"]
    cut = int(len(train_idx) * 0.8)
    tr, iv = train_idx[:cut], train_idx[cut:]

    def pool(idxs, target):
        Xp = RF[idxs].reshape(-1, RF.shape[2])
        yp = target[idxs].reshape(-1)
        mp = sm[idxs].reshape(-1) & np.isfinite(yp)
        return Xp[mp], yp[mp]

    out = {}
    coefs = {}
    for head, tgt, metric in [("sigma_hat", panel["fwd_vol5"], "qlike"),
                              ("beta_hat", panel["fwd_beta63"], "mse")]:
        Xtr, ytr = pool(tr, tgt)
        Xiv, yiv = pool(iv, tgt)
        hd, lam, loss = _fit_ridge_head(Xtr, ytr, Xiv, yiv, metric)
        coefs[head] = (hd, lam, loss)
        if len(val_idx):
            Xv = RF[val_idx].reshape(-1, RF.shape[2])
            out[head] = _predict_head(hd, Xv).reshape(len(val_idx), -1)
    # book head: features = cross-sectional means; target fwd_book_vol5 [N]
    bX = RF.mean(axis=1)
    bt = panel["fwd_book_vol5"]
    mtr = np.isfinite(bt[tr]); miv = np.isfinite(bt[iv])
    hd, lam, loss = _fit_ridge_head(bX[tr][mtr], bt[tr][mtr],
                                    bX[iv][miv], bt[iv][miv], "qlike")
    coefs["book_vol_hat"] = (hd, lam, loss)
    if len(val_idx):
        out["book_vol_hat"] = _predict_head(hd, bX[val_idx])
    out["coefs"] = coefs
    out["n_coefs"] = int(sum(len(c[0]["coef"]) for c in coefs.values()))
    return out
