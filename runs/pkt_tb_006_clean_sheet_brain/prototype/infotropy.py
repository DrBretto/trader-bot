"""PKT-TB-006 — Infotropy mechanisms (BUILD_SPEC §9).

Transfer B (§9.2): the panel precomputes ``w_rec_score`` (raw record score in
[0,1]); ``w_rec(score, eps)`` re-applies any gene eps (record_weight_eps,
default 0.25, floor 0.20).

Transfer A (§9.1): per training fold, each event feature FAMILY entering
EventHead/GBM passes iff R1 ∧ R2 ∧ R3 (family-level, never per-event):

  R1 persistence      mean over family-active days of
                      1{|regime_stat(t+h_r1) − regime_stat(t)| > k·sd_base}, k=1,
                      h_r1=10; regime_stat = bucket 21d realized vol AND 21d
                      drift — both must pass at the 0.5 level on average.
  R2 non-self-encode  1 − OOS R² of (family intensity ~ lagged bucket returns
                      [t−5,t−1]) ≥ 0.5, walk-forward inside the fold's training
                      window (fit on first 70%, OOS on last 30%).
  R3 downstream reuse OOS IC lift of a cheap per-bucket ridge proxy (EventHead
                      shape) with vs without the family > 0; OOS = the last 20%
                      of the fold's TRAINING window (never the fold itself —
                      the screen routes the fold model, so reading the fold
                      would leak).

Forced choices (logged in validation_looks.jsonl):
  - family intensity_{t,k} = mean over the family's B columns of |value|;
    family-active day = intensity above its per-bucket trailing 75th pct
    (expanding, past-only).
  - R2/R3 aggregated across buckets by sample-weighted mean.

Inference-time R2-only down-weight (§9.1, leak-free): rolling, PAST-ONLY
regression of family intensity on the bucket's own trailing returns
[t−5,t−1]; where the trailing out-of-sample R² > 0.5 the event-day row is
down-weighted ×0.25 in EventHead's design matrix.

Falsifier hook: ``screen_pass(..., variant="r3_only")`` gives the R3-only
twin's routing for the conjunctive-gate falsifier.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent

# registered constants (§16): k=1, h=10, R2 bar 0.5, runtime ×0.25
R1_K = 1.0
R1_H = 10
R2_BAR = 0.5
RUNTIME_DOWNWEIGHT = 0.25
RECORD_WEIGHT_EPS_DEFAULT = 0.25
RECORD_WEIGHT_EPS_FLOOR = 0.20

# family -> B-panel columns (B_cols of store/panel.npz)
FAMILIES: Dict[str, List[str]] = {
    "G1_themes": ["g1_z"],
    "G2_country": ["g2_goldstein", "g2_conflict"],
    "G3_tone": ["g3_tone", "g3_tone_dispersion"],
    "G4_novelty_burst": ["g4_burst_z", "g4_novelty"],
    "G5_concentration": ["g5_hhi_loc", "g5_hhi_org"],
    "LLM_event_flags": [f"llm_event_{i:02d}_x" for i in range(12)],
}
# LLM sent/conf/sal columns are the organ's continuous surface, not a screened
# event family (§9.1 list); they ride the llm_available mask instead.


# ----------------------------------------------------------------- Transfer B

def w_rec(score: np.ndarray, eps: float = RECORD_WEIGHT_EPS_DEFAULT) -> np.ndarray:
    """clip(eps + score, eps, 1); eps = record_weight_eps gene (floor 0.20)."""
    eps = max(float(eps), RECORD_WEIGHT_EPS_FLOOR)
    w = np.clip(eps + np.nan_to_num(score, nan=0.0), eps, 1.0)
    return np.where(np.isnan(score), np.nan, w)


# ------------------------------------------------------------- bucket returns

def bucket_daily_returns(panel: dict, bucket_map: dict, symbols: List[str],
                         buckets: List[str]) -> np.ndarray:
    """[N,K] equal-weight member daily close returns (from close_px, ≤ D)."""
    px = panel["close_px"]                       # [N,S] split-adjusted close at D
    with np.errstate(invalid="ignore", divide="ignore"):
        r = px[1:] / px[:-1] - 1.0
    r = np.vstack([np.full((1, px.shape[1]), np.nan), r])
    out = np.full((px.shape[0], len(buckets)), np.nan)
    for k, b in enumerate(buckets):
        js = [symbols.index(s) for s in bucket_map[b] if s in symbols]
        if js:
            with np.errstate(invalid="ignore"):
                out[:, k] = np.nanmean(r[:, js], axis=1)
    return out


def _regime_stats(br: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vol21, drift21) [N,K] from bucket daily returns."""
    df = pd.DataFrame(br)
    vol = (df.rolling(21).std() * np.sqrt(252)).to_numpy()
    drift = df.rolling(21).mean().to_numpy()
    return vol, drift


def family_intensity(B: np.ndarray, b_cols: List[str], family: str) -> np.ndarray:
    """[N,K] mean |value| over the family's columns."""
    idx = [b_cols.index(c) for c in FAMILIES[family]]
    return np.abs(B[:, :, idx]).mean(axis=2)


def _active_mask(intensity: np.ndarray, q: float = 0.75,
                 min_hist: int = 126) -> np.ndarray:
    """Family-active days: intensity above its per-bucket expanding (past-only)
    75th percentile. Zero-variance (all-zero) families are never active."""
    N, K = intensity.shape
    df = pd.DataFrame(intensity)
    thr = df.shift(1).expanding(min_periods=min_hist).quantile(q).to_numpy()
    act = (intensity > thr) & np.isfinite(thr)
    act &= intensity > 1e-12
    return act


# ----------------------------------------------------------------- R1 / R2 / R3

def _r1(intensity_active: np.ndarray, vol: np.ndarray, drift: np.ndarray,
        train_rows: np.ndarray) -> dict:
    """R1 persistence on family-active TRAINING days."""
    out = {}
    for name, stat in [("vol", vol), ("drift", drift)]:
        df = pd.DataFrame(stat)
        d = df.diff(R1_H)                                  # stat(t) - stat(t-10)
        sd_base = d.shift(1).rolling(252, min_periods=126).std().to_numpy()
        shift = np.full_like(stat, np.nan)
        shift[:-R1_H] = np.abs(stat[R1_H:] - stat[:-R1_H])  # over [t+1, t+10]
        with np.errstate(invalid="ignore"):
            ind = shift > R1_K * sd_base
        ok = intensity_active & np.isfinite(shift) & np.isfinite(sd_base)
        ok[~train_rows] = False
        n = int(ok.sum())
        out[name] = float(ind[ok].mean()) if n else np.nan
        out[f"n_{name}"] = n
    out["pass"] = bool((out["vol"] is not np.nan) and (out["n_vol"] >= 30)
                       and out["vol"] >= 0.5 and out["drift"] >= 0.5)
    return out


def _r2(intensity: np.ndarray, br: np.ndarray, train_rows: np.ndarray,
        split: float = 0.7) -> dict:
    """Walk-forward OOS R² of intensity ~ lagged bucket returns [t−5,t−1]."""
    rows = np.nonzero(train_rows)[0]
    cut = rows[int(len(rows) * split)]
    K = intensity.shape[1]
    sse = sst = 0.0
    n_tot = 0
    for k in range(K):
        y = intensity[:, k]
        lags = np.stack([pd.Series(br[:, k]).shift(l).to_numpy()
                         for l in range(1, 6)], axis=1)
        X = np.column_stack([np.ones(len(y)), lags])
        m_fit = train_rows & (np.arange(len(y)) < cut)
        m_oos = train_rows & (np.arange(len(y)) >= cut)
        m_fit &= np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        m_oos &= np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if m_fit.sum() < 60 or m_oos.sum() < 30 or np.nanstd(y[m_fit]) < 1e-12:
            continue
        beta, *_ = np.linalg.lstsq(X[m_fit], y[m_fit], rcond=None)
        e = y[m_oos] - X[m_oos] @ beta
        sse += float(np.sum(e ** 2))
        sst += float(np.sum((y[m_oos] - y[m_fit].mean()) ** 2))
        n_tot += int(m_oos.sum())
    if n_tot == 0 or sst <= 0:
        return {"oos_r2": np.nan, "n": 0, "pass": False}
    r2 = max(0.0, 1.0 - sse / sst)
    return {"oos_r2": r2, "n": n_tot, "pass": bool(1.0 - r2 >= R2_BAR)}


def _ridge_ic(Bm: np.ndarray, y: np.ndarray, fit_rows: np.ndarray,
              oos_rows: np.ndarray, lam: float = 10.0) -> float:
    """Pooled per-bucket ridge proxy -> Spearman IC on OOS rows."""
    from scipy.stats import spearmanr
    K = Bm.shape[1]
    preds, acts = [], []
    for k in range(K):
        Xk, yk = Bm[:, k, :], y[:, k]
        ok = np.all(np.isfinite(Xk), axis=1) & np.isfinite(yk)
        f, o = fit_rows & ok, oos_rows & ok
        if f.sum() < 60 or o.sum() < 20:
            continue
        mu, sd = Xk[f].mean(0), Xk[f].std(0)
        sd[sd < 1e-12] = 1.0
        Xf, Xo = (Xk[f] - mu) / sd, (Xk[o] - mu) / sd
        G = Xf.T @ Xf + lam * np.eye(Xf.shape[1])
        b = np.linalg.solve(G, Xf.T @ (yk[f] - yk[f].mean()))
        preds.append(Xo @ b)
        acts.append(yk[o])
    if not preds:
        return np.nan
    p, a = np.concatenate(preds), np.concatenate(acts)
    return float(spearmanr(p, a).statistic) if len(p) > 10 else np.nan


def _r3(B: np.ndarray, b_cols: List[str], family: str, y5_bucket: np.ndarray,
        train_rows: np.ndarray, split: float = 0.8) -> dict:
    """Per-family ablation lift of the cheap proxy on the training tail."""
    rows = np.nonzero(train_rows)[0]
    cut = rows[int(len(rows) * split)]
    fit_rows = train_rows & (np.arange(B.shape[0]) < cut)
    oos_rows = train_rows & (np.arange(B.shape[0]) >= cut)
    drop = [b_cols.index(c) for c in FAMILIES[family]]
    keep = [i for i in range(len(b_cols)) if i not in drop]
    ic_with = _ridge_ic(B, y5_bucket, fit_rows, oos_rows)
    ic_without = _ridge_ic(B[:, :, keep], y5_bucket, fit_rows, oos_rows)
    lift = (ic_with - ic_without) if np.isfinite(ic_with) and np.isfinite(ic_without) else np.nan
    return {"ic_with": ic_with, "ic_without": ic_without, "lift": lift,
            "pass": bool(np.isfinite(lift) and lift > 0)}


# ------------------------------------------------------------- the fold screen

def run_screen(panel: dict, dates: np.ndarray, out_path: Optional[Path] = None,
               llm_pending: bool = True) -> dict:
    """Transfer-A R1∧R2∧R3 family screen per fold -> store/infotropy_a_screen.json."""
    import folds as F
    b_cols = [str(c) for c in panel["B_cols"]]
    buckets = [str(b) for b in panel["buckets"]]
    symbols = [str(s) for s in panel["symbols"]]
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    br = bucket_daily_returns(panel, bucket_map, symbols, buckets)
    vol, drift = _regime_stats(br)
    y5b = panel["y5_bucket"]
    B = panel["B"]

    result: dict = {"constants": {"R1_k": R1_K, "R1_h": R1_H, "R2_bar": R2_BAR,
                                  "runtime_downweight": RUNTIME_DOWNWEIGHT},
                    "families": {}}
    for fam in FAMILIES:
        inten = family_intensity(B, b_cols, fam)
        act = _active_mask(inten)
        fam_res = {}
        for f in range(1, 7):
            tm, _ = F.fold_train_val(dates, f)
            if fam == "LLM_event_flags" and llm_pending:
                fam_res[f"F{f}"] = {"status": "pending_llm_merge", "pass": False,
                                    "note": "LLM columns are zeros until the "
                                            "Bedrock backfill merges; re-run the "
                                            "screen post-merge"}
                continue
            r1 = _r1(act, vol, drift, tm)
            r2 = _r2(inten, br, tm)
            r3 = _r3(B, b_cols, fam, y5b, tm)
            fam_res[f"F{f}"] = {
                "R1": r1, "R2": r2, "R3": r3,
                "pass": bool(r1["pass"] and r2["pass"] and r3["pass"]),
                "pass_r3_only": bool(r3["pass"]),
            }
        result["families"][fam] = fam_res
    if out_path is not None:
        Path(out_path).write_text(json.dumps(_jsonable(result), indent=1))
    return result


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def screen_pass(screen: dict, family: str, fold: int,
                variant: str = "conjunctive") -> bool:
    """Routing read for EventHead/GBM design matrices.

    variant='conjunctive' (production R1∧R2∧R3) or 'r3_only' (falsifier twin)."""
    e = screen["families"][family][f"F{fold}"]
    if e.get("status") == "pending_llm_merge":
        return False
    return bool(e["pass_r3_only"] if variant == "r3_only" else e["pass"])


# -------------------------------------------- inference-time R2-only down-weight

def runtime_r2_downweight(intensity: np.ndarray, br: np.ndarray,
                          window: int = 252, min_hist: int = 126,
                          bar: float = R2_BAR) -> np.ndarray:
    """[N,K] design-matrix row weights ∈ {1, RUNTIME_DOWNWEIGHT}.

    Rolling, PAST-ONLY: for each (t,k), fit intensity ~ lagged bucket returns
    [t−5,t−1] on the trailing `window` days strictly before t, and score the
    trailing 21 pre-t days out-of-sample; if that rolling OOS R² > bar, day t's
    row is down-weighted ×0.25. Only past returns + past intensity are read —
    leak-free by construction. Evaluated every 21 days per bucket (rolling
    refit; weights forward-filled inside the refit block) for wall-clock."""
    N, K = intensity.shape
    w = np.ones((N, K), dtype=np.float32)
    lagmat = {k: np.stack([pd.Series(br[:, k]).shift(l).to_numpy()
                           for l in range(1, 6)], axis=1) for k in range(K)}
    step = 21
    for k in range(K):
        y = intensity[:, k]
        X = np.column_stack([np.ones(N), lagmat[k]])
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        for t0 in range(min_hist + step, N, step):
            lo = max(0, t0 - window - step)
            fit = np.zeros(N, dtype=bool)
            fit[lo:t0 - step] = True            # strictly pre-OOS
            oos = np.zeros(N, dtype=bool)
            oos[t0 - step:t0] = True            # trailing 21d, strictly < t0
            fit &= ok
            oos &= ok
            if fit.sum() < 60 or oos.sum() < 10 or np.std(y[fit]) < 1e-12:
                continue
            beta, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
            e = y[oos] - X[oos] @ beta
            ss = np.sum((y[oos] - y[fit].mean()) ** 2)
            if ss <= 0:
                continue
            r2 = 1.0 - np.sum(e ** 2) / ss
            if r2 > bar:
                w[t0:min(N, t0 + step), k] = RUNTIME_DOWNWEIGHT
    return w
