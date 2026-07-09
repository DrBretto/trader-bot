"""PKT-TB-006 — solo-book rule (BUILD_SPEC §6, fixed) + trust ledger.

Solo-book rule (deterministic, no parameters):

    mu_z    = clip(mu_m, ±q95 cross-sectional)        # q95 of |mu| fixed per day
    prec[s] = 1 / max(sigma_m[s], sigma_floor=0.25)^2
    w_raw   = max(mu_z * prec, 0)                     # long-only
    w_m     = min(w_raw / sum(w_raw), w_cap=0.10) renormalized to unit gross

Capping uses the standard waterfall (clip at cap, renormalize the uncapped
remainder, repeat) so the emitted book satisfies BOTH the cap and unit gross.
If fewer than ceil(1/w_cap) names are positive, unit gross is infeasible and
gross = w_cap * n_pos (remainder cash) — logged forced choice.

Trust ledger (§6 + meta proposal §3.3/§4): per (date, member)
    u_realized  — U_h of the member's solo book at reference sizing f=0.7 + vol cap,
                  through the half-spread cost table (src/utils/transaction_costs.py)
    ewma21/63   — EWMA of u (half-lives 21/63 td), per-fold standardized
    hit_rate    — rolling 63d fraction of u > 0
    cf_drawdown — rolling 63d max drawdown of the cumulative counterfactual return
    LAG RULE    — stats at decision date D use only u(D') with D' <= D - h - 1
                  (window fully realized by D-1); h = 5.

OOF member-matrix contract (oof/fold_<f>_<member>.npz):
    dates[D] (U10), mu[D,64], sigma[D,64], c[D],
    manifest = JSON string with at least {"walk_forward": true, "member": <name>}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# ---- registered constants (BUILD_SPEC §6/§16) -------------------------------
SIGMA_FLOOR = 0.25
W_CAP = 0.10
Q95 = 0.95
H = 5                      # member/executive horizon (td)
REFERENCE_F = 0.7          # ledger reference sizing
SIGMA_CAP_REF = 0.10       # reference vol cap (B0 vol_target_ann)
LAMBDA_DN_REF = 2.0        # B0 risk_aversion_lambda (ledger uses the B0 default)
EWMA_HALFLIVES = (21, 63)
ROLL_WINDOW = 63           # hit_rate / cf_drawdown rolling window (forced choice, logged)

MEMBERS = ("cast", "gbm_cond", "event_head")

_PROTO = Path(__file__).resolve().parent
_REPO = _PROTO.parents[2]


# ---- cost table --------------------------------------------------------------
def load_half_spread_bps(universe_csv: Path | None = None) -> np.ndarray:
    """Per-symbol half-spread bps from src/utils/transaction_costs + universe sectors."""
    import sys
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from chassis.utils.transaction_costs import get_half_spread_bps

    path = universe_csv or (_REPO / "config" / "universe.csv")
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append(get_half_spread_bps(row["sector"], row.get("asset_class", "equity")))
    return np.asarray(out, dtype=np.float64)


# ---- solo-book rule ------------------------------------------------------------
def solo_book(mu: np.ndarray, sigma: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """BUILD_SPEC §6 fixed rule for one member on one date. mu,sigma: [64]."""
    mu = np.asarray(mu, dtype=np.float64).copy()
    sigma = np.asarray(sigma, dtype=np.float64)
    if active is not None:
        mu = np.where(active, mu, 0.0)
    finite = np.isfinite(mu)
    mu = np.where(finite, mu, 0.0)
    q = np.quantile(np.abs(mu[finite]), Q95) if finite.any() else 0.0
    mu_z = np.clip(mu, -q, q)
    prec = 1.0 / np.maximum(sigma, SIGMA_FLOOR) ** 2
    prec = np.where(np.isfinite(prec), prec, 0.0)
    w_raw = np.maximum(mu_z * prec, 0.0)
    tot = w_raw.sum()
    if tot <= 0:
        return np.zeros_like(w_raw)
    w = w_raw / tot
    # waterfall cap: cap & renormalize the uncapped remainder until stable
    for _ in range(64):
        over = w > W_CAP + 1e-12
        if not over.any():
            break
        excess_budget = 1.0 - W_CAP * over.sum()
        under_sum = w[~over].sum()
        w[over] = W_CAP
        if under_sum <= 1e-15 or excess_budget <= 0:
            break
        w[~over] *= max(excess_budget, 0.0) / under_sum
    return w


def solo_books(mu: np.ndarray, sigma: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """Vector over dates: mu,sigma [D,64] -> books [D,64]."""
    D = mu.shape[0]
    out = np.zeros_like(mu, dtype=np.float64)
    for d in range(D):
        a = active[d] if active is not None else None
        out[d] = solo_book(mu[d], sigma[d], a)
    return out


# ---- book vol estimate ---------------------------------------------------------
def est_book_vol(w: np.ndarray, sigma_hat_ann: np.ndarray) -> float | np.ndarray:
    """Annualized book-vol estimate: full-correlation upper bound sum_s w_s*sigma_s.

    Forced choice (logged): E4 emits book_vol_hat for the EQUAL-WEIGHT book only;
    an arbitrary-w estimate is required for the vol cap, and the linear
    full-correlation bound is the conservative parameter-free mapping.
    """
    return np.asarray(w) @ np.asarray(sigma_hat_ann)


def vol_cap_scale(w: np.ndarray, sigma_hat_ann: np.ndarray, sigma_cap: float) -> float:
    v = float(est_book_vol(w, sigma_hat_ann))
    if v <= sigma_cap or v <= 0:
        return 1.0
    return sigma_cap / v


# ---- counterfactual utility U_h -------------------------------------------------
def u_h(w_tgt: np.ndarray, w_prev: np.ndarray, rets_5d: np.ndarray,
        half_spread_bps: np.ndarray, lam_dn: float = LAMBDA_DN_REF,
        slip_bps: float = 0.0) -> float:
    """U_h(D) per BUILD_SPEC §7.2 for one date. rets_5d: [5,64] daily returns D..D+4."""
    r_book = rets_5d @ w_tgt                                # [5]
    growth = np.log1p(np.clip(r_book, -0.99, None)).mean()
    downside = (np.minimum(0.0, r_book) ** 2).mean()
    cost = float(np.abs(w_tgt - w_prev) @ (half_spread_bps + slip_bps)) / 1e4
    return float(growth - lam_dn * downside - cost / H)


# ---- trust ledger ----------------------------------------------------------------
def realized_u_series(books: np.ndarray, rets: np.ndarray, sigma_hat_ann: np.ndarray,
                      half_spread_bps: np.ndarray, f_ref: float = REFERENCE_F,
                      sigma_cap: float = SIGMA_CAP_REF,
                      lam_dn: float = LAMBDA_DN_REF) -> np.ndarray:
    """u_m(D) for every decision date: solo book at reference sizing through costs.

    books [D,64]; rets [D,64] daily close returns aligned so rets[d] is the return
    realized over decision date d's first holding day; sigma_hat_ann [D,64].
    u(D) needs rets[D..D+4]; trailing dates get NaN.
    """
    D = books.shape[0]
    u = np.full(D, np.nan)
    w_prev = np.zeros(books.shape[1])
    for d in range(D):
        w_tgt = f_ref * books[d]
        w_tgt = w_tgt * vol_cap_scale(w_tgt, sigma_hat_ann[d], sigma_cap)
        if d + H <= D:
            u[d] = u_h(w_tgt, w_prev, rets[d:d + H], half_spread_bps, lam_dn)
        w_prev = w_tgt
    return u


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    out = np.full_like(x, np.nan, dtype=np.float64)
    acc, wsum = 0.0, 0.0
    for i, v in enumerate(x):
        if np.isfinite(v):
            acc = (1 - alpha) * acc + alpha * v
            wsum = (1 - alpha) * wsum + alpha
        out[i] = acc / wsum if wsum > 1e-12 else np.nan
    return out


def build_trust_ledger(dates: np.ndarray, u_by_member: dict[str, np.ndarray],
                       fold_ids: np.ndarray) -> dict:
    """Lagged + per-fold-standardized ledger stats.

    Returns dict member -> {ewma21, ewma63, hit_rate, cf_drawdown, u} arrays [D]
    where row D contains stats computed from u(D') for D' <= D - H - 1 only.
    ewma21/ewma63 standardized within fold (SK §0.5); fold_ids: int [D] (-1 = no fold).
    """
    D = len(dates)
    lag = H + 1
    out = {}
    for m, u in u_by_member.items():
        e21_raw, e63_raw = _ewma(u, 21), _ewma(u, 63)
        hit = np.full(D, np.nan)
        dd = np.full(D, np.nan)
        cum = np.nancumsum(np.where(np.isfinite(u), u, 0.0))
        for i in range(D):
            lo = max(0, i - ROLL_WINDOW + 1)
            win = u[lo:i + 1]
            win = win[np.isfinite(win)]
            if len(win) >= 5:
                hit[i] = float((win > 0).mean())
                c = cum[lo:i + 1]
                dd[i] = float((c - np.maximum.accumulate(c)).min())
        # lag: stats AT decision date D come from index D - lag
        def lagged(a):
            la = np.full(D, np.nan)
            la[lag:] = a[:-lag]
            return la
        e21, e63, hit_l, dd_l = lagged(e21_raw), lagged(e63_raw), lagged(hit), lagged(dd)
        # per-fold standardization of the EWMAs
        e21_z, e63_z = np.full(D, np.nan), np.full(D, np.nan)
        for f in np.unique(fold_ids):
            if f < 0:
                continue
            sel = fold_ids == f
            for src, dst in ((e21, e21_z), (e63, e63_z)):
                v = src[sel]
                mu_, sd = np.nanmean(v), np.nanstd(v)
                dst[sel] = (v - mu_) / (sd if sd > 1e-12 else 1.0)
        out[m] = {"u": u, "ewma21": e21_z, "ewma63": e63_z,
                  "ewma21_raw": e21, "ewma63_raw": e63,
                  "hit_rate": hit_l, "cf_drawdown": dd_l}
    return out


def ledger_to_parquet(dates: np.ndarray, ledger: dict, tau_assigned: dict | None,
                      path: Path) -> None:
    import pandas as pd
    rows = []
    for m, st in ledger.items():
        df = pd.DataFrame({
            "date": dates, "member": m, "u_realized": st["u"],
            "ewma21": st["ewma21"], "ewma63": st["ewma63"],
            "hit_rate": st["hit_rate"], "cf_drawdown": st["cf_drawdown"],
            "tau_assigned": (tau_assigned or {}).get(m, np.full(len(dates), np.nan)),
        })
        rows.append(df)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


# ---- OOF loading -----------------------------------------------------------------
def load_oof(oof_dir: Path, fold: int, member: str, require_walk_forward: bool = True) -> dict:
    """Load one member OOF matrix; refuses non-walk-forward manifests (BUILD_SPEC §5)."""
    p = Path(oof_dir) / f"fold_{fold}_{member}.npz"
    z = np.load(p, allow_pickle=True)
    manifest = json.loads(str(z["manifest"]))
    if require_walk_forward and not manifest.get("walk_forward", False):
        raise RuntimeError(
            f"REFUSED: {p} manifest lacks walk_forward: true — the executive trains "
            f"only on walk-forward OOF member opinions (BUILD_SPEC §5/§7.2).")
    return {"dates": np.asarray(z["dates"]).astype(str), "mu": np.asarray(z["mu"], dtype=np.float64),
            "sigma": np.asarray(z["sigma"], dtype=np.float64), "c": np.asarray(z["c"], dtype=np.float64),
            "manifest": manifest}
