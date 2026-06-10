"""PKT-TB-006 — SYNTHETIC smoke OOF generator (plumbing validation only).

Generates seeded member OOF matrices (mu/sigma/c per date) with a realistic
correlation structure: a common latent factor shared by members, member-specific
skill loading on the true forward return, and idiosyncratic noise. Manifests
carry ``walk_forward: true`` so the executive's stacking-discipline assertion
passes — they also carry ``synthetic: true`` so no final artifact can quietly
train on them.

Two worlds:
  make_synth_world(...)        — fully synthetic dates/returns (hermetic tests)
  world_from_panel(panel_path) — REAL panel dates/prices/Z (end-to-end smoke)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import books as bk

N_SYM = 64
N_Z = 24
MEMBERS = bk.MEMBERS

# member skill / correlation profile (arbitrary but fixed)
_SKILL = {"cast": 0.08, "gbm_cond": 0.05, "event_head": 0.03}
_COMMON = {"cast": 0.5, "gbm_cond": 0.5, "event_head": 0.2}
_ABSTAIN_P = {"cast": 0.0, "gbm_cond": 0.0, "event_head": 0.35}  # event head abstains


def make_synth_world(n_dates: int = 320, seed: int = 7, start: str = "2020-02-03") -> dict:
    rng = np.random.default_rng(seed)
    dates = np.array([str(d)[:10] for d in
                      np.array(np.datetime64(start) + np.arange(n_dates * 2), dtype="datetime64[D]")
                      if np.is_busday(d)][:n_dates])
    assert len(dates) == n_dates
    # factor returns: market + sector-ish factor + idio
    mkt = rng.normal(0.0002, 0.009, n_dates)
    load = 0.6 + 0.6 * rng.random(N_SYM)
    fac = rng.normal(0, 0.004, (n_dates, 4))
    fload = rng.normal(0, 1, (4, N_SYM)) * 0.5
    idio = rng.normal(0, 0.006, (n_dates, N_SYM))
    fwd1 = mkt[:, None] * load[None, :] + fac @ fload + idio    # fwd1[d] = day-d holding return
    vol21 = np.full((n_dates, N_SYM), 0.012)
    for d in range(1, n_dates):
        vol21[d] = 0.95 * vol21[d - 1] + 0.05 * np.abs(fwd1[d - 1])
    sigma_hat = vol21 * np.sqrt(252) * 1.25
    z = rng.normal(0, 1, (n_dates, N_Z)).astype(np.float64)
    w_rec_score = np.clip(rng.beta(2, 5, (n_dates, N_SYM)), 0, 1)
    return {
        "dates": dates, "fwd1": fwd1, "sigma_hat": sigma_hat, "z": z,
        "w_rec_score": w_rec_score,
        "half_spread": np.full(N_SYM, 3.0),
        "symbol_mask": np.ones((n_dates, N_SYM), dtype=bool),
        "seed": seed, "synthetic": True,
    }


def world_from_panel(panel_path: Path) -> dict:
    """Real dates/prices/Z from store/panel.npz; member opinions stay synthetic."""
    p = np.load(panel_path, allow_pickle=True)
    dates = np.asarray(p["dates"]).astype(str)
    close = np.asarray(p["close_px"], dtype=np.float64)
    n, s = close.shape
    fwd1 = np.full((n, s), np.nan)
    fwd1[:-1] = close[1:] / close[:-1] - 1.0          # decision d's first holding day
    fwd1 = np.where(np.isfinite(fwd1), fwd1, 0.0)
    # trailing 21d vol (past-only), annualized, scaled as a forward-vol proxy
    r_hist = np.zeros_like(close)
    r_hist[1:] = np.where(close[:-1] > 0, close[1:] / close[:-1] - 1.0, 0.0)
    r_hist = np.where(np.isfinite(r_hist), r_hist, 0.0)
    vol = np.full((n, s), 0.012)
    for d in range(1, n):
        vol[d] = 0.95 * vol[d - 1] + 0.05 * np.abs(r_hist[d])
    sigma_hat = vol * np.sqrt(252) * 1.25
    z = np.nan_to_num(np.asarray(p["Z"], dtype=np.float64))
    w_rec_score = np.nan_to_num(np.asarray(p["w_rec_score"], dtype=np.float64))
    return {
        "dates": dates, "fwd1": fwd1, "sigma_hat": sigma_hat, "z": z,
        "w_rec_score": w_rec_score,
        "half_spread": bk.load_half_spread_bps(),
        "symbol_mask": np.asarray(p["symbol_mask"], dtype=bool),
        "seed": 0, "synthetic": False,
    }


def gen_member_oof(world: dict, date_mask: np.ndarray, member: str, fold: int,
                   seed_base: int = 1000) -> dict:
    """Synthetic mu/sigma/c for one (fold, member) with cross-member correlation."""
    member_id = int(hashlib.sha256(member.encode()).hexdigest()[:6], 16) % 1000
    rng = np.random.default_rng(seed_base + 97 * fold + member_id + world["seed"])
    idx = np.where(date_mask)[0]
    n = len(idx)
    fwd1 = world["fwd1"]
    # 5d forward return z (the true signal members partially see)
    n_all = fwd1.shape[0]
    f5 = np.zeros((n, N_SYM))
    for k, d in enumerate(idx):
        hi = min(d + bk.H, n_all)
        f5[k] = fwd1[d:hi].sum(axis=0)
    truth = (f5 - f5.mean(axis=1, keepdims=True)) / (f5.std(axis=1, keepdims=True) + 1e-9)
    common_seed = np.random.default_rng(seed_base + 7919 * fold + world["seed"])
    common = common_seed.normal(0, 1, (n, N_SYM))
    own = rng.normal(0, 1, (n, N_SYM))
    a, b_ = _SKILL[member], _COMMON[member]
    mu = a * truth + b_ * common + np.sqrt(max(1e-9, 1 - a * a - b_ * b_)) * own
    mu = (mu - mu.mean(axis=1, keepdims=True)) / (mu.std(axis=1, keepdims=True) + 1e-9)
    sigma = np.clip(0.8 + 0.4 * rng.random((n, N_SYM)), bk.SIGMA_FLOOR, None)
    c = np.clip(0.5 + 0.2 * rng.normal(0, 1, n), 0.05, 0.95)
    if _ABSTAIN_P[member] > 0:
        quiet = rng.random(n) < _ABSTAIN_P[member]
        mu[quiet] = 0.0
        c[quiet] = 0.05
    return {"dates": world["dates"][idx], "mu": mu, "sigma": sigma, "c": c}


def write_smoke_oofs(world: dict, oof_dir: Path, fold_masks: dict[int, np.ndarray]) -> list[Path]:
    """Write oof/fold_<f>_<member>.npz per the member-lane contract."""
    oof_dir = Path(oof_dir)
    oof_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f, mask in fold_masks.items():
        for m in MEMBERS:
            d = gen_member_oof(world, mask, m, f)
            manifest = {"member": m, "fold": f, "walk_forward": True, "synthetic": True,
                        "model_version": "synthetic_smoke_v1",
                        "seeds": [int(world["seed"])], "code_sha": "smoke"}
            p = oof_dir / f"fold_{f}_{m}.npz"
            np.savez_compressed(p, dates=d["dates"], mu=d["mu"], sigma=d["sigma"],
                                c=d["c"], manifest=json.dumps(manifest))
            paths.append(p)
    return paths
