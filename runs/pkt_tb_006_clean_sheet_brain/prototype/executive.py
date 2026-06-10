"""PKT-TB-006 — the executive (BUILD_SPEC §7; ≈560 params, hard ceiling 3k).

Architecture (§7.1):
    phi (30->8 tanh, SHARED across members) on [r_m(4), c_m, agree_m, z(24)]
    trust logit s_m = v·phi_m + b + b_m ;  tau = softmax(s/T) ; floor 0.95·tau + 0.05/3
    blend (convex, book space): w_unit = Σ_m tau_m · w_m
    sizing psi (35->8->1) on [z 24, g 3, |mu_blend| stats 3, tau entropy 1,
                              ledger means 2, book_vol_hat 1, own drawdown 1]
    f = f_max · sigmoid(psi_gain·psi_raw + psi_bias + f_bias) ; w_tgt = f·w_unit
    vol cap: scale so est book vol <= sigma_cap (gene vol_target_ann, B0 0.10)

Loss (§7.2): differentiable decision replay; U_h with lambda_dn; half-spread cost
table + U(−2,+2) bps slippage noise; beta_to turnover; beta_tr KL trust alignment;
entropy-floor penalty; weight decay. Teacher-forced first epoch then sequential
w_prev (w_prev path recomputed with current params each epoch, detached — the
gradient flows through w_tgt(D), not through the w_prev chain; forced choice
logged in validation_looks.jsonl).

Trains ONLY on OOF member books whose manifests carry walk_forward: true.
Fine-tune (§7.6): exactly {b_m×3, log_T, psi_out_gain, psi_out_bias, f_sigmoid_bias}.
Audit record: meta_decision.json per PROPOSAL_META_EVALUATOR §6 (16-step IG by group).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import books as bk
import folds as fd

torch.set_num_threads(4)

# ---- registered constants (§7/§16) -------------------------------------------
N_MEMBERS = 3
N_Z = 24
TRUST_IN = 4 + 1 + 1 + N_Z        # r_m(4), c_m, agree_m, z -> 30
PSI_IN = N_Z + 3 + 3 + 1 + 2 + 1 + 1   # -> 35
F_MAX = 1.0
EPS_TAU = 0.05
BETA_TO = 0.5
BETA_TR = 0.1
BETA_H = 1.0                       # entropy-floor penalty weight (forced choice)
H_FLOOR = 0.5 * float(np.log(3))   # entropy floor threshold (forced choice)
LAMBDA_DN_B0 = 2.0
SIGMA_CAP_B0 = 0.10
WD = 1e-3                          # weight decay (matches spec's other wd; forced choice)
DEPLOY_SEEDS = (11, 13, 17, 19, 23)
PARAM_CEILING = 3000
FINE_TUNE_PARAMS = ("b_m", "log_T", "psi_out_gain", "psi_out_bias", "f_sigmoid_bias")

# psi input layout (audit grouping depends on this)
PSI_SLICES = {"z": (0, 24), "g": (24, 27), "mu_stats": (27, 30), "entropy": (30, 31),
              "ledger_means": (31, 33), "book_vol_hat": (33, 34), "drawdown": (34, 35)}


# ================================ models ======================================
class Executive(nn.Module):
    def __init__(self, seed: int = 11):
        super().__init__()
        torch.manual_seed(seed)
        self.phi = nn.Linear(TRUST_IN, 8)
        self.v = nn.Parameter(torch.randn(8) * 0.3)
        self.b = nn.Parameter(torch.zeros(1))
        self.b_m = nn.Parameter(torch.zeros(N_MEMBERS))
        self.log_T = nn.Parameter(torch.zeros(1))
        self.psi1 = nn.Linear(PSI_IN, 8)
        self.psi2 = nn.Linear(8, 1)
        self.psi_out_gain = nn.Parameter(torch.ones(1))
        self.psi_out_bias = nn.Parameter(torch.zeros(1))
        self.f_sigmoid_bias = nn.Parameter(torch.zeros(1))
        self.z_dropout = nn.Dropout(0.1)

    def trust(self, x_trust: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_trust [..., 3, 30] -> (tau [...,3], logits [...,3])."""
        e = torch.tanh(self.phi(x_trust))
        s = e @ self.v + self.b + self.b_m
        T = torch.exp(self.log_T).clamp(0.1, 10.0)
        tau = torch.softmax(s / T, dim=-1)
        tau = (1 - EPS_TAU) * tau + EPS_TAU / N_MEMBERS
        return tau, s

    def sizing(self, x_psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.psi2(torch.tanh(self.psi1(x_psi))).squeeze(-1)
        f = F_MAX * torch.sigmoid(self.psi_out_gain * raw + self.psi_out_bias
                                  + self.f_sigmoid_bias)
        return f, raw

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LinearGate(nn.Module):
    """~100-param linear twin (§7.3): tau = softmax(a·[r,c,agree] + B·z + b_m)."""
    def __init__(self, seed: int = 11):
        super().__init__()
        torch.manual_seed(seed)
        self.a = nn.Parameter(torch.zeros(6))
        self.B = nn.Parameter(torch.zeros(N_MEMBERS, N_Z))
        self.b_m = nn.Parameter(torch.zeros(N_MEMBERS))
        self.log_T = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(PSI_IN))
        self.d = nn.Parameter(torch.zeros(1))
        self.psi_out_gain = nn.Parameter(torch.ones(1))
        self.psi_out_bias = nn.Parameter(torch.zeros(1))
        self.f_sigmoid_bias = nn.Parameter(torch.zeros(1))
        self.z_dropout = nn.Dropout(0.0)

    def trust(self, x_trust: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xm = x_trust[..., :6]
        z = x_trust[..., 0, 6:]                       # z identical across members
        s = xm @ self.a + z @ self.B.t() + self.b_m
        T = torch.exp(self.log_T).clamp(0.1, 10.0)
        tau = torch.softmax(s / T, dim=-1)
        tau = (1 - EPS_TAU) * tau + EPS_TAU / N_MEMBERS
        return tau, s

    def sizing(self, x_psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = x_psi @ self.c + self.d
        f = F_MAX * torch.sigmoid(self.psi_out_gain * raw + self.psi_out_bias
                                  + self.f_sigmoid_bias)
        return f, raw

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================== data assembly =================================
@dataclass
class ExecData:
    dates: np.ndarray          # [N] decision dates (str)
    seg: np.ndarray            # [N] int segment id (fold) — w_prev resets per segment
    fold_of: np.ndarray        # [N] fold number 1..6
    books: np.ndarray          # [N,3,64] solo books (OOF)
    mu: np.ndarray             # [N,3,64]
    c: np.ndarray              # [N,3]
    r: np.ndarray              # [N,3,4] lagged ledger stats (ewma21z, ewma63z, hit, cf_dd)
    agree: np.ndarray          # [N,3]
    g: np.ndarray              # [N,3]   (mean pairwise rank corr, mean pairwise L1, gross disp)
    z: np.ndarray              # [N,24]
    rets5: np.ndarray          # [N,5,64] forward daily returns D..D+4
    u_real: np.ndarray         # [N,3] realized counterfactual member utility (KL target)
    w_rec: np.ndarray          # [N] scalar record weight (cross-sectional mean; forced choice)
    w_rec_raw: np.ndarray      # [N] raw record score (pre-eps; the EA's eps gene rebuilds w_rec)
    book_vol_hat: np.ndarray   # [N] equal-weight book vol estimate (E4 proxy)
    sigma_hat: np.ndarray      # [N,64] annualized per-symbol vol estimate
    half_spread: np.ndarray    # [64]
    ledger: dict | None = None


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a))
    return ranks


def _book_stats(wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """wb [3,64] -> (agree[3], g[3])."""
    M = wb.shape[0]
    rc = np.zeros((M, M))
    l1 = np.zeros((M, M))
    rk = np.stack([_rankdata(wb[m]) for m in range(M)])
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            ri, rj = rk[i], rk[j]
            denom = ri.std() * rj.std()
            rc[i, j] = ((ri - ri.mean()) * (rj - rj.mean())).mean() / denom if denom > 1e-12 else 0.0
            l1[i, j] = np.abs(wb[i] - wb[j]).sum()
    agree = rc.sum(axis=1) / (M - 1)
    pair = [(i, j) for i in range(M) for j in range(i + 1, M)]
    g = np.array([np.mean([rc[i, j] for i, j in pair]),
                  np.mean([l1[i, j] for i, j in pair]),
                  float(np.std(wb.sum(axis=1)))])
    return agree, g


def assemble_exec_data(world: dict, oof_dir: Path, fold_list: list[int],
                       end_date: str | None = None) -> ExecData:
    """OOF npz files + world prices -> training table. Enforces walk_forward."""
    dates_all = world["dates"]
    hs = world["half_spread"]
    segs = []
    for f in fold_list:
        oofs = {m: bk.load_oof(oof_dir, f, m) for m in bk.MEMBERS}
        d0 = oofs[bk.MEMBERS[0]]["dates"]
        for m in bk.MEMBERS:
            assert np.array_equal(oofs[m]["dates"], d0), f"OOF date mismatch fold {f}"
        if end_date is not None:
            keep = d0 <= end_date
            d0 = d0[keep]
            for m in bk.MEMBERS:
                oofs[m] = {**oofs[m], "mu": oofs[m]["mu"][keep],
                           "sigma": oofs[m]["sigma"][keep], "c": oofs[m]["c"][keep]}
        gidx = np.searchsorted(dates_all, d0)
        assert np.array_equal(dates_all[gidx], d0), f"OOF dates not in world (fold {f})"
        n = len(d0)
        booksf = np.stack([bk.solo_books(oofs[m]["mu"], oofs[m]["sigma"]) for m in bk.MEMBERS], axis=1)
        # ledger u per member within the fold (reference sizing walk)
        rets_seq = world["fwd1"][gidx]                       # [n,64] day-d holding returns
        sh = world["sigma_hat"][gidx]
        u_by = {m: bk.realized_u_series(booksf[:, i], rets_seq, sh, hs)
                for i, m in enumerate(bk.MEMBERS)}
        ledger = bk.build_trust_ledger(d0, u_by, np.full(n, f))
        r = np.stack([np.stack([ledger[m]["ewma21"], ledger[m]["ewma63"],
                                ledger[m]["hit_rate"], ledger[m]["cf_drawdown"]], axis=1)
                      for m in bk.MEMBERS], axis=1)          # [n,3,4]
        r = np.nan_to_num(r)
        agree = np.zeros((n, N_MEMBERS))
        gstat = np.zeros((n, 3))
        for d in range(n):
            agree[d], gstat[d] = _book_stats(booksf[d])
        rets5 = np.full((n, bk.H, 64), np.nan)
        for k, gi in enumerate(gidx):
            if gi + bk.H <= len(dates_all):
                rets5[k] = world["fwd1"][gi:gi + bk.H]
        u_real = np.stack([np.nan_to_num(u_by[m]) for m in bk.MEMBERS], axis=1)
        eq_book = np.ones(64) / 64
        bvh = sh @ eq_book
        segs.append(dict(
            dates=d0, seg=np.full(n, f), fold_of=np.full(n, f), books=booksf,
            mu=np.stack([oofs[m]["mu"] for m in bk.MEMBERS], axis=1),
            c=np.stack([oofs[m]["c"] for m in bk.MEMBERS], axis=1),
            r=r, agree=agree, g=gstat, z=world["z"][gidx], rets5=rets5,
            u_real=u_real,
            w_rec=np.clip(0.25 + world["w_rec_score"][gidx].mean(axis=1), 0.25, 1.0),
            w_rec_raw=world["w_rec_score"][gidx].mean(axis=1),
            book_vol_hat=bvh, sigma_hat=sh, ledger=ledger))
    cat = {k: np.concatenate([s[k] for s in segs], axis=0)
           for k in segs[0] if k != "ledger"}
    valid = ~np.isnan(cat["rets5"]).any(axis=(1, 2))
    cat = {k: v[valid] for k, v in cat.items()}
    return ExecData(half_spread=hs, ledger=None, **cat)


# ============================ batched forward =================================
def _to_t(a):
    return torch.as_tensor(a, dtype=torch.float32)


class ExecTensors:
    def __init__(self, data: ExecData, idx: np.ndarray):
        d = data
        self.idx = idx
        self.books = _to_t(d.books[idx])
        self.mu = _to_t(d.mu[idx])
        self.c = _to_t(d.c[idx])
        self.r = _to_t(d.r[idx])
        self.agree = _to_t(d.agree[idx])
        self.g = _to_t(d.g[idx])
        self.z = _to_t(d.z[idx])
        self.rets5 = _to_t(d.rets5[idx])
        self.u_real = _to_t(d.u_real[idx])
        self.w_rec = _to_t(d.w_rec[idx])
        self.bvh = _to_t(d.book_vol_hat[idx])
        self.sigma_hat = _to_t(d.sigma_hat[idx])
        self.hs = _to_t(d.half_spread)
        self.seg = d.seg[idx]
        self.n = len(idx)


def trust_inputs(t: ExecTensors, model: nn.Module, train: bool,
                 noise_rng: torch.Generator | None = None) -> torch.Tensor:
    r = t.r
    if train and noise_rng is not None:   # input noise on ledger stats
        r = r + 0.1 * torch.randn(r.shape, generator=noise_rng)
    z = t.z
    if train:
        z = model.z_dropout(z)
    zz = z.unsqueeze(1).expand(-1, N_MEMBERS, -1)
    return torch.cat([r, t.c.unsqueeze(-1), t.agree.unsqueeze(-1), zz], dim=-1)


def psi_inputs(t: ExecTensors, tau: torch.Tensor, dd: torch.Tensor,
               z: torch.Tensor | None = None) -> torch.Tensor:
    mu_blend = (tau.unsqueeze(-1) * t.mu).sum(dim=1)            # [N,64]
    am = mu_blend.abs()
    mu_stats = torch.stack([am.mean(-1), am.std(-1), am.amax(-1)], dim=-1)
    ent = -(tau * (tau + 1e-12).log()).sum(-1, keepdim=True)
    led = torch.stack([t.r[:, :, 0].mean(-1), t.r[:, :, 1].mean(-1)], dim=-1)
    zz = t.z if z is None else z
    return torch.cat([zz, t.g, mu_stats, ent, led, t.bvh.unsqueeze(-1),
                      dd.unsqueeze(-1)], dim=-1)


def forward_batch(model: nn.Module, t: ExecTensors, w_prev: torch.Tensor,
                  dd: torch.Tensor, sigma_cap: float, no_trade_band: float,
                  train: bool, noise_rng: torch.Generator | None = None):
    """Batched forward with externally-supplied (detached) w_prev / drawdown state."""
    tau, logits = model.trust(trust_inputs(t, model, train, noise_rng))
    w_unit = (tau.unsqueeze(-1) * t.books).sum(dim=1)            # convex blend [N,64]
    f, psi_raw = model.sizing(psi_inputs(t, tau, dd))
    w_tgt = f.unsqueeze(-1) * w_unit
    vol = (w_tgt * t.sigma_hat).sum(-1)                          # linear book-vol bound
    scale = sigma_cap / torch.clamp(vol, min=sigma_cap)
    w_tgt = w_tgt * scale.unsqueeze(-1)
    delta = w_tgt - w_prev
    keep = (delta.abs() > no_trade_band).float()
    w_emit = w_tgt - (delta * (1 - keep)).detach()               # straight-through band
    return {"tau": tau, "logits": logits, "f": f, "psi_raw": psi_raw,
            "w_tgt": w_tgt, "w_emit": w_emit, "w_unit": w_unit}


@torch.no_grad()
def sequential_state(model: nn.Module, t: ExecTensors, sigma_cap: float,
                     no_trade_band: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential pass with current params -> (w_prev[N,64], drawdown[N]) states.

    w_prev resets to 0 at each segment boundary; drawdown is the running
    drawdown-from-peak of cumulative first-holding-day book returns.
    """
    tau, _ = model.trust(trust_inputs(t, model, train=False))
    w_unit = (tau.unsqueeze(-1) * t.books).sum(dim=1)
    w_prev = torch.zeros(t.n, 64)
    dd = torch.zeros(t.n)
    wp = torch.zeros(64)
    equity, peak = 0.0, 0.0
    prev_seg = None
    for i in range(t.n):
        if t.seg[i] != prev_seg:
            wp = torch.zeros(64)
            equity, peak = 0.0, 0.0
            prev_seg = t.seg[i]
        w_prev[i] = wp
        dd[i] = peak - equity
        x = psi_inputs(ExecSlice(t, i), tau[i:i + 1], dd[i:i + 1])
        f, _ = model.sizing(x)
        w = f.unsqueeze(-1) * w_unit[i:i + 1]
        vol = (w * t.sigma_hat[i:i + 1]).sum(-1)
        w = w * (sigma_cap / torch.clamp(vol, min=sigma_cap)).unsqueeze(-1)
        delta = w[0] - wp
        wp = wp + delta * (delta.abs() > no_trade_band).float()
        r1 = float((wp * t.rets5[i, 0]).sum())
        equity += np.log1p(max(r1, -0.99))
        peak = max(peak, equity)
    return w_prev, dd


class ExecSlice:
    """Single-row view adapter for psi_inputs during the sequential pass."""
    def __init__(self, t: ExecTensors, i: int):
        self.mu = t.mu[i:i + 1]
        self.g = t.g[i:i + 1]
        self.r = t.r[i:i + 1]
        self.z = t.z[i:i + 1]
        self.bvh = t.bvh[i:i + 1]


# ================================ loss ========================================
def compute_loss(model: nn.Module, out: dict, t: ExecTensors, w_prev: torch.Tensor,
                 lam_dn: float, T_u: float, rng: torch.Generator | None,
                 train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, mean weighted utility). §7.2 exact."""
    w = out["w_emit"]
    sabs = lambda x: torch.sqrt(x * x + 1e-8)                    # smooth-abs
    r_book = (t.rets5 * w.unsqueeze(1)).sum(-1)
    growth = torch.log1p(torch.clamp(r_book, min=-0.99)).mean(-1)
    downside = torch.clamp(r_book, max=0.0).pow(2).mean(-1)
    if train and rng is not None:
        eta = (torch.rand(t.n, generator=rng) * 4.0) - 2.0       # U(−2,+2) bps
    else:
        eta = torch.zeros(t.n)
    cost = (sabs(w - w_prev) * (t.hs + eta.unsqueeze(-1))).sum(-1) / 1e4
    U = growth - lam_dn * downside - cost / bk.H
    util = (t.w_rec * U).mean()
    # turnover smoothness on consecutive targets within segments
    same = torch.as_tensor((t.seg[1:] == t.seg[:-1]).astype(np.float32))
    to = (sabs(out["w_tgt"][1:] - out["w_tgt"][:-1]).sum(-1) * same).sum() / same.sum().clamp(min=1)
    # trust alignment KL vs realized counterfactual utilities
    p = torch.softmax(t.u_real / T_u, dim=-1)
    kl = (p * ((p + 1e-12).log() - (out["tau"] + 1e-12).log())).sum(-1).mean()
    ent = -(out["tau"] * (out["tau"] + 1e-12).log()).sum(-1)
    ent_pen = torch.relu(H_FLOOR - ent).mean()
    loss = -util + BETA_TO * to + BETA_TR * kl + BETA_H * ent_pen
    return loss, util


# ============================== training ======================================
def lofo_train_indices(data: ExecData, exclude_fold: int | None,
                       val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Global (train_idx, val_idx) after LOFO exclusion + embargoed split."""
    keep = np.ones(len(data.dates), dtype=bool)
    if exclude_fold is not None:
        keep &= data.fold_of != exclude_fold
    idx_all = np.where(keep)[0]
    tr_i, va_i = fd.embargoed_split(len(idx_all), val_frac=val_frac)
    return idx_all[tr_i], idx_all[va_i]


def train_executive(data: ExecData, seed: int = 11, model_cls=Executive,
                    sigma_cap: float = SIGMA_CAP_B0, lam_dn: float = LAMBDA_DN_B0,
                    no_trade_band: float = 0.010, max_epochs: int = 200,
                    patience: int = 10, lr: float = 0.01, exclude_fold: int | None = None,
                    val_frac: float = 0.15, verbose: bool = False,
                    loss_log: list | None = None) -> nn.Module:
    """Train one executive (or LOFO when exclude_fold is set). Seeded, CPU."""
    tr_idx, va_idx = lofo_train_indices(data, exclude_fold, val_frac)
    t_tr = ExecTensors(data, tr_idx)
    t_va = ExecTensors(data, va_idx)
    T_u = float(max(np.std(data.u_real[tr_idx]), 1e-4))
    model = model_cls(seed=seed)
    if model.n_params() > PARAM_CEILING:
        raise RuntimeError(f"param ceiling exceeded: {model.n_params()} > {PARAM_CEILING}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    rng = torch.Generator().manual_seed(seed * 7919 + 1)
    best_u, best_state, bad = -np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        # teacher-forced epoch 0 / sequential thereafter: state from current params
        w_prev, dd = sequential_state(model, t_tr, sigma_cap, no_trade_band)
        out = forward_batch(model, t_tr, w_prev, dd, sigma_cap, no_trade_band,
                            train=True, noise_rng=rng)
        loss, util = compute_loss(model, out, t_tr, w_prev, lam_dn, T_u, rng, True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            wp_v, dd_v = sequential_state(model, t_va, sigma_cap, no_trade_band)
            out_v = forward_batch(model, t_va, wp_v, dd_v, sigma_cap, no_trade_band, False)
            _, util_v = compute_loss(model, out_v, t_va, wp_v, lam_dn, T_u, None, False)
        loss_f, util_f = float(loss.detach()), float(util.detach())
        if loss_log is not None:
            loss_log.append({"epoch": epoch, "train_loss": loss_f,
                             "train_util": util_f, "val_util": float(util_v)})
        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:3d}  loss {loss_f:+.6f}  "
                  f"train_U {util_f:+.6f}  val_U {float(util_v):+.6f}")
        if float(util_v) > best_u + 1e-9:
            best_u, bad = float(util_v), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_deploy_ensemble(data: ExecData, seeds=DEPLOY_SEEDS, **kw) -> list[nn.Module]:
    return [train_executive(data, seed=s, **kw) for s in seeds]


def train_lofo_executives(data: ExecData, fold_list: list[int], seed: int = 11,
                          **kw) -> dict[int, nn.Module]:
    """Six executives, each trained WITHOUT fold f — the EA scores fold f with it."""
    return {f: train_executive(data, seed=seed, exclude_fold=f, **kw) for f in fold_list}


# ------------------------------ fine-tune (§7.6) ------------------------------
def fine_tune(model: nn.Module, data: ExecData, start: str = fd.FINE_TUNE_START,
              cutoff: str = fd.FINE_TUNE_CUTOFF, lr: float = 0.001,
              max_epochs: int = 50, patience: int = 8,
              sigma_cap: float = SIGMA_CAP_B0, lam_dn: float = LAMBDA_DN_B0,
              no_trade_band: float = 0.010) -> dict:
    """Tune EXACTLY {b_m, log_T, psi_out_gain, psi_out_bias, f_sigmoid_bias};
    every other parameter must stay bit-identical (unit-tested)."""
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for name, p in model.named_parameters():
        p.requires_grad_(name in FINE_TUNE_PARAMS)
    sel = (data.dates >= start) & (data.dates <= cutoff)
    if not sel.any():
        raise RuntimeError(f"fine-tune window {start}..{cutoff} has no data")
    idx = np.where(sel)[0]
    n_va = max(1, int(round(len(idx) * 0.25)))
    t_tr = ExecTensors(data, idx[:-n_va])
    t_va = ExecTensors(data, idx[-n_va:])
    T_u = float(max(np.std(data.u_real[idx]), 1e-4))
    params = [p for n, p in model.named_parameters() if n in FINE_TUNE_PARAMS]
    opt = torch.optim.Adam(params, lr=lr)
    rng = torch.Generator().manual_seed(4242)

    def val_util():
        with torch.no_grad():
            wp, dd = sequential_state(model, t_va, sigma_cap, no_trade_band)
            out = forward_batch(model, t_va, wp, dd, sigma_cap, no_trade_band, False)
            _, u = compute_loss(model, out, t_va, wp, lam_dn, T_u, None, False)
        return float(u)

    pre_val = val_util()
    # never ship a fine-tune that degrades validation: baseline state is the best-so-far
    best_u, bad = pre_val, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(max_epochs):
        model.train()
        wp, dd = sequential_state(model, t_tr, sigma_cap, no_trade_band)
        out = forward_batch(model, t_tr, wp, dd, sigma_cap, no_trade_band, True, rng)
        loss, _ = compute_loss(model, out, t_tr, wp, lam_dn, T_u, rng, True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        u = val_util()
        if u > best_u + 1e-9:
            best_u, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    post_val = val_util()
    changed = [k for k in pre if not torch.equal(pre[k], model.state_dict()[k])]
    illegal = [k for k in changed if k not in FINE_TUNE_PARAMS]
    if illegal:
        raise RuntimeError(f"fine-tune touched frozen params: {illegal}")
    return {"pre_val_util": pre_val, "post_val_util": post_val,
            "delta": post_val - pre_val, "changed_params": changed,
            "window": [start, cutoff]}


# ============================ audit record (§7.4) ==============================
def _integrated_gradients(fn, x: torch.Tensor, steps: int = 16) -> torch.Tensor:
    """IG with zero baseline; deterministic 16-step left Riemann midpoint."""
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        xi = (base + alpha * (x - base)).detach().requires_grad_(True)
        y = fn(xi)
        g = torch.autograd.grad(y.sum(), xi)[0]
        total = total + g
    return (x - base) * total / steps


TRUST_GROUPS = {"ewma21": [0], "ewma63": [1], "hit": [2], "cf_dd": [3],
                "confidence": [4], "agree": [5], "context": list(range(6, 30))}
SIZING_GROUPS = {
    "vol_level": [1, 7, 8, 13, 33], "sentiment_agg": [14, 15, 16, 17],
    "disagreement": [24, 25, 26, 30], "edge_magnitude": [27, 28, 29],
    "trust_record": [31, 32], "drawdown": [34],
    "context": [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 18, 19, 20, 21, 22, 23],
}
MEMBER_NAMES = {"cast": "xsec_transformer", "gbm_cond": "gbm_cond", "event_head": "event_head"}


def write_meta_decision(model: nn.Module, data: ExecData, i: int, out_dir: Path,
                        sigma_cap: float = SIGMA_CAP_B0, no_trade_band: float = 0.010,
                        w_prev: np.ndarray | None = None, dd: float = 0.0,
                        model_version: str = "syn1_exec", code_sha: str = "dev") -> dict:
    """meta_decision.json per PROPOSAL_META_EVALUATOR §6 (16-step IG by group)."""
    t = ExecTensors(data, np.array([i]))
    wp = torch.zeros(1, 64) if w_prev is None else _to_t(w_prev).reshape(1, 64)
    ddt = torch.tensor([float(dd)])
    out = forward_batch(model, t, wp, ddt, sigma_cap, no_trade_band, train=False)
    tau = out["tau"][0].detach().numpy()
    logits = out["logits"][0].detach().numpy()
    x_trust = trust_inputs(t, model, train=False)[0]            # [3,30]
    experts = []
    for m_i, m in enumerate(bk.MEMBERS):
        xm = x_trust[m_i]

        def logit_m(xi, m_i=m_i):
            e = torch.tanh(model.phi(xi)) if isinstance(model, Executive) else None
            if e is not None:
                return e @ model.v + model.b + model.b_m[m_i]
            return xi[..., :6] @ model.a + xi[..., 6:] @ model.B[m_i] + model.b_m[m_i]

        ig = _integrated_gradients(lambda xi: logit_m(xi.unsqueeze(0)), xm).detach().numpy()
        attrib = {k: float(ig[v].sum()) for k, v in TRUST_GROUPS.items()}
        attrib["bias"] = float(logits[m_i] - ig.sum())
        wb = data.books[i]
        others = [j for j in range(N_MEMBERS) if j != m_i]
        rk_m = _rankdata(wb[m_i])
        corr = []
        for j in others:
            rk_j = _rankdata(wb[j])
            den = rk_m.std() * rk_j.std()
            corr.append(((rk_m - rk_m.mean()) * (rk_j - rk_j.mean())).mean() / den if den > 1e-12 else 0.0)
        experts.append({
            "name": MEMBER_NAMES[m], "confidence": float(data.c[i, m_i]),
            "rolling": {"ewma21": float(data.r[i, m_i, 0]), "ewma63": float(data.r[i, m_i, 1]),
                        "hit": float(data.r[i, m_i, 2]), "cf_dd": float(data.r[i, m_i, 3])},
            "trust": float(tau[m_i]), "trust_logit": float(logits[m_i]),
            "logit_attrib": attrib, "solo_book_corr": float(np.mean(corr)),
        })
    # sizing attribution on psi pre-activation
    x_psi = psi_inputs(t, out["tau"], ddt)[0]

    def psi_fn(xi):
        f, raw = model.sizing(xi)
        return raw if raw.dim() > 0 else raw.unsqueeze(0)

    ig_s = _integrated_gradients(lambda xi: psi_fn(xi.unsqueeze(0)), x_psi).detach().numpy()
    sizing_attrib = {k: float(ig_s[v].sum()) for k, v in SIZING_GROUPS.items()}
    with torch.no_grad():
        _, raw0 = model.sizing(x_psi.unsqueeze(0))
    sizing_attrib["bias"] = float(raw0[0]) - float(ig_s.sum())
    w = out["w_emit"][0].detach().numpy()
    held = np.where(w > 1e-6)[0]
    # expert share per holding: tau_m * w_m_s / w_unit_s (book-space blend = exact)
    w_unit = out["w_unit"][0].detach().numpy()
    top = held[np.argsort(-w[held])][:10]
    contributions = []
    for s_i in top:
        denom = max(w_unit[s_i], 1e-12)
        share = {MEMBER_NAMES[m]: float(tau[m_i] * data.books[i, m_i, s_i] / denom * w[s_i])
                 for m_i, m in enumerate(bk.MEMBERS)}
        contributions.append({"symbol": int(s_i), "w": float(w[s_i]), "expert_share": share})
    eq = np.ones(64) / 64 * 0.7
    ewma_raw = data.r[i, :, 0]
    rec = {
        "date": str(data.dates[i]), "model_version": model_version, "code_sha": code_sha,
        "input_hash": hashlib.sha256(np.ascontiguousarray(data.z[i]).tobytes()).hexdigest()[:16],
        "experts": experts,
        "trust_entropy": float(-(tau * np.log(tau + 1e-12)).sum()),
        "deployment_fraction": float(out["f"][0]),
        "sizing_attrib": sizing_attrib,
        "book": {"gross": float(w.sum()), "n_positions": int((w > 1e-6).sum()),
                 "top_contributions": contributions},
        "counterfactuals": {"equal_weight_baseline_book_gross": float(eq.sum()),
                            "divergence_from_baseline_L1": float(np.abs(w - eq).sum())},
        "expected": {"U_h_hat": float((tau * ewma_raw).sum())},
        "realized": None,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta_decision.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ----------------------------- entrypoint -------------------------------------
def main():
    """Final-train entrypoint (run after real member OOFs land in oof/):

        .venv/bin/python executive.py --oof-dir oof --out exec_out
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default="oof")
    ap.add_argument("--out", default="exec_out")
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()
    import synth_oof as so
    proto = Path(__file__).resolve().parent
    world = so.world_from_panel(proto / "store" / "panel.npz")
    fold_list = [int(x) for x in args.folds.split(",")]
    data = assemble_exec_data(world, proto / args.oof_dir, fold_list,
                              end_date=fd.FITNESS_END)
    out_dir = proto / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"executive params: {Executive().n_params()} (ceiling {PARAM_CEILING}); "
          f"linear twin: {LinearGate().n_params()}")
    models = []
    for s in DEPLOY_SEEDS:
        log: list = []
        m = train_executive(data, seed=s, max_epochs=args.max_epochs, loss_log=log)
        torch.save(m.state_dict(), out_dir / f"executive_seed{s}.pt")
        models.append((s, m, log))
        print(f"seed {s}: best val util epoch trace in exec_out/loss_seed{s}.json")
        with open(out_dir / f"loss_seed{s}.json", "w") as fh:
            json.dump(log, fh)
    twin = train_executive(data, seed=11, model_cls=LinearGate, max_epochs=args.max_epochs)
    torch.save(twin.state_dict(), out_dir / "linear_twin.pt")
    lofo = train_lofo_executives(data, fold_list, max_epochs=args.max_epochs)
    for f, m in lofo.items():
        torch.save(m.state_dict(), out_dir / f"lofo_fold{f}.pt")
    print("done:", out_dir)


if __name__ == "__main__":
    main()
