"""PKT-TB-006 — E1 primary reads (TOURNAMENT §4.3; BUILD_SPEC §13).

E1 primary statistic per organ: the with-organ vs without-organ DAILY UTILITY
difference series on identical dates over the pooled F1–F6 out-of-fold panel,
HAC-adjusted t (Newey–West, 10 lags; stats.hac_tstat). The walk is ea.py's
vectorized fold walk (LOFO executive per fold, TR S2) re-emitted as a per-day
return series at cost scenario 1.0 — ``E1WalkEngine.fold_daily_returns`` mirrors
``ea.FitnessEngine.fold_utility`` exactly (consistency unit-tested: the min over
cost scenarios of the utility recomputed from these series equals
``fold_utility`` to float tolerance).

Supported arm shapes (one ``E1Side`` per side of the comparison):
  - member drops          genome.member_gate transform (trust renorm in-walk)
  - ridge-in-slot         OOF member map cast -> ridge_twin (+ RT-2 LOFO execs)
  - LLM-neutral           OOF suffix retrain map + feature_gate LLM_block off
  - GDELT-ablated         OOF suffix retrain map + G1/G4G5 gates off
  - infotropy-B uniform   OOF *_uniform map + w_rec uniform + RT-5 execs
  - B0 vs champion        genome swap
  - equal-trust + f=0.7   executive bypass walk (R08)
  - RiskNet+ vs trailing  sigma_hat source swap (R07)

FORCED CHOICES (logged once here, printed in every E1 manifest):
  - The E1 daily-utility series is the walk's daily BOOK RETURN at cost
    scenario 1.0 (the same series whose Sharpe/MaxDD defines the EA's U_f).
  - Base sides use sigma_source='trailing21' — the FROZEN deployed sigma
    convention (FREEZE_SYN1.md: trailing-21 proxy is the replay default), so
    E1 and E2 compare the same brain. The R07 organ read is therefore
    risknet (with) vs trailing21 (without/deployed): positive t = the RiskNet
    E4 instrument ADDS over the deployed trailing-21 proxy. [supersedes the
    earlier 'risknet base' choice — re-logged in validation_looks.jsonl]
  - Base sides walk LOFO executives of the FROZEN gate class. The rung that
    ships for an exec dir is read from <exec_dir>/ladder.json ("ships":
    "linear_twin"|"mlp"); linear_twin rungs load lofo_twin_fold<f>.pt
    (LinearGate), mlp rungs load lofo_fold<f>.pt (Executive MLP). Missing
    ladder.json falls back to the MLP LOFO files (legacy layout).
  - The R08 bypass replaces the WHOLE trust/sizing stack with tau = 1/M over
    gated members and f = 0.7 (no genome trust tilt, no event_weight_cap),
    keeping the genome rail ladder (gross/cash, max-symbol, vol cap, dd brake,
    abstain, no-trade band) so the read isolates the learned executive.
"""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import books as bk
import ea
import executive as ex
import folds as fd
import stats as st

PROTO = Path(__file__).resolve().parent


# ============================ side configuration ===============================
@dataclass
class E1Side:
    label: str
    genome: "ea.Genome"
    oof_member_map: dict | None = None      # e.g. {"cast": "ridge_twin"}
    exec_dir: Path | None = None            # LOFO executive dir (lofo_fold<f>.pt)
    models: dict | None = None              # {fold: nn.Module} injected (tests)
    sigma_source: str = "trailing21"        # 'trailing21' (FROZEN deployed) | 'risknet'
    uniform_w_rec: bool = False             # infotropy-B without-side
    bypass: dict | None = None              # {"equal_trust": True, "f_fixed": 0.7}
    feature_gate_off: tuple = ()            # ea.FEATURE_GATE_NAMES entries forced 0
    warnings: list = field(default_factory=list)

    def effective_genome(self) -> "ea.Genome":
        g = ea.Genome.from_dict(self.genome.to_dict())
        if self.feature_gate_off:
            fg = list(g.feature_gate)
            for fam in self.feature_gate_off:
                fg[ea.FEATURE_GATE_NAMES.index(fam)] = 0
            g.feature_gate = fg
        return g


@contextmanager
def member_map_ctx(member_map: dict | None):
    """Redirect bk.load_oof member names (slot -> oof file name) for assembly."""
    if not member_map:
        yield
        return
    orig = bk.load_oof

    def patched(oof_dir, fold, member, require_walk_forward=True):
        return orig(oof_dir, fold, member_map.get(member, member), require_walk_forward)

    bk.load_oof = patched
    try:
        yield
    finally:
        bk.load_oof = orig


def apply_sigma_source(data: ex.ExecData, oof_dir: Path, fold_list: list[int],
                       source: str, warnings: list) -> ex.ExecData:
    """sigma_source='risknet': overwrite sigma_hat/book_vol_hat rows from the
    E4 OOF instrument files; 'trailing21': keep the world trailing proxy."""
    if source == "trailing21":
        return data
    if source != "risknet":
        raise ValueError(f"unknown sigma_source {source!r}")
    data = copy.deepcopy(data)
    n_hit = 0
    for f in fold_list:
        p = Path(oof_dir) / f"fold_{f}_risknet.npz"
        if not p.exists():
            warnings.append(f"risknet OOF missing for fold {f}; trailing-21d proxy kept")
            continue
        z = np.load(p, allow_pickle=True)
        rd = np.asarray(z["dates"]).astype(str)
        rs = np.asarray(z["sigma_hat"], dtype=np.float64)
        rb = np.asarray(z["book_vol_hat"], dtype=np.float64)
        lut = {d: i for i, d in enumerate(rd)}
        for row, d in enumerate(data.dates):
            j = lut.get(str(d))
            if j is not None and data.fold_of[row] == f:
                data.sigma_hat[row] = rs[j]
                data.book_vol_hat[row] = rb[j]
                n_hit += 1
    if n_hit == 0:
        warnings.append("risknet sigma source matched 0 rows; trailing proxy in effect")
    return data


def shipped_rung(exec_dir: Path) -> str:
    """The §7.3 ladder rung that SHIPS for this exec dir: <dir>/ladder.json
    {'ships': 'linear_twin'|'mlp'}; missing file = legacy MLP layout."""
    p = Path(exec_dir) / "ladder.json"
    if p.exists():
        return json.loads(p.read_text())["ships"]
    return "mlp"


def load_lofo(exec_dir: Path, fold_list: list[int]) -> dict:
    """LOFO executives of the SHIPPED gate class for this dir (frozen-gate
    mirroring: E1 walks the same gate class the replay deploys)."""
    rung = shipped_rung(exec_dir)
    cls, pat = ((ex.LinearGate, "lofo_twin_fold{f}.pt") if rung == "linear_twin"
                else (ex.Executive, "lofo_fold{f}.pt"))
    models = {}
    for f in fold_list:
        m = cls()
        m.load_state_dict(torch.load(Path(exec_dir) / pat.format(f=f)))
        m.eval()
        models[f] = m
    return models


def build_side(side: E1Side, world: dict, oof_dir: Path,
               fold_list: list[int]) -> "E1WalkEngine":
    with member_map_ctx(side.oof_member_map):
        data = ex.assemble_exec_data(world, Path(oof_dir), fold_list,
                                     end_date=fd.FITNESS_END)
    data = apply_sigma_source(data, oof_dir, fold_list, side.sigma_source,
                              side.warnings)
    if side.uniform_w_rec:
        data = copy.deepcopy(data)
        data.w_rec_raw = np.ones_like(data.w_rec_raw)
        data.w_rec = np.ones_like(data.w_rec)
    if side.bypass:
        models = side.models or {f: ex.Executive() for f in fold_list}  # unused in walk
    elif side.models is not None:
        models = side.models
    elif side.exec_dir is not None:
        models = load_lofo(side.exec_dir, fold_list)
    else:
        raise RuntimeError(f"E1 side {side.label!r}: no LOFO executives "
                           f"(exec_dir or models required)")
    return E1WalkEngine(data, models, fold_list)


# ============================ the daily walk ===================================
class E1WalkEngine(ea.FitnessEngine):
    """ea.FitnessEngine whose walk also emits the per-day return series."""

    def fold_daily_returns(self, genome: "ea.Genome", f: int,
                           cost_scale: float = 1.0,
                           bypass: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(dates[n], r[n]) — daily book returns of the genome walk on fold f at
        one cost scenario. Mirrors FitnessEngine.fold_utility exactly."""
        idx = self.fold_idx[f]
        dates = np.asarray(self.data.dates[idx]).astype(str)
        hs = self.data.half_spread
        d = self.data
        if bypass:
            mg = np.asarray(genome.member_gate, dtype=float)
            if mg.sum() <= 0:
                raise RuntimeError("bypass walk with all members gated off")
            n = len(idx)
            tau_row = mg / mg.sum()
            coef_n = np.tile(tau_row, (n, 1))
            books = d.books[idx]
            mu = d.mu[idx]
            w_unit = np.einsum("nm,nms->ns", coef_n, books)
            mu_blend = np.einsum("nm,nms->ns", coef_n, mu)
            am = np.abs(mu_blend)
            f_fixed = float(bypass.get("f_fixed", 0.7))
            fwd1 = d.rets5[idx, 0, :]
            r_out = np.zeros(n)
            w_prev = np.zeros(64)
            equity, peak = 0.0, 0.0
            for i in range(n):
                dd = peak - equity
                w = f_fixed * w_unit[i]
                w, w_emit, traded = self._rails(w, w_prev, dd, am[i], genome,
                                                d.sigma_hat[idx[i]])
                cost = np.abs(traded) @ hs * cost_scale / 1e4
                r = float(w_emit @ fwd1[i]) - cost
                r_out[i] = r
                equity += np.log1p(max(r, -0.99))
                peak = max(peak, equity)
                w_prev = w_emit
            return dates, r_out
        # ---- full genome walk: identical math to FitnessEngine.fold_utility ----
        c = self._cached(f, tuple(genome.feature_gate))
        n = c["s"].shape[0]
        lam = 1.0 - 0.5 ** (1.0 / genome.trust_halflife_days)
        w_rec = np.clip(genome.record_weight_eps + c["w_rec_raw"],
                        genome.record_weight_eps, 1.0)
        tilt = np.zeros((n, 3))
        acc, wsum = np.zeros(3), 0.0
        lag = bk.H + 1
        for i in range(n):
            tilt[i] = acc / wsum if wsum > 1e-9 else 0.0
            j = i - lag
            if j >= 0:
                acc = (1 - lam) * acc + lam * w_rec[j] * c["u_real"][j]
                wsum = (1 - lam) * wsum + lam
        tilt = tilt / self._u_std[f]
        logits = c["s"] + np.asarray(genome.trust_prior) + tilt
        T = c["T_exec"] * genome.conviction_temp
        e = np.exp(logits / T - logits.max(axis=1, keepdims=True) / T)
        tau = e / e.sum(axis=1, keepdims=True)
        tau = (1 - ex.EPS_TAU) * tau + ex.EPS_TAU / 3
        mg = np.asarray(genome.member_gate, dtype=float)
        tau = tau * mg
        norm = tau.sum(axis=1, keepdims=True)
        tau = np.divide(tau, norm, out=np.zeros_like(tau), where=norm > 1e-12)
        capf = np.ones(3)
        capf[ea.EVENT_MEMBER_IDX] = genome.event_weight_cap
        coef = tau * capf
        cnorm = coef.sum(axis=1, keepdims=True)
        coef_n = np.divide(coef, cnorm, out=np.zeros_like(coef), where=cnorm > 1e-12)
        w_unit = np.einsum("nm,nms->ns", coef_n, c["books"])
        mu_blend = np.einsum("nm,nms->ns", coef_n, c["mu"])
        am = np.abs(mu_blend)
        mustats = np.stack([am.mean(1), am.std(1), am.max(1)], axis=1)
        ent = -(tau * np.log(tau + 1e-12)).sum(axis=1)
        pre = c["A_static"] + mustats @ c["W_mu"].T + ent[:, None] * c["w_ent"][None, :]
        r_out = np.zeros(n)
        w_prev = np.zeros(64)
        equity, peak = 0.0, 0.0
        for i in range(n):
            dd = peak - equity
            h_pre = pre[i] + c["w_dd"] * dd
            hidden = np.tanh(h_pre) if c["act"] == "tanh" else h_pre
            f_dep = 1.0 / (1.0 + np.exp(-(c["gain"] * (hidden @ c["v2"] + c["b2"])
                                          + c["bias"])))
            w = f_dep * w_unit[i]
            w, w_emit, traded = self._rails(w, w_prev, dd, am[i], genome,
                                            self.data.sigma_hat[idx[i]])
            cost = np.abs(traded) @ hs * cost_scale / 1e4
            r = float(w_emit @ c["fwd1"][i]) - cost
            r_out[i] = r
            equity += np.log1p(max(r, -0.99))
            peak = max(peak, equity)
            w_prev = w_emit
        return dates, r_out

    @staticmethod
    def _rails(w: np.ndarray, w_prev: np.ndarray, dd: float, am_i: np.ndarray,
               genome: "ea.Genome", sigma_hat_row: np.ndarray):
        """The genome rail ladder, byte-order identical to FitnessEngine."""
        gross = w.sum()
        gmax = min(genome.gross_target, 1.0 - genome.cash_floor)
        if gross > gmax > 0:
            w = w * (gmax / gross)
        w = np.minimum(w, genome.max_symbol_weight)
        vol = w @ sigma_hat_row
        if vol > genome.vol_target_ann > 0:
            w = w * (genome.vol_target_ann / vol)
        if dd > genome.dd_brake_threshold:
            brake = 1.0 - genome.dd_brake_strength * min(
                1.0, (dd - genome.dd_brake_threshold) / max(genome.dd_brake_threshold, 1e-6))
            w = w * brake
        if am_i.mean() < genome.abstain_threshold:
            w_emit = w_prev
            traded = np.zeros(64)
        else:
            delta = w - w_prev
            mask = np.abs(delta) > genome.no_trade_band
            traded = delta * mask
            w_emit = w_prev + traded
        return w, w_emit, traded

    def utility_from_series(self, genome: "ea.Genome", f: int) -> float:
        """min over cost scenarios of U_f recomputed from the daily series —
        must equal FitnessEngine.fold_utility (consistency test)."""
        us = []
        for sc in ea.COST_SCENARIOS:
            _, r = self.fold_daily_returns(genome, f, cost_scale=sc)
            sd = r.std()
            shrp = np.sqrt(252) * r.mean() / sd if sd > 1e-12 else 0.0
            eq = np.cumsum(np.log1p(np.clip(r, -0.99, None)))
            peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
            maxdd = float(np.max(np.concatenate([[0.0], peak - eq])))
            us.append(shrp - 0.5 * (1.0 - np.exp(-maxdd)) / 0.10)
        return float(min(us))


# ============================ the E1 comparison ================================
def pooled_series(engine: E1WalkEngine, genome: "ea.Genome", fold_list: list[int],
                  bypass: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    ds, rs = [], []
    for f in fold_list:
        d, r = engine.fold_daily_returns(genome, f, cost_scale=1.0, bypass=bypass)
        ds.append(d)
        rs.append(r)
    return np.concatenate(ds), np.concatenate(rs)


def run_e1(side_with: E1Side, side_without: E1Side, world: dict, oof_dir: Path,
           fold_list: list[int]) -> dict:
    """The §4.3 E1 primary read: pooled F1–F6 with-vs-without daily utility
    difference on identical dates, HAC t. Returns the full stat block."""
    eng_w = build_side(side_with, world, oof_dir, fold_list)
    eng_wo = build_side(side_without, world, oof_dir, fold_list)
    dw, rw = pooled_series(eng_w, side_with.effective_genome(), fold_list,
                           bypass=side_with.bypass)
    dwo, rwo = pooled_series(eng_wo, side_without.effective_genome(), fold_list,
                             bypass=side_without.bypass)
    dates, a, b = st.align_paired(dw, rw, dwo, rwo)
    diff = a - b
    per_fold = {}
    for f in fold_list:
        sel = np.isin(dates, eng_w.data.dates[eng_w.fold_idx[f]])
        if sel.any():
            per_fold[str(f)] = {"n": int(sel.sum()),
                                "mean": float(diff[sel].mean()),
                                "hac_t": st.hac_tstat(diff[sel])}
    return {
        "with": side_with.label, "without": side_without.label,
        "n": int(len(diff)),
        "n_with_only": int(len(rw) - len(diff)),
        "n_without_only": int(len(rwo) - len(diff)),
        "mean": float(diff.mean()) if len(diff) else float("nan"),
        "sd": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "hac_t": st.hac_tstat(diff),
        "naive_t": st.naive_t(diff),
        "ci95": st.ci95(diff),
        "mde": st.mde(diff),
        "per_fold": per_fold,
        "warnings": side_with.warnings + side_without.warnings,
        "spec": {"statistic": "pooled F1-F6 OOF daily utility difference, "
                              "HAC t (Newey-West, 10 lags), cost scenario 1.0",
                 "folds": fold_list},
    }


def save_e1(result: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
