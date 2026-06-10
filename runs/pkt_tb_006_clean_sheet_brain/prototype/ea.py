"""PKT-TB-006 — evolution (BUILD_SPEC §8; PROPOSAL_EVOLUTION machinery).

(mu+lambda) GA: P=28, G=14, elitism 2, tournament 3, uniform crossover p_swap=0.5
on 70% of offspring, float mutation sigma=0.10 annealed ×0.85/gen (p=0.30/gene),
bit-flip 0.05, early stop after 4 stagnant generations,
seed = int(sha256("PKT-TB-006-EA"+window_end)[:8], 16).
Every genome + fitness logged to ea/generation_<k>.jsonl.

Fitness: per fold f, a fast vectorized walk over cached OOF matrices using the
LOFO executive for fold f (TR S2); U_f = sqrt(252)·mean(r)/std(r) − 0.5·MaxDD/0.10;
min over cost scenarios ×{1.0, 1.5}; FITNESS = mean_f − 0.5·std_f − 0.02·#active
gates (#active = gates set to 0, i.e. actively removing a member/family);
4-of-6 fold subsample per generation, elites + champion rescored on all folds.
Fitness data ends 2026-02-06 (the ExecData is clipped upstream).

Live genes IN THE WALK: trust_prior / trust_halflife / conviction_temp /
member_gate (zero member + trust renorm) / feature_gate (zero gated z columns
into the LOFO executive) / record_weight_eps (re-weights the trust-tilt EWMA's
u contributions from the stored RAW record score) / event_weight_cap (scales the
event member's contribution to the blend and to mu_blend) / all allocator rails
(gross_target, vol_target_ann, max_symbol_weight, dd_brake_*, no_trade_band,
cash_floor, abstain_threshold). risk_aversion_lambda parameterizes the NEXT
executive-training cycle (and the B0 default already trained with it) — dormant
in the walk by construction.

Baselines: B0 DEFAULT_GENOME in the population every generation; adoption gate
champion ships only if FITNESS(champ) − FITNESS(B0) > 1.0 × cross-fold sd of the
paired per-fold differences; B1 budget-matched random search (K=366, same
folds/seeds).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

import books as bk
import executive as ex

# ---- GA constants (§8, fixed — not genes) -------------------------------------
P_DEFAULT = 28
G_DEFAULT = 14
ELITISM = 2
TOURNAMENT = 3
P_SWAP = 0.5
CROSSOVER_FRAC = 0.70
MUT_SIGMA0 = 0.10
MUT_ANNEAL = 0.85
P_MUT = 0.30
P_FLIP = 0.05
STAGNATION = 4
FOLD_SUBSAMPLE = 4
COST_SCENARIOS = (1.0, 1.5)
GATE_PENALTY = 0.02
B1_K = 366


def ea_seed(window_end: str) -> int:
    return int(hashlib.sha256(f"PKT-TB-006-EA{window_end}".encode()).hexdigest()[:8], 16)


# ---- genome (27 genes; §8.1 table verbatim) ------------------------------------
# (name, length, kind, lo, hi, b0)   kind ∈ {float, log, bit}
GENE_SPECS = [
    ("trust_prior",         3, "float", -2.0,  2.0,  0.0),
    ("trust_halflife_days", 1, "log",    5.0, 60.0, 21.0),
    ("member_gate",         3, "bit",    0,    1,    1),
    ("feature_gate",        8, "bit",    0,    1,    1),
    ("gross_target",        1, "float",  0.30, 1.00, 0.60),
    ("vol_target_ann",      1, "log",    0.06, 0.18, 0.10),
    ("max_symbol_weight",   1, "float",  0.02, 0.15, 0.08),
    ("dd_brake_threshold",  1, "float",  0.05, 0.20, 0.10),
    ("dd_brake_strength",   1, "float",  0.00, 1.00, 0.50),
    ("no_trade_band",       1, "float",  0.000, 0.030, 0.010),
    ("conviction_temp",     1, "log",    0.25, 4.0,  1.0),
    ("cash_floor",          1, "float",  0.00, 0.30, 0.10),
    ("risk_aversion_lambda", 1, "log",   0.5,  8.0,  2.0),
    ("abstain_threshold",   1, "float",  0.0,  0.5,  0.10),
    ("record_weight_eps",   1, "float",  0.20, 0.50, 0.25),
    ("event_weight_cap",    1, "float",  0.5,  2.0,  1.0),
]
N_GENES = sum(s[1] for s in GENE_SPECS)
assert N_GENES == 27, N_GENES

FEATURE_GATE_NAMES = ["G1_themes", "G2_country", "G3_tone", "G4G5_novelty_conc",
                      "LLM_block", "CBOE_S1", "FRED_S3", "COT_S2"]
# z-column indices each family touches in the executive's z[24] (forced mapping,
# logged; families with no z presence still gate member inputs at retrain time
# and still count in the gate penalty).
FEATURE_GATE_Z_MAP = {
    "G1_themes": [22], "G2_country": [], "G3_tone": [],
    "G4G5_novelty_conc": [18, 21, 23],
    "LLM_block": [14, 15, 16, 17, 19, 20],
    "CBOE_S1": [10, 11, 12, 13], "FRED_S3": [5], "COT_S2": [],
}
EVENT_MEMBER_IDX = bk.MEMBERS.index("event_head")


@dataclass
class Genome:
    trust_prior: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    trust_halflife_days: float = 21.0
    member_gate: list = field(default_factory=lambda: [1, 1, 1])
    feature_gate: list = field(default_factory=lambda: [1] * 8)
    gross_target: float = 0.60
    vol_target_ann: float = 0.10
    max_symbol_weight: float = 0.08
    dd_brake_threshold: float = 0.10
    dd_brake_strength: float = 0.50
    no_trade_band: float = 0.010
    conviction_temp: float = 1.0
    cash_floor: float = 0.10
    risk_aversion_lambda: float = 2.0
    abstain_threshold: float = 0.10
    record_weight_eps: float = 0.25
    event_weight_cap: float = 1.0

    # ---- (de)serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(**{k: d[k] for k in cls().to_dict()})

    def to_json(self, path: Path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def from_json(cls, path: Path) -> "Genome":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    # ---- flat encoding: floats normalized to [0,1] (log genes in log space) ----
    def to_vector(self) -> tuple[np.ndarray, np.ndarray]:
        floats, bits = [], []
        for name, ln, kind, lo, hi, _ in GENE_SPECS:
            v = getattr(self, name)
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if kind == "bit":
                    bits.append(int(x))
                elif kind == "log":
                    floats.append((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)))
                else:
                    floats.append((x - lo) / (hi - lo))
        return np.asarray(floats, dtype=np.float64), np.asarray(bits, dtype=np.int64)

    @classmethod
    def from_vector(cls, floats: np.ndarray, bits: np.ndarray) -> "Genome":
        g = cls()
        fi, bi = 0, 0
        for name, ln, kind, lo, hi, _ in GENE_SPECS:
            vals = []
            for _ in range(ln):
                if kind == "bit":
                    vals.append(int(np.clip(bits[bi], 0, 1)))
                    bi += 1
                else:
                    x = float(np.clip(floats[fi], 0.0, 1.0))
                    fi += 1
                    if kind == "log":
                        vals.append(float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))))
                    else:
                        vals.append(float(lo + x * (hi - lo)))
            setattr(g, name, vals if ln > 1 else vals[0])
        return g

    @classmethod
    def b0(cls) -> "Genome":
        return cls()

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        nf = sum(s[1] for s in GENE_SPECS if s[2] != "bit")
        nb = sum(s[1] for s in GENE_SPECS if s[2] == "bit")
        return cls.from_vector(rng.random(nf), (rng.random(nb) < 0.85).astype(int))

    def n_active_gates(self) -> int:
        return int(sum(1 - g for g in self.member_gate) + sum(1 - g for g in self.feature_gate))


# ============================ fitness engine ===================================
class FitnessEngine:
    """Vectorized fold walk over cached OOF matrices through the LOFO executives."""

    def __init__(self, data: ex.ExecData, lofo: dict[int, torch.nn.Module],
                 fold_list: list[int] | None = None):
        self.data = data
        self.lofo = lofo
        self.fold_list = fold_list or sorted(lofo.keys())
        self.fold_idx = {f: np.where(data.fold_of == f)[0] for f in self.fold_list}
        self._gate_cache: dict = {}
        self._u_std = {f: max(float(np.std(data.u_real[self.fold_idx[f]])), 1e-6)
                       for f in self.fold_list}

    # ---- per (fold, gate-pattern) cache: exec logits + psi static part --------
    def _cached(self, f: int, gate: tuple) -> dict:
        key = (f, gate)
        if key in self._gate_cache:
            return self._gate_cache[key]
        d, idx = self.data, self.fold_idx[f]
        zmask = np.ones(ex.N_Z)
        for gname, on in zip(FEATURE_GATE_NAMES, gate):
            if not on:
                zmask[FEATURE_GATE_Z_MAP[gname]] = 0.0
        model = self.lofo[f]
        t = ex.ExecTensors(d, idx)
        zg = t.z * torch.as_tensor(zmask, dtype=torch.float32)
        with torch.no_grad():
            zz = zg.unsqueeze(1).expand(-1, ex.N_MEMBERS, -1)
            x_tr = torch.cat([t.r, t.c.unsqueeze(-1), t.agree.unsqueeze(-1), zz], dim=-1)
            _, s = model.trust(x_tr)
            T_exec = float(torch.exp(model.log_T).clamp(0.1, 10.0))
            # psi decomposition: hidden = act(A_static + W_mu·mustats + w_ent·ent + w_dd·dd)
            # act = tanh for the Executive MLP; identity for the LinearGate twin
            # (raw = x_psi·c + d decomposes into the same static/dynamic parts,
            # so the SAME walk serves both gate classes — required since the
            # SYN-1 freeze ships the linear twin as the deployed gate and the
            # E1 reads must walk the same gate class; EA fitness walks remain
            # MLP-LOFO per TR S2/FREEZE, unchanged).
            if isinstance(model, ex.Executive):
                act = "tanh"
                W1 = model.psi1.weight.detach().numpy()          # [8,35]
                b1 = model.psi1.bias.detach().numpy()
                v2 = model.psi2.weight.detach().numpy()[0]
                b2 = float(model.psi2.bias.detach().numpy()[0])
            elif isinstance(model, ex.LinearGate):
                act = "linear"
                W1 = model.c.detach().numpy()[None, :]           # [1,35]
                b1 = model.d.detach().numpy().reshape(1)
                v2 = np.ones(1)
                b2 = 0.0
            else:
                raise NotImplementedError(
                    f"FitnessEngine cannot walk a {type(model).__name__}")
            gain = float(model.psi_out_gain.detach())
            bias = float(model.psi_out_bias.detach()) + float(model.f_sigmoid_bias.detach())
        led = np.stack([d.r[idx, :, 0].mean(1), d.r[idx, :, 1].mean(1)], axis=1)
        x_static = np.zeros((len(idx), ex.PSI_IN))
        x_static[:, 0:24] = zg.numpy()
        x_static[:, 24:27] = d.g[idx]
        x_static[:, 31:33] = led
        x_static[:, 33] = d.book_vol_hat[idx]
        A_static = x_static @ W1.T + b1                          # [n,8]
        cache = {"s": s.numpy(), "T_exec": T_exec, "A_static": A_static,
                 "W_mu": W1[:, 27:30], "w_ent": W1[:, 30], "w_dd": W1[:, 34],
                 "v2": v2, "b2": b2, "gain": gain, "bias": bias, "act": act,
                 "books": d.books[idx], "mu": d.mu[idx],
                 "fwd1": d.rets5[idx, 0, :], "sigma_hat": d.sigma_hat[idx],
                 "u_real": d.u_real[idx], "w_rec_raw": d.w_rec_raw[idx]}
        self._gate_cache[key] = cache
        return cache

    # ---- the walk ---------------------------------------------------------------
    def fold_utility(self, genome: Genome, f: int) -> float:
        c = self._cached(f, tuple(genome.feature_gate))
        n = c["s"].shape[0]
        hs = self.data.half_spread
        # trust tilt: lagged EWMA of eps-record-weighted counterfactual utility
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
        # trust: exec logits + genome prior + tilt, genome temperature, floor, gates
        logits = c["s"] + np.asarray(genome.trust_prior) + tilt
        T = c["T_exec"] * genome.conviction_temp
        e = np.exp(logits / T - logits.max(axis=1, keepdims=True) / T)
        tau = e / e.sum(axis=1, keepdims=True)
        tau = (1 - ex.EPS_TAU) * tau + ex.EPS_TAU / 3
        mg = np.asarray(genome.member_gate, dtype=float)
        tau = tau * mg
        norm = tau.sum(axis=1, keepdims=True)
        tau = np.divide(tau, norm, out=np.zeros_like(tau), where=norm > 1e-12)
        # blend with event_weight_cap on the event member
        capf = np.ones(3)
        capf[EVENT_MEMBER_IDX] = genome.event_weight_cap
        coef = tau * capf
        cnorm = coef.sum(axis=1, keepdims=True)
        coef_n = np.divide(coef, cnorm, out=np.zeros_like(coef), where=cnorm > 1e-12)
        w_unit = np.einsum("nm,nms->ns", coef_n, c["books"])
        mu_blend = np.einsum("nm,nms->ns", coef_n, c["mu"])
        am = np.abs(mu_blend)
        mustats = np.stack([am.mean(1), am.std(1), am.max(1)], axis=1)
        ent = -(tau * np.log(tau + 1e-12)).sum(axis=1)
        pre = c["A_static"] + mustats @ c["W_mu"].T + ent[:, None] * c["w_ent"][None, :]
        rs = {sc: np.zeros(n) for sc in COST_SCENARIOS}
        w_prev = {sc: np.zeros(64) for sc in COST_SCENARIOS}
        equity = {sc: 0.0 for sc in COST_SCENARIOS}
        peak = {sc: 0.0 for sc in COST_SCENARIOS}
        maxdd = {sc: 0.0 for sc in COST_SCENARIOS}
        for i in range(n):
            for sc in COST_SCENARIOS:
                dd = peak[sc] - equity[sc]
                h_pre = pre[i] + c["w_dd"] * dd
                hidden = np.tanh(h_pre) if c["act"] == "tanh" else h_pre
                f_dep = 1.0 / (1.0 + np.exp(-(c["gain"] * (hidden @ c["v2"] + c["b2"])
                                              + c["bias"])))
                w = f_dep * w_unit[i]
                gross = w.sum()
                gmax = min(genome.gross_target, 1.0 - genome.cash_floor)
                if gross > gmax > 0:
                    w = w * (gmax / gross)
                w = np.minimum(w, genome.max_symbol_weight)
                vol = w @ self.data.sigma_hat[self.fold_idx[f][i]]
                if vol > genome.vol_target_ann > 0:
                    w = w * (genome.vol_target_ann / vol)
                if dd > genome.dd_brake_threshold:
                    brake = 1.0 - genome.dd_brake_strength * min(
                        1.0, (dd - genome.dd_brake_threshold) / max(genome.dd_brake_threshold, 1e-6))
                    w = w * brake
                if am[i].mean() < genome.abstain_threshold:
                    w_emit = w_prev[sc]
                    traded = np.zeros(64)
                else:
                    delta = w - w_prev[sc]
                    mask = np.abs(delta) > genome.no_trade_band
                    traded = delta * mask
                    w_emit = w_prev[sc] + traded
                cost = np.abs(traded) @ hs * sc / 1e4
                r = float(w_emit @ c["fwd1"][i]) - cost
                rs[sc][i] = r
                equity[sc] += np.log1p(max(r, -0.99))
                peak[sc] = max(peak[sc], equity[sc])
                maxdd[sc] = max(maxdd[sc], peak[sc] - equity[sc])
                w_prev[sc] = w_emit
        us = []
        for sc in COST_SCENARIOS:
            r = rs[sc]
            sd = r.std()
            sharpe = np.sqrt(252) * r.mean() / sd if sd > 1e-12 else 0.0
            mdd_frac = 1.0 - np.exp(-maxdd[sc])
            us.append(sharpe - 0.5 * mdd_frac / 0.10)
        return float(min(us))

    def fitness(self, genome: Genome, folds_used: list[int]) -> dict:
        u = {f: self.fold_utility(genome, f) for f in folds_used}
        vals = np.array(list(u.values()))
        fit = float(vals.mean() - 0.5 * vals.std() - GATE_PENALTY * genome.n_active_gates())
        return {"fitness": fit, "fold_u": {str(f): float(v) for f, v in u.items()},
                "n_active_gates": genome.n_active_gates()}


def adoption_gate(champ_res: dict, b0_res: dict, fold_list: list[int]) -> dict:
    """§8 adoption gate: champion ships only if FITNESS(champ) − FITNESS(B0)
    > 1.0 × cross-fold sd (sd of the PAIRED per-fold utility differences)."""
    diffs = np.array([champ_res["fold_u"][str(f)] - b0_res["fold_u"][str(f)]
                      for f in fold_list])
    sd = float(diffs.std(ddof=0))
    margin = float(champ_res["fitness"] - b0_res["fitness"])
    return {"margin": margin, "cross_fold_sd": sd, "adopted": bool(margin > 1.0 * sd)}


# ================================ GA ==========================================
def _tournament_pick(rng, scored: list[dict]) -> dict:
    picks = [scored[rng.integers(len(scored))] for _ in range(TOURNAMENT)]
    return max(picks, key=lambda r: r["fitness"])


def _crossover(rng, a: Genome, b: Genome) -> Genome:
    fa, ba = a.to_vector()
    fb, bb = b.to_vector()
    fm = rng.random(len(fa)) < P_SWAP
    bm = rng.random(len(ba)) < P_SWAP
    return Genome.from_vector(np.where(fm, fb, fa), np.where(bm, bb, ba))


def _mutate(rng, g: Genome, sigma: float) -> Genome:
    fv, bv = g.to_vector()
    mut = rng.random(len(fv)) < P_MUT
    fv = np.clip(fv + mut * rng.normal(0, sigma, len(fv)), 0.0, 1.0)
    flips = rng.random(len(bv)) < P_FLIP
    bv = np.where(flips, 1 - bv, bv)
    return Genome.from_vector(fv, bv)


def run_ea(engine: FitnessEngine, out_dir: Path, window_end: str,
           P: int = P_DEFAULT, G: int = G_DEFAULT,
           fold_subsample: int = FOLD_SUBSAMPLE, verbose: bool = True) -> dict:
    """Full GA + B0-every-generation + adoption gate. Logs ea/generation_<k>.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ea_seed(window_end))
    all_folds = engine.fold_list
    b0 = Genome.b0()
    pop = [b0] + [Genome.random(rng) for _ in range(P - 1)]
    sigma = MUT_SIGMA0
    champion, champ_fit_full, stagnant = None, -np.inf, 0
    history = []
    for gen in range(G):
        k = min(fold_subsample, len(all_folds))
        folds_used = sorted(rng.choice(all_folds, size=k, replace=False).tolist())
        scored = []
        for gnm in pop:
            res = engine.fitness(gnm, folds_used)
            scored.append({"genome": gnm, **res})
        scored.sort(key=lambda r: r["fitness"], reverse=True)
        # elites + champion + B0 rescored on ALL folds
        full_scores = {}
        rescore = scored[:ELITISM] + [r for r in scored if r["genome"] is b0]
        for r in rescore:
            fr = engine.fitness(r["genome"], all_folds)
            full_scores[id(r["genome"])] = fr
            r["fitness_full"] = fr["fitness"]
            r["fold_u_full"] = fr["fold_u"]
        gen_best = scored[0]
        best_full = gen_best.get("fitness_full", -np.inf)
        if champion is not None:
            ch = engine.fitness(champion, all_folds)
            champ_fit_full = ch["fitness"]
        if best_full > champ_fit_full:
            champion, champ_fit_full = gen_best["genome"], best_full
            stagnant = 0
        else:
            stagnant += 1
        with open(out_dir / f"generation_{gen}.jsonl", "w") as fh:
            for r in scored:
                fh.write(json.dumps({
                    "gen": gen, "folds_used": folds_used, "sigma": sigma,
                    "fitness": r["fitness"], "fold_u": r["fold_u"],
                    "fitness_full": r.get("fitness_full"),
                    "n_active_gates": r["n_active_gates"],
                    "is_b0": r["genome"] is b0,
                    "genome": r["genome"].to_dict()}) + "\n")
        if verbose:
            print(f"  gen {gen:2d} folds {folds_used} best {scored[0]['fitness']:+.4f} "
                  f"champ(full) {champ_fit_full:+.4f} stagnant {stagnant}")
        if stagnant >= STAGNATION:
            history.append({"gen": gen, "early_stop": True})
            break
        # next generation: elites + offspring; B0 re-inserted EVERY generation
        nxt = [r["genome"] for r in scored[:ELITISM]]
        while len(nxt) < P - 1:
            pa = _tournament_pick(rng, scored)["genome"]
            if rng.random() < CROSSOVER_FRAC:
                pb = _tournament_pick(rng, scored)["genome"]
                child = _crossover(rng, pa, pb)
            else:
                child = pa
            nxt.append(_mutate(rng, child, sigma))
        nxt.append(b0)
        pop = nxt
        sigma *= MUT_ANNEAL
        history.append({"gen": gen, "best": scored[0]["fitness"], "sigma": sigma})
    # adoption gate (paired cross-fold sd)
    champ_res = engine.fitness(champion, all_folds)
    b0_res = engine.fitness(b0, all_folds)
    gate = adoption_gate(champ_res, b0_res, all_folds)
    margin, sd, adopted = gate["margin"], gate["cross_fold_sd"], gate["adopted"]
    ship = champion if adopted else b0
    result = {
        "window_end": window_end, "seed": ea_seed(window_end),
        "champion_fitness": champ_res["fitness"], "b0_fitness": b0_res["fitness"],
        "champion_fold_u": champ_res["fold_u"], "b0_fold_u": b0_res["fold_u"],
        "margin": margin, "cross_fold_sd": sd, "adopted_champion": adopted,
        "shipped": "champion" if adopted else "B0",
        "champion": champion.to_dict(), "b0": b0.to_dict(),
        "shipped_genome": ship.to_dict(), "history": history,
    }
    ship.to_json(out_dir / f"genome_{window_end}.json")
    with open(out_dir / f"ea_manifest_{window_end}.json", "w") as fh:
        json.dump(result, fh, indent=1)
    return result


def run_random_baseline(engine: FitnessEngine, window_end: str, K: int = B1_K,
                        out_path: Path | None = None) -> dict:
    """B1: budget-matched random search, same seed stream / folds."""
    rng = np.random.default_rng(ea_seed(window_end) ^ 0xB1B1B1)
    all_folds = engine.fold_list
    best, best_fit = None, -np.inf
    for _ in range(K):
        g = Genome.random(rng)
        r = engine.fitness(g, all_folds)
        if r["fitness"] > best_fit:
            best, best_fit = g, r["fitness"]
    res = {"K": K, "best_fitness": best_fit, "best_genome": best.to_dict()}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


# ----------------------------- entrypoint --------------------------------------
def main():
    """Final-run entrypoint (after final member OOFs + LOFO executives exist):

        .venv/bin/python ea.py --oof-dir oof --exec-dir exec_out --window-end 2026-02-06
    """
    import argparse
    import folds as fd
    import synth_oof as so
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default="oof")
    ap.add_argument("--exec-dir", default="exec_out")
    ap.add_argument("--window-end", default=fd.FITNESS_END)
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--b1", action="store_true", help="also run the B1 random baseline")
    args = ap.parse_args()
    proto = Path(__file__).resolve().parent
    fold_list = [int(x) for x in args.folds.split(",")]
    world = so.world_from_panel(proto / "store" / "panel.npz")
    data = ex.assemble_exec_data(world, proto / args.oof_dir, fold_list,
                                 end_date=fd.FITNESS_END)
    lofo = {}
    for f in fold_list:
        m = ex.Executive()
        m.load_state_dict(torch.load(proto / args.exec_dir / f"lofo_fold{f}.pt"))
        m.eval()
        lofo[f] = m
    engine = FitnessEngine(data, lofo, fold_list)
    res = run_ea(engine, proto / "ea", args.window_end)
    print(json.dumps({k: res[k] for k in
                      ("champion_fitness", "b0_fitness", "margin", "cross_fold_sd",
                       "adopted_champion", "shipped")}, indent=1))
    if args.b1:
        b1 = run_random_baseline(engine, args.window_end,
                                 out_path=proto / "ea" / f"b1_{args.window_end}.json")
        print("B1 best:", b1["best_fitness"])


if __name__ == "__main__":
    main()
