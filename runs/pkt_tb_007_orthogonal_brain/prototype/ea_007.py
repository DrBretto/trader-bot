"""PKT-TB-007 — ORB-1 evolution (BUILD_SPEC_007 §4; TOURNAMENT_007 §4.4.3–4.4.6).

The (μ+λ) GA carried from TB-006 ea.py machinery, retargeted:
  genome   12 genes max (genome_007.Genome007; organ_trust genes exist ONLY for
           shipped organs; B0 = neutral = bit-for-bit incumbent).
  fitness  floored paired IR over the tilt series (§4.2, verbatim
           PROPOSAL_EVOLUTION_007 §1): per fold f, cost scenario c ∈ {1.0,1.5},
           U_f,c = √252·mean(Δr)/max(sd(Δr), 2bp/day) on the PAIRED daily
           brain-vs-neutral differences through THE SAME projection/tilt code
           path as the harness adapter (tilt_adapter.solve_tilt — one
           implementation); U_f = min_c; FITNESS = mean_f U_f
           − 1.0·max(0, −min_f U_f) − 0.05·Σ(g_i − g_B0,i)² (unit² gene space).
           FITNESS(B0) ≡ 0 by construction — asserted at engine init.
  search   production: P=28 G=14 K≤400, 4-of-6 fold subsampling, elitism 2,
           tournament 3, uniform crossover p=0.5 on 70%, σ=0.10×0.85^gen,
           p_mut=0.30, B0 = individual #0 every generation, early stop 4.
  champion median-of-top-8 full-fold-rescored (L2-dedupe < 0.05 unit space),
           re-evaluated fresh; if below the best max-min-over-folds top-8
           member, that member ships.
  gates    adoption: champion > max(1.4, √(2·ln K_eff)) × cross-fold sd of its
           U_f (K_eff = L2-deduped distinct genomes evaluated, printed).
           rotation (§4.4.4): 6 reruns (P=28 G=8), fold f FULLY withheld from
           fitness/subsampling/elite re-scoring; per-rotation masks + roster
           consumed from the wave-2a contract files; pass iff ≥4/6 unseen-fold
           ΔU_f ≥ 0 AND pooled unseen mean > 1×se. Gates are AND; B0 ships on
           any failure. Boundary-pin audit (FB2) in the champion manifest.
  controls B1 budget-matched uniform random search (same fitness/folds, seeded).
           Blend-0 sensitivity = one reduced-budget rerun (--blend0; needs
           blend-0 packs built first).
  logging  every variant (rotations included) to ea/<run>/generation_<k>.jsonl;
           deterministic from master seed 4242; wall cap with the §8 shrink
           ladder (rotation G → production G → P; rotations never dropped).

WAVE-2A CONTRACT (consumed here; synthetic stand-ins are flagged everywhere):
  store/nightly_007_folds/<D>.json       organ_inputs_007.v1, OOF-true per date
  store/rotation_masks_007/rotation_<f>.json
      {rotation, withheld_fold, derived_from_folds, roster, masks[, synthetic]}
  store/roster_007.json                  {"roster": ["M1", ...]} (production)

ONE-COMMAND PRODUCTION ENTRYPOINT (orchestrator; AFTER organs land + anchor PASS):
  .venv/bin/python ea_007.py --mode full \
      --fold-nightly-dir store/nightly_007_folds \
      --rotation-masks-dir store/rotation_masks_007 \
      --roster-file store/roster_007.json
SMOKE (synthetic organs; what wave-2b ran):
  .venv/bin/python ea_007.py --mode smoke
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB006_PROTO = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
for p in (str(PROTO), str(REPO), str(TB006_PROTO)):
    if p not in sys.path:
        sys.path.append(p)

import tilt_adapter as TA                                          # noqa: E402
from genome_007 import (Genome007, GENE_RANGES, ORGAN_TRUST_RANGE,  # noqa: E402
                        DIRECTIONAL_ORGANS)
import surrogate_007 as SG                                          # noqa: E402

# ---- GA constants (§4.3, FIXED — never genes) ----------------------------------
P_PROD, G_PROD = 28, 14
P_ROT, G_ROT = 28, 8
K_CAP = 400
ELITISM = 2
TOURNAMENT = 3
P_SWAP = 0.5
CROSSOVER_FRAC = 0.70
MUT_SIGMA0 = 0.10
MUT_ANNEAL = 0.85
P_MUT = 0.30
STAGNATION = 4
FOLD_SUBSAMPLE = 4
LAMBDA_REG = 0.05
DEDUPE_L2 = 0.05
TOP_K = 8
MASTER_SEED = 4242
WALL_CAP_MIN_DEFAULT = 90.0
EA_DIR = PROTO / "ea"
LOG_GENES = ["conviction_temp"]        # log-encoded floats; all others linear


def run_seed(tag: str) -> int:
    h = hashlib.sha256(f"PKT-TB-007-EA::{MASTER_SEED}::{tag}".encode()).hexdigest()
    return int(h[:8], 16)


# ================================ unit-space encoding ===========================
def gene_names(roster: Sequence[str]) -> List[str]:
    return ([f"organ_trust.{o}" for o in sorted(roster)]
            + list(GENE_RANGES.keys()))


def gene_bounds(name: str):
    if name.startswith("organ_trust."):
        lo, hi, b0 = ORGAN_TRUST_RANGE
    else:
        lo, hi, b0 = GENE_RANGES[name]
    return lo, hi, b0


def to_unit(genome: Genome007, roster: Sequence[str]) -> np.ndarray:
    out = []
    for nm in gene_names(roster):
        lo, hi, _ = gene_bounds(nm)
        if nm.startswith("organ_trust."):
            v = genome.organ_trust[nm.split(".", 1)[1]]
        else:
            v = getattr(genome, nm)
        if nm in LOG_GENES:
            out.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
        else:
            out.append((v - lo) / (hi - lo))
    return np.asarray(out, dtype=np.float64)


def from_unit(vec: np.ndarray, roster: Sequence[str]) -> Genome007:
    trust: Dict[str, float] = {}
    kwargs: Dict[str, float] = {}
    for x, nm in zip(vec, gene_names(roster)):
        lo, hi, _ = gene_bounds(nm)
        x = float(np.clip(x, 0.0, 1.0))
        v = (float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
             if nm in LOG_GENES else float(lo + x * (hi - lo)))
        if nm.startswith("organ_trust."):
            trust[nm.split(".", 1)[1]] = v
        else:
            kwargs[nm] = v
    return Genome007(organ_trust=trust, **kwargs)


def b0_unit(roster: Sequence[str]) -> np.ndarray:
    return to_unit(Genome007.b0(tuple(roster)), roster)


def random_genome(rng: np.random.Generator, roster: Sequence[str]) -> Genome007:
    return from_unit(rng.random(len(gene_names(roster))), roster)


def boundary_pin_fraction(genome: Genome007, roster: Sequence[str],
                          tol: float = 0.05) -> dict:
    """FB2: fraction of genes within `tol` of a range edge (unit space)."""
    u = to_unit(genome, roster)
    pinned = [nm for nm, x in zip(gene_names(roster), u)
              if x <= tol or x >= 1 - tol]
    frac = len(pinned) / len(u)
    return {"fraction_pinned": round(frac, 4), "pinned_genes": pinned,
            "noise_flag": bool(frac > 1 / 3),
            "rule": "FB2: >1/3 of genes within 5% of an edge => noise flag"}


# ================================ fitness engine ================================
class FitnessEngine007:
    """Floored paired-IR fitness over cached day-packs through the shared
    tilt_adapter solve path (surrogate_007.walk_fold_paired)."""

    def __init__(self, packs: Dict[int, dict], organs_dir: Path,
                 masks: Dict[str, Dict[str, float]], roster: Sequence[str],
                 fold_list: Optional[List[int]] = None):
        self.packs = packs
        self.organs_dir = Path(organs_dir)
        self.masks = masks
        self.roster = tuple(sorted(roster))
        self.fold_list = sorted(fold_list or packs.keys())
        self._b0u = b0_unit(self.roster)
        self._cache: Dict[tuple, dict] = {}      # (hash, fold) -> walk result
        self.n_evals = 0
        # FITNESS(B0) ≡ 0 by construction — assert through the real walk
        b0 = Genome007.b0(self.roster)
        r = self.fitness(b0, self.fold_list)
        assert r["fitness"] == 0.0 and all(v == 0.0 for v in r["fold_u"].values()), \
            f"FITNESS(B0) != 0: {r}"

    def fold_utility(self, genome: Genome007, f: int) -> float:
        key = (genome.hash(), f)
        if key not in self._cache:
            w = SG.walk_fold_paired(self.packs[f], genome, self.organs_dir,
                                    self.masks)
            self._cache[key] = w
            self.n_evals += 1
        w = self._cache[key]
        return SG.fold_U(w["dr_gross"], w["cost1"])

    def shrinkage(self, genome: Genome007) -> float:
        d = to_unit(genome, self.roster) - self._b0u
        return float(LAMBDA_REG * (d @ d))

    def fitness(self, genome: Genome007, folds_used: List[int]) -> dict:
        u = {f: self.fold_utility(genome, f) for f in folds_used}
        vals = np.array(list(u.values()))
        fit = float(vals.mean() - 1.0 * max(0.0, -vals.min())
                    - self.shrinkage(genome))
        return {"fitness": fit,
                "fold_u": {str(f): float(v) for f, v in u.items()},
                "shrinkage": round(self.shrinkage(genome), 6)}


# ================================ gates =========================================
def adoption_gate(champ_res: dict, k_eff: int) -> dict:
    """§4.4.3: champion ships only if FITNESS(champ) − FITNESS(B0) >
    max(1.4, √(2·ln K_eff)) × cross-fold sd of the champion's U_f.
    FITNESS(B0) ≡ 0 so margin = champion fitness."""
    u = np.array(list(champ_res["fold_u"].values()), dtype=float)
    sd = float(u.std(ddof=1)) if len(u) > 1 else 0.0
    bar_mult = max(1.4, float(np.sqrt(2 * np.log(max(k_eff, 2)))))
    margin = float(champ_res["fitness"])
    bar = bar_mult * sd
    return {"margin": margin, "cross_fold_sd": sd, "k_eff": int(k_eff),
            "bar_multiplier": round(bar_mult, 4), "bar": round(bar, 6),
            "adopted": bool(margin > bar)}


def rotation_gate(delta_u: Dict[int, float]) -> dict:
    """§4.4.4: ≥4/6 unseen-fold ΔU_f ≥ 0 AND pooled unseen mean > 1×se."""
    v = np.array([delta_u[f] for f in sorted(delta_u)], dtype=float)
    n_nonneg = int((v >= 0).sum())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("inf")
    mean = float(v.mean())
    return {"delta_u": {str(f): float(delta_u[f]) for f in sorted(delta_u)},
            "n_nonneg": n_nonneg, "n_rotations": len(v),
            "pooled_mean": round(mean, 6), "pooled_se": round(se, 6),
            "passed": bool(n_nonneg >= 4 and mean > 1.0 * se),
            "rule": ">=4/6 dU>=0 AND pooled mean > 1x se (surrogate space)"}


def dedupe_count(unit_vecs: List[np.ndarray], tol: float = DEDUPE_L2) -> int:
    """K_eff: L2-deduped distinct genomes (greedy cluster count)."""
    reps: List[np.ndarray] = []
    for v in unit_vecs:
        if not any(np.linalg.norm(v - r) < tol for r in reps):
            reps.append(v)
    return len(reps)


# ================================ GA ============================================
def _tournament_pick(rng, scored):
    picks = [scored[rng.integers(len(scored))] for _ in range(TOURNAMENT)]
    return max(picks, key=lambda r: r["fitness"])


def _crossover(rng, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = rng.random(len(a)) < P_SWAP
    return np.where(m, b, a)


def _mutate(rng, v: np.ndarray, sigma: float) -> np.ndarray:
    mut = rng.random(len(v)) < P_MUT
    return np.clip(v + mut * rng.normal(0, sigma, len(v)), 0.0, 1.0)


def select_champion(engine: FitnessEngine007, archive: List[dict],
                    rng: np.random.Generator) -> dict:
    """Median-of-top-8 rule (§4.3): full-fold-rescore the leading archive,
    L2-dedupe < 0.05, per-gene median, fresh evaluation; ship the best
    max-min-over-folds top-8 member instead if the median scores below it."""
    roster = engine.roster
    folds = engine.fold_list
    # full-fold rescore the best ~3*TOP_K by recorded (subsample) fitness
    pool = sorted(archive, key=lambda r: r["fitness"], reverse=True)[:3 * TOP_K]
    seen = set()
    rescored = []
    for r in pool:
        g = Genome007.from_dict(r["genome"])
        if g.hash() in seen:
            continue
        seen.add(g.hash())
        fr = engine.fitness(g, folds)
        rescored.append({"genome": g, "unit": to_unit(g, roster), **fr})
    rescored.sort(key=lambda r: r["fitness"], reverse=True)
    # L2-dedupe, keep best of each cluster, take top 8
    top8 = []
    for r in rescored:
        if not any(np.linalg.norm(r["unit"] - t["unit"]) < DEDUPE_L2
                   for t in top8):
            top8.append(r)
        if len(top8) == TOP_K:
            break
    med = from_unit(np.median(np.stack([t["unit"] for t in top8]), axis=0),
                    roster)
    med_res = engine.fitness(med, folds)
    # best max-min-over-folds member of the top-8
    mm = max(top8, key=lambda r: min(r["fold_u"].values()))
    if med_res["fitness"] >= mm["fitness"]:
        choice, choice_res, rule = med, med_res, "median_of_top8"
    else:
        choice, choice_res, rule = mm["genome"], \
            {k: mm[k] for k in ("fitness", "fold_u", "shrinkage")}, \
            "best_maxmin_top8_member"
    return {"champion": choice, "result": choice_res, "rule": rule,
            "top8": [{"hash": t["genome"].hash(), "fitness": t["fitness"],
                      "fold_u": t["fold_u"]} for t in top8],
            "median_result": med_res}


def run_ga(engine: FitnessEngine007, out_dir: Path, tag: str,
           P: int, G: int, fold_subsample: int = FOLD_SUBSAMPLE,
           verbose: bool = True) -> dict:
    """One GA run (production or rotation): B0 = individual #0 every
    generation; every variant logged; archive returned for championization."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(run_seed(tag))
    roster = engine.roster
    all_folds = engine.fold_list
    b0 = Genome007.b0(roster)
    pop: List[Genome007] = [b0] + [random_genome(rng, roster)
                                   for _ in range(P - 1)]
    sigma = MUT_SIGMA0
    archive: List[dict] = []
    unit_log: List[np.ndarray] = []
    best_full, stagnant = -np.inf, 0
    n_genome_evals = 0
    for gen in range(G):
        k = min(fold_subsample, len(all_folds))
        folds_used = sorted(rng.choice(all_folds, size=k,
                                       replace=False).tolist())
        scored = []
        for gnm in pop:
            res = engine.fitness(gnm, folds_used)
            n_genome_evals += 1
            rec = {"genome": gnm, "is_b0": gnm.to_dict() == b0.to_dict(), **res}
            scored.append(rec)
        scored.sort(key=lambda r: r["fitness"], reverse=True)
        # elites + B0 rescored on ALL training folds
        for r in scored[:ELITISM] + [r for r in scored if r["is_b0"]]:
            fr = engine.fitness(r["genome"], all_folds)
            r["fitness_full"] = fr["fitness"]
            r["fold_u_full"] = fr["fold_u"]
        gen_best_full = scored[0].get("fitness_full", -np.inf)
        if gen_best_full > best_full + 1e-12:
            best_full, stagnant = gen_best_full, 0
        else:
            stagnant += 1
        with open(out_dir / f"generation_{gen}.jsonl", "w") as fh:
            for r in scored:
                fh.write(json.dumps({
                    "tag": tag, "gen": gen, "folds_used": folds_used,
                    "sigma": round(sigma, 5), "fitness": r["fitness"],
                    "fold_u": r["fold_u"], "shrinkage": r["shrinkage"],
                    "fitness_full": r.get("fitness_full"),
                    "is_b0": r["is_b0"], "genome_hash": r["genome"].hash(),
                    "genome": r["genome"].to_dict()}) + "\n")
        for r in scored:
            archive.append({"fitness": r["fitness"], "fold_u": r["fold_u"],
                            "genome": r["genome"].to_dict()})
            unit_log.append(to_unit(r["genome"], roster))
        if verbose:
            b0row = next(r for r in scored if r["is_b0"])
            print(f"  [{tag}] gen {gen:2d} folds {folds_used} "
                  f"best {scored[0]['fitness']:+.4f} "
                  f"b0 {b0row['fitness']:+.4f} "
                  f"best_full {best_full:+.4f} stagnant {stagnant}")
        if stagnant >= STAGNATION:
            break
        nxt = [r["genome"] for r in scored[:ELITISM]]
        while len(nxt) < P - 1:
            pa = to_unit(_tournament_pick(rng, scored)["genome"], roster)
            if rng.random() < CROSSOVER_FRAC:
                pb = to_unit(_tournament_pick(rng, scored)["genome"], roster)
                child = _crossover(rng, pa, pb)
            else:
                child = pa
            nxt.append(from_unit(_mutate(rng, child, sigma), roster))
        nxt.append(b0)
        pop = nxt
        sigma *= MUT_ANNEAL
    champ = select_champion(engine, archive, rng)
    k_eff = dedupe_count(unit_log)
    return {"tag": tag, "seed": run_seed(tag), "P": P, "G": G,
            "n_genome_evals": n_genome_evals, "k_eff": k_eff,
            "archive_n": len(archive), "champion": champ}


def run_b1(engine: FitnessEngine007, K: int, tag: str = "b1") -> dict:
    """B1: budget-matched uniform random search, same fitness/folds, seeded."""
    rng = np.random.default_rng(run_seed(tag) ^ 0xB1B1B1)
    best, best_res = None, {"fitness": -np.inf}
    for _ in range(K):
        g = random_genome(rng, engine.roster)
        r = engine.fitness(g, engine.fold_list)
        if r["fitness"] > best_res["fitness"]:
            best, best_res = g, r
    return {"K": K, "best_fitness": best_res["fitness"],
            "best_fold_u": best_res["fold_u"], "best_genome": best.to_dict()}


# ================================ rotations =====================================
def load_rotation_spec(masks_dir: Path, f: int) -> dict:
    p = Path(masks_dir) / f"rotation_{f}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"rotation masks missing: {p} — wave-2a contract file "
            f"(store/rotation_masks_007/rotation_<f>.json); for machinery "
            f"smoke generate synthetic ones via "
            f"`surrogate_007.py --synth-masks`")
    doc = json.loads(p.read_text())
    assert doc["withheld_fold"] == f
    return doc


def run_rotations(packs: Dict[int, dict], organs_dir: Path, masks_dir: Path,
                  out_root: Path, P: int = P_ROT, G: int = G_ROT,
                  rotations: Sequence[int] = (1, 2, 3, 4, 5, 6)) -> dict:
    """§4.4.4/§4.4.6: per rotation, fold f fully excluded from fitness,
    subsampling and elite re-scoring; rotation champion evaluated on the
    unseen fold under that rotation's own masks/roster; ΔU_f = U_f(champ)
    − U_f(B0) = U_f(champ) since FITNESS(B0) ≡ 0."""
    delta_u: Dict[int, float] = {}
    per_rot = {}
    any_synth = False
    for f in rotations:
        spec = load_rotation_spec(masks_dir, f)
        any_synth |= bool(spec.get("synthetic"))
        train_folds = [k for k in packs if k != f]
        engine = FitnessEngine007(packs, organs_dir, spec["masks"],
                                  spec["roster"], fold_list=train_folds)
        res = run_ga(engine, out_root / f"rotation_{f}", f"rotation_{f}",
                     P=P, G=G,
                     fold_subsample=min(FOLD_SUBSAMPLE, len(train_folds)))
        champ = res["champion"]["champion"]
        # unseen-fold evaluation under THIS rotation's masks/roster
        unseen_engine = FitnessEngine007(packs, organs_dir, spec["masks"],
                                         spec["roster"], fold_list=[f])
        du = unseen_engine.fold_utility(champ, f)
        delta_u[f] = float(du)
        per_rot[f] = {"champion_hash": champ.hash(),
                      "champion_rule": res["champion"]["rule"],
                      "train_fitness": res["champion"]["result"]["fitness"],
                      "unseen_fold_dU": float(du),
                      "k_eff": res["k_eff"], "P": P, "G": G,
                      "masks_synthetic": bool(spec.get("synthetic")),
                      "roster": spec["roster"]}
        print(f"  rotation {f}: unseen dU = {du:+.4f} "
              f"(train fitness {res['champion']['result']['fitness']:+.4f})")
    gate = rotation_gate(delta_u) if len(delta_u) >= 2 else \
        {"passed": None, "note": "fewer than 2 rotations (smoke)",
         "delta_u": {str(k): v for k, v in delta_u.items()}}
    return {"rotations": per_rot, "gate": gate,
            "masks_synthetic_any": any_synth}


# ================================ orchestration =================================
def load_packs(packs_dir: Path, folds=(1, 2, 3, 4, 5, 6)) -> Dict[int, dict]:
    packs = {}
    for f in folds:
        p = Path(packs_dir) / f"pack_F{f}.pkl"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — build with `surrogate_007.py --build-packs`")
        with open(p, "rb") as fh:
            packs[f] = pickle.load(fh)
    return packs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "production", "rotations", "b1",
                             "blend0", "full"])
    ap.add_argument("--packs-dir", default=str(SG.STORE))
    ap.add_argument("--fold-nightly-dir",
                    default=str(PROTO / "store" / "nightly_007_folds"))
    ap.add_argument("--rotation-masks-dir",
                    default=str(PROTO / "store" / "rotation_masks_007"))
    ap.add_argument("--roster-file",
                    default=str(PROTO / "store" / "roster_007.json"))
    ap.add_argument("--roster", default="",
                    help="comma list override (smoke default: M1,M2,M5)")
    ap.add_argument("--masks-file", default="",
                    help="production masks json override (default: frozen "
                         "PRODUCTION_MASKS)")
    ap.add_argument("--out", default=str(EA_DIR))
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--rotations", default="1,2,3,4,5,6")
    ap.add_argument("--wall-cap-min", type=float, default=WALL_CAP_MIN_DEFAULT)
    args = ap.parse_args(argv)
    t0 = time.time()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    smoke = args.mode == "smoke"

    organs_dir = Path(args.fold_nightly_dir)
    masks_dir = Path(args.rotation_masks_dir)
    if smoke:
        organs_dir = PROTO / "store" / "nightly_007_folds_SYNTH"
        masks_dir = PROTO / "store" / "rotation_masks_007_SYNTH"
        if not organs_dir.exists():
            SG.gen_synthetic_organs(organs_dir)
        if not masks_dir.exists():
            SG.gen_synthetic_rotation_masks(masks_dir)
    if not organs_dir.exists():
        raise SystemExit(f"organ inputs missing: {organs_dir} (wave-2a "
                         f"contract: store/nightly_007_folds/<D>.json)")

    if args.roster:
        roster = tuple(args.roster.split(","))
    elif Path(args.roster_file).exists():
        roster = tuple(json.loads(Path(args.roster_file).read_text())["roster"])
    elif smoke:
        roster = ("M1", "M2", "M5")
    else:
        raise SystemExit(f"roster missing: {args.roster_file} (wave-2a writes "
                         f"it after the acceptance gates)")
    masks = (json.loads(Path(args.masks_file).read_text()) if args.masks_file
             else TA.PRODUCTION_MASKS)
    synthetic_inputs = smoke or "SYNTH" in str(organs_dir)

    packs = load_packs(args.packs_dir)
    P = args.p or (8 if smoke else P_PROD)
    G = args.g or (2 if smoke else G_PROD)
    rot_list = [int(x) for x in args.rotations.split(",")]
    if smoke and args.rotations == "1,2,3,4,5,6":
        rot_list = [5, 6]

    manifest = {"mode": args.mode, "generated":
                _dt.datetime.now().isoformat(timespec="seconds"),
                "master_seed": MASTER_SEED, "roster": list(roster),
                "organs_dir": str(organs_dir),
                "synthetic_inputs": synthetic_inputs,
                "surrogate_space": True,
                "packs_dir": args.packs_dir}

    if args.mode in ("production", "full", "smoke", "blend0"):
        engine = FitnessEngine007(packs, organs_dir, masks, roster)
        tag = "blend0" if args.mode == "blend0" else \
            ("production_smoke" if smoke else "production")
        Pg, Gg = (P_PROD, G_ROT) if args.mode == "blend0" else (P, G)
        res = run_ga(engine, out_root / tag, tag, P=Pg, G=Gg)
        champ = res["champion"]
        gate = adoption_gate(champ["result"], res["k_eff"])
        pin = boundary_pin_fraction(champ["champion"], roster)
        b1 = run_b1(engine, K=max(res["n_genome_evals"], 10),
                    tag=f"{tag}_b1")
        manifest.update({
            "production": {
                "P": Pg, "G": Gg, "k_eff": res["k_eff"],
                "n_genome_evals": res["n_genome_evals"],
                "champion_rule": champ["rule"],
                "champion_hash": champ["champion"].hash(),
                "champion_fitness": champ["result"]["fitness"],
                "champion_fold_u": champ["result"]["fold_u"],
                "champion_genome": champ["champion"].to_dict(),
                "top8": champ["top8"]},
            "adoption_gate": gate,
            "boundary_pin_audit": pin,
            "b1_random_search": {"K": b1["K"],
                                 "best_fitness": b1["best_fitness"],
                                 "ga_beats_b1": bool(
                                     champ["result"]["fitness"]
                                     > b1["best_fitness"])},
        })
        print(json.dumps({"adoption_gate": gate, "boundary_pin": pin,
                          "b1_best": b1["best_fitness"]}, indent=1))

    if args.mode in ("rotations", "full", "smoke"):
        # wall-cap shrink ladder (§8): shrink rotation G first; never drop
        elapsed_min = (time.time() - t0) / 60.0
        g_rot = args.g or (2 if smoke else G_ROT)
        if not smoke and elapsed_min > 0.5 * args.wall_cap_min:
            g_rot = max(6, g_rot - 2)
            manifest["wall_cap_shrink"] = {"rotation_G": g_rot,
                                           "elapsed_min": round(elapsed_min, 1)}
        rot = run_rotations(packs, organs_dir, masks_dir, out_root,
                            P=P if smoke else P_ROT, G=g_rot,
                            rotations=rot_list)
        manifest["rotation"] = rot
        print(json.dumps(rot["gate"], indent=1))

    if args.mode == "b1":
        engine = FitnessEngine007(packs, organs_dir, masks, roster)
        b1 = run_b1(engine, K=K_CAP)
        manifest["b1_random_search"] = b1

    # ---- ship decision (gates are AND; B0 ships on any failure) ---------------
    if args.mode in ("production", "full", "smoke"):
        ag = manifest.get("adoption_gate", {}).get("adopted")
        rg = manifest.get("rotation", {}).get("gate", {}).get("passed")
        both_known = ag is not None and rg is not None
        adopted = bool(ag and rg) if both_known else False
        shipped = (manifest["production"]["champion_genome"]
                   if adopted else Genome007.b0(roster).to_dict())
        manifest["ship"] = {
            "gates_are_AND": True, "adoption_passed": ag,
            "rotation_passed": rg, "adopted_champion": adopted,
            "shipped": "champion" if adopted else "B0",
            "shipped_genome": shipped,
            "note": ("SYNTHETIC-INPUT SMOKE — ship decision is machinery "
                     "exercise only, never evidence" if synthetic_inputs
                     else "production")}

    manifest["wall_clock_min"] = round((time.time() - t0) / 60.0, 2)
    name = ("champion_manifest.json" if args.mode in ("production", "full")
            else f"{args.mode}_manifest.json")
    (out_root / name).write_text(json.dumps(manifest, indent=1, default=str))
    print(f"-> {out_root / name} ({manifest['wall_clock_min']} min)")
    return manifest


if __name__ == "__main__":
    main()
