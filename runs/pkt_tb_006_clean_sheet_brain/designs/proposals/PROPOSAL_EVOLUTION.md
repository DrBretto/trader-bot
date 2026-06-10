# PROPOSAL — EVOLUTION ENGINEER (panel role 8)
# PKT-TB-006 Clean-Sheet Trader's Brain — where the EA bites

**Author:** Evolution Engineer, 2026-06-10. Written blind to the incumbent (sealed-incumbent
rule); sources: packet, ASSIGNMENT_BRIEF.md, committee/EVIDENCE_PROTOCOL.md only.
**Audience:** Architects Alpha/Beta/Gamma — graft this organ into your design. Everything below
is architecture-agnostic: it assumes only that your brain has (a) N gradient-trained ensemble
members emitting per-day, per-symbol outputs, (b) a meta-evaluator consuming them, and (c) an
allocator turning scores into trade intents. The EA evolves the connective tissue between them.

---

## 0. The one-sentence thesis

**Evolution owns the brain's dispositions; gradients own its perceptions; the contract is fixed.**
The EA never touches a neural weight. It evolves a small (≤40-gene) genome of trust priors,
organ/feature gates, and allocator risk parameters, scored directly on cost-adjusted walk-forward
trading utility — the objective backprop cannot reach because it is non-differentiable (discrete
intents, transaction costs, path-dependent drawdown) and risk-shaped (we care about the
distribution of outcomes, not mean prediction error).

## 1. Division of labor — EVOLVED vs GRADIENT vs FIXED

**The principled rule (apply it to any organ your design adds):**

1. **GRADIENT-TRAINED** iff a dense, differentiable, per-sample loss exists AND the parameter
   count is supportable by brief §7's sample budget. Perception is dense: every symbol-day is a
   training example for "what happens next." That is where the ~189k symbol-days can feed
   thousands–millions of (heavily shared) weights without memorizing.
2. **EVOLVED** iff the true objective is the trading utility itself — non-differentiable through
   discrete BUY/SELL/REDUCE intents, the cost model, and max-drawdown — AND the search space is
   small (≤~40 dimensions). Disposition is sparse: one portfolio path per history, so only a
   handful of distinct regimes (~6) constrain it. Low-dimensional direct search is the only
   honest optimizer at that sample size; a gradient surrogate would optimize the wrong loss.
3. **FIXED** iff it is a contract, a safety rail, or part of the measurement instrument. The
   optimizer never holds the ruler: fitness hyperparameters, fold/embargo scheme, cost model,
   and hard risk ceilings are pre-registered constants, not genes.

**Concrete assignment:**

| Component | Regime | Why |
|---|---|---|
| Ensemble member weights (transformer + other heads) | GRADIENT | dense supervised losses (forward returns / rank / utility targets); per-fold retrain is the leakage firewall (§3) |
| Meta-evaluator network weights | GRADIENT | trained on decision-grade targets (realized forward utility), dense per-day loss — per Meta-Evaluator Designer |
| Per-member **trust priors** + daily trust-adaptation speed | **EVOLVED** | "how much to listen to whom" has no per-sample gradient; it is a risk posture |
| **Organ/feature-block gates** (ensemble composition; GDELT theme blocks; LLM-signal inclusion) | **EVOLVED** | discrete; exactly the "ensemble composition" the packet names |
| **Allocator risk genes** (gross exposure, vol target, per-symbol cap, drawdown brake, no-trade band, conviction temperature) | **EVOLVED** | objective = cost-adjusted path utility; non-differentiable; risk-shaped |
| Meta-evaluator's **risk-aversion λ and abstain threshold** (scalar knobs, not weights) | **EVOLVED** | score→size mapping shape is a disposition, not a perception |
| Feature engineering code; fold/embargo scheme; fitness λs; cost model; replay contract | **FIXED** | measurement instrument |
| Hard rails: gross ≤ 1.0 (no leverage), harness cluster cap, universe = config/universe.csv | **FIXED** | evolution may tighten inside a rail, never loosen past it |

Note on gates and the assignment scorecard: evolution is **allowed** to gate a feature block to
zero or drive an organ's trust to its floor. That is not a silent omission — gate states and
trust floors are reported in the dossier, and the leave-one-out attribution arm (Phase D)
independently measures each organ. If evolution finds GDELT worthless, the packet says that is
an acceptable evidence-backed finding; this design surfaces it rather than hiding it.

## 2. Genome design — exact fields, encoding, operators

### 2.1 Genome schema (decoded form = `genome.json`, the interaction contract)

All genes stored internally as a flat float vector in [0,1]^L; decoded by per-gene affine or
log maps. M = number of ensemble members (architect-dependent, assume M ≤ 6). With M=5 and
B=8 feature blocks, L = 5+1+5+8+8+2 = 29 genes. Hard cap L ≤ 40 (see §6).

```
GENOME (29–40 genes):
  # --- trust (M + 1 genes) ---
  trust_prior[m]        m=1..M    float, logit in [-2.0, +2.0]      # softmax over members at day 0
  trust_halflife_days             float, log-scale in [5, 60]       # speed of daily trust adaptation (§5)
  # --- composition (M + B genes, binary) ---
  member_gate[m]        m=1..M    {0,1}                             # member in/out of ensemble
  feature_gate[b]       b=1..B    {0,1}   B≈8 blocks                # e.g. GDELT-themes, GDELT-tone-dynamics,
                                                                    # GDELT-actor, LLM-sentiment, vol-structure,
                                                                    # macro-rates, cross-asset-flow, trend-block
  # --- allocator risk (8 genes) ---
  gross_target                    float [0.30, 1.00]                # target gross exposure (rail: ≤1.0)
  vol_target_ann                  float, log [0.06, 0.18]           # annualized portfolio vol target
  max_symbol_weight               float [0.02, 0.15]
  dd_brake_threshold              float [0.05, 0.20]                # trailing-peak drawdown that arms the brake
  dd_brake_strength               float [0.0, 1.0]                  # fraction of gross shed when armed
  no_trade_band                   float [0.000, 0.030]              # weight-space band; below it, don't trade (turnover control)
  conviction_temp                 float, log [0.25, 4.0]            # temperature mapping meta-evaluator scores -> weights
  cash_floor                      float [0.00, 0.30]
  # --- meta-evaluator shaping (2 genes) ---
  risk_aversion_lambda            float, log [0.5, 8.0]             # in the score->size utility
  abstain_threshold               float [0.0, 0.5]                  # below this |score|, size to 0
```

Decoded genome is written as JSON with the gene names above; any architect's allocator reads it
by name. Architects MAY add genes (e.g. a horizon-blend weight) but must keep L ≤ 40 and declare
ranges; ranges are part of the pre-registration.

### 2.2 Algorithm, sizes, operators, determinism

(μ+λ)-style generational GA — chosen over CMA-ES because the genome mixes binaries with floats.

```
population P = 28, generations G = 14, elitism E = 2
total genome evaluations K ≈ 28 + 13×26 ≈ 366   (hard cap K ≤ 400, see §6)

init:        P genomes ~ U[0,1]^L, EXCEPT individual #0 := DEFAULT_GENOME (§4) — the
             hand-set baseline always starts in the population (evolution must beat it from within)
selection:   tournament, size 3, on fitness (§3); deterministic tie-break by genome SHA
crossover:   uniform per-gene, p_swap = 0.5, applied to ⌈0.7·(P−E)⌉ offspring pairs
mutation:    floats:  Gaussian σ=0.10 in [0,1]-space, per-gene prob 0.30, reflect at bounds
             binaries: bit-flip prob 0.05
             σ annealed ×0.85 per generation
elitism:     top-2 carried unchanged
early stop:  best fitness improves < 0.01 for 4 consecutive generations → stop

determinism: master_seed = sha256("PKT-TB-006-EA" + train_window_end_date)[:8] as int;
             ONE numpy Generator(master_seed) drives init/selection/crossover/mutation/fold
             subsampling; per-evaluation cost-rng seeds derived as child seeds. Identical
             inputs ⇒ identical champion, bit-for-bit. All seeds in the run manifest.
```

**Compute shape (the trick that makes this fit the 1–2 h window):** gradient heads are trained
FIRST (per fold, §3.1) and frozen; their per-day outputs are cached as numpy matrices
(days × symbols × members). A genome evaluation is then a pure-numpy vectorized portfolio walk
over precomputed outputs — no model inference, no S3, no replay engine. Measured-order estimate:
~20–80 ms per fold-pass on Apple Silicon ⇒ 366 genomes × 6 folds × 2 cost scenarios ≈ 4,400
fold-passes ≈ **3–8 min wall-clock**. EA budget cap: 25 min; if exceeded, shrink G before P
(packet: shrink seeds/epochs before dropping organs).

## 3. Fitness — walk-forward utility on TRAINING folds only; the holdout is never touched

### 3.1 Fold layout (FIXED, pre-registered; E2 firewall)

Holdout starts 2026-03-11 (EVIDENCE_PROTOCOL). Evolution's world ends 21 trading days before
it: **fitness data ≤ ~2026-02-06**. The E2 holdout is read exactly once, at bake-off, by the
orchestrator — never by the EA, never by any baseline run.

```
Six evaluation folds, ~250 trading days each, tiled backward from 2026-02 (dates snapped to
trading calendar at run time; layout FIXED):
  F1 2020-02→2021-02   F2 2021-02→2022-02   F3 2022-02→2023-02
  F4 2023-02→2024-02   F5 2024-02→2025-02   F6 2025-02→2026-02
Head-training for fold f: expanding window 2014-08-29 → (fold_f.start − 21 trading days).
Purging/embargo: 21 trading days between head-train end and fold start — longer than the
longest label horizon (21d forward), so no label leakage into fitness folds.
2014–2019 is head-training substrate only, never a fitness fold (oldest regimes least like
the deployment regime; they inform perception, not disposition).
```

Each fold sits in a different regime chunk (2020 crash/rebound, 2021 bull, 2022 bear, 2023
recovery, 2024 bull, 2025-26 recent), so **cross-fold dispersion is a regime-robustness
measurement**, which the fitness function spends.

Genome evaluation on fold f consumes ONLY out-of-fold head outputs (heads that never saw fold
f). The six per-fold head sets are a by-product of the monthly walk-forward training the
gradient organs need anyway — the EA adds no head-training cost.

### 3.2 Fitness function (FIXED constants; not genes — no self-grading)

```
per fold f, genome g:
  simulate daily: member outputs → trust-weighted meta-evaluator score → allocator(genome)
                  → target weights → intents where |Δw| > no_trade_band
                  → net daily returns r_t after the transaction-cost model
                  (half-spread table from get_cost_config_snapshot(); slippage at its
                   expectation, 0 bps, in the fast sim — the full seeded-slippage model
                   applies at bake-off on the real harness)
  U_f(g) = sqrt(252)·mean(r)/std(r)  −  0.5 · MaxDD_f / 0.10
           # cost-adjusted annualized Sharpe minus 0.5 fitness-points per 10% max drawdown

robustness scenarios: evaluate each fold under cost multiplier c ∈ {1.0, 1.5};
  U_f(g) := min over c                       # the genome must survive being wrong about costs

FITNESS(g) = mean_f U_f(g)  −  0.5 · std_f U_f(g)  −  0.02 · (# active gates)
             # mean−λ·std across regime-folds (penalize fold-pickers)
             # parsimony pressure: each open gate must pay ≥0.02 fitness for its complexity
```

### 3.3 Anti-overfit controls on the search itself

- **Fold subsampling per generation:** each generation scores its offspring on a seeded random
  4-of-6 fold subset; elites and the final champion are re-scored on all 6. (Mini-batching the
  folds — a genome cannot win by tuning to one fixed fold panel all run long.)
- **Default-in-population:** DEFAULT_GENOME competes every generation; the champion's margin
  over it is observed in-run, not post-hoc.
- **Adoption gate (pre-registered):** the champion ships only if
  `FITNESS(champion) − FITNESS(default) > 1.0 × std_f U_f(champion)` (one cross-fold sd).
  Otherwise **the default genome ships** and the dossier says evolution failed to clear noise
  that month. Evolution must earn its slot every cycle.
- **Cost-scenario min** (§3.2) and **parsimony term** push toward robust, simple genomes.
- **Every variant logged** (EVIDENCE_PROTOCOL): all ≤400 genomes + fitness tables land in the
  run dir (`ea/generation_<k>.jsonl`), so "we report only the winner" is structurally blocked,
  and the holdout-looks ledger can state exactly how many candidates the search consumed.

## 4. Meaningful-not-ceremonial — the pre-registered proof design

Three baselines, all defined BEFORE the first evolution run:

- **B0 — DEFAULT_GENOME (evolution OFF):** hand-set, written into the run dir now: uniform
  trust priors (0.0 logits), all gates ON, trust_halflife 21d, gross_target 0.60,
  vol_target 0.10, max_symbol_weight 0.08, dd_brake 0.10/0.50, no_trade_band 0.01,
  conviction_temp 1.0, cash_floor 0.10, risk_aversion 2.0, abstain 0.10. This is "a sane quant
  set it by hand in five minutes" — the honest opponent.
- **B1 — budget-matched random search:** K=366 genomes drawn uniform from the same space, same
  fitness, same folds, seeded; take its best. Tests whether selection/crossover/mutation add
  anything over raw evaluation budget.
- **B2 — the E2 read:** at bake-off, run the full brain twice on the IDENTICAL replay harness:
  champion genome vs B0 genome. One pre-registered configuration each; paired daily-difference
  stats on identical dates; holdout-only verdict. This is the attribution number that goes in
  the final-line scorecard (`evolution=<attr>`).

**Pre-committed verdict rules (what counts, what is ceremonial):**

1. Evolution **carries weight** iff: champion > B0 on E1 full walk-forward (cross-fold mean
   ΔU > 0 with paired t across folds) AND the E2 holdout paired-daily mean Δ is ≥ 0 with
   ΔSharpe ≥ +0.10. (Holdout is ~62 trading days — wide error bars; that is why E1 cross-fold
   evidence is required too, and why the claimed effect size is modest.)
2. Evolution is **ceremonial as an optimizer** if champion fitness ≤ B1 best fitness (GA adds
   nothing over random search at equal budget) — reported as such even if the champion beats B0.
3. Evolution is **ceremonial as an organ** if the champion fails the §3.3 adoption gate or
   loses to B0 on the holdout: the brain ships with B0 and the scorecard reads
   `evolution≈0`, honestly.

These three sentences go verbatim into TOURNAMENT.md's pre-registration before any building.

## 5. Interaction contract + cadence

**Inputs to the EA (from the rest of the brain, per fold):** cached out-of-fold output matrices
from every gradient organ — ensemble member scores, meta-evaluator scores as a function of
member inputs (the meta-evaluator is trained per-fold by its designer; the EA treats it as a
frozen score function whose risk knobs it shapes), LLM-sentiment and GDELT feature blocks
(gateable), prices for the fast cost-aware simulator.

**Outputs (monthly artifacts, S3 `models/` convention: versioned file + latest.json pointer):**

```
genome_<YYYYMMDD>.json        # decoded champion (or B0 if adoption gate failed) + schema version
ea_manifest_<YYYYMMDD>.json   # master seed, K evaluated, fitness tables, B0/B1 scores,
                              # adoption-gate verdict, fold dates, code SHA, wall-clock
ea/generation_*.jsonl         # every genome + fitness (protocol: every variant logged)
```

**Cadence — what evolves monthly vs what adapts daily:**

- **Monthly (local Mac, in the training window):** retrain gradient heads → rebuild OOF
  matrices → run EA → adoption gate → upload genome + manifest. Nothing evolves intra-month.
- **Daily (Lambda, nightly):** the genome is STATIC; what moves daily is the deterministic
  trust-update rule the genome parameterizes:
  `trust_m ← decay(trust_m, trust_halflife_days) + realized-utility credit for member m`,
  renormalized, floored so no member's listen-weight hits exactly 0 mid-month. Daily adaptation
  is a fixed function of evolved parameters — auditable, replayable, no online learning in
  Lambda. Lambda reads one ~5 KB JSON; zero architecture change to the nightly shape.

## 6. Sample budget — why ≤400 evaluations and ≤40 genes (brief §7)

The fitness signal is 6 fold-scores per genome, drawn from ~1,500 fold-days spanning ~6 regime
chunks — and brief §7 is explicit that the 189k symbol-days are heavily cross-correlated and
~11.7y covers only a handful of regimes. The search's selection bias is governed by the number
of effective independent trials vs the number of independent fitness observations: with K=366
correlated trials, expected max-of-K noise uplift on a 6-fold mean is ≈ σ_fold·√(2·ln K)/√6 ≈
1.4·σ_fold — which is exactly why the adoption gate demands a >1.0·σ_fold margin over B0 and
why fitness subtracts 0.5·std. Doubling K or L buys more noise-mining, not more signal: at 40
genes / 400 evaluations the searcher is already near what 6 regime-folds can discipline.
Monthly re-runs do not re-mine a frozen set forever — the fold panel rolls forward as history
accrues, so each month's search faces partially fresh data. Program-lifetime holdout looks from
this organ: exactly one per bake-off (B2), counted in the looks ledger.

## 7. Cost line

- **AWS:** $0.00/month. The EA runs entirely on the operator's Mac inside the existing monthly
  launchd training window (EA wall-clock 3–8 min typical, 25 min cap, within the 1–2 h budget;
  OOF matrices are a by-product of head training). Lambda's only new work is reading one ~5 KB
  genome JSON from S3 nightly (≪ $0.01/mo in GET/storage). No Bedrock, no new infra, no ECR
  growth (pure numpy, already in the training image).
- Worksheet entry for the Feasibility Auditor: `Evolution organ: $0.00 AWS / ~10 min local
  monthly`.

## 8. Implementation sketch (pin-down pseudocode)

```python
def evolve(oof: FoldOutputs, prices: FoldPrices, seed: int) -> Champion:
    rng = np.random.default_rng(seed)
    pop = [DEFAULT_GENOME_UNIT] + [rng.uniform(size=L) for _ in range(P - 1)]
    best_hist = []
    for gen in range(G):
        folds = all_folds if gen == G - 1 else rng.choice(6, 4, replace=False)
        fit = [fitness(decode(g), oof, prices, folds) for g in pop]   # §3.2, vectorized numpy
        log_generation(gen, pop, fit)                                  # every variant logged
        if early_stop(best_hist, fit): break
        elite = top_k(pop, fit, E)
        kids = mutate(crossover(tournament(pop, fit, rng), rng), rng, sigma=0.10 * 0.85**gen)
        pop = elite + kids
    champ = rescore_on_all_folds(top1(pop))
    gate_ok = champ.fitness - rescore_on_all_folds(DEFAULT_GENOME).fitness > champ.fold_std
    return Champion(genome=champ if gate_ok else DEFAULT_GENOME, gate_ok=gate_ok, ...)
```

Fitness inner loop: `weights = allocator(genome, meta_scores(trust(genome, oof_f)))`, banded by
`no_trade_band`, costed per trade via the half-spread table, compounded daily — one matrix pass
per fold. The bake-off itself never uses this fast simulator: champion and B0 run through the
real replay engine (`run_variant`) with the seeded cost model, per the harness contract.

---
*End of proposal. 29-gene reference genome; (μ+λ) GA, P=28, G=14, K≤400; fitness =
min-over-cost-scenarios of (Sharpe − DD penalty), mean − 0.5·std across six regime folds,
parsimony-pressured; champion must beat the hand-set default by one cross-fold sd or the
default ships; ceremonial verdicts pre-committed against B0/B1/B2; $0 AWS.*
