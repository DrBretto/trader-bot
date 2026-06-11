# PROPOSAL — EVOLUTION ENGINEER (role 7) — PKT-TB-007 ORTHOGONAL BRAIN

**Author:** Evolution Engineer, 2026-06-10. Sources: PKT-TB-007 packet (assignment item 3,
constraint C3), ANALYST_AUDIT.md, UNIVERSE_EXPLOITABILITY.md, TB-006 PROPOSAL_EVOLUTION.md +
`prototype/ea.py` + `ea/` logs + COMMITTEE_REPORT.md (the evolution paradox, finding 3).
**Carries forward verbatim:** the (μ+λ) GA machinery, B0-in-population, B1 budget-matched
random search, the 1×cross-fold-sd adoption gate, B0-ships-on-gate-failure, every-variant
logging, determinism contract, fixed-fitness-constants rule. **Changes:** the fitness target,
the genome, the champion-selection rule, and a new generalization gate.

---

## 0. The TB-006 autopsy, in one paragraph

The EA was a real optimizer: champion fitness 1.092 vs B0 0.216 (margin 0.876 = 1.37× the
0.638 gate bar) and it beat budget-matched random search (B1 best 0.838). It then read
**0-vs-B0 (measured)** at E1/E2; B0 beat it on the replay (−0.96). It did not malfunction —
it answered the question it was asked. Its fitness was the Sharpe-minus-drawdown of the
brain's OWN book, min-over-cost-scenarios, mean−0.5σ across folds. The highest-Sharpe book
available to a weak-signal brain is a de-risked one: the champion converged to 26% gross,
4 trades/65 days, dd_brake_strength pinned at 1.0 with threshold near the floor (0.08),
conviction_temp pinned at the 4.0 ceiling. The verdict, meanwhile, was a paired daily t of
(brain − incumbent) — a game about *relative selection*, in which de-risking is a losing
move. **Fitness optimized defense; the verdict graded offense (C3's origin).** TB-007 fixes
this at the root: the fitness IS the verdict statistic, computed the only honest place it
can be — on training folds, with the holdout untouched.

## 1. Verdict-aligned fitness — the exact statistic

The pre-registered verdict is a paired daily t of (incumbent+brain − incumbent) on identical
dates. The in-sample analogue is the **paired tilt utility series**: on each fold, walk the
SAME expression channel twice — once with the genome's tilt applied, once with the neutral
genome (tilt gain 0, which reproduces the base arm exactly, §3) — and difference the daily
net returns: `Δr_t = r_t(brain arm) − r_t(neutral arm)`, both arms costed identically.

```
per fold f, genome g, cost scenario c ∈ {1.0, 1.5}:
  Δr_t  = paired daily tilt return after incremental transaction costs × c
  U_f,c(g) = sqrt(252) · mean_t(Δr_t) / max( sd_t(Δr_t), s_min )      # FLOORED PAIRED IR
  U_f(g)   = min_c U_f,c(g)                                            # survive being wrong about costs

FITNESS(g) = mean_f U_f(g)
             − 1.0 · max(0, −min_f U_f(g))                             # worst-fold floor: a fold the
                                                                       #   genome LOSES is paid for twice
             − λ_reg · Σ_i (g_i − g_B0,i)²                             # shrinkage toward neutral (§2.3)
s_min = 2 bp/day · sqrt-of-nothing (daily units before annualization); λ_reg = 0.05.
All constants FIXED, pre-registered, never genes.
```

**Why this statistic (and not plain mean, and not a pure t):**
- **The paired difference removes the market component.** Both arms hold the same gross/risk
  budget by construction (C1), so beta, regime calls, and the chassis's exposure behavior
  cancel in Δr. The only way to raise mean(Δr) is to select/size better names than the
  incumbent's own ordering — the TB-006 de-risking exploit is not in the action space at all.
  This kills the failure mode at the root rather than penalizing it.
- **The IR shape matches the verdict's geometry.** The verdict bar is t ≥ +2.0 =
  mean/(sd/√n); a fitness in mean/sd units optimizes the same ratio the gate will grade.
  TB-006's book-Sharpe shared the *form* but not the *random variable* — that is the lesson.
- **The sd floor `s_min` is the honest variance treatment.** A pure t-ratio is scale-free: a
  microscopic but consistent tilt scores the same as a bold one (mean and sd shrink
  together), leaving tilt magnitude an unidentified flat direction the search would fill
  with noise — and violating C3 (the layer must SPEND signal). Below the floor the fitness
  degrades to mean(Δr)/s_min, i.e. pure mean — rewarding spend; above it, consistency rules.
  One constant buys both C3 pressure and t-alignment, and fixes the 0/0 at neutral
  (FITNESS(B0) ≡ 0 exactly — see §3, the zero point).
- **Worst-fold floor instead of mean−0.5σ.** TB-006's −0.5σ penalty mixed "volatile across
  regimes" with "wrong in one regime." For a paired tilt the asymmetry matters: a genome
  positive in 5 folds and negative in 1 is regime-fragile in a way symmetric σ understates.
  Losing folds are penalized at full weight on top of their drag on the mean.
- **Costs:** both arms pay the same per-trade cost model (half-spread table +
  expectation-slippage in the fast sim; the seeded full model, seed 4242, only at bake-off
  on the real harness). The Δ series therefore carries exactly the tilt's *incremental*
  turnover cost; min over c∈{1.0,1.5} carried verbatim from TB-006.

## 2. Why TB-006's champion overfit, and the countermeasures

### 2.1 Diagnosis (from `ea/generation_*.jsonl` + the manifest)

1. **Boundary pinning.** Champion genes at or against range edges: conviction_temp = 4.0
   (exact upper bound), dd_brake_strength = 1.0 (upper bound), dd_brake_threshold = 0.080
   (vs floor 0.05). Edge-pinned genes mean the fitness gradient pointed out of the legal
   box — the optimizer was buying an extreme disposition (max conviction-sharpening, total
   de-risk at the first 8% drawdown) that happened to fit the six fold draws.
2. **The fold pool was the whole world.** 4-of-6 subsampling only noised the *offspring*
   scores; elites and the champion were re-scored on all 6 every generation, so over 14
   generations the elite lineage optimized the full fixed panel anyway. The top-10 logged
   genomes are one basin (temp 3.8–4.0, eps 0.37–0.40, dd 0.08/1.0) — converged by gen
   8–10, then elitism + annealed σ locked it in. Nothing in the loop ever asked "does this
   work on a fold you never scored on?"
3. **The gate bar sat below the noise ceiling.** TB-006's own §6 computed the expected
   max-of-K noise uplift at ≈1.4σ_fold; the adoption gate demanded 1.0σ. A pure noise-miner
   could clear the gate — and the champion's 1.37× margin is uncomfortably close to exactly
   that prediction.
4. **Argmax champion.** Best-of-392-correlated-trials is, by construction, the most
   noise-favored point of the cloud.
5. (Root cause, already fixed in §1: even a perfectly generalizing optimum of the WRONG
   fitness loses the verdict.)

### 2.2 Primary countermeasure — leave-one-fold-out rotation gate (chosen)

The right object to certify is not one genome but **the search procedure**: "does this GA,
at this budget, on this fitness, produce genomes that carry to regimes it never scored?"

- Run the full GA 7 times: once on all 6 folds (the production run) and 6 rotations, each
  with fold f held out entirely (fitness on the other 5; the held-out fold never touches
  selection, subsampling, or elite re-scoring in that rotation).
- **Generalization gate (new, pre-registered):** for each rotation, evaluate that rotation's
  champion on its unseen fold: `ΔU_f = U_f(rot-champion) − U_f(B0) = U_f(rot-champion)`
  (B0 ≡ 0). Gate passes iff **ΔU_f ≥ 0 in ≥ 4 of 6 rotations AND mean over the 6 unseen-fold
  ΔU_f > 0.** Fail ⇒ B0 ships, dossier says the search does not generalize at this budget —
  regardless of how good the production champion looks on its own folds.
- Rotations run at reduced budget (P=28, G=8, K≈190 each) — they certify the procedure,
  not hunt a champion; the production run keeps the full budget.

Why this over the alternatives considered: it is the only mechanism that *measures*
generalization instead of merely *hoping* a regularizer induces it, it reuses the existing
fold machinery, and it produces a yes/no artifact the Skeptic can audit. It directly
repairs diagnosis (2) and supplies the out-of-pool evidence the 1.0σ gate (3) lacked.

### 2.3 Supporting countermeasures (all adopted; cheap, attack different failure modes)

- **Shrinkage toward B0 in fitness** (§1, λ_reg = 0.05 per unit² in [0,1] gene space, float
  genes only): distance from neutral must pay for itself in paired IR. Boundary points are
  maximal-distance points — this directly taxes the pinning in diagnosis (1). B0 is the
  incumbent here (§3), so this is shrinkage toward "do nothing," the correct prior for a
  tilt layer.
- **Tighter gene ranges around neutral** (§3 table): TB-006's [0.25, 4.0] conviction_temp
  invited the pin; 007 ranges are set so the neutral value sits mid-range and the edges are
  ±1 honest disposition, not ±1 order of magnitude. Evolution may tighten parity tolerances
  within C1's hard bounds, never loosen (range tops = the hard bounds).
- **Champion = median-of-top-K, not argmax:** take the top 8 full-fold-rescored genomes,
  dedupe at L2 < 0.05 in unit space (the TB-006 top-10 were near-duplicates — dedupe is
  load-bearing), build the per-gene median (majority vote for binaries), evaluate it fresh
  on all 6 folds. If the median genome scores below the best max-min-over-folds member of
  the top-8, ship that member instead (guards against an incoherent cross-basin chimera).
  Either way the shipped champion is a *re-evaluated* genome, never the raw argmax.
- 4-of-6 fold subsampling per generation is retained (cheap selection-noise injection) but
  is explicitly NOT load-bearing for generalization any more — the rotation gate is.

## 3. The 007 genome — what evolution governs, and the neutral-recovery zero point

The brain (per the packet's frame): incumbent chassis + M orthogonal organs (final set from
the Orthogonality Engineer; assume M ≤ 6) expressed through the ranking socket (primary;
rank-permutation variant per ANALYST_AUDIT A5) and/or post_decision reshaping. The genome is
the balancing organ: which organs to trust, how hard to tilt, where.

| Gene | Count | Range (encoding) | B0 default | Governs |
|---|---|---|---|---|
| `organ_trust[m]` | M ≤ 6 | logit [−2, +2] | 0.0 (uniform) | softmax listen-weights over organ convictions (static per month; TB-006's daily trust adaptation is dropped — its three kill criteria all fired) |
| `tilt_gain` κ | 1 | [0.0, 1.0] | **0.0** | master conviction→strength gain; THE zero point |
| `conviction_temp` | 1 | log [0.5, 2.0] | 1.0 | shape of the conviction→tilt curve (TB-006 range [0.25,4.0] pinned; halved span, neutral mid-range) |
| `dead_zone` | 1 | [0.0, 0.3] | 0.1 | \|conviction\| below this ⇒ no tilt (turnover control) |
| `cap_core` | 1 | [0, T_core] | T_core/2 | per-name tilt cap on the 10-name tilt core (UNIVERSE_EXPLOITABILITY §4); T_core = Expression Architect's C1 hard bound |
| `cap_conditional` | 1 | [0, T_cond] | T_cond/2 | cap on the conditional adds (VIXY/IYR/SHY/KRE); T_cond < T_core |
| `defensive_fraction` | 1 | [0.0, 0.5] | 0.25 | share of tilt budget allowed into the defensive sleeve (TLT/AGG/MUB/SHY) inside the Architect's risk-parity rail — the disguised-de-risk caveat, §4 C1 note |
| `event_veto_threshold` | 1 | [0.6, 1.0] | 1.0 (never fires) | LLM/event conditioner gate (role 6's organ): event-risk above this vetoes new tilts that day |
| `event_damp_strength` | 1 | [0.0, 1.0] | 0.0 | how hard event-risk damps tilt magnitude |
| `parity_tol_gross` | 1 | [0.25, 1.0]×ε_gross | 1.0 | tightener on C1 gross-parity tolerance; range top = the hard bound — tighten only |
| `parity_tol_risk` | 1 | [0.25, 1.0]×ε_risk | 1.0 | same, risk-budget parity |
| `channel_mix` | 1 | [0.0, 1.0] | 1.0 | ranking-socket vs post_decision blend (only if both ship; else FIXED 1.0) |

L = M + 11 ≤ 17 genes (hard cap 24; vs TB-006's 27). Ballast tilt cap is **FIXED at 0**, not
a gene — the exploitability audit's mega-clone/factor sleeve is untiltable by pre-registered
construction (NetEdge −8 to −17 bp/bet; giving evolution that dial only sells noise).
Universe tiers (core/conditional/ballast) are FIXED inputs from the Universe Selector, not
evolved — evolution sets how hard to lean, never re-derives where.

**The neutral-recovery zero point.** At `tilt_gain = 0` (with parity tolerances at 1.0 and
event gates inert — all B0 defaults), the brain's emitted ranking scores ARE the incumbent's
deployed-MLP scores unchanged (or the identity permutation, in rank-permutation expression);
post_decision passes intents through untouched. **B0 therefore reproduces the incumbent
bit-for-bit on the same harness and seeds** — pre-registered as a unit check in Phase C:
replay both arms under B0, assert identical trade lists, before any EA run. Consequences:
`Δr_t ≡ 0` ⇒ `FITNESS(B0) ≡ 0` identically — the fitness scale is anchored at "the
incumbent," every fitness point is paired improvement over it, and the adoption gate
becomes "champion must clear +1 cross-fold sd above the incumbent itself." B0 shipping on
gate failure now literally means "the brain ships as a pass-through and changes nothing" —
the honest-zero outcome with zero deployment risk.

## 4. Controls — carried verbatim, plus the new gate

- **B0-in-population:** individual #0 := the neutral genome, every generation, every run.
- **B1 — budget-matched random search:** K genomes uniform from the same space, same
  fitness, same folds, seeded; best taken. GA ≤ B1 ⇒ "ceremonial as an optimizer," verbatim.
- **Adoption gate (verbatim TB-006):** champion ships only if
  `FITNESS(champion) − FITNESS(B0) > 1.0 × cross-fold sd of champion's U_f` — with
  FITNESS(B0) ≡ 0 this is champion > 1×sd_f. Fail ⇒ B0 ships.
- **Generalization gate (new, §2.2):** unseen-fold ΔU ≥ 0 in ≥4/6 rotations AND pooled
  unseen-fold mean > 0. Fail ⇒ B0 ships. Both gates must pass; gates are AND, not OR.
- **E2:** exactly one holdout read, at bake-off, champion-vs-B0 arms inside the brain-vs-
  incumbent run, by the orchestrator. Looks ledger entry: 1.
- **Every variant logged** (`ea/generation_*.jsonl`, all rotations included), manifest with
  seeds/SHAs/wall-clock, deterministic replay of the whole search from the master seed.
- **Pre-committed verdict sentences** (TOURNAMENT_007, before build): (i) evolution carries
  weight iff champion clears BOTH gates AND E2 paired Δ ≥ 0; (ii) ceremonial-as-optimizer
  iff champion fitness ≤ B1 best; (iii) ceremonial-as-organ iff either gate fails or B0
  beats it on holdout — ships B0, scorecard reads `evolution≈0`, honestly.

## 5. Fitness substrate — folds, the fast paired walk, and the emulator honesty check

- **Fold layout carried from TB-006** (F1–F6, 2020-02→2026-02, 21-day embargo, holdout
  2026-03-11 firewall; fitness data ends 2026-02-06). Organ OOF outputs per fold from the
  retargeted members (Orthogonality Engineer's set) — same cached-matrix discipline.
- **The fast paired walk:** genome evaluation is two passes of a vectorized chassis
  surrogate over precomputed inputs — base arm (neutral scores) and tilt arm — differenced
  daily. The surrogate reproduces the expression geometry: base ranking from the deployed
  MLP run on historical features (RANKING_FEATURES exist across the whole window, audit
  B3.3), top-N selection, vol-adjusted sizing, matched gross by construction, half-spread
  costs. It does NOT pretend to be the full replay engine; it preserves the verdict's
  *geometry* (paired, matched-exposure, selection-channel) — which is what fitness must
  align with. The base-arm pass is genome-independent: computed once per fold, cached;
  only the tilt arm runs per genome.
- **Emulator honesty checks (pre-registered, before the first EA run):** (a) neutral-
  recovery unit check (§3) on the real harness; (b) anchor check — on the artifact window
  inside the fitness era (2025-08-04→2026-02-06, real chassis inputs, the Analyst's
  epoch-shim), champion and B0 are run through the REAL replay engine and the paired ΔU
  must agree in sign with the fast sim's read on the same dates; sign disagreement ⇒ the
  fast sim is misspecified ⇒ stop, fix, re-run (a finding, not a tweak).

## 6. Cadence + cost

- **Monthly, local Mac, inside the existing training window:** retrain organs → rebuild OOF
  matrices → production GA (P=28, G=14, K≤400) + 6 rotation runs (G=8, K≈190 each) + B1 →
  gates → champion-or-B0 → `genome_<date>.json` + manifest + generation logs to S3.
- **Wall-clock:** the paired walk is one extra vectorized pass vs TB-006's single-book walk
  (base arm cached), so per-eval cost is TB-006-comparable: ~20–100 ms/fold-pass. Budget:
  production run ≈ 400×6×2 ≈ 4,800 fold-passes ≈ 5–15 min; rotations ≈ 6×190×5×2 ≈ 11,400
  passes ≈ 10–25 min; B1 ≈ 5–15 min. **Total ≈ 20–55 min typical, cap 90 min**; if exceeded,
  shrink rotation G before production G before P (never drop the rotation gate — it is the
  generalization instrument, not a luxury).
- **Daily (Lambda):** unchanged shape — reads one static ~5 KB genome JSON; the genome is
  constant intra-month (no daily trust adaptation in 007). $0.00 AWS for the EA; worksheet
  line for the Feasibility Auditor: `Evolution organ: $0.00 AWS / ~20–55 min local monthly`.

---
*Summary for synthesis: fitness = floored paired IR (√252·mean/max(sd, 2bp) of the daily
brain-minus-neutral tilt series, min over cost×{1.0,1.5}, worst-fold floor, shrinkage toward
B0) — the verdict's own random variable, with the market component cancelled so de-risking
cannot win; generalization = leave-one-fold-out rotation gate on the search procedure (≥4/6
unseen folds non-negative + pooled mean > 0) + median-of-top-8 champion + tightened ranges +
B0-shrinkage; genome = ≤17 genes (organ trust, tilt gain/curve, per-tier caps with ballast
FIXED at 0, event-conditioner gates, parity tighteners within C1 hard bounds, channel mix);
zero point = B0 at tilt_gain 0 reproduces the incumbent bit-for-bit, FITNESS(B0) ≡ 0; all
TB-006 controls verbatim; both gates AND; B0 ships on any failure; ~20–55 min monthly, $0 AWS.*
