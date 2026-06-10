# ATTACK — FEASIBILITY AUDITOR (panel role 12) — cost + buildability, all three designs

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — Phase B tournament — 2026-06-10
**Charter:** own the cost worksheet and the envelope (≤$10/mo target, ~$15/mo hard fail);
kill/shrink anything that breaches it BEFORE build effort is spent. Two audits: (A) deployed
monthly cost per design; (B) Phase C buildability in THIS committee run on the operator's Mac.

**Verified facts used (probes run 2026-06-10, no bulk downloads, no Bedrock calls):**
- GDELT 15-min GKG file `20260609120000.gkg.csv.zip` Content-Length = **5,375,643 B (5.4 MB)**;
  live `lastupdate.txt` GKG = 7.3 MB, events = 99 KB. The "~3 MB/file" planning figure is low for
  2026 files; I carry **4–7 MB/GKG file** below.
- Operator Mac venv: **python 3.13.3, torch 2.10.0** (verified import). Note: the pinned
  `torch==2.1.2` in requirements-training has no py3.13 wheels — the prototype runs on the
  venv's torch 2.x. Cost impact: none. Reproducibility note for manifests: record 2.10.0.
- Cost of record (docs/OPERATIONS.md): Lambda ~$6, S3 ~$1, Secrets ~$1, CloudWatch ~$1 = **~$9/mo**.
  **Anomaly flagged:** list-price arithmetic for the recorded shape (22 nights × 110 s + 22
  mornings × 15 s at 2.94 GB ≈ 8.1k GB-s × $0.0000166667) = **~$0.14/mo**, not $6. DEPLOY.md's
  "$0.20/day" night-run figure is ~37× the GB-s list price. Either the real bill is dominated by
  non-GB-s lines (CloudWatch log ingest, ECR, transfer) folded into "Lambda," or the doc is a
  conservative carry. Consequence for this audit: **all three designs' absolute worksheets carry
  the conservative $6 baseline and are therefore upper bounds; the marginal cost of the new brain
  is the decision-grade number**, and it is computed at list price below.
- Pricing assumed (us-east-1 list): Lambda $0.0000166667/GB-s; S3 $0.023/GB-mo, GET $0.0004/1k,
  PUT $0.005/1k; ECR $0.10/GB-mo; Bedrock Claude 3 Haiku $0.25/MTok in, $1.25/MTok out (packet
  figure, matches LLM proposal); S3 egress to the Mac falls inside the 100 GB/mo account free tier.

---

## A. Deployed-design monthly cost worksheets

All three designs add the same shape to the same 3008 MB container: GDELT 15-min pulls (8
zips/night ≈ 30–55 MB at verified 2026 sizes), one batched Bedrock call, small CSV/API pulls,
and <5 s of model inference (45k/30k/45k params total — CPU forward passes are milliseconds).

### A.1 DESIGN_ALPHA — "Forecast First"

| Line | Arithmetic (list price) | Marginal $/mo | Absolute carry |
|---|---|---|---|
| Lambda night | 110→175 s (+25 s GDELT, +40 s LLM funnel/call incl. retry, +5 s experts/exec, +5 s CSVs): 22 × 175 s × 2.94 GB = 11.3k GB-s = $0.19; baseline shape $0.13 | **+$0.06** | $6.20 (record-carried) |
| Lambda morning | unchanged 22 × 15 s | +$0.00 | incl. |
| Bedrock Haiku | (10k in + 1.5k out)/night × 22 × 1.5 headroom = 330k in / 50k out → $0.083 + $0.062 | **+$0.15** | $0.15 |
| S3 storage | +~20–25 MB steady (gdelt_rich, opinions, ledger, meta_decision, llm json) × $0.023/GB | +$0.001 | $1.06 |
| S3 requests | ~30 PUT + ~40 GET/day → ~$0.005 | +$0.01 | incl. |
| ECR | no new heavy deps (torch/sklearn already in image) | +$0.00 | $0.12 |
| Secrets / CloudWatch / SNS | unchanged | +$0.00 | $2.00 |
| Data transfer | inbound free; outbound trivial | +$0.00 | $0.01 |
| **TOTAL** | | **+$0.22/mo marginal** | **≈$9.5 absolute** |

**Verdict: PASS.** Marginal +$0.22; absolute $9.5 even with the conservative $6 Lambda carry.
At honest list-price Lambda the absolute is ≈$3.6. Headroom to hard fail: ≥$5.5.

### A.2 DESIGN_BETA — "Bookwright"

| Line | Arithmetic | Marginal $/mo | Absolute carry |
|---|---|---|---|
| Lambda night | 110→170 s (+40 s LLM funnel, +20 s ingests): 22 × 170 × 2.94 = 11.0k GB-s = $0.18 | **+$0.06** | $6.50 |
| Bedrock Haiku | identical organ to Alpha (proposal adopted wholesale) | **+$0.15** | $0.15 |
| S3 | +~15 MB parquets + ~60 KB/day artifacts (candidate_books, meta_decision, ledger) | +$0.01 | $1.10 |
| ECR / Secrets / CW / transfer | unchanged | +$0.00 | $3.14 |
| **TOTAL** | | **+$0.22/mo marginal** | **≈$9.9 absolute** |

**Verdict: PASS** (their own "$9.90 tight" self-read is an artifact of the conservative carry;
marginal is identical to Alpha's). Their named lever (merge LLM-funnel GKG parse into the
gdelt_rich fetch pass, −20 s ≈ −$0.01 list / −$0.40 carried) is real and free — take it in
synthesis regardless of which design wins.

### A.3 DESIGN_GAMMA — "Information Funnel"

| Line | Arithmetic | Marginal $/mo | Absolute carry |
|---|---|---|---|
| Lambda night | 110→~180 s (funnel F0–F6 parse-once ≈ +35 s, LLM ≈ +40 s): 22 × 180 × 2.94 = 11.6k GB-s = $0.19 | **+$0.07** | $6.00 (under-carried: design kept base at $6.00 without the increment line — corrected here) |
| Bedrock Haiku | §6.3 event-token annotations add ~15 tok × top clusters ≈ +1.8k out/night → (10k in + 3.3k out) × 22 × 1.5 = $0.083 + $0.136 | **+$0.22** | $0.22 |
| S3 | gdelt_rich ~10 MB + feature parquets ~5 MB + event_tokens/field_state artifacts | +$0.01 | $1.06 |
| ECR / Secrets / CW / transfer | unchanged | +$0.00 | $3.12 |
| **TOTAL** | | **+$0.30/mo marginal** | **≈$9.4–9.6 absolute** |

**Verdict: PASS.** Gamma's own $9.4 sketch slightly understates Lambda (+35 s funnel not priced)
and Bedrock (event-token output tokens not priced) — both corrected above; still comfortably
inside the target. Marginal is the highest of the three but by $0.08/mo. Note the IAM-widening
contingency (Haiku-4.5-class at $1/$5 per MTok) moves every design's Bedrock line to
$0.55–0.80/mo — in envelope for all three; flag stands, not a differentiator.

**Cross-check on the cluster of "unchanged" lines:** Secrets Manager at list is 4 secrets ×
$0.40 = $1.60 (record carries $1.00) — pre-existing, not a design delta. No design adds a
secret, an ECR layer with new heavy deps, an SNS topic, or a CloudWatch metric stream.
**No design breaches; no kill on audit A. The envelope is not the discriminator in this
tournament — buildability is.**

---

## B. Phase C buildability in THIS run (Mac CPU, replay over 2026-01-31→06-10, ~89 days; holdout 2026-03-11→, ~62 days)

### B.1 One-time backfills (shared shape, all designs)

| Job | Bytes | Wall-clock | $ | Notes |
|---|---|---|---|---|
| **GDELT top-up 2026-02-05→06-10** (4 GKG + 4 events/day × ~125 days ≈ 1,000 files) | 500 × 4–7 MB + 500 × ~100 KB ≈ **2.0–3.5 GB** | **<1–1.5 h** (4–8 workers, restartable) | $0 | MANDATORY all designs — unblocks holdout GDELT; build task #0 |
| GDELT deep re-pull 2015-02→ (16.5k GKG + 16.5k events) | **80–110 GB** | **10–20 h** | $0 | DOES NOT FIT a committee session as a foreground task — see shrink ladders; prototype-scale substitute: 2 files/day pre-2023 + 4/day 2023→ ≈ 40–55 GB, 5–10 h background; or panel-start 2019-01 at 2/day ≈ 16–25 GB, 2–5 h |
| LLM backfill pilot (10 d) | — | minutes | ~$0.05 | all designs |
| LLM Tier 1 (89 d, covers replay window + entire holdout) | ~11 GB GKG already pulled by top-up | ~1 h | **$0.35–0.40** (Alpha/Beta) / **$0.59** (Gamma, +annotation tokens) | MANDATORY — without it the item-3 attribution is a proxy measurement |
| LLM Tier 2 (523 d, training depth) | covered by deep pull | ~2 h | **$2.05** (Alpha/Beta) / **$3.45** (Gamma as designed) | conditional — see cap arithmetic |

**$5 Bedrock cap arithmetic (the one near-breach found):** Alpha/Beta: pilot 0.05 + T1 0.40 +
T2 2.05 ≈ **$2.50 → $2.50 headroom** for schema-repair retries and prototype iteration. Fine.
**Gamma as designed: 0.05 + 0.59 + 3.45 ≈ $4.09 → $0.91 headroom.** One funnel/prompt bug
discovered after Tier 2 starts forces a re-run that breaches the cap. **SHRINK (binding on
Gamma): event-token annotations Tier-1 + live only; Tier 2 at bucket-score price** → total
≈ $2.69, headroom $2.31. The ETT trains with `annotation_available` masked off pre-2026 (it
already consumes masks by design §9). This is a shrink, not an organ drop: the annotation
channel still exists everywhere the holdout reads.

### B.2 Training wall-clock per full cycle (CPU, prototype scale; my estimates carry a 1.5× risk factor over the designs' own)

| Design | Components per cycle | Design's est. | Audited est. | Dominant item |
|---|---|---|---|---|
| **Alpha** | CAST 6 folds × 3 seeds (+5-seed deploy), XGB 6 folds, EventHead/RiskNet, MetaTrigger 6 folds, exec 5 seeds, EA K≤400 | 95–110 min | **2–3 h** | CAST: 45k params × 6 folds × 3 seeds ≈ 18 transformer trainings |
| **Beta** | library: P1 ×2 variants ×3 seeds ×6 folds = **36 P1 trainings** + P2×8 + P3×2 + P4×2, exec twins, EA | 90 min (warm-start steady-state) | **2.5–3.5 h cold** (no warm start exists in Phase C) | the library grid; §10's record/uniform twin requirement doubles member training (+~1 h) |
| **Gamma** | ETT 6 folds × 5 seeds = 30 + ISRE 30 + GBM 6 + exec + EA | 35–70 min | **1.5–2.5 h** | ETT seeds; lightest training of the three |

### B.3 Attribution battery — counting the arms (the session-budget driver)

Replay run ≈ 89 days × (~3 S3 objects + brain inference) ≈ **1–3 min cold, <1 min with a local
disk cache** (S3Cache is in-memory per run — wrap it with a disk cache for the battery; ~45 MB
of prices.parquet total, one-time). Replays are cheap. **Training cycles attached to arms are
not.** Counts as designed:

| | Replay arms (incl. bake-off pair) | Retrain cycles beyond base | Battery compute |
|---|---|---|---|
| **Alpha** (§11) | full + incumbent + {CAST-drop, ridge-swap, XGB-drop, Event-drop, RiskNet-replace, MetaTrigger-identity, exec-B0, EA-B0(B2), LLM-retrained, LLM-neutralized, GDELT-ablated, GDELT-G1-sub, InfotropyB-uniform} = **15 replays** | ~8 (3 exec-only ~5 min ea; LLM ~25 min; GDELT ~25; **InfotropyB-uniform re-runs CAST ≈ 60–75 min**; InfotropyA + ridge ~10) | **≈2.5–3.5 h retrain + ~0.5 h replay** |
| **Beta** (§14) | full + incumbent + {P1-twin, P1-off, P2s-off, P2f-off, P3-off, P4-off, EA-B0, LLM-retrain, LLM-neutralized, GDELT, exec-baseline, InfoB-uniform, InfoA-R3only, S1, S2, S3 blocks} + 2 dumb-twin validation runs = **20 replays** — the largest battery | ~6 (member drops are trust-renorm, free; LLM/GDELT retrain P4/P3/exec ~30 min; InfoB twins already in base but double stage 1) | **≈2–3 h retrain + ~0.75 h replay** |
| **Gamma** (§12) | full + incumbent + {ETT-twin, ISRE-off, GBM-off, LLM-member-off, contrarian-off, EA-B0, LLM-neutral, GDELT-G1–G5, G1-sub, exec-baseline, Infotropy-A+B-off} = **14 replays** | ~5 (LLM-neutral retrains ETT ~30 min; GDELT retrains everything ~60; Info-off ~60) | **≈2.5–3.5 h retrain + ~0.5 h replay** |

**Seed multiplication — the 30–60-cycle trap, and the ruling:** all three designs put seeds
INSIDE each run (5-seed inference ensembles averaged at decision time) rather than replicating
replays per seed. Seed-replicating the battery (×3) would give Alpha 45 / Beta 60 / Gamma 42
replay+retrain cycles ≈ **12–20 h of pure battery** — that blows the session and buys nothing
the protocol needs: paired daily-difference stats require identical dates and identical
slippage seeds, not seed-replicated arms. **Pre-register: internal seed-ensembles only; plus
2 extra full-brain replays at alternate master seeds (seed-sensitivity line for the dossier);
no per-arm seed replication.**

**Honest minimum battery (answers "what does each organ individually buy, in numbers"):**
bake-off pair (2) + one arm per scorecard organ — transformer-twin, per-ensemble-member drops
(2–3), LLM-retrained-neutral, GDELT-block, EA-champion-vs-B0, exec-vs-fixed-baseline,
infotropy-off (1 combined A+B arm) = **9–10 arms** + 2 seed-sensitivity reps = **13–14 replay
runs, 5–6 retrain cycles ≈ 3–4 h battery compute**. Anything below this stops answering the
acceptance test; anything above ~20 replays/9 cycles is spending session on decimal places the
62-day holdout cannot resolve anyway (min detectable |ΔSharpe| ≈ 0.4–0.5 per Beta §15.3 —
correct arithmetic, applies to all three).

### B.4 Total Phase C session budget per design (backfills backgroundable)

| Design | Engineering build | Training (base + battery retrains) | Replays | Foreground total | Biggest blow-the-budget risk |
|---|---|---|---|---|---|
| **Alpha** | feature store + 5 organs + exec wiring — moderate, all standard supervised | 2–3 h + 2.5–3.5 h | 0.5 h | **≈6–8 h** | CAST retrain cycles, esp. the Infotropy-B uniform twin (a second full CAST run); shrink rung exists |
| **Beta** | **heaviest**: bespoke differentiable-replay core, sequential w_prev training, library plumbing, slot-selector EA genes | 2.5–3.5 h (×~1.4 for record/uniform twins) + 2–3 h | 0.75 h | **≈8–11 h** | library × twin combinatorics + 20-arm battery + most novel machinery to debug |
| **Gamma** | **heaviest data layer**: theme→sector dictionary, ACTOR_MAP, country map, event-token pipeline, record-grade gate | 1.5–2.5 h + 2.5–3.5 h | 0.5 h | **≈6–9 h** | funnel engineering wall-clock; LLM cap (fixed in B.1); R3-as-written (below) |

**Gamma correctness/feasibility flag (binding):** §6.4's R3 leg
(`AUC_oos(model|with e) − AUC_oos(model|without e)`) read literally is **per-event** — one
ablation model fit per candidate event is combinatorially infeasible (hundreds of fits/day).
It must be computed **per feature-family/bucket** (~10–20 ablation fits per training cycle,
minutes). Gamma's own "one ablation on data already ingested" phrasing supports this reading;
the prototype must implement the per-family version. Not a kill — a forced clarification.

### B.5 Shrink ladders (ordered; every rung preserves all organs + the bake-off, per packet)

**ALPHA:** 1) CAST OOF seeds 3→2, deploy ensemble 5→3 (−~35% of the dominant item);
2) tighten CAST early-stop patience / epoch cap; 3) drop to the pre-declared CAST-22k
(d_model 32) as primary; 4) EA G 14→8 (K≈230 — adoption gate unchanged); 5) deep GDELT 2/day
pre-2023 (40–55 GB) or panel-start 2019 (16–25 GB); 6) CAST/RiskNet pretrain start 2014→2016.
The Infotropy-B uniform-twin replay arm is kept (it is a scorecard organ); its CAST retrain is
the last thing shrunk, via rungs 1–3 applying to it too.

**BETA:** 1) **library P1 variants 2→1** (kills 18 of 36 P1 trainings; the EA loses one
slot-selector gene — a disposition shrink, not an organ drop); 2) library seeds 3→2;
3) Infotropy-B twins: executive twin replayed, member twins read at validation only (one
replay arm survives); 4) merge S1/S2/S3 column arms into one data-edge block arm (3→1
replays); 5) EA G 14→8; 6) GDELT 2/day pre-2023; pretrain start 2016.

**GAMMA:** 1) event-token annotations Tier-1+live only (BINDING, cap fix, B.1); 2) ETT/ISRE
seeds 5→3; 3) R3 per-family (BINDING, B.4); 4) Tier-2 LLM backfill conditional on cumulative
Bedrock spend < $3 (the LLM proposal's own rule — enforce it); 5) bag-of-events twin at 3
seeds; 6) GDELT 2/day pre-2023; 7) contrarian Member-5 trained but its LOO arm skipped if the
EA gates it off (gate state reported instead — packet-compliant: gate states are dossier
content).

---

## C. Cross-design comparison and verdicts

| | ALPHA | BETA | GAMMA |
|---|---|---|---|
| **Audit A verdict (deployed $/mo)** | **PASS** — +$0.22 marginal, ≈$9.5 absolute | **PASS** — +$0.22 marginal, ≈$9.9 absolute (carry artifact; same marginal as Alpha) | **PASS** — +$0.30 marginal, ≈$9.4–9.6 absolute (after pricing the funnel seconds + annotation tokens Gamma omitted) |
| **Cheapest to RUN** | tied-1st | tied-1st | 3rd by $0.08/mo — immaterial |
| **Cheapest to BUILD-AND-PROVE** | 2nd (6–8 h) | 3rd (8–11 h) | 1st-equal on compute (6–9 h) but highest engineering variance |
| **Cost risk concentrates in** | CAST training cycles (18 transformer fits/cycle; doubled by the Infotropy-B twin) | library×twin combinatorics, 20-arm battery, most bespoke machinery to debug | funnel/dictionary engineering hours; the $5 Bedrock cap (fixed by shrink rung 1); R3 leg as written |
| **Battery as designed** | 15 replays / 8 retrains | 20 replays / 6 retrains | 14 replays / 5 retrains |

**No design fails the envelope. The discriminator is Phase C buildability, where Beta is the
most expensive to prove and carries the most novel machinery per unit of attribution.**

## D. Recommendation to synthesis — the cost/buildability-optimal organ set (grafts)

1. **Spine from the shared proposals (all three adopted them — zero graft cost):** the
   meta-evaluator (~600 params, ~5 min/cycle), the EA on cached OOF matrices ($0 AWS, 3–8 min),
   the LLM organ with Tier-1 backfill mandatory ($0.40) and Tier-2 conditional on the <$3
   cumulative rule.
2. **Transformer slot: take Gamma's ETT.** It is the cheapest transformer in the tournament to
   train (28k params, minutes/fold), the cheapest to attribute (bag-of-events MLP twin ≈ a
   3-minute training arm vs Alpha's full ridge-ladder on CAST), and the one whose food
   (event/theme tokens) a transformer is least replaceable on. Alpha's CAST is the single most
   expensive training item in any design and its own author flags ridge-parity as the likely
   outcome; Beta's P1 is inseparable from the differentiable-replay machinery.
3. **Tabular member: take ONE GBM** (Alpha's XGB-Cond and Gamma's ranker are the same
   sklearn family; minutes to train). Two GBMs = one redundant LOO arm.
4. **Risk organ: take Alpha's RiskNet** (90-coef ridge — the cheapest organ anywhere in the
   tournament), adding Gamma's realized-correlation target as a second ridge head. Promote to
   Gamma's ISRE GRU only on demonstrated validation lift (it costs 30 trainings/cycle).
5. **Take Beta's P2 linear pair as the dumb-twin floor baselines, not as trust-head members**
   (~30 params each, free) — keeps M small, keeps Beta's best audit idea, adds zero LOO arms.
   Also take Beta's pre-registered turnover budget line and the shared-fetch Lambda lever (A.2).
6. **Do NOT take Beta's policy library + slot-selector genes** (the largest single training
   multiplier across all designs) **nor per-member differentiable replay** — utility-gradient
   training lives in the executive only; members stay supervised (Alpha/Gamma style). This
   halves the bespoke machinery the run must debug.
7. **GDELT:** top-up first (<1.5 h — unblocks everything); deep backfill at prototype scale
   (2/day pre-2023, 5–10 h background) with the density break in the manifest.
8. **Infotropy:** Transfer B everywhere it is free (sample weights; A/B twin at validation +
   one replay arm); Transfer A per-family (B.4).
9. **Battery:** pre-register the honest minimum from B.3 — **bake-off pair + 9–10 organ arms +
   2 seed-sensitivity reps ≈ 13–14 replay runs, 5–6 retrain cycles**; hard cap 20 replays / 9
   retrain cycles; internal seed-ensembles, no per-arm seed replication; one local disk cache
   over S3Cache for the whole battery.

Grafted this way, the synthesis brain's deployed cost is **+$0.25–0.30/mo marginal
(≈$9.4–9.7 absolute on the conservative carry, ≈$3.6 at list)** and its Phase C bill is
**≈5–7 h foreground compute + ≤$3.10 Bedrock** — inside one committee session with the deep
backfill running in the background.

*— Feasibility Auditor, 2026-06-10*
