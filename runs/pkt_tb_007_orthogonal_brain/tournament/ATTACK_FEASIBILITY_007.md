# ATTACK — FEASIBILITY AUDITOR (role 9) — build effort, battery budget, cost envelope, both designs

**Packet:** PKT-TB-007-ORTHOGONAL-BRAIN — Phase B attack — 2026-06-10
**Inputs:** DESIGN_MIN, DESIGN_MAX, ANALYST_AUDIT (asset-reuse inventory §C; blocking
mechanics: model-cache hazard `replay_engine.py:147-149` ⇒ separate process per arm; the
2025 epoch shim — as-coded the backfill era replays ZERO days), all four proposals, TB-006
ATTACK_FEASIBILITY + REVIEW_FEASIBILITY_PHASE_D (my own calibration record), TB-006
COST_WORKSHEET.

**Calibration discipline, stated first.** My TB-006 Phase B wall-clock estimates ran
**1–2 orders of magnitude high** (Phase D §3.2): I budgeted 0.5 h of replays and 4–6 h of
training; actuals were 15 replays = **54.5 s total** (3.9 s for a full 84-date replay incl.
brain inference, disk-cached), warm-cache final-grade training cycle ≈ **80 s** (cold
35–50 min), battery retrains in minutes. My GDELT byte estimate was 12× high. Consequence
for TB-007: **compute is not the budget. Engineering hours and debugging variance are.**
Every audited number below prices replays at 1–3 min cold / seconds warm, member trainings
at minutes-to-1 h (CAST cold-cache worst case), and treats the designs' own compute
figures as the fears I myself used to publish.

---

## 1. Build-effort truth — itemized against the reuse inventory

The Analyst's rebuild list (§C bottom line): everything data-shaped imports (~3.5 GB
final-grade: GDELT cache, OHLCV, CBOE/FRED/COT edges, panel, LLM artifacts/features,
battery+stats machinery, replay runner, folds). New spend = retargeted members + OOF +
nightly precompute + EA fitness + expression layer + shims + lot fix. Per item:

### 1.1 Shared core (both designs build these — the honest common bill)

| Item | What's actually new | Audited wall-clock |
|---|---|---|
| Tilt adapter + parity projection (Σ Δw=0 ∩ β=0 ∩ |σ|≤ε on ≤15 names) + conviction path + expression_log + per-decision organ_attribution | New module; TB-006 `strategy_adapter.py` is a lineage start, not a copy (it replaced intents; this edits at margin) | **0.5–1 day** build+test; closed-form projection, no solver risk |
| Lot fix, 2 sites (adapter aggregation `strategy_adapter.py:320,:335-339`; harness `_execute_intents` monkeypatch, SELL iterates lots) + `test_lot_aggregation.py` | Specified to the line in PROPOSAL_EXPRESSION §3 | **1–2 h** |
| B0-EXPR neutral-recovery check (sha256(timeline) == incumbent) | Cheap to run; **the failure mode is a half-day float/ordering debug** — budget it | 1 h run + **0–0.5 day debug reserve** |
| Panel target-slice regeneration (y_rot D+5→D+21 rank, dispersion, bucket exceedance; y1_z exists) | `feature_store.py` imports; new slice code | **2–4 h** |
| Member retrains M1–M5 + referee/control nets (CAST at TB-006 shrink-rung config 16k/2-seed from the start; GBM/HAR/enet minutes; M5 computed) | All training machinery imports (`train_members.py`, `folds.py`) | **CAST ≤1 h cold-cache; rest <30 min combined.** MIN's "2–4 h" is the pre-calibration fear |
| OOF matrix rebuild (84 npz pattern exists) | rerun, new members | **minutes–1 h** |
| C2 gate harness (G-signal, G-book, G-scalar, per-fold + pooled, member zero row) + commit `c2_gate_matrix.json` | New measurement code; statistics specified | **~0.5 day** code; minutes to run |
| Nightly precompute rebuild (TB-006 `store/nightly*/` 93 dirs, member-specific) | adapt `precompute_nightly.py` to new member set | **2–4 h** code; compute trivial at TB-006 actuals |
| EA: fitness swap to floored paired-IR + fast paired walk (base arm cached/fold) + rotation driver ×6 + B1 | `ea.py` machinery imports; the surrogate's tilt pass is new | **0.5–1 day** code; **20–55 min compute, cap 90** (proposal's own numbers are post-calibration sane: 4,800+11,400 fold-passes at 20–100 ms) |
| Battery runner (separate process per arm — REQUIRED, model-cache hazard) + disk cache | `run_replay.py`/`battery.py` import; arm manifests | **2–4 h**; replay compute is seconds-warm/minutes-cold per arm |

**Shared-core subtotal: ≈ 2–3 working days**, of which < 3 h is compute. This is the floor
under BOTH designs.

### 1.2 DESIGN_MIN increments (on top of shared core)

| Item | Audited |
|---|---|
| Acceptance-gate harness (5 pre-registered evidence gates on OOF) | 2–3 h |
| Anchor check on live pre-holdout leg (2 real-harness arms × ~27 dates, no shim) | **minutes**, not the design's "~30 min" — TB-006 replay actuals |
| Deletions vs the menu: NO epoch shim, NO permutation machinery, NO hybrid, NO M6, NO executive | $0, −risk |

**MIN total: ≈ 2–2.5 working days.** The design's own "8–14 h compute, 2–3 calendar days"
is honest on calendar, inflated ~4–10× on compute (harmless direction). **Schedule risk
concentration: the projection + neutral-recovery exactness** — bit-for-bit B0 recovery and
the no-shorting/min_order/rounding edge cases in the tilt adapter. One module, bounded.

### 1.3 DESIGN_MAX increments (on top of shared core)

| Item | What it really costs | Audited |
|---|---|---|
| **M6 GAP-GRU** | New OHLCV-derived feature block (overnight/intraday decomposition, gap_fill_rate, amihud — 8 cols, all from imported cache) + GRU 2.5k params + OOF + C2 matrix to 21 pairs | **~0.5 day** (feature engineering is the cost; training is minutes; arithmetic checks out — densest target in the system) |
| **Executive linear state-gate** (30 params, 4 reconstructible observables) + gradient training on surrogate objective + frozen twin + static baseline | observables from imported caches (2–3 h) + trainer + two control evals | **~0.5 day**; compute minutes |
| **Epoch shim** (backfill prices pairing inputs@D/fill@D+1 + 2026-01-28..02-02 stitch) — built for the sign-anchor + one context read | pairing-logic change inside the replay plan + validation that 122 days actually replay | **~0.5 day incl. validation**; tail risk if the stitch fights the SPY-bar check |
| **Counterfactual permutation socket (1a) + hybrid (1c)** — the "hidden week?" item | Exact local replica of the incumbent's candidate pipeline (blend→multiplier→clip→threshold/health/vol-bucket/LLM-veto/panic), score solve-back r' = (target/mult − 0.65h)/0.35 with |r'|≤3 re-repair, predicted `compute_position_size` for both assignments, greedy β/σ guard + repair loop, high-vol-gate support exclusion, + surrogate extension so the hybrid genome is EA-representable | **Not a hidden week of compute — a 1–2 day engineering item with the fattest tail in either design.** The exactness claim ("EXACT, not estimated") must be VALIDATED arm-vs-engine; the failure mode is serial divergence-chasing against `decision_engine` edge cases (cluster caps `:1309-1315`, cash clipping `:430-432`, panic filter). MIN's own read — "the largest single build item on the menu" — is correct. And it buys **2 non-verdict arms** (hybrid-secondary + perm ablation) |
| M4-A/B two-variant decision | one extra enet training + one pre-registered AUC compare on existing `llm_features.parquet` | **minutes, $0** |
| +3 LOO arms, hybrid + perm replay arms | replay compute trivial | ~0 |

**MAX total: ≈ 3.5–4.5 working days — the design's own estimate is honest** (it was
engineering-dominated, not compute-dominated, so calibration doesn't deflate it). But
**~40% of MAX's increment over MIN (the permutation machinery + shim ≈ 1.5–2.5 days) buys
arms that can never carry the verdict.** Schedule risk concentration: the exact
counterfactual replica — MAX §11.3 names integration debugging as its own most-likely
overrun, and I concur with that self-diagnosis.

### 1.4 The hidden-multiplier audit (reruns nobody prices)

- **C2 remediation ladder:** a failed pair ⇒ junior re-partition → retrain → OOF rebuild →
  C2 rerun → (if shipped set changes) EA rerun. Each loop ≈ minutes-to-1 h compute but
  **0.5 day of attention**. UNPRICED in both designs. **Cap pre-registered: ≤2 full
  remediation passes per pair, then demotion-by-default** (the ladder's own rung 4).
- **EA rotation reruns ×6:** priced and sane (≈10–25 min for all six at 190 evals × 5
  folds × 2 passes). The trap is re-running production+rotations+B1 after every member
  change — with the remediation cap above, worst case 3 full EA cycles ≈ ≤4.5 h compute.
- **MAX ambiguity flagged:** if the hybrid arm's genome (live `channel_mix`) needs its OWN
  production run + rotations rather than riding the verdict-arm champion, that is +7 EA
  runs ≈ +30–60 min compute (cheap) but one more pre-registration surface. Synthesis must
  pin this to ONE EA production run.
- **Precompute reruns:** once after final member acceptance; re-run only if remediation
  fires post-acceptance (the cap bounds it).

## 2. Battery + process budget

Replay arithmetic at TB-006 actuals: ~67-date arm ≈ 3–5 s warm / 1–3 min cold, separate
process each (mandatory — global `_get_ranking_model` cache). Battery compute is noise;
the budget that matters is ARMS (multiplicity + attention) and RETRAIN CYCLES.

| | MIN | MAX |
|---|---|---|
| Replay arms as designed | incumbent, brain, B0-EXPR, ≤5 LOO, R-LLM, B1-compare, ≤2 challenger = **≤12** | incumbent, verdict-tilt, B0-EXPR, hybrid, perm-ablation, LOO×6, B-disp, R-LLM = **14, +3 contingent ordered-pair = ≤17** |
| Non-replay reads | EA gates (F1–F6), acceptance gates, anchor ×2 arms ×27d | EA gates, M4-A/B AUC, executive twin, sign-anchor ×2 arms ×122d (shim), context read |
| Retrain cycles | ~6 trainings (M1+twin, M2, M3, M4+control, M5 computed) + remediation ≤2 | ~9 (+M6, +M4-B, +executive+frozen twin) + remediation ≤2 |
| Total replay processes | ~16 | ~22–24 |
| Battery wall-clock | **<1 h** incl. cold first pass | **<1.5 h** |

**Proposed caps (pre-register in TOURNAMENT_007):** MIN-shaped: ≤14 replay arms / ≤7
retrain cycles. MAX-shaped: **≤18 replay arms / ≤9 retrain cycles** (matches TB-006's
20/9 precedent), EA ≤3 full cycles, remediation ≤2 passes per pair, internal
seed-ensembles only + 2 master-seed sensitivity reps (TB-006 ruling carried), one shared
read-only disk cache. TB-006 conformance ran 15/5 against 20/9 — these caps will not bind
unless something is wrong, which is what caps are for.

## 3. Deployed-shape cost (vs the ~$9.5 conservative-carry baseline)

Both designs deploy as: unchanged nightly Lambda + organ inference + tilt computation +
static ~5 KB genome JSON (monthly, $0 AWS — EA is local).

- **Organ inference:** M1–M5(+M6) are all sub-second CPU forwards. TB-006 measured a FULL
  84-date replay incl. brain inference at 3.9 s ⇒ single-night inference ~50 ms; the
  worksheet's 5 s line has ~100× headroom. Tilt projection on ≤15 names: ms. **+$0.00–0.01.**
- **Data pulls — VERIFIED against TB-006 COST_WORKSHEET §1:** the Lambda-night line
  already prices "+5 s CBOE/FRED/COT CSVs" and the +25 s GDELT 8-zip pull. M2 (FRED/
  breadth), M3 (CBOE daily), M5 (CFTC weekly) introduce **no new pull** beyond that priced
  line; M6 needs only prices already in the nightly. **Nothing new to add.**
- **LLM:** monitoring emission only, **$0.14–0.20/mo** measured-rate ($0.00418/call × 22
  × 1.5 headroom) — identical both designs (R5 retirement; no decision-path calls).
- **S3:** expression_log + genome + organ artifacts ≈ +25 MB steady + ~70 req/day ≈ +$0.01.

| | MIN | MAX |
|---|---|---|
| Marginal $/mo | **+$0.21** | **+$0.21–0.23** (M6+executive inference is ms-scale) |
| Absolute $/mo | **≈$9.5** conservative carry (≈$3.6 at list) | **≈$9.5** (≈$3.6 list) |
| Verdict vs ≤$10 | **PASS** | **PASS** |

The cost envelope is **not a discriminator** — both designs are the same Lambda shape.
Standing caveat carried from TB-006 (still unreconciled): the $6/mo Lambda baseline is
~37× its list-price arithmetic; the marginal column is the decision-grade number.

## 4. Bedrock / cap verification — $0 claims CHECKED, both designs

- **M4-A/B (MAX):** both variants train on existing `store/llm_features.parquet`
  (890×155) + GDELT features parquet — **$0, verified** (columns exist on disk; F5–F6
  coverage confirmed by the Role-Finder's probe, which itself made zero model calls).
- **R-LLM falsifier arm (both):** `llm_disag` = std over existing `llm_sent_*` columns,
  252d z — **$0**.
- **De-chattering (monitoring surface):** post-hoc transform on existing artifacts — $0;
  the prompt-side alternative (~$2.9 re-score) is explicitly NOT recommended and NOT in
  either design.
- **2025-leg LLM coverage:** the organ's artifacts cover 2025-08→2026-01 already (178
  Tier-2 days on disk) — no respend even if the context read wants them.
- **No other Bedrock surface exists in either design.** The $3.00 default cap stands
  **unspent**; ledger stays open in case the operator overrules the R5 retirement (the
  wired-in conditioner would still be $0 — it consumes existing features).

**Verdict: $0 new Bedrock claims are TRUE for both designs. Nothing hidden.**

## 5. Kill/shrink list before build

1. **KILL the epoch shim's anchor role (MAX §8b) — adopt MIN's relocated anchor** (live
   pre-holdout leg, ~27 dates, native plan). The shim's anchor validates the surrogate on
   a 25-name record expressing 2/10 tilt-core names — weak validation per engineering
   dollar. Saves ~0.5 day + a validation tail. The 2025 "context read" dies with it and
   was worth ~0 (both designs already exclude that leg from the verdict).
2. **TIMEBOX-THEN-CUT the permutation/hybrid machinery (MAX §3):** if synthesis keeps it,
   pre-register a 1-day build+validation timebox with an exactness gate (predicted buy
   list must match the real engine on every smoke date); on failure, drop BOTH non-verdict
   arms and print the forfeit (MIN §10.1 already words it). Do not let the largest-
   variance item in the run live on the critical path of the verdict arms — build it LAST,
   after the verdict pair is green.
3. **KEEP M6 GAP-GRU** (~0.5 day, densest target, junior-most, demotion costs nothing,
   adds the production-native model type) and **KEEP the executive linear gate + twin**
   (~0.5 day, 30 params, the one genuinely new intelligence claim, fails safe to static
   trust). These are the cheap 80% of MAX's intelligence increment.
4. **KEEP M4-A/B** (minutes, $0, one pre-registered look) — it is the honest discharge of
   "LLM at max value, wherever it is."
5. **CAP the hidden multipliers:** remediation ≤2 passes per pair then demote; EA ≤3 full
   cycles; ONE EA production run (hybrid genome rides it or the hybrid dies); battery
   ≤18 replays / ≤9 retrains.
6. **Start CAST at the TB-006 shrink-rung config** (16k, 2 OOF seeds) — known-good; do not
   re-fight the 45k battle.
7. **MIN's compute lines** (2–4 h CAST, 30 min anchor, 2–4 h battery): carry them as
   written — they are over-estimates in the safe direction and cost nothing.

## 6. Verdicts + graft recommendation to synthesis

| | DESIGN_MIN | DESIGN_MAX |
|---|---|---|
| **Build wall-clock (audited)** | **≈2–2.5 working days** (claim "2–3 days" honest; compute sub-claims ~4–10× inflated, safe direction) | **≈3.5–4.5 working days** (claim honest — engineering-dominated; ~1.5–2.5 days of the increment buys non-verdict arms) |
| **Battery** | ≤12 arms / ~6 retrains / <1 h compute | 14–17 arms / ~9 retrains / <1.5 h compute |
| **Deployed $/mo** | +$0.21 marginal, ≈$9.5 absolute — **PASS** | +$0.21–0.23 marginal, ≈$9.5 absolute — **PASS** |
| **Phase C cash** | $0 Bedrock, $0 AWS | $0 Bedrock, $0 AWS |
| **Schedule risk concentrates in** | parity projection + bit-for-bit neutral recovery (one bounded module) | the exact counterfactual chassis replica for the socket arms (open-ended divergence-chasing) |

**Graft recommendation: MIN's build plan is the honest core.** Its deletions (epoch shim,
permutation machinery) remove exactly the two highest-variance engineering items, and its
anchor relocation validates the surrogate in the world the brain is graded in. Import from
MAX, in order of value-per-day: **(1) M6 GAP-GRU** (+0.5 day — six types incl. the
chassis's own GRU family, densest target, free demotion); **(2) the executive linear
state-gate + mandatory twin** (+0.5 day — the only meta-structure the sample supports,
gated to a measured zero); **(3) M4-A/B** (+minutes, $0); **(4) the graded verdict ladder
(TIES-POSITIVE cell)** — free, and the honest cell given the Grinold ceiling both designs
accept. The permutation/hybrid channel enters ONLY under the §5.2 timebox, built last,
never on the verdict path. Grafted this way the synthesis brain is **≈3–3.5 working days,
≤16 replay arms / ≤8 retrains, +$0.21–0.23/mo marginal (≈$9.5 absolute, ≈$3.6 at list),
$0 Bedrock** — cheaper than TB-006 on every line, as the packet requires.

**The single biggest schedule risk in the synthesis:** the exactness chain at the zero
point — B0-EXPR must hash-match the incumbent bit-for-bit THROUGH the new tilt adapter,
lot-fix monkeypatch, and separate-process runner before anything else can be trusted; if
the permutation channel is also kept, its exact-replica validation is the same risk class
doubled. Budget a half-day debug reserve at the zero point and run it FIRST.

*— Feasibility Auditor (role 9), Phase B attack, 2026-06-10*
