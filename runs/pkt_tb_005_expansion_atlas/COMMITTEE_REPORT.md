# PKT-TB-005 — Committee Report (Expansion Atlas)

Packet: `committee/packets/PKT-TB-005-EXPANSION-ATLAS-V1-20260609.md`
Run dir: `runs/pkt_tb_005_expansion_atlas/` · Branch: `ai/expansion-atlas`
Panel: declared in `PANEL.md` before deliberation; six real subagent seats
(invocation receipts below). Artifacts: `ATLAS.md` (triage ledger),
`PILOT_EVIDENCE.md` (E1+E2 numbers + manifests), `ROADMAP.md` (sequenced
follow-ons), `panel_returns/` (verbatim seat returns), `pilots/` (all code,
results, and pinned data cache).

## Brief Rundown

The committee enumerated **56 expansion candidates** across data (24), model
(16), and strategy-shape (16) families; the Skeptic and Feasibility Auditor
then triaged them to **7 piloted / 40 parked-promising (each with a named
revival condition) / 9 killed**. Five pilots ran to E1+E2 replay evidence on a
pipeline-faithful harness validated byte-identical to `optimizer/replay.py`,
under a pre-registered honesty regime (12 holdout arm-reads budget, 11
consumed; paired-daily t ≥ 3.0 evidence bar). No pilot cleared the evidence
bar — and that is the correct reading of a 45-day, zero-Monday, rally-only
holdout window, not a failure of the candidates. The run's largest returns
were not the pilots but five run-level discoveries about the system itself
(below), three of which affect how all PRIOR evidence should be read.

## Final Verdict

1. **Expansion should proceed by accrual, not by pilots, for one more
   quarter.** The current holdout window cannot statistically support any
   alpha claim (MDE ≈ 7-37pp/yr depending on arm). The committee ships
   zero-evidence-claim riders now (two ingest bundles, GDELT path fix,
   artifact-clock starts, engineering port) and pre-registers next-window
   reads in ROADMAP.md.
2. **The mandatory baseline check is a statistical tie** — the deep regime
   ensemble does not measurably beat the rule labeler it was distilled from
   (and vice versa); under the pre-registered removal asymmetry this licenses
   removing the torch regime path on complexity grounds (dissent recorded).
3. **The trained health autoencoder is now the model under the most
   suspicion** (rule health beat it by +14.5pp holdout endpoint at ¼ the
   drawdown; below the bar, first in line for the next-window read).
4. **The active ranking layer (blend 0.35) is the run's most urgent operator
   decision**: live in production with zero clean out-of-sample evidence,
   in-sample IC that collapses at its own training boundary, and a 3× worse
   max-drawdown path when active on this harness — with the contamination
   asymmetry running in its favor.
5. **Two removals are proven harmless and win on simplicity** (VIXY ejection,
   universe dedup); production adoption is a future packet per the
   evidence-only constraint.

## Key Tensions

- **Breadth vs the look budget.** The enumerators marked 29 candidates
  pilot-now; the Skeptic demoted all but 7 because 45 perforated holdout days
  cannot answer 29 questions. Resolved by inventing the `ship-ingest rider`
  category: data flows now, evidence claims wait for the next window.
- **Removal asymmetry vs affirmative-harm.** The Skeptic pre-registered "a tie
  kills the torch regime path"; that reading converts the baseline tie into a
  simplification mandate. The opposing position (keep ML layers absent proof
  of harm) is dissent #5 and the operator owns the call.
- **Production-faithful control vs clean control.** The active bundle carries
  the suspect ranking layer; replaying the canon WITH it (faithful) gives a
  control that may itself be damaged, while WITHOUT it (clean) is not what
  production runs. Resolved by running both and reporting the contrast as its
  own finding — which became the run's headline.
- **Harness object vs dashboard object.** This run measures the
  pipeline-faithful optimizer object; the displayed champion line lives on the
  overlay object (`three_line_replay` + top-up + real 3/11 seed). Absolute
  numbers differ by construction; the beat-champion +11.85% holdout and this
  run's numbers are not in contradiction — they are different objects. Every
  verdict here is arm-vs-arm within one object.

## Non-Obvious Findings (run-level; each outranks any single candidate)

1. **The stored "deep ensemble track record" is mostly the rule labeler.**
   128/194 stored inference days (2025-08-04→2026-02-05) are rule-fallback
   one-hots; genuine deep output exists only from ~2026-01-31. Every prior
   analysis of the stored series — including champion-selection evidence —
   read the rule labeler for two-thirds of its length.
2. **`paper_trader.execute_trade` SELL is a partial-sell landmine**: it
   credits cash for the sold shares but pops the ENTIRE holding. Production is
   safe only because engine sells are full-position today; the sleeve pilot
   lost 97% to it before diagnosis. Any future scale-in/partial-exit work hits
   it. (Bug lane.)
3. **Production GDELT ingestion has been silently dead** — the hardcoded v2
   URL 404s (v1 works); `gdelt_available=False` placeholder zeros have been
   feeding the context features. (Bug lane; verified independently by two
   seats.)
4. **~30% of recent trading days have no stored decision artifacts** — zero
   Mondays in the entire holdout window plus a 13-day May hole. This biases
   every window-level metric the system reports and makes weekend/Monday
   candidates untestable. (Ops lane.)
5. **Teacher-version skew:** current-code rule labels agree with stored deep
   labels 77.8% on holdout but only 28.6% on deep-era pre-holdout days — the
   deployed students were distilled from a different labeler calibration than
   the one in the repo today. Pin labeler versions in training manifests.
6. **Disagreement throttle vindicated (directionally):** gru-vs-transformer
   disagreement does not predict forward vol (ρ=-0.08) but does predict
   regime-label instability (ρ=+0.37, p=0.003, n=66) — the one ML-stack
   component this run found positive evidence FOR.

## What Changed From Prior Assumptions

- *"The deep pair is the system's regime engine"* → it has been the rule
  labeler for two-thirds of the stored record, and where it is genuinely
  active it is statistically indistinguishable from the rules.
- *"The ranking blend was validated by the beat-champion work"* → the
  beat-champion winning change never touched ranking; the blend rode in from
  the April shadow-deployment work, was trained inside the current holdout,
  and has no clean out-of-sample read anywhere.
- *"Expert signals are the layer to re-examine"* → the expert-signal
  production-only design survived re-examination (sample-size grounds); the
  TRAINED-model layers (health AE, ranking MLP) are where the evidence is
  thinnest.
- *"More data sources = the obvious expansion"* → all 24 data candidates
  either ride as no-claim ingest bundles or park; the binding constraint is
  the failure surface (Yahoo fragility) and the holdout's statistical power,
  not data availability.
- *"VIXY is a meaningful portfolio question"* → it traded 3 times in 194 days;
  ejecting it is free.

## Implementation Shape (what a follow-on packet executes)

Zero production changes shipped by this packet (hard constraint). The
follow-on lanes, in order:
1. **Bug lane (immediate):** GDELT v1 path fix (config-driven); paper_trader
   partial-SELL fix.
2. **Ops lane:** Monday/May artifact-gap diagnosis (nightly persistence).
3. **Operator decision:** ranking blend 0.35 keep/disable pending the clean
   read (ROADMAP 1.1 pre-designs it on both harness objects).
4. **Ingest riders (two PRs max):** vol-surface Yahoo module; FRED expansion
   (D-04/D-05/D-07/D-21 + optional D-22), with the storage pattern in ATLAS §
   ingest-riders. Universe artifact-clock commit (S-11/S-12 symbols) and the
   S-03 engineering port may ride alongside.
5. **Adoption candidates for a future packet:** S-05 + S-10 (proven harmless
   removals).
6. **Next-window reads (pre-registered in ROADMAP Wave 1):** baseline-block
   second read with the health arm first; ranking clean read after the next
   training cutoff.

## Dissent

1. (Skeptic, standing) The 2026-03-11 holdout is not virgin — prior champion
   work tuned against it; all this run's reads inherit unknown prior looks,
   which strengthens the t≥3.0 standard.
2. (Skeptic, standing) Data-ingest candidates must never claim holdout
   evidence on this window; if future runs replay them here, the Skeptic
   dissents in advance.
3. (Skeptic, pre-registered before results) A baseline-check tie is read as a
   kill of the torch regime path; members requiring affirmative harm before
   removal hold the opposing position. The tie occurred; the tension is now
   live and the operator owns the disposition.
4. (Analyst, recorded in ATLAS) The locked slate charges the S-05+S-10 bundle
   3 looks where the Skeptic budgeted 2; the 11-of-12 total uses the locked
   arithmetic.
5. (Orchestrator, on M-13) The health finding's magnitude (+14.5pp endpoint,
   ¼ drawdown) deserves more weight than its t-stat conveys; recorded as a
   minority lean, not a verdict — the standard says noise-with-direction and
   the report says exactly that.
No other dissent; all other seats converged on the triage as written.

## Panel effectiveness note (one row per declared panelist)

| Seat | Effectiveness |
|---|---|
| Analyst | High — clean 56-candidate synthesis under the resolution rules; invented nothing, reconciled everything; one budget-arithmetic catch recorded as dissent #4. |
| Skeptic | Exceptional — the run's three load-bearing contributions (fallback-era discovery, holdout MDE regime, baseline-design repairs R1-R4) all came from this seat; demotions were specific and numeric, never reflexive. |
| Information Scout | High — 24 candidates with live probes (TESTED-OK/FAIL per source), one production bug found (GDELT 404); the probe discipline is what made the Feasibility and Skeptic passes cheap. |
| Model Architect | High — the distillation-honesty framing (M-01/M-13 as student-vs-teacher) and the 2×2 label/sizing decomposition became the pilot design; over-optimistic on pilot-now counts, corrected by the Skeptic as designed. |
| Portfolio Strategist | High — grounded every claim in engine code (top-up impossibility, dead leveraged stack, duplicate sets); top picks survived triage nearly intact. |
| Feasibility Auditor | High — independent GDELT verification, the aggregate-creep discipline (2 bundles max, 1/quarter), and the immutable-backfill storage pattern; zero BREAKS among 29 graded = the enumerators' envelope hygiene held, which the audit confirmed rather than assumed. |

## Subagent invocation receipts (committee-kind requirement)

Real subagents via the Agent tool; outputs preserved verbatim in
`panel_returns/` (digest = sha256-16 of the saved return file).

| agent_id | seat | output file | output_digest |
|---|---|---|---|
| ae7ef7ed991b12dbd | Information Scout | INFORMATION_SCOUT.md | 42d91dcc1b76fa50 |
| ae898f40277600de7 | Model Architect | MODEL_ARCHITECT.md | 81d3679ab161ab8c |
| aa0562f74fe27fa7e | Portfolio Strategist | STRATEGIST.md | 1451f326a1c0e5ad |
| a137aaa3b720258e2 | Skeptic | SKEPTIC.md | 83678f395ddd4e55 |
| ad1e77ca926af1964 | Feasibility Auditor | FEASIBILITY_AUDITOR.md | 0e0e1d0b0d643d38 |
| a5496835cad889bfd | Analyst (synthesis) | (synthesized ATLAS.md body) | see ATLAS.md |

Prompt digests: each seat's charter is reproduced in full in `PANEL.md` plus
the per-seat prompts embedded in the session transcript; charters bound each
seat to a distinct lens and the shared substrate brief.

## Executor concerns (routing_receipt.executor_concern)

1. **Shared-checkout concurrency hazard.** A concurrent PKT-TB-004 session
   switched this clone's checked-out branch mid-run and committed code changes
   to modules this run's replays import (`decision_engine.py`,
   `paper_trader.py`; flags default-OFF). Behavior-neutrality was verified
   empirically (control replays byte-identical across the boundary), and this
   run's commits were made by temp-index plumbing against `ai/expansion-atlas`
   without touching the other session's checkout — but parallel packet
   sessions in one working tree is a collision class the orchestrator should
   close (git worktrees per session, or serialized repo access).
2. Findings 1-5 in Non-Obvious Findings each need a lane (bug/ops/operator);
   this packet routes them in Implementation Shape but executes none.
3. The packet named `holdout_start: 2026-03-11` as the E2 boundary; the
   discovery that prior work tuned against this same window (Skeptic dissent
   #1) means future packets should roll the boundary forward when the next
   quarter accrues.

## Final line

ATLAS: 56 candidates → 7 piloted / 40 parked / 9 killed; PILOT VERDICTS: baseline-regime=tie (Δret -1.9pp holdout, t=-0.68, sign flips across ranking variants @E2), health-rules-vs-AE=+14.5pp holdout ret / ΔSharpe +5.8, t=+1.18 noise-with-direction @E2, ranking-layer=-7.5pp holdout ret / ΔmaxDD -10.7pp vs blend-0, t=-0.65 + no clean OOS read exists @E2-contaminated-in-its-favor, cash-sleeve=evidence-incomplete (wiring confounded by production partial-SELL landmine, t=-2.32 @E1), universe-removals=Δ≈0 (|t|≤0.2) → harmless, removals win @E2; BASELINE CHECK: ensemble DOES NOT BEAT dumb baseline by any measurable margin (best |t|=1.05 < 2.0 on 45 holdout days, both directions, both harness variants).
