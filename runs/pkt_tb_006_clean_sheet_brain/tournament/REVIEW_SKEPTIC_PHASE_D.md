# REVIEW_SKEPTIC_PHASE_D — evidence-review pass (Skeptic, panel role 2)

**Scope:** did the executed evidence honor TOURNAMENT.md §4, and does the dossier say
anything the numbers don't support. Sources read: TOURNAMENT.md §4, BAKEOFF.md,
ATTRIBUTION.md, COST_WORKSHEET.md, SCORECARD.md, FREEZE_SYN1.md, RUN_JOURNAL.md, both
look ledgers, `evidence/*/comparison.json` spot-checks (R01vR02, R03, R05, R19),
`e1_reads.json`, `runs_battery/R01|R02/manifest.json` + `daily_series.csv`,
`bedrock_spend.jsonl`, `battery_plan.json`, `stats.py` verdict functions, git log.
Every number below was recomputed or traced; nothing was written outside this file.

---

## Findings

**F1 — Pre-registration ordering and the one-config freeze. CLEAN.**
Git sequence verified: pre-registration commit `eda53a8` precedes all build; freeze
commit `4afc606` (17:39) precedes the first holdout look (18:44, ledger); the §5
precondition (iii) is met (`battery_plan.json` generated 18:10, `planned_arms_runnable:
true`, `replays_consumed: 0`); all E1 reads computed 18:03 from fold data only (zero
holdout content). FREEZE_SYN1.md states every rung decision with its pre-registered
deciding rule, including the ones that cut against the build (linear twin over MLP,
GBM uniform, conjunctive Transfer-A gate DEAD, fine-tune de-claimed at delta=0).

**F2 — Look ledgers and battery caps. CLEAN.**
15 holdout looks (R01–R14 + contingency R19) ≤ 20; 5 retrains ≤ 9; ledger rows match
`runs_battery/` one-for-one; degenerate R07 honestly kept as a consumed look.
Validation ledger: 78 entries, count printed in the §4.5 line as required. Note the
ledger over-counts in the conservative direction (it logs forced engineering choices
alongside true F5/F6-reading selections) — acceptable.

**F3 — Holdout n=44 vs pre-registered ~62: the shortfall is systematic and
under-explained. CONCERN (the headline one).**
Spot-check of `runs_battery/R01/daily_series.csv`: 65 decision dates vs 88 expected
trading days 2026-02-04→06-10 — 23 missing, of which **every Monday in the window
(14 of them)** plus the journaled 05-11→05-22 gap and two stray days. The dossier
explains the shortfall only as "snapshot gap 2026-05-11→05-22 + window math"
(BAKEOFF §7); the journaled gap accounts for ~9 of 19 missing holdout days. The
systematic Monday absence (a property of the snapshot store) is nowhere disclosed.
Consequences a hostile reader will find: (a) the verdict series contains **no Monday
returns** — a non-random day-of-week filter on the comparison (both arms identically
affected, so pairing is intact, but the sample is not "the holdout period"); (b)
multi-day gaps are treated as single observations (the 05-08→05-21 step, 13 calendar
days, is one "daily" diff — horizon mixing inflates sd); (c) **verdict sensitivity:**
at the same mean (−8.28 bp/day) and sd (59.50), the pre-registered n≈62 gives
t ≈ −1.10 → LOSES, not TIES. *Resolves it:* COMMITTEE_REPORT.md states the Monday
absence and gap-spanning observations explicitly, and carries the n-sensitivity of
the TIES/LOSES boundary in limitations. No re-run is required — the data does not
exist; the disclosure does not exist either, and must.

**F4 — The TIES verdict mechanics. CLEAN as applied.**
Recomputed from `comparison.json`: t = −0.9228 ✓, dSharpe −0.3889 ✓; `beats_ties_loses`
implements §4.2 verbatim including the honest UNDEFINED rule-gap branch (not coerced).
|t| < 1 → TIES regardless of sign — my own rule, applied to a result it happens to
flatter. The honesty line and the computed MDE (20.71 bp/day on n=44, vs the
pre-registration's ≈5 bp/day estimate — 4× optimistic) are printed side-by-side; the
negative endpoint deltas are printed with the verdict, not buried. The required
reading, which the committee report must not soften: every point estimate is
negative; TIES here means "underpowered to call LOSES," and F3's sensitivity sits
directly on the boundary.

**F5 — Cost-model implementation deviates from §4.1. CONCERN (disclosed, forced).**
§4.1 pre-registered "same cost model `src/utils/transaction_costs.py` ... same seeded
slippage RNG" inside the identical harness. The harness applies no costs
(post-registration wiring finding); execution used an identical post-hoc cost overlay
on both arms, raw + cost-adjusted both reported, journaled before any replay. Symmetric
and benign — but it is a change after the §4 commit, and §4's own preamble says such
changes "are protocol violations and must be reported as such in COMMITTEE_REPORT.md."
Currently it lives only in RUN_JOURNAL.md and the runner docstring. *Resolves it:* the
committee report lists it as an executed deviation with the adjudication.

**F6 — Verdict-pair seeding: executed contrary to the journaled adjudication.
CONCERN.**
The binding wiring adjudication (RUN_JOURNAL 2026-06-10; `run_replay.py` header #1)
says: E2 **verdict pair = native 03-11-seeded holdout-window runs** (live book at
holdout start); full-period context pair = seed-date-override runs consuming 2
contingency slots. What actually ran: only the override full-window pair (R01/R02
manifests: `window=full`, `start_portfolio_date_used=2026-02-03`), with the verdict
read as the holdout slice. No native pair exists in `runs_battery/`; I found no logged
supersession in the journal or either ledger. Effect: each arm enters the holdout with
its own replayed book instead of the common live book (it saved 2 looks and is
symmetric, but it is the opposite of what the record says the verdict would be).
*Resolves it:* a supersession record if one exists; otherwise COMMITTEE_REPORT.md
reports it as an executed deviation and states the direction-of-effect argument (SYN-1
traded 4 round trips over the window, so its book-path dependence is small; the
incumbent's is not obviously small).

**F7 — R07→R19 repair: legitimate contingency, not a quiet second look. CLEAN, with a
required plain statement.**
R07 ran the planned trailing-21 replacement and came out byte-identical to R01 because
the frozen deployed sigma convention IS trailing-21 — the planned contrast was null by
construction, the look was honestly consumed and kept, and R19 used a §4.4 reserved
contingency slot, logged, with the orientation flip ("positive flatters the CANDIDATE")
disclosed in both BAKEOFF and ATTRIBUTION. The trigger was a structural degeneracy
(zero contrast), not an unfavorable first result — there is no second-look gaming
surface. **However**, the underlying fact must be stated plainly in the committee
report: in the shipped configuration, RiskNet+ outputs are not consumed by the replay
decision path at all (sigma and book-vol come from the trailing-21 proxy, per FREEZE).
The mandated risk member is structurally dormant at serve time; its scorecard line
(+0.00, indeterminate) measures a **non-shipped candidate** config, and E1 (+1.39)
mildly favors the candidate the freeze did not ship. The freeze rationale (executive
train/serve consistency) is a-priori and journaled — fine — but "risknet=indeterminate"
must not be presentable as evidence about an organ wired into the shipped brain.

**F8 — GBM verdict (E1 +2.48, E2 sign-disagree → indeterminate). CLEAN.**
`organ_verdict` applies §4.3 verbatim: POSITIVE requires E2 holdout mean Δ ≥ 0; the E2
read is −20.8 bp/day (t=−1.45), so indeterminate. Recomputed pooled HAC t 2.4825 from
`e1_reads.json` ✓. This is the exact rule pre-registered to stop E2 from being an
escape hatch, here applied against the only organ that cleared E1 — the opposite of
gaming. Note printed honestly: fold F3's contrast is exactly zero (t = n/a). The one
sentence a hostile reader must never find in the report: "GBM works." It doesn't say
that anywhere; keep it so.

**F9 — The three negative-leaning indeterminates (executive −1.02, LLM −1.08,
GDELT −1.79). CLEAN in artifacts; watch the report prose.**
§4.3 has no negative tag until t ≤ −2.0, so these print as indeterminate with CIs and
MDEs — mechanically correct. The journal already says "negative-leaning"; the committee
report must keep that qualifier. Describing these as bare "indeterminate" without the
sign would be the only place the dossier's prose could lean more favorable than the
numbers, and it currently does not.

**F10 — Scorecard special prints and gate consequences. CLEAN.**
Diversity floor TRIGGERED (0.941 ≥ 0.90) with the pre-registered equal-trust-tie
statement printed beside the executive read; all three executive §7 kill criteria fire
and are reported unreinterpreted; the G1 placebo FAIL consequence (`0 (measured)`
regardless of block arm) is applied even though the champion genome had already gated
G1 off; the LLM Stage-1 chattiness partial-fail is logged and the organ shipped per the
letter of the pre-registered kill condition (8/8 scheduled events hit). Each consequence
is the pre-committed one.

**F11 — §4.6.6 train-vs-harness gap diagnostic NOT PROVIDED. VIOLATION (low severity,
transparently labeled).**
A pre-registered evidence gate (Skeptic B-K2; TOURNAMENT §4.6 item 6: "the dossier
reports the gap between training utility and harness replay utility on identical
dates") was not delivered; SCORECARD.md prints "NOT PROVIDED." Honest labeling does not
discharge a pre-registered reporting obligation. *Resolves it:* compute it from the
executive training-utility artifacts and R01's replay utility on identical dates before
the report converges; failing that, COMMITTEE_REPORT.md reports the violation with this
cite.

**F12 — Costs and the Bedrock cap. CLEAN.**
`bedrock_spend.jsonl` sums to $2.873 ≤ $3.10 cap; 707 ledger rows = 686 primary calls +
20 schema repairs + 1 Phase-A verification, all on the single pinned Haiku model (the
686/687 counts quoted in ATTRIBUTION/COST_WORKSHEET are consistent conventions over the
same ledger). Tier-2 window reduction logged and carried to the final line; monthly
worksheet $9.50 inside the ≤$10 envelope with every shrink rung named on the final line
as §4.4 requires.

**F13 — Multiplicity line. CLEAN, one stale figure to gloss.**
Printed verbatim with actual counts (15 looks / 78 decisions) substituted as designed.
The verbatim text's "holdout MDE ≈ 5 bp/day" was the pre-registration's estimate; the
computed MDE is 20.71 bp/day and is printed beside it. The report should say in one
line that the pre-registered power estimate was ~4× optimistic (consequence of F3),
so no reader anchors on 5.

**F14 — Named-failure / sealed-incumbent check. CLEAN.**
Grep of every dossier surface: the incumbent appears only as its score line, manifest
citation, turnover/cost row, and equity-curve column — no strategy-logic content, no
evaluative prose about the incumbent anywhere. The seal record (RUN_JOURNAL, dated
before the registration commit, "seal incidents: none") is intact. Keep the committee
report's framing on what SYN-1 did and did not demonstrate; the turnover table
(incumbent 35.4 bp drag vs 2.8) is admissible score-line context but must not become
"the incumbent overtrades" commentary.

**F15 — Assignment-vs-result tension. CLEAN.**
The packet mandated the organs; the measured outcome is: no organ POSITIVE, five
`0 (measured)`, five `indeterminate`, `infotropy=no-transfer`, and a TIES-with-negative-
point-estimates bake-off. The dossier reports exactly that, mechanically: CAST ships
with its measured zero under the mandated-presence rule (§4.3 row 1, SPECIAL PRINT
present); the "item fails honestly" interpretations were pre-committed in §4.3 and the
packet itself declared honest no-signal acceptable. No prose anywhere upgrades an
indeterminate to a positive or a zero to "promising." The final line follows §4.7
exactly, with the build-time-journaled member-tagged compound for the ensemble slot.

---

## Dissent / limitations statement (for COMMITTEE_REPORT.md)

The Skeptic finds the execution honest and the mechanical verdicts correctly computed,
and enters four limitations that must travel with the headline. First, the holdout
verdict rests on n=44 paired days, not the pre-registered ~62, and the shortfall is
systematic, not just the journaled May snapshot gap: every Monday in the window is
absent from the snapshot store, so the verdict series contains no Monday returns and
treats gap-spanning multi-day returns as single observations; at the same mean and sd,
the pre-registered n would have crossed the LOSES bar (t ≈ −1.10) — "TIES" is therefore
a statement about power, not parity, and every point estimate (return, Sharpe, paired
mean) favors the incumbent. Second, two §4.1 wiring items executed differently than
registered or journaled (post-hoc cost overlay in place of the in-harness cost model;
verdict read as the holdout slice of full-window runs rather than the journal's native
03-11-seeded pair) — both symmetric across arms, both judged benign, both reportable
deviations rather than silent ones. Third, one pre-registered evidence gate (§4.6.6
train-vs-harness gap) was not delivered. Fourth, the shipped configuration does not
consume RiskNet+ outputs at decision time, so the risknet scorecard line characterizes
a non-shipped candidate; and the single E1-passing organ (GBM-Cond, t=+2.48) failed
holdout sign confirmation — under the pre-registered rule nothing in this dossier
licenses the claim that any mandated organ demonstrably carries weight.

## Verdict on the packet's acceptance test

**PASS, as an evidence product** — conditional on COMMITTEE_REPORT.md carrying F3, F5,
F6, F7, and F11 explicitly (the report's Dissent/limitations paragraph above is the
minimum text). The protocol was honored in substance: pre-registration before build,
one-config freeze before any holdout look, look budgets respected, verdicts computed by
the registered rules including every one that cut against the build, honest zeros and
indeterminates printed unspun, seal intact, envelope met. The honest measured outcome —
the brain TIES (underpowered, negative-leaning) and no mandated organ achieved a
POSITIVE attribution — is exactly the outcome class the packet pre-authorized as
acceptable, and the dossier reports it as such. What the return may not be presented
as: evidence that the organs work, or that TIES means parity.

*— Skeptic, panel role 2, Phase D evidence review, 2026-06-10*
