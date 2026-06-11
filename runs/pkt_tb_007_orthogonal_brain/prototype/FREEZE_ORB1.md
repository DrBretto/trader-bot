# FREEZE_ORB1 — the one bake-off configuration (PKT-TB-007, Phase C close)

**Status: READY FOR ORCHESTRATOR REVIEW — not committed until reviewed.**
Frozen 2026-06-11 at the close of Phase C wave 3. Everything below is the single
configuration the Phase D bake-off (E1 full-live read + the ONE E2 holdout read)
runs against. No knob below may move without a journaled chair adjudication.

---

## 1. Shipping genome: **B0** (hash `3df8053a6c42`)

B0 ships by the chair's pre-stated anchor-remediation rule (§5 below) — not by an
EA gate failure: the EA production sequence never ran because no fitness
instrument survived certification.

| Gene | Shipped value | Range | Note |
|---|---|---|---|
| organ_trust.M1 | 0.0 | [−2, +2] | sole roster gene (roster = [M1]) |
| tilt_gain | **0.0** | [0, 1] | THE zero point — structurally neutral |
| conviction_temp | 1.0 | log [0.5, 2.0] | |
| dead_zone | 0.1 | [0, 0.3] | |
| disp_gain | 0.5 | [0, 2.0] | |
| cap_core | 0.0125 | [0, 0.025] | |
| cap_conditional | 0.00625 | [0, 0.0125] | |
| defensive_fraction | 0.25 | [0, 0.5] | |
| event_damp_strength | 0.0 | [0, 1] | M4 not in trust ⇒ no-op |

`tilt_gain = 0` ⇒ the adapter returns the chassis intents object unchanged.
**B0-EXPR exactness chain (wave 1, ledgered): ORB-1-at-B0 vs incumbent is
bit-for-bit identical** — sha256(timeline.json), sha256(daily_series.csv),
sha256(cost_overlay.json) all equal; 51 executed actions identical
(`runs_battery_007/B0_EXPR/b0_expr_result.json`, BUILD_SPEC §8 step 3).
The bake-off ORB arm at this genome is therefore the incumbent plus the full
organ/measurement program riding alongside in expression logs.

Machine-readable copy: `ea/ship_decision_007.json`.

## 2. Roster and organ dispositions (acceptance gates, wave 2a — all ledgered)

| Organ | Disposition | Rule cite |
|---|---|---|
| **M1** | **SHIPS** — sole roster member (pooled weekly IC .1076, +.042 over member zero, NetEdge 2.6) | TOURNAMENT §4.4.1 mechanical map; gates/acceptance_007.json |
| M2 | challenger, status OPEN (FM6: fail at 29% power = "not measurable yet") | acceptance G-gate, ledgered look |
| M3 | **B-disp constant** — existence PASS, materiality FAIL (.011 bp/day); disp enters conviction as the B-disp sigmoid, no gene | TOURNAMENT M3 materiality rule |
| M4 | challenger — **M4-A variant trained/shipped as challenger** (AUC .616 but ΔAUC vs vol-control −.002; M4-B lost the one-look A/B: ΔAUC(B−A) −.020 vs bar +.011) | G8 one ledger look (ledgers/gates.json) |
| M5 | **disabled** (12/19 sign tally, p = .18; tally printed) | acceptance gate, expected-DISABLED power note |
| M6 | **forecast-altitude line only** — never a roster member (IC .0224, 6/6 folds positive; PASS as a forecast verdict row) | TOURNAMENT §4.4.x "forecast-altitude verdict line, never a roster gate" |
| LLM | **retired from the decision path with receipts** (46 pre-registered screens, 0 BH-FDR survivors); monitoring emission stays; R-LLM falsifier arm in the battery only; 900-day re-test trigger | TOURNAMENT G14 / §4.4.9 |

Roster contract file: `store/roster_007.json` (roster = ["M1"], M3 = "B-disp",
M4_variant = "M4-A", M5 = "disabled").

## 3. Expression channel (frozen — none of this is a gene)

Channel 1b cash-neutral tilt overlay (`tilt_adapter.py`):
- Support tiers (Phase-0 frozen, G15/A9): CORE = ITA SOXX XRT TLT AGG MUB FXE
  USO FXI RSP; COND = IYR SHY KRE; **VIXY excluded entirely**; FXE hard cap
  1.25% (ADV); DEFENSIVE sleeve = TLT AGG MUB SHY.
- T_max = 0.08 NAV one-sided; conviction a = 1.0, θ = 0.5;
  c_t = σ((a·agree + disp_gain·disp + q_sum − θ)/temp), T_t = tilt_gain·c_t·T_max.
- Projection: Σdw = 0 (cash-neutral), β-neutral, σ-budget ε = 0.10·Σ|dw|·σ,
  per-name tier caps, no-short, min_order $250 with whole-share rounding on new
  legs (order edits floor-free), PROJ_MAX_ITER 5.
- Masks: frozen production per-organ/per-name table (`PRODUCTION_MASKS`);
  M1 row only is live given the roster.
- Lot fixes (C5) applied to BOTH arms' processes (`lot_fix_007.harness_lot_patch`).
- Known bandwidth ceiling (ledgered, carries into the Phase D read): cash-neutral
  tilt cannot express on zero-funding-capacity days (panic force-sell regimes) —
  6/21 preholdout smoke days.

Bake-off replay invocation (both arms, separate OS processes, cost seed 4242):
`run_replay_007.py --arm {incumbent|orb1} --window live` with the ORB arm at the
B0 genome above and `--nightly-dir store/nightly_007`. E2 = the one
`--window holdout --from-live` subset read, orchestrator-fired,
`PKT_TB_007_HOLDOUT_AUTHORIZED=1`.

## 4. C2 orthogonality gate — chair adjudication 1 (commit 0b52981)

Signal-space leg (binding per the adjudication): max pooled |ρ| = **.202 ≤ .70
PASS** (TB-006 was .941 — the orthogonality engineering worked). The registered
G-book instrument (TB-006 §6 long-only unit-gross solo books) saturates on the
market factor (all pairs .88–.97 including ρ≈0 signal pairs) and was ruled
**mechanically inapplicable** — a pre-registration defect, reported in the
dossier; market-residualized book diagnostic .498 ≤ .70 printed. The remediation
ladder was NOT fired (10 simultaneous instrument-driven breaches ≠ 10
redundancies). Vindication path stated in the journal: if Phase D shows M1-tilt
and a challenger tilt producing near-identical books, this call was wrong.
Artifacts: `designs/c2_gate_matrix.json`; ledgered in
`validation_looks_007.jsonl` (C2 family entries).

## 5. Anchor disposition — chair adjudication 2 applied mechanically

Pre-stated remediation ladder (journaled BEFORE the re-run, commit 0b52981):
re-run at champion-representative tilt scale, then **Pearson ≥ .8** → fold-era
surrogate fitness as registered; **.5–.8** → EA fitness restricted to the
live-era real-record pack (fold-era demoted to rotation-sanity, restriction
printed everywhere); **< .5** → **B0 ships outright**.

**Leg 1** (starved test tilt, synthetic organs): Pearson .4996 (raw-decomp .626),
slope .82 PASS, mean-equivalence PASS → FAIL finding
(`store/surrogate_007/anchor_007.json`).

**Leg 2 — representative scale** (documented per the rule): tilt_gain 0.5
(mid-range — the chair's own example), organ_trust {M1: +1.0}, disp_gain 1.0
(mid-range), dead_zone 0.05, temp 1.0, caps at B0 mids; realized conviction
.62–.86 (median .77) on REAL organ inputs (`store/nightly_007`), T_t ≈ 2.5–3.4%
NAV, 11/21 real-arm active days, 74 executed actions vs incumbent 51,
paired-delta sd ≈ 2.2 bp/day (≈2.8× leg 1 — statistic no longer starved).
Result, n = 20 paired deltas (`store/surrogate_007/anchor_007_rerun.json`):

- **Pearson .2818 — FAIL, and < .5** (LOO range [.24, .40]; Spearman diag .59)
- slope .8248 — PASS, no D4 magnitude caution
- mean-equivalence — PASS (gap 0.8e-5 vs 2×se 1.0e-4)
- raw-value decomposition Pearson **.2862 ≈ .2818** ⇒ the leg-1 cost-rng
  noise-floor hypothesis is **eliminated**; the disagreement is structural, and
  fidelity DEGRADED with scale (.50 → .28) — coherent with the itemized L2c
  divergence: the surrogate's carried tilt never feeds back into chassis state,
  so coupling error grows with tilt magnitude.

**Branch fired: < .5 ⇒ B0 SHIPS OUTRIGHT.** Consequences, applied as pre-stated:
- The EA production sequence (GA P28/G14, 6 rotations + rotation gate, B1
  control, adoption gate, boundary-pin audit) **did not run** — no fitness pack
  is certified at any level; a champion searched on an uncertified instrument
  could not ship and would only manufacture pressure against the rule.
  ea_cycles budget burned: **0 of 3**.
- The fold-era AND live-era surrogate remain **(surrogate space), diagnostics
  only**, for the life of this packet.
- This is a **reported limitation, never silently absorbed** (chair rule text);
  it prints in the dossier and in every downstream output via
  `ea/ship_decision_007.json`.
- Wave-2b machinery validation stands as machinery-only (smoke EA: 41/41 tests
  green, gates exercised on synthetic inputs — never evidence).

## 6. Rung-by-rung decision record (rule cites)

| Rung | Decision | Pre-registered rule |
|---|---|---|
| B0-EXPR exactness | PASS first attempt (3-way sha256 match) | BUILD_SPEC §8 step 3 |
| Lot fix C5 | applied both arms; incumbent no-op verified by timeline sha | BUILD_SPEC §3 |
| Acceptance gates | M1 ships; M2/M4 challengers; M3 B-disp; M5 disabled; M6 forecast row | TOURNAMENT §4.4.1 + per-organ gates; BH-FDR(10%) printed |
| M4-A/B one look | M4-A (ΔAUC(B−A) −.020 < bar +.011) | G8, 1 ledger look |
| C2 gate | PASS on the binding signal-space leg (.202 ≤ .70); G-book instrument defect escalated, ladder not fired | TOURNAMENT §4.4.2 + chair adjudication 1 |
| L7 firewall | PASS by repo forensics (training-data end 2026-01-14 < 2026-03-11); manifest absence itself ledgered | TOURNAMENT §4.4.7 G17 |
| L2a memorization | fold ≫ live confirmed (gap +.18) — disclosed, paired-cancellation argument printed | BUILD_SPEC §5.3 check 1 |
| Anchor leg 1 | FAIL .4996 → stop-and-fix finding | BUILD_SPEC §5.3 check 3 |
| Anchor remediation | re-run at representative scale → .2818 → **B0 ships outright** | chair adjudication 2 (pre-stated, binding) |
| EA production | NOT RUN (no certified fitness instrument) | chair adjudication 2, <.5 branch |
| Adoption / rotation gates / B1 / boundary-pin | moot — no champion exists to gate | BUILD_SPEC §4.3 (gates are AND; B0 ships) |
| Blend-0 sensitivity | moot — no production EA to compare against | BUILD_SPEC §5.3 check 2 |

## 7. Budget state at freeze

- Holdout looks: 0 used / 1 (the E2 read remains, orchestrator-fired).
- Replay arms: ledgered in `ledgers/replay_arms.json`; anchor re-run arm added.
- EA cycles: 0 / 3. Retrains: within ledger. Remediation rungs: 1 / 4
  (`ledgers/remediation.json`).
- Holdout firewall intact: nothing ≥ 2026-03-11 touched fitness or selection;
  `PKT_TB_007_HOLDOUT_AUTHORIZED` unset throughout this wave.

— end of freeze —

---

## ADDENDUM (chair, 2026-06-11, journaled as adjudication 3 BEFORE any performance read)

The registered verdict pair runs at B0 exactly as above (hash-equality demonstration; the
registered final line reads the degenerate cells honestly). ADDITIONALLY, a **deviation
battery** runs at the value-blind A-PRIORI genome — tilt_gain 0.5, organ_trust {M1:+1.0},
disp_gain 1.0, dead_zone 0.05, all caps at B0 mids, conviction_temp 1.0 — labeled
"(deviation: a-priori genome, instrument-failed EA)" on every artifact and never replacing
the registered verdict. Value-blind attestation in RUN_JOURNAL. Since roster = [M1], the
deviation pair (apriori vs B0/incumbent) IS the M1 utility-attribution arm. Challenger-in
arms (M2, M4-A, M5), the LLM-width falsifier, and cost-seed sensitivity run on top of the
a-priori genome under the same labeling. PERM-DESC contingency: cut, reported not-run.
