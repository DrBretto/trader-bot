# PKT-TRADER-BOT-DIAGNOSTIC-AND-RECALIBRATION-20260429

Packet Owner:
- Claude (fresh executor session, instantiated in `/Users/drbretto/Desktop/Projects/trader-bot/`)

Date:
- 2026-04-29

Authority surface:
- `CLAUDE.md` (project workflow, plan-doc discipline, AWS cost rules)
- `docs/PLAN.md` (current roadmap; Phase 6 expert-engine spec)
- `docs/PHASE3_PROPOSAL.md` (origins of fragility / vol_uncertainty / entropy signals)
- `docs/DEPLOY.md` (canonical deploy procedure)

Authority level:
- execution

Discovery mode:
- discovery_bearing (executor is expected to surface bugs not named in this packet, not just confirm the four named below)

Scope:
- Diagnose the trader-bot's underperformance since 2026-04-02. Find every bug or miscalibration in the daily decision pipeline that contributed to the lag. Fix them with rollback safety. Recalibrate throttle thresholds against the corrected signals. Re-backtest the fixed system on the full 900-day historical dataset. Do not cut over to live until the backtest passes the gate metrics in §Acceptance test. The end state is: algorithm fixed, backtested, golden, ready to ship.

Stop condition:
- A new decision-params bundle exists, all Phase-4 fixes are merged to main with passing tests, walk-forward verification on the 900-day dataset passes the gate metrics defined in §Acceptance test, the Phase-6 shadow day has been produced and reviewed, and `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` ends with the literal line:
  ```
  TRADER-BOT DIAGNOSTIC + RECALIBRATION COMPLETE — AWAITING OPERATOR LIVE-CUTOVER DECISION
  ```

Acceptance test:
- The operator opens the return doc and within 30 minutes can answer: *what was actually wrong, what was fixed, and would the fixed system have made me money over the last 30 days where the broken one lost me 9 points to SPY?*
- Every dollar number on the dashboard either matches the broker truth or has a documented, traceable cashflow gap. No silent paper-vs-broker drift.
- No signal in `timeseries.json` is at a fallback value > 50% of the time. No signal is monotone for > 30 days. Every signal in production has been characterized with a one-line health card.
- Walk-forward verification on the 900-day dataset shows the new code beating SPY in the Aug-2025 → Mar-2026 down-leg AND capturing ≥ 60% of SPY in the Apr-2026 → today up-leg, with max drawdown ≤ 5% across the span.
- Three new postmortem entries in `docs/POSTMORTEMS.md`: silent-fallback-signal class, one-way-ratchet class, first-pass-audit-framing-trap.

Reference outputs:
- Negative reference (do NOT produce): a doc that frames the lag as "doing what it was designed to do, just over-conservative." That framing was the failure mode in the prior audit (`docs/plans/2026-04-28-performance-audit-since-hybrid.md`) — it soothed away three concrete bugs. If at any point a finding looks "actually fine, just conservative," that is the trap; double-check with code-level evidence before accepting.
- Negative reference (do NOT produce): a single bundled commit that mixes multiple fixes. Each fix gets its own branch, its own test, its own commit. Operator reviews each independently.
- Negative reference (do NOT produce): a fix that lowers the bar of `docs/POSTMORTEMS.md` (e.g., adds a "we changed this" entry without the *why* and the *how to prevent the class*).
- Positive reference shape — finding row: `finding_id | symptom_in_data | code_location (file:line) | reproducer (small Python snippet on the data) | recommended_fix_summary | risk_class`
- Positive reference shape — fix branch: separate branch named `ai/<short-task>`; one focused commit; one unit test; one paragraph in the return doc explaining what changed and why.
- Positive reference (existing) — `docs/plans/2026-03-12-alpaca-paper-integration-plan.md` shows the right plan-doc shape this project uses; the return doc should match its rigor.

Write surface:
- `docs/plans/2026-04-29-phase1-bug-audit-findings.md` (Phase 1 output)
- `docs/plans/2026-04-29-phase2-signal-diagnostic.md` (Phase 2 output)
- `docs/plans/2026-04-29-phase3-pnl-reconciliation.md` (Phase 3 output)
- `docs/plans/2026-04-29-phase5-calibration-recommendation.md` (Phase 5 output)
- `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` (the final stop-condition file with the literal final line above)
- `src/` and `tests/` per the fix branches in Phase 4
- `config/` per the new decision-params bundle in Phase 5
- `docs/POSTMORTEMS.md` (three new entries)
- `docs/PLAN.md` (Phase 9 entry summarizing what was done)
- Specifically PROHIBITED: any live deploy or live-param mutation outside Phase 6; any change to `frontend/src/` (operator has separate uncommitted WIP there — leave alone); any retroactive edit of the prior bad audit doc `docs/plans/2026-04-28-performance-audit-since-hybrid.md` (preserve as historical record of the framing trap).

Read-first list:
- `CLAUDE.md`
- `docs/PLAN.md` (esp. Phase 6 expert-engine + Phase 8 Alpaca sections)
- `docs/PHASE3_PROPOSAL.md`
- `docs/POSTMORTEMS.md`
- `docs/plans/2026-04-28-performance-audit-since-hybrid.md` — the prior bad audit. **Read this first.** It documents the framing trap this packet exists to avoid; understand what the prior pass got wrong before producing a new one.
- `src/signals/fragility.py`
- `src/signals/vol_uncertainty.py`
- `src/signals/regime_fusion.py`
- `src/signals/compute_signals.py`
- `src/steps/decision_engine.py`
- `src/steps/morning_executor.py`
- `src/steps/publish_artifacts.py`
- `src/steps/build_features.py`
- `src/brokers/alpaca_client.py`
- `scripts/run_hybrid_walk_forward.py` (existing harness for Phase 5)
- Production data already snapshotted at `$TMPDIR/trader-bot-audit/` (`dashboard.json`, `timeseries.json`, `latest.json`); refresh by `AWS_PROFILE=personal aws s3 cp s3://investment-system-data/dashboard/{dashboard,timeseries}.json` if older than 24h.

Flow position:
- This packet is the algorithm fix, re-calibration, and re-backtest cycle. The final phase is the operator-gated live cutover after the 900-day backtest passes the §Acceptance test gate.

## Objective

The bot has been losing 9 points to SPY over the last 30 days where for the prior 7 months it beat SPY by 7 points. The prior audit framed this as conservative-by-design; that framing was wrong. Initial follow-up surfaced four concrete issues (vol_uncertainty silently-disabled-then-flipped-on; fragility one-way ratchet; vvix/skew never wired; per-position dollar P&L bug). The discovery is to find every bug in the same surface, fix them in safe order with rollback discipline, recalibrate throttles against the corrected signals, and verify on the 900-day historical span before any live cutover. The discovery is *discovery_bearing* — additional bugs are expected to surface during execution and must be added to the finding list, not silently fixed.

## Required method

The executor chooses how to deliberate, in what order, and what intermediate artifacts are useful, within the following minimal procedural floor:

1. **Read the Read-first list.** Especially the prior bad audit. Internalize what framing trap to avoid.

2. **Phase 1 — Bug audit.** For each of F-1 through F-4 below, plus a sweep of adjacent decision-pipeline code, produce a code-level confirmed/disconfirmed finding tied to file:line and a small Python reproducer against the snapshotted data. Add any new findings (F-5+) discovered during the sweep. Write `docs/plans/2026-04-29-phase1-bug-audit-findings.md`.

3. **Phase 2 — Comprehensive signal diagnostic.** For every column in `timeseries.json` (26 fields), compute a one-line health card: distinct values, monotonicity, step changes, fallback frequency, correlation with regime. Flag every dead/stuck/ratchet/step signal and trace to its source in `src/signals/`. Write `docs/plans/2026-04-29-phase2-signal-diagnostic.md`.

4. **Phase 3 — P&L and broker reconciliation.** Pull broker truth from Alpaca paper. Reconcile every dollar number on the dashboard against broker. Trace TWR-vs-raw $10k cashflow gap to its source. Write `docs/plans/2026-04-29-phase3-pnl-reconciliation.md`.

5. **Phase 4 — Fix sequence.** For each confirmed finding, create a separate branch (`ai/<short-task>`), a focused commit, and a unit test that locks in expected behavior. No bundling. Order by independence (smallest blast radius first). Each branch is independently reviewable.

6. **Phase 5 — Calibration sweep.** With Phase 4 fixes merged to a verification branch, run `scripts/run_hybrid_walk_forward.py` across three variants (current production / fixes only / fixes + recalibrated thresholds) on the 900-day dataset. Search throttle thresholds and cash-deployment rules. Recommendation must pass the §Acceptance test gate metrics. Write `docs/plans/2026-04-29-phase5-calibration-recommendation.md`.

7. **Phase 6 — Verification + staged go-live.** Pre-deploy checklist (all Phase 4 tests green; Phase 2 health table re-runs clean on backtest; Phase 3 reconciliation clean on backtest; Phase 5 gate passes). Shadow-deploy: new code live, OLD decision params, log what new params would have decided, for one trading day. Operator reviews. If sane, switch live. Day-1 / Day-3 / Day-7 monitoring.

8. **Postmortem capture.** Three new entries in `docs/POSTMORTEMS.md`: (a) silent-fallback-signals class (the vvix/skew never-wired and vol_uncertainty disabled-then-flipped patterns); (b) one-way-ratchet class (fragility update rule); (c) first-pass-audit-framing-trap (specifically the rationalize-the-anomaly failure). Each entry must include the *why* and the *how to prevent recurrence* — not just *what changed*.

9. **Update `docs/PLAN.md`** with a new Phase 9 entry summarizing the diagnostic and the fixes.

10. **Stop.** Write `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` ending with the literal final line in §Stop condition.

## Constraints

- **Read-only on `frontend/src/`** — operator has separate uncommitted WIP there; do not touch.
- **Read-only on the prior audit doc** (`docs/plans/2026-04-28-performance-audit-since-hybrid.md`) — preserve as historical record of the framing trap.
- **No live deploy or live-param mutation outside Phase 6.** Phase 6 itself is operator-gated.
- **No bundled commits.** One fix = one branch = one commit = one test. Operator must be able to revert any fix individually.
- **No "good enough" P&L math.** Either the dashboard matches the broker or it does not get displayed.
- **No silent fixes during the audit.** Every finding gets logged in the Phase-1 doc, even ones tempting to silently fix on the side. The point of the discovery is to know what was wrong, not just to ship a working result.
- **Read code, not memory.** Every claim about behavior must cite `file:line` or commit hash. Memory of how the system "should" work is not evidence.
- **Trust the broker number, not the paper number.** If the dashboard's `total_value` and the broker's account equity disagree, the broker is right by default. The dashboard is wrong until proven otherwise.
- **No skipping verification.** Phase 5 walk-forward gate must pass before Phase 6 begins. If it doesn't pass, that is the finding — surface it; do not weaken the gate.
- **AWS cost rule** (per CLAUDE.md): no S3 versioning; no new persistent infra without operator approval. Calibration runs use existing optimizer harness.

## Required return

Under the write surface above:

1. `docs/plans/2026-04-29-phase1-bug-audit-findings.md` — per-finding (F-1 through F-N) with: code location, reproducer snippet against snapshotted data, recommended fix summary, risk class.
2. `docs/plans/2026-04-29-phase2-signal-diagnostic.md` — full 26-row signal health table; flagged-signals follow-up list.
3. `docs/plans/2026-04-29-phase3-pnl-reconciliation.md` — reconciliation table broker vs dashboard; gap explanation per row.
4. Phase 4 fix-branches merged to main (or to a single verification branch pending operator approval), each with one passing test, one focused commit, one paragraph of rationale in the commit message.
5. `docs/plans/2026-04-29-phase5-calibration-recommendation.md` — sweep table; recommended thresholds; sensitivity analysis (±10%); 900-day gate-metric pass/fail per variant.
6. New decision-params bundle file (path per project convention; current bundles live in `config/`).
7. Phase 6 shadow-day output (the log of what new params would have decided on shadow day) plus operator-reviewable diff.
8. Three `docs/POSTMORTEMS.md` entries: silent-fallback-signals, one-way-ratchet, first-pass-audit-framing-trap.
9. `docs/PLAN.md` Phase 9 entry summarizing the work.
10. `docs/plans/2026-04-29-trader-bot-diagnostic-RETURN.md` — synthesis doc: what was wrong, what was fixed, what the system now does differently, what to monitor day-1/day-3/day-7, ending with the literal final line in §Stop condition.

---

## Findings to confirm or disconfirm (starting points, not exhaustive)

These are starting points from the prior follow-up audit. The Phase-1 sweep is expected to add F-5+ findings.

### F-1 — `vol_uncertainty` was effectively disabled for 8 months, then turned ON 2026-04-02

`timeseries.json` evidence: vol_uncertainty was a flat `0.10` from 2025-08-04 through 2026-04-01 (entire 8-month live history). On 2026-04-02 it stepped to `0.88` and has been firing daily since (range 0.52–0.90).

Likely commit: `e61f9c5 (2026-04-01 16:25 ET) feat: bounded midday check — trailing stops, VIX circuit breaker, skipped-buy re-check`. Needs source confirmation.

Effect on live: the throttle gate that silently never fired for 8 months turned on the day after this commit, dropping `position_size_modifier` from baseline ~0.86 to ~0.49. Performance lag begins exactly at this date.

### F-2 — `fragility_score` is a one-way ratchet

Across 187 days of live history: 125 flat days, 38 up days, **only 23 down days**. Biggest one-day drop in fragility ever recorded: **−0.038** on 2026-02-11. Currently `0.98`, has not dropped below `0.97` in 6 weeks. A signal that physically cannot decline more than 4 percentage points in a day in a `[0,1]` range is not a fragility detector — it is a smoothed running max. Confirm by reading `src/signals/fragility.py`.

### F-3 — Vol "complex" is a single-input signal in disguise

Phase 6 spec called for VIX + VVIX + SKEW + term-structure as the volatility complex. Across 187 live days:
- `vvix_percentile`: **1 distinct value (0.5 fallback)** — never wired up
- `skew_value`: **1 distinct value (0.0 fallback)** — never wired up
- `vix_percentile`: 20 distinct values, identical to `vol_uncertainty_score`

The composite IS just VIX-percentile padded with two dead inputs. Phase 6 was incomplete in production.

### F-4 — Per-position dollar P&L is wrong on at least one holding

VUG row in dashboard.json today: `entry_price 73.48, current_price 82.77, shares 102.77`, `unrealized_pnl_pct +12.64%` (consistent with prices), `unrealized_pnl −$6,097.78` (sign and magnitude both wrong; expected +$954). Either the pct is right or the dollars are right. May propagate into headline `total_value`.

### F-5+ — additional bugs surfaced during execution

Add findings here. Required: each one gets a code location (`file:line`), a reproducer, a fix summary, a risk class. Do not silently fix. Do not absorb into another finding.

---

## Hard rules during execution

- Read code before claiming behavior. `file:line` for every claim.
- Each fix = one branch = one commit = one test.
- Three postmortems are non-negotiable. They prevent the same trap.
- Operator gates Phase 6 live cutover. Do not deploy to live without explicit approval.
- If a finding feels small and rationalize-able, that is the trap. Surface it harder.
