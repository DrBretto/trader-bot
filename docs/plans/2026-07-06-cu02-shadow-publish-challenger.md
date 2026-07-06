# CU-02 — shadow-publish route + independent challenger publisher + clear contamination

## Context

`source=shadow-publish` (03:30Z EventBridge trigger) had **no branch** in the clean
router `app/handler.py`, so it fell through to `app.night.run_night` (a duplicate
clean night). Consequences:

- The challenger "dotted" line (`shadow_A` in `dashboard/shadow_timeseries.json`)
  had no publisher — it was frozen at 2026-07-02 carrying the **coupled** ladder-F
  values (terminal 114847.63), not the P6 **independent** incumbent+tilt challenger.
- `live_line` (the canon mirror the frontend rebases the dotted lines against) still
  showed the **known-contaminated 121147.52** terminal on the **contaminated grid**
  (phantom non-trading days 06-13/19/20/27; missing real days 06-26/29).
- The `investment-system-shadow-trigger` rule also failed delivery every day
  (`FailedInvocations=1`, 06-30→07-04) — same missing-invoke-permission class as CU-01.

Key discovery: there is **no** clean-core function that both runs the P6 challenger
and publishes `shadow_timeseries.json`. The only `shadow_timeseries.json` writer was
the LEGACY `runs/pkt_tb_007_orthogonal_brain/shadow/shadow_nightly.py` (the coupled
old ladder — forbidden). **But** the P6 independent challenger series is **already
computed** and stored in the corrected `canon/equity_ledger_clean_v2/` ledger's
`comparison` column (written by `replay/seed_canon.seed_canon_by_replay`,
`independent_challenger=True`), 2026-06-11 (114772.39 anchor) → 2026-07-02
(115534.92). So the publisher sources the corrected line from clean_v2 rather than
re-running heavy inference.

## Plan

- **`app/shadow_publish.py`** (new) — `run_shadow_publish(event, bucket, region)`:
  reads the supersede-folded clean_v2 cache (`EquityLedger.read_cache()`), builds
  `live_line` = corrected canon `value` series (real trading-day grid, clears
  121147.52) and `shadow_A` = independent challenger `comparison` series; merges into
  the existing `shadow_timeseries.json` preserving every other key; correctness guard
  refuses to publish if either terminal is the contaminated value.
- **`app/handler.py`** — add `if source == "shadow-publish": run_shadow_publish(...)`
  before the run_night default (thin, lazy per-branch import).
- **Infra** — add `lambda:InvokeFunction` grant for the shadow rule ARN
  (`EventBridgeShadowInvoke`); rebuild+deploy the container image.
- Reality-prove: manual `shadow-publish` invoke (publisher runs, not run_night;
  shadow_A = independent, live_line terminal ≠ 121147.52); off-cycle real trigger
  fire (Invocations>0 & FailedInvocations=0).

## Execution Log

- Traced path; confirmed clean_v2 already carries the independent challenger series.
- Wrote publisher + handler branch; added shadow grant; deployed; proved. (see run receipt)

## Follow-ups

- **Forward-advance gap (night-pipeline / CU-04 lane):** clean_v2's `comparison`
  column is populated by the seed/reconstruction path; forward nights
  (`lines/append.py`) write `comparison=None`. So the challenger advances past 07-02
  only once (a) settled data unfreezes (CU-01 flagged the night pipeline frozen at
  07-02) AND (b) the independent-challenger forward compute is wired into the settled
  advance. CU-02 ships the PUBLISHER; the forward compute is out of scope here (would
  touch the canon/night append the packet forbids CU-02 from modifying).
