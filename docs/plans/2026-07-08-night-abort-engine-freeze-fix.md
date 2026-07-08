# 2026-07-08 — Night abort: engine cold-start FREEZE fix (GATE-A1)

## Context
The scheduled night (`app.night.run_night` via the clean thin router) aborted in
~5–9s to the incumbent on 07-07 and 07-08, producing no `daily/` artifacts — so
neither line advanced past the 07-02 replay seed (GATE-A1 failed). The abort reason
is not in stdout (it goes to the abort SNS).

Diagnosis (real evidence, live reads + a non-destructive `night` probe invoke):
the abort reason is
`FREEZE assertion failed (ABORT): engine_sha mismatch: got 870ade52… expected 4ae7b79d…`.
`run_cutover`'s cold-start freeze (`forecast.freeze.assert_cold_start`) hashes
`trader-bot-core/engine/*.py` and compares to `brain/FREEZE_ORB1.json`
(`4ae7b79d` = the pre-registered `src/brain/engine`, LIVE_PREREG line 51 / §3,
operator authority `D-AUTO-20260616`). The deployed clean engine had drifted:

1. `engine/hysteresis.py` — a replay/marking helper (imported only by
   `replay/driver.py`, the offline seed tool) had leaked into the frozen engine
   package (untracked; baked via `COPY trader-bot-core/`), poisoning `engine_sha`.
2. `engine/allocation.py` + `engine/contracts.py` — the P9 `target_weights` output
   field (a marking-layer concern read only by `replay/driver.py`) was added to the
   frozen decision engine without re-freezing.

Neither is used by any LIVE path (night / morning / shadow-publish CU-02 / watchdog);
both are used only by the offline `replay/driver.py`. The freeze correctly caught the
leak. `model_sha` matched, so `engine_sha` was the only freeze problem.

Fix direction: honor LIVE_PREREG §3 (the deployed engine MUST be byte-identical to the
pre-registered `src/brain/engine` = `4ae7b79d`) — do NOT re-freeze (that would override
a baked operator pre-registration decision and weaken the gate to match a drifted
deploy). Instead restore byte-preservation by relocating the replay/marking helpers out
of the frozen decision engine into the marking/replay layer.

## Plan
- [x] Move `engine/hysteresis.py` → `replay/hysteresis.py`; update `replay/driver.py`
      import + docstring.
- [x] Revert `engine/allocation.py` + `engine/contracts.py` to the frozen
      `src/brain/engine` bytes (remove the `target_weights` field) → `engine_sha` = 4ae7b79d.
- [x] Add `publish.challenger.canon_target_weights(engine_out)` — derives the canon
      marking weights in the marking layer from the frozen engine output (Stage-1
      `sel.w_target` at the Stage-2 realized gross; same base the challenger tilts).
- [x] Update `replay/driver.py:559` to call it instead of `engine_out.allocation.target_weights`.
- [x] Verify `assert_cold_start()` OK (engine_sha 4ae7b79d, freeze OK); all imports resolve;
      engine/invariant unit tests pass.
- [ ] Deploy the image; grant the Lambda role `cloudwatch:PutMetricData` (done in IAM).
- [ ] Prove GATE-A1: a real night for run_date 2026-07-07 (store max 07-02, lag 3 =
      freshness tolerance) advances the canon line off the 07-02 seed.

## Execution Log
- Diagnosed the exact abort via `forecast-diag` + a non-destructive `night` probe invoke
  (returned the freeze `engine_sha` mismatch reason). Store max bar = 07-02;
  `max_available_bar` = 07-02 (no settled `prices.parquet` for 07-06/07-07 — see Follow-ups).
- Applied the byte-preservation fix; `assert_cold_start` now returns OK / `freeze OK`.

## Follow-ups
- **Substrate publisher gap (separate root):** settled `daily/<D>/prices.parquet` has not
  been written since 07-03 23:03 (the clean thin router replaced the old
  `src/steps/publish_artifacts` bar writer and does not write settled bars). The store is
  frozen at 07-02. The freshness gate correctly tolerates a 07-07 advance (lag 3), but the
  scheduled nights will re-stall (STALE) on run_date 07-08+ until settled-bar production
  resumes. Needs a dedicated packet (wire settled-bar production into the clean spine, or
  restore an EOD bar job). NOT a forced advance on stale data — the 07-07 advance is within
  the gate's designed holiday-weekend tolerance.
- `replay/driver.py` canon marking now uses Stage-1 `w_target` at realized gross (frozen
  output) instead of the removed post-lot `d_final/nav`. The current 07-02 seed in
  `clean_v2` is untouched; a future re-seed would use the new derivation.
