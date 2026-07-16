# 2026-07-16 - Health canary runtime repair

## Context

The July 16 night completed settled-price production, inference, and artifact writes,
then failed its post-pipeline gate with exit code 1. The independent three-line
watchdog remained green: promoted TILT, SPY, and the two-stage comparison all reached
July 15 with matching ledger and dashboard values. The failing process was the live
reality-canary subprocess, whose deployed image did not contain pytest.

The incident exposed two additional false-green risks. The live canary still read the
retired clean-v2 ledger instead of the promoted TILT ledger, and the forecast-rotation
check selected the newest two dates that happened to contain inference artifacts even
when those artifacts were weeks behind the settled frontier.

## Plan

- [x] Package the canary runner in the Lambda image and lock that deployment contract.
- [x] Point reality checks at the promoted ledger and make the forecast fingerprint
  understand the current `mu` record schema.
- [x] Persist the successful clean-core forecast record every night and require the
  forecast canary to reach the current settled session.
- [x] Surface CBOE refresh failures that are currently swallowed and inspect the live
  Lambda-context result.
- [x] Run focused and broad tests, commit and push only the owned files, deploy the
  image, then prove the live canary and three-line health checks are green.

## Execution Log

- 2026-07-16: Reproduced the inbox error from CloudWatch. Three EventBridge attempts
  failed after otherwise successful night work because `/var/lang/bin/python3.11`
  reported `No module named pytest`.
- 2026-07-16: Invoked the production canary directly and reproduced the exact exit 1.
  Separately verified both post-replay and post-morning three-line health checks green
  through July 15, with TILT `114848.72179181811`, SPY `109417.65600035332`, and
  two-stage comparison `114480.69094528201`.
- 2026-07-16: Confirmed all 11 reliability alarms are currently OK. The Lambda-errors
  alarm correctly entered ALARM for the failed night and recovered after retries.
- 2026-07-16: Found the live canary pinned to `equity_ledger_clean_v2`, while the
  watchdog correctly follows `lines.ledger.CACHE_KEY` to promoted TILT canon.
- 2026-07-16: Found no `daily/<D>/inference.json` artifacts after July 2. The clean
  night computes current `mu` but drops the record before persistence, allowing the
  rotation canary to pass against July 1 and July 2 indefinitely.
- 2026-07-16: Found CBOE history stuck at June 9 in Lambda while all seven official
  endpoints are current through July 15 from a local fetch. Per-index HTTP failures
  are returned by the fetcher but discarded by the forward path, leaving no usable
  production diagnosis.
- 2026-07-16: Added the production pytest dependency, carried the exact inference
  record through `CutoverResult`, persisted it as `daily/<D>/inference.json`, and
  made the canary reject any artifact frontier behind the latest settled session.
- 2026-07-16: Made forecast fingerprints canonical over predictions rather than
  timestamps, imported the promoted ledger pointer instead of duplicating a retired
  prefix, and updated the promotion-era model-id and comparison-line assertions.
- 2026-07-16: Added a browser-header CBOE primary plus a named Yahoo/yfinance tail
  fallback, and returned per-index status through the non-destructive forecast diag.
- 2026-07-16: Verification passes: 14 focused reliability tests, the 40-test commit
  canary selection (39 passed, one expected xfail), and 32 active clean-core replay,
  single-book, publish-date, and market-calendar tests. The pre-deploy live canary
  now has exactly one red: its newest forecast is July 2 while the settled frontier
  is July 15, the production artifact gap this deployment is intended to close.
- 2026-07-16: Rechecked infrastructure. All scheduled trader-bot rules remain in
  their expected enabled/disabled states and all 11 current trader-bot reliability
  alarms are OK.
- 2026-07-16: Deployed commit `6e550fd` as immutable image digest
  `sha256:5321f44a172010f1c71bc248ae810f82ec34bdae6a0b3c4896f303f21d2ea184`.
  A fresh Lambda canary proved pytest is present and correctly rejected only the
  stale July 2 forecast before the repair run.
- 2026-07-16: Re-ran the settled July 15 night. It persisted the exact 64-symbol
  forecast, kept the replay-owned TILT terminal unchanged at `114848.72179181811`,
  and returned both reality and config canaries green. An independent invocation
  then passed all 10 live canaries under Lambda's Python 3.11 runtime.
- 2026-07-16: The Lambda-context forecast diagnostic is current and green: OHLCV
  through July 15, 64 forecasts, no settled gap, and all seven CBOE series refreshed
  from their official CSVs through July 15.
- 2026-07-16: The post-morning health probe exposed one final retry edge case. A
  successful night rerun after morning could overwrite the later operational pointer
  date/phase with the settled night date. Added a monotonic pointer merge and tests;
  today’s pointer will be restored through the resumable morning checkpoint.
- 2026-07-16: The completed-checkpoint fast path correctly skipped execution but
  also skipped pointer repair. Added a metadata-only repair on completed replay so a
  regressed/missing pointer is restored without touching fills, portfolio state, or
  trade logs.
- 2026-07-16: Deployed the final code checkpoint `2b1cc57` as immutable digest
  `sha256:2272a54a67bcd177c0d17e92cf9c4afd05c2ebc6e32efd2eb38a19bec701f133`.
  The completed July 16 checkpoint returned `idempotent_replay:true` and
  `pointer_repaired:true`; `daily/latest.json` is back on the July 16 morning
  snapshot and the existing trade artifact remains six rows, last modified at the
  original 09:45 execution.
- 2026-07-16: Final independent production acceptance is green: all 10 live canaries
  pass under Lambda Python 3.11, all three displayed lines have zero settled-session
  lag, the public morning snapshot and receipt are current, and all 11 trader-bot
  CloudWatch alarms are OK. Public values match the promoted ledger: TILT
  `114848.72179181811`, SPY `109417.65600035332`, and two-stage comparison
  `114480.69094528201` (published to cents as `114480.69`).
- 2026-07-16: One operator-induced diagnostic alert was sent when the first manual
  health probe explicitly forced `today=2026-07-16` before that market session had
  settled. The subsequent real-clock health probe is green and the health-red alarm
  returned to OK; this was not a line or pipeline failure.

## Closeout

The exit-code incident is closed. The production image contains the canary runtime,
nightly forecasts persist from the exact decision record, canaries follow promoted
TILT canon and require the current settled frontier, CBOE is current and visible,
and retry paths cannot regress or strand the morning pointer. No duplicate execution
occurred during repair.
