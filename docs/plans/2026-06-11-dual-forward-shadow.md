# 2026-06-11 — Dual Forward Shadow (PKT-TB-007 follow-on, operator-approved)

## Context

Operator approved the committee's ranked follow-on from PKT-TB-007: a costless dual forward
shadow. Two legs, both written nightly BEFORE outcomes exist: (1) paper shadow book(s) — the
incumbent's book with the brain's tilt applied on paper, marked nightly, rendered as a second
line on the dashboard next to the live line; (2) forecast-IC records — M1's nightly rankings,
scored at +5 trading days. Settles "is the forecast IC real forward" in ~7 months (Realist
spec); banks fresh holdout for any future brain. Configs: the value-blind a-priori genome
(shadow book A) and a-priori + M4-A damping (shadow book B, the pre-registered confirmation
arm). Verdict reads pre-registered in runs/pkt_tb_007_orthogonal_brain/shadow/SHADOW_PREREG.md.

## Plan

- [x] Shadow engine under runs/pkt_tb_007_orthogonal_brain/shadow/ (imports prototype machinery):
      nightly catch-up-capable deterministic job — pull daily/<D> artifacts, run M1 (+M4-A) inference,
      compute both paper books via the tilt adapter, mark to close, append forecast records, score
      matured IC, persist state to S3 shadow/pkt_tb_007/ + local mirror, publish
      dashboard/shadow_timeseries.json
- [x] SHADOW_PREREG.md — read dates, IC bar, utility equivalence band, M4-A sub-window rule (Realist spec)
- [x] Tests (determinism, catch-up idempotence, no-look-ahead timestamps, S3 round-trip)
- [x] launchd plist com.traderbot.shadow.plist (nightly, after the night pipeline lands)
- [x] First night run end-to-end (manual), verify artifacts + JSON
- [x] Frontend: render shadow line(s) when shadow_timeseries.json exists (graceful absent)
- [ ] STOP: ask operator before frontend deploy (CLAUDE.md checkpoint)

## Execution Log

- 2026-06-11: Branch ai/forward-shadow from ai/orthogonal-brain. Plan doc created.
- 2026-06-11: Shadow engine built (shadow/{shadow_lib,forward_inference,shadow_nightly}.py).
  Imports, not forks: tilt_adapter/genome_007/lot_fix_007/risk_stats_007/run_replay_007.cost_overlay
  (TB-007), FeatureStore/features_gdelt/gdelt_backfill/data_layer.fetch_cboe (TB-006),
  replay_engine dataclasses+executors (src, untouched). Copied minimally (noted in headers):
  seed_portfolio (date-parameterized), make_targets_007 evt_ctl/disp_lags/cal_dummies blocks,
  trailing_pct. Shadow-local caches under shadow/state (TB-006 caches seeded by copy; forward
  OHLCV spliced from S3 daily prices; GDELT/CBOE fetched forward, fail-soft to trained masking).
- 2026-06-11: Nightly parity self-check (recompute vs frozen prototype store/nightly_007) wired
  as an abort-gate. First run caught a real divergence: the frozen panel's turn-of-month dummy
  labeled 2026-06-09/10 month-end (truncation artifact). Forward convention corrected to the
  true NYSE calendar (projected month-end trading days); disp_z parity moved to complete-month
  dates. Final parity: mu 4.9e-07, disp_z 1.7e-03, p_exceed 5.0e-05, q 3.8e-05 — green.
- 2026-06-11: SHADOW_PREREG.md registered (genome hashes 952a2f5a565e / c9b8ee96c859; IC read
  at n=31 weeks ≈ 2027-01-27, bar t≥2 one-sided; utility reads at 164 td ≈ 2027-02-08 and
  291 td ≈ 2027-08-10 with equivalence band CI ⊂ ±1.5 bp/day; M4-A valid only on
  base-tilt-positive sub-windows, one arm, drop on miss; no-mid-stream-changes clause).
- 2026-06-11: 7 tests green (forward-only refusal, catch-up byte-idempotence, fresh-reprocess
  determinism, timestamp-before-outcome invariant incl. stale-record exclusion, JSON schema,
  book divergence, armed-skeleton publish).
- 2026-06-11: launchd com.traderbot.shadow loaded (23:30 ET Mon–Fri), state=not running
  (awaiting schedule), exit code never.
- 2026-06-11: First run end-to-end: pending=[2026-06-11]; forecast record written (64-symbol mu,
  recorded 12:49 UTC, late_record=false, feature+model shas); books unseeded as designed
  (daily/2026-06-12 not yet written — 2026-06-11 settles tonight); published
  dashboard/shadow_timeseries.json skeleton (armed) + 4 mirror objects under
  s3://investment-system-data/shadow/pkt_tb_007/.

- 2026-06-11: Frontend shadow overlay. New hook useShadowData.ts (same idioms as
  useTimeseriesData: VITE_DATA_URL same-prefix fetch, ./data/ fallback, cache-bust,
  fail-soft null on 404/parse/schema mismatch; types live in the hook since the payload is
  shadow-job-owned, also avoids the in-flight types/index.ts edit from another task).
  PerformanceChart: shadow_A solid amber 1.5px, shadow_B dashed amber 1px (subordinate),
  merged by date into the existing equity rows, tooltip rows, legend chips
  "Shadow: brain tilt (paper)" / "Shadow: +event damp (paper)"; armed-empty payload renders a
  single "Shadow (paper): armed — accruing" legend chip + neutral stats note (days accrued,
  mean IC (t), utility diff bp/day with 95% CI when present; read dates 2027-01-27 / 2027-08-10
  hardcoded with comment — prereg_pointer carries the doc path, not dates). App.tsx wires the
  hook into the desktop PerformanceChart. Armed skeleton committed as
  public/data/shadow_timeseries.json dev fixture. DEPLOY.md + FRONTEND.md sync commands gained
  --exclude "shadow_timeseries.json" (without it, `aws s3 sync --delete` would delete the
  pipeline-written shadow JSON on every frontend deploy). Checks: tsc+vite build green; eslint
  not installed (repo lint script skips by design); no frontend unit tests exist
  (playwright-verify.mjs / diag-runtime.mjs target the live site post-deploy). Local vite dev +
  Playwright verification: armed fixture → chip+note, zero console errors; fixture absent →
  nothing shadow-rendered, zero console errors (vite SPA-fallback HTML caught by schema guard;
  prod S3 404 caught by response.ok); populated fixture → both amber lines from start date
  forward + full stats note, zero console errors. BUILD ONLY — not deployed.

## Follow-ups

- Read dates land automatically per SHADOW_PREREG (IC read ~31 weeks; utility reads at 8/14 months).
