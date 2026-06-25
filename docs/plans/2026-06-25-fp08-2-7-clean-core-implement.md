# 2026-06-25 — FP-08-2..7 Clean-core implement (stored equity ledger as the line)

## Context
The displayed equity line is still a nightly recompute (`extend_dashboard` re-anchor +
`dashboard_metrics` re-chain) that drifts/reverts. FP-08-1 built the ledger primitive
(`src/canon/equity_ledger.py`) but wired it into nothing. This finishes the clean core
(dossier FP-08-2..7): make the displayed line a stored, append-only, content-addressed
fact — seeded from the current certified line, read by the chart, never recomputed,
advancing one settled point per trading day, never reverting.

Authority: operator override of the divergence-committee seed gate — "seed from the
current live line" (R0; current terminal = $116,126.12, the live S3 dashboard.json
2026-06-25 night snapshot). Pin the seed terminal to the CURRENT line ($116,126.12),
not the dossier's stale $117,873.57.

## Plan
- [ ] **Ledger primitive** (`src/canon/equity_ledger.py`): make `comparison` Optional
      (the dotted comparison only exists for the recent shadow window); add supersede-chain
      resolution for FP-08-5. Keep all FP-08-1 guards/tests green.
- [ ] **FP-08-2 seed** (`src/canon/equity_seed.py`): read live S3 `dashboard/dashboard.json`
      (`equity_curve[].value`,`[].benchmark`,segment) + `dashboard/shadow_timeseries.json`
      `shadow_A` (raw comparison). Photograph into write-once leaves. terminal pin
      == 116126.12. assert_ledger_parity EXACT. write `meta.json`. NO calls to
      extend_dashboard/champion_freeze_map.
- [ ] **Line read-side** (`src/canon/equity_line.py`): pure fold of the ledger cache into
      {equity_curve, drawdowns, monthly_returns, line_metrics}; `assert_ledger_parity`
      gate (parity-or-hold).
- [ ] **FP-08-3 repoint**: `build_dashboard_data` reads the line from the ledger
      (equity_line), keeps trade/exposure stats from current state. parity-or-hold gate
      replaces `_verify_extension_or_alarm`. Rendered chart byte-identical.
- [ ] **Append-one-point** (`src/canon/equity_append.py`, lightweight; no torch/pandas,
      no execute/morning/midday import — G-APPEND-IMPORT-WALL): night path emits
      `daily/<date>/canon_point.json` (daily_return + benchmark + comparison from the
      regime-restored engine's settled marks — NEVER the persisted sim_book_value);
      append carries capital from yesterday's STORED leaf × (1+return). One leaf/day.
- [ ] **FP-08-4 delete**: delete `extender.py`, `replay_engine.py` (live path),
      `_load_daily_states`/`_apply_canonical_overrides`/`_select_active_segment`/
      `_build_return_rows`/`_build_continuity_rows`, the `:309` sim_book read; rewrite
      `_run_republish_dashboard` to pure re-serve; delete/rewrite
      `tests/test_continuity_repair_invariants.py`. Prove no reachable import from handlers.
- [ ] **FP-08-5 corrections**: corrections.py supersede-chain as ledger resolver; a
      correction = new attributed supersede leaf that sticks across regenerate.
- [ ] **FP-08-6 model socket**: `Model` protocol around `select()`; registry dispatch
      replacing `if engine_name=="tilt_adapter"` (publish_artifacts.py:146), validated vs
      FREEZE_ORB1 engine hash; tilt `select()` adapter.
- [ ] **Band-aid cleanup**: DELETE `config/new_brain_forward_freeze_20260625.json`
      (repo+S3), `scripts/build_new_brain_forward_freeze.py`; revert the forward-freeze
      loader in `canonical_replay_anchor.py`. KEEP paper_trader entry_date `.get()` fix.
      Preserve `champion_freeze_20260611.json` + universe manifest byte-identical.
- [ ] **Docker-popup fix**: stop `docker buildx build` surfacing Docker Desktop GUI.
- [ ] **FP-08-7 deploy + verify**: deploy; forced-night-run append-without-revert proof;
      verify live on trader-bot.infotrope.io; gated infra intact.

## Execution Log
- 2026-06-25: Read full publish path (handler, publish_artifacts, dashboard_metrics,
  extender, canonical_replay_anchor, corrections, equity_ledger). Confirmed current
  forward line == freeze table == resim (regime-ON) terminal 116126.12. Seed source =
  live dashboard.json equity_curve (248 rows 2025-08-04..2026-06-25) + shadow_A (7 pts).

## Follow-ups
- Append canon-return source (regime-restored engine settled marks) flagged for operator
  confirmation in the run receipt (executor_concern): the core stored/non-revert/advance
  property holds regardless; the return source is cleanly swappable via canon_point.json.
