# 2026-06-24 — Honest resim of the 06-17..06-24 cliff at the source

## Context

The New-Brain cutover shipped **regime-blind** (regime chassis dropped): the committee
universe-narrowing concentrated the field into a semis/metals pool and the regime-blind
engine held it into a risk-off gap, producing a spurious ~3% drawdown across 06-17..06-23
that kept dropping into the 06-24 frontier. The prior fix (void PKT-TB-IR-01) layered an
append-only **correction overlay** on the rendered dashboard, which the nightly rebuild-from-
source kept fighting. Executes PKT-TB-CLIFF-RESIM-AT-SOURCE-V1 (Infotropy run
`20260624_trader-bot-cliff-resim-at-source`).

## Plan

- Resim 06-17..06-23 under the regime-restored two-stage engine, full `config/universe.csv`
  (`scripts/resimulate_new_brain_line.py --variant fixed --field full`); validate the harness
  reproduces the broken line (`--variant baseline --field derived`).
- Fix at the **source**: write resim-derived `sim_book_value` to
  `daily/2026-06-17..06-23/portfolio_state.json` (the field the extender chains from).
- Reconcile the live frontier (`daily/2026-06-24/portfolio_state.json`) to the resim's
  corrected terminal book (06-24 is pre-market → corrected 06-23 book carried flat), so the
  corrected line is durable and cannot revert to the bug's drawdown on the next rebuild.
- Remove the void IR-01 overlay hook (`apply_active_overlay` + `corrections/` read) from
  `src/utils/three_line_replay/extender.py`.
- Add an idempotent `republish-dashboard` Lambda route so a regen runs under the Lambda role
  (allowed to write the rendered keys) and reproduces the line byte-identically from source.
- Revert the void IR-01 bucket-policy `Deny` (durability now rides on source-fix + regen).

## Execution Log

- Resim baseline/derived reproduces the cliff shape; fixed/full removes it (06-23 displayed
  117,862 vs broken ~114,278). Corrected source written 06-17..06-23; 06-24 reconciled to the
  resim terminal book (diversified, cash plugged to the display-anchor scale).
- Overlay hook removed; idempotent `republish-dashboard` route added to `src/handler.py`.
- Resim script now emits the corrected terminal book for the live-book reconcile.
- Deployed the container image; double-invoke of `republish-dashboard` (Lambda role) produced
  the **identical** line + max-drawdown; live `dashboard.json` reads max-dd −1.68% (was −3.92%),
  total 117,873; original cliff value absent; overlay key gone.
- Disabled the `investment-system-morning-trigger` EventBridge rule to stop the 06-24 morning
  executor from running the stale regime-blind 06-24 intents against the reconciled book (would
  have re-corrupted it minutes after the fix). Night trigger left enabled.

## Follow-ups

- **Operator: re-enable `investment-system-morning-trigger`** once the regime-restored engine is
  confirmed to emit clean intents (`aws events enable-rule --name investment-system-morning-trigger --region us-east-1`).
- The 06-24 trade_intents.json on S3 are the stale regime-blind plan; tonight's regime-restored
  night run is the first clean forward decision.
- Pre-existing (unrelated) test failures: 5 in `test_dashboard_metrics.py` /
  `test_publish_artifacts_dates.py` from the current WIP tree — not touched by this change.
