# Fix: portfolio_state.json reset to empty $100k book

## Context

The live dashboard showed the SPY benchmark "dropping ~10% overnight" on 2026-06-18.
SPY did not move — the benchmark line reverted to exactly $100,000 (the seed value).

Root cause: commit `530564f` (PKT-TB-012, 2026-06-16) changed the night and morning
portfolio-state writes in `publish_artifacts.py` to route through
`paper_trader.to_published_state(portfolio_state)`, but **that function was never
defined**. Every night/morning `portfolio_state.json` write has since thrown
`AttributeError`, caught silently by the surrounding `try/except` ("Failed to publish
portfolio_state.json"). The only successful writer is the midday check
(`midday_checker.py:372`), which writes the raw portfolio dict.

Continuity therefore depends entirely on `latest.json` pointing at a day whose
`portfolio_state.json` happens to exist. The reset triggers when the morning run
advances `latest.json` to today (06-18) but cannot persist the state — the next
`load_portfolio_state()` finds no file at `daily/<today>/portfolio_state.json` and
falls through to the hardcoded empty $100k default (`paper_trader.py:67-74`). Midday
then persists that empty stub. Holdings + benchmark tracking are wiped.

The same broken commit also left `tests/test_single_book_invariant.py` failing
(10 tests) — it is the regression lock / spec for the missing code, including the
email-summary label change ("Portfolio (canon line):").

## Plan

- [ ] `src/steps/paper_trader.py`: add `SIM_BOOK_ROLE`, `to_published_state()`,
      `_restore_internal_keys()`; wire `_restore_internal_keys` into
      `load_portfolio_state()` so in-memory state always carries `portfolio_value`.
- [ ] `src/steps/midday_checker.py`: route the portfolio_state write through
      `to_published_state()` (currently leaks raw `portfolio_value` — violates the
      single-book invariant).
- [ ] `src/utils/sns_alerts.py`: rename "Portfolio Value:" → "Portfolio (canon line):"
      in night/morning summaries; emit "unavailable" (no `$`) when value is None.
- [ ] Run `tests/test_single_book_invariant.py` + full suite; all green.
- [ ] Deploy Lambda per docs/DEPLOY.md.
- [ ] Data repair: restore `daily/2026-06-18/portfolio_state.json` from the last
      good book (06-17) so tonight's night run continues from real holdings instead
      of the empty stub.

## Execution Log

- 2026-06-18: Diagnosed. `to_published_state` referenced at publish_artifacts.py:758
  & :1019 but undefined (`hasattr` False). Confirmed via S3: 06-15/16/17 states good,
  06-18 is a 269-byte empty stub written by midday at 18:00 UTC. Test spec found.

## Follow-ups

- The 06-18 morning executed 5 buys (SLV/SOXX/MTUM/VLUE/XLC) against the phantom
  empty $100k book. Continuity restore from 06-17 discards those phantom trades.
- Consider a fail-loud guard in `load_portfolio_state`: never silently seed-reset a
  live account when `latest.json` points at a date whose state file is missing.
