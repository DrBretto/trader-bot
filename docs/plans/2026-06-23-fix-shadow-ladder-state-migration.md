# Fix shadow nightly KeyError 'R' — legacy→ladder state migration

## Context

The shadow nightly job (`runs/pkt_tb_007_orthogonal_brain/shadow/shadow_nightly.py`,
launchd `com.traderbot.shadow`) has crashed every night since 2026-06-17 with
`KeyError: 'R'` at `settle_dates` line 177:

```
books = {b: book_from_dict(state["books"][b]) for b in BOOKS}
```

Root cause: the ladder was upgraded from the legacy 3-book scheme
`BOOKS_LEGACY = ("I","A","B")` to the 5-rung organ ladder
`BOOKS = ("I","R","F","E","U")`, but the PERSISTED state (local `state/state.json`
+ its S3 mirror `s3://investment-system-data/shadow/pkt_tb_007/state.json`) and
the ledger files still carry only `I/A/B`. So `state["books"]["R"]` KeyErrors
before the job can settle or publish. The live `dashboard/shadow_timeseries.json`
is frozen at `as_of 2026-06-16` (schema v1, books A/B/I) — last good publish
predates the ladder change.

Persisted state at time of fix: `start_date=2026-06-11`,
`last_settled_date=2026-06-13`, `n_settled=3`, books `{I,A,B}` (all evolved over
3 settled days 06-11..06-13).

## Plan

- Add an idempotent, forward-only `migrate_state_books(state, ledgers_dir)` to
  `shadow_lib.py` that self-heals legacy state on load:
  - book state: keep `I`; rename `A`→`F`, `B`→`E` (the `LEGACY_BOOK_ALIAS`
    mapping — identical genomes); seed the two genuinely-new rungs from their
    ladder parents on the EVOLVED books: `R` from `I` (R is the regime throttle;
    over 06-11..06-13 the regimes `high_vol_panic`/`risk_on_trend` are not in
    `REGIME_EXPOSURE` → identity → R==I exactly), `U` from `E` (no
    `selected_universe` published in that window → U==E exactly). This is
    identical to how a fresh D0 seed sets all books equal, so the migration is
    faithful.
  - ledger files: rename `actions_A.jsonl`→`actions_F.jsonl`,
    `actions_B.jsonl`→`actions_E.jsonl`; copy `actions_I.jsonl`→`actions_R.jsonl`
    and `actions_E.jsonl`→`actions_U.jsonl` (R==I, U==E over the migrated
    window); rewrite `equity_ledger.jsonl` rows renaming `nav_A/n_actions_A`→`_F`,
    `nav_B/_B`→`_E`, and adding `nav_R=nav_I`, `nav_U=nav_E`, `n_actions_R=0`,
    `n_actions_U=n_actions_E`.
  - Guard: idempotent — if books already carry the full ladder set, no-op.
- Call the migration in `shadow_nightly.load_state` so it self-heals on every
  load before `settle_dates` touches `state["books"][b]`.
- Add a regression test `test_legacy_state_migration` to `tests/test_shadow.py`.
- Run the job to catch up 06-17..06-23 and publish; verify fresh `as_of` and
  populated ladder series; run the shadow test suite; confirm launchd loaded.

## Execution Log

- Confirmed root cause: local + S3 state both `{I,A,B}`; live timeseries frozen
  `as_of 2026-06-16`. Regimes over settled window don't bite REGIME_EXPOSURE →
  R==I; no selected_universe → U==E. Migration via parent-seed is exact.
- Added `migrate_state_books` + `_migrate_ledgers` to `shadow_lib.py`; wired into
  `shadow_nightly.load_state` (persists + logs on apply). Added regression
  `test_legacy_state_migration`. 17/17 tests green.
- Dry-run (`--no-publish`) on production state: migration applied, caught up
  06-16..06-20 (06-15/06-22 non-decision dirs; 06-19/06-20 SKIP-FLAT held;
  06-23 provisional), exit 0, no KeyError.
- Real publish run: exit 0, 11 S3 objects, migration a no-op (idempotent).
  Live `shadow_timeseries.json` now `as_of 2026-06-23`, schema v2, all 5 ladder
  books populated, organ_ledger 4 rungs (U-E "positive"), n_settled=8,
  mean_ic=0.5866. S3 state mirror migrated; removed orphan legacy
  actions_A/B.jsonl from S3.
- launchd `com.traderbot.shadow` confirmed loaded (registered with active
  calendarinterval streams, 23:30 Mon–Fri); absent from `launchctl list` only
  because it's an idle calendar-interval agent. No reload needed.

## Follow-ups

- None; the guard makes the migration a permanent no-op once applied.
