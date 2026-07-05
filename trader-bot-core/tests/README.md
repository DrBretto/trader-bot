# Reality Test Suite (P7)

The old 570-test suite is **discarded as the acceptance surface**: it mocked every
input and asserted the failure value (`gdelt_doc_count == 0`) as correct, so a
two-week frozen production line coexisted with a green suite. This is its
replacement — a **three-tier, reality-asserting** suite.

**Load-bearing rule:** no test asserts a value it (or a fixture) planted. Every
green traces to an **independent reality** — the live S3 bucket
`investment-system-data`, the live Yahoo v8 feed, the live GDELT v2 files, or a
**dated real recording** under `fixtures/`. The `test_no_planted_value_asserts.py`
guard enforces this mechanically.

## Tiers

| Tier | Marker | Cadence | What it proves |
|---|---|---|---|
| unit-fixture | `unit` | every commit | parser/engine logic on **real recorded** data |
| live-canary | `live_canary` | **nightly + post-pipeline** | TODAY's real S3 output isn't frozen/degraded |
| external-cross-check | `external_check` | nightly | stored values **== the real feed** (Yahoo/GDELT) |

## The 14 tests

- **unit:** `price_parser_on_recorded_fixture`, `regime_rerank_invariant`,
  `selection_set_invariant_under_sizing` (+ xfail incumbent neg-control),
  `no_silent_degrade_aborts` (fault-injected on the real publish guard).
- **live-canary:** `gdelt_nonzero_and_varying`, `ohlcv_store_advances`,
  `forecast_rotates_day_over_day` (hard-fail if byte-identical),
  `canon_line_is_not_sim_book`, `challenger_line_present_nonempty`,
  `freshness_gate_on_real_store`, `regime_chassis_loaded_nontrivial`.
- **external-check:** `price_equals_external_feed` (S3 OHLCV vs Yahoo v8, <0.5%),
  `gdelt_ingest_matches_source` (±5%), `canon_line_equals_recompute`
  (line == Σ shares·settled, <0.5%).

Each live-canary has a `fault_injection` twin that feeds a deliberately broken
reality (frozen store / byte-identical mu / empty publish) and asserts the canary
goes **RED** — proving it catches the real failure modes that went undetected.

## Salvaged legacy tests (the only survivors, demoted to unit)

`selection_set_invariant_under_sizing` (re-pointed to `engine`), the regime unit
(`regime_rerank_invariant`), `transaction_costs`, and the freshness/monitor unit
half (`freshness_and_store_advance`). Everything else in the old `tests/` is
retired as the acceptance surface.

## Running

```bash
export AWS_PROFILE=personal AWS_REGION=us-east-1     # live S3 creds
python trader-bot-core/tests/run_canaries.py commit    # Tier-1, fast
python trader-bot-core/tests/run_canaries.py live      # Tier-2 (post-pipeline)
python trader-bot-core/tests/run_canaries.py external  # Tier-3
python trader-bot-core/tests/run_canaries.py nightly   # live + external
```

Post-pipeline hook: `monitors.canary_gate.run_post_pipeline_canaries()` (fires SNS
CRITICAL on red). Nightly cron: `.github/workflows/reality-canaries-nightly.yml`.
Wiring into the prod handler is P8/P9 (this packet does not cut prod over).
