# New Brain canon-line re-simulation (2026-06-23)

Harness: scripts/resimulate_new_brain_line.py. Re-runs the New-Brain forward book
06-17..06-23 (the engine-driven canon days), seeded from the real 06-16 book, with
NO look-ahead: recorded forecast-ledger mu (parity 1e-7), each day's REAL derived
pool (brain_selected_universe.json — restricting to it reproduces the live
selection EXACTLY), real regimes, real prices, morning-executor fill semantics.

## The 06-22 -> 06-23 drop (raw book NAV)
| variant | 06-22 | 06-23 | one-day move |
|---|---|---|---|
| ACTUAL (live, deployed bug) | 97,932 | 94,078 | **-3.94%** |
| BASELINE (re-sim of the bug) | 98,284 | 95,584 | -2.75% |
| FIXED (corrected algorithm) | 98,215 | 96,861 | **-1.38%** |

Harness tracks the actual line within ~1.6% (fill/cost/rounding + the displayed
line is continuity-adjusted, not raw NAV). Within the harness (apples-to-apples)
the corrected algorithm roughly HALVES the drop.

## What actually drove the improvement — and what did NOT
- The regime SELECTION tilt was a **no-op here**: the derived pool is 10 names and
  N=10, so selection takes all 10 regardless of tilt. It can reorder, not exclude.
- The regime EXPOSURE cut did not fire on 06-23 (regime was risk_on, mult 1.0).
- The win came from the **cluster cap binding**: fixed groups SMH/SOXX/XLK/ARKK ->
  tech_growth and GLD/SLV -> precious_metals, capping each at 35% of NAV. Baseline
  (raw sectors) left them uncapped. Less concentration -> less gap damage.

## Root cause the re-sim exposed (upstream of the regime chassis)
The concentration was baked into the **universe derivation**: the daily pool
itself was ARKK,GLD,MTUM,SLV,SMH,SOXX,VLUE,VWO,XLC,XLK — semis+tech+metals. The
regime chassis operates WITHIN the pool; restoring it mitigates (cluster cap) but
does not de-concentrate a pool that was concentrated before selection ran. THIS is
the "wrong pool" — and it is a separate fix from the regime restoration.
