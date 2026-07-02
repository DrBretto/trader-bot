# True Replay: Canon + Shadow reconstruction (PKT-TRADER-BOT-TRUE-REPLAY-CANON-AND-SHADOW-V1)

## Context
PKT-5 (seamfree-reconstruct) re-marked the *broken* engine's held holdings
(`daily/<D>/portfolio_state.json` at settled prices) instead of replaying the
corrected `production_forecaster` to recompute what it would have picked/held each
day. The lines barely moved (±0.5%) and still reflect frozen-forecast / wrong-universe
picks. This packet OVERRIDES that: replay the corrected engine over as-of-D substrate
for BOTH canon and shadow, 06-17 → latest settled, one seam-free code path; and fix
the forward stale-substrate false-abort.

## Plan
- [ ] **A. Fix the forward false-abort.** The freshness gate's invariant 2
  (`gap non-empty AND bars_added<=0`) misfires when a `daily/<today>/` folder exists
  with only morning artifacts (no settled `prices.parquet`) — a not-yet-closed day is
  read as the frozen-substrate signature. Fix: count only gap dates that carry a
  settled `prices.parquet` (`gap_settled`). Phantom morning-only folders + Sunday
  folders are excluded; a genuine freeze (settled bar exists, unspliced) still fires
  loud. Files: `src/brain/runtime.py` (`_extend_ohlcv_from_s3`, `_freshness_gate_verdict`).
  Test: `tests/`.
- [ ] **B. Real-engine replay harness.** For each D in the real NY grid (06-17→latest
  settled): `build_panel([D])` (as-of-D, leak-free) → `run_inference([D])` (real
  `production_forecaster` mu) → `run_cutover(D, forecaster=production-mu)` (regime
  chassis + Stage-1/Stage-2) → held_symbols/intents → book sim (settled-price marks).
  Same code path as live forward. Must ROTATE picks day-over-day. Driver in the run dir.
- [ ] **C. Reconstruct canon + shadow.** canon = engine book from replayed picks marked
  at settled prices; shadow = M1-tilt of the SAME replayed inference (0.0-diff mu parity).
  Both anchored at preserved 06-16 display value; champion_freeze byte-immutable; R0
  anchor/scaling/styling preserved.
- [ ] **D. External price verification.** SOXX 06-30→07-01 −6.4%, SMH −5.4%, SPY −0.1%.
- [ ] **E. Persist append-only + supersede, ship (no gate), deploy clean SHA image.**

## Execution Log
- 2026-07-02: Diagnosed the false-abort — confirmed `daily/2026-07-02/` in S3 holds only
  morning artifacts (no `prices.parquet`); it enters `gap` and splices 0 bars →
  invariant-2 misfire. Fix designed: `gap_settled` (prices.parquet-present only).

## Follow-ups
- Frontier append-only date-set: phantom leaves 06-19/20/27 and missing real days
  06-26/29 (PKT-5 finding) — carry forward if it blocks a clean grid.
