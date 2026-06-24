# FP-08-1 — Canonical Equity Ledger Primitive

## Context

Step 1 of 7 of the clean-core rebuild (CLEAN_CORE_ARCHITECTURE_DOSSIER §2/§5/§8).
The displayed equity line is currently recomputed + re-anchored on every publish, so
"history moves every other day." The cure is to make the line a **stored fact**: a
canonical, append-only, content-addressed equity ledger that the chart reads and
nothing recomputes.

This task builds the **primitive in isolation, unit-tested only**. It does NOT seed
from production, does NOT wire into any live render path, and deletes no existing code.
Those are FP-08-2/-3/-4. Nothing here changes any displayed number or live behavior.

## Plan

- [x] Read corrections.py (reuse its write-once / content-addressing primitives) +
      dossier §2 schema and §5 history-protection semantics.
- [x] New module `src/canon/equity_ledger.py`:
  - `equity_point.v1` leaf schema (value/benchmark/comparison + segment, model_id,
    source, prev_date, prev_content_hash, content_sha, issued_by, written_at, supersedes).
  - content-addressing via canonical bytes (deterministic JSON, `allow_nan=False`),
    reusing `corrections.content_sha` / `_canonical_bytes`.
  - write-once frontier append via `PutObject(IfNoneMatch="*")`.
  - frontier gate: only an append at a date strictly after the frontier advances the
    chain; same date + identical content is an idempotent no-op; same date + different
    content is rejected (G-APPEND-ONLY-FRONTIER); date < frontier is rejected.
  - ordered `_manifest.json` chain of record (the frontier), hash-linked leaf-to-leaf.
  - `equity_history.jsonl` cache as a **pure deterministic fold** over leaves+manifest,
    never written independently (G-CACHE-PROJECTION).
  - restore-from-leaves: rebuild manifest+cache by listing the `points/` prefix.
- [x] Unit suite `tests/test_equity_ledger.py` against an in-memory FakeS3 (the
      IfNoneMatch write-once stub already used by test_correction_overlay), covering
      every Acceptance-test item + both binding guards.
- [x] Run new suite + full existing suite; no regressions introduced.
- [ ] Commit + push the slice; write the return receipt.

## Execution Log

- Baseline (before any change): collectable suite = 345 passed / 12 failed / 1 skipped /
  1 xfailed; 5 modules error on collection from pre-existing missing deps
  (`requests`, `torch`). All unrelated to this primitive (pure JSON+S3, no heavy deps).
- Built `src/canon/equity_ledger.py` + `src/canon/__init__.py`.
- Built `tests/test_equity_ledger.py`.

## Follow-ups

- FP-08-2 seeds this ledger from the live rendered chart (G-SEED-SOURCE, G-TERMINAL-PIN).
- Supersede-chain head resolution for corrections lands in FP-08-5; this packet's
  restore walks the prev-hash frontier chain only (no corrections exist yet).
