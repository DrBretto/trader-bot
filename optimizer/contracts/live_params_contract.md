# Live Params Contract

## Runtime Source of Truth
The live bot loads decision configuration from exactly one file:

- `config/decision_params.active.json`

This file is loaded in `src/handler.py` and must include:

- `decision_params`
- `regime_compatibility`

Optional optimizer-managed sections:

- `signals`
- `regime_fusion`
- `decision_engine`
- `ensemble`
- `transaction_costs`

## Active Bundle Schema
```json
{
  "schema_version": "1",
  "version_id": "opt-...",
  "source_run_id": "run-...",
  "updated_at": "ISO-8601",
  "decision_params": {},
  "regime_compatibility": {},
  "signals": {},
  "regime_fusion": {},
  "decision_engine": {},
  "ensemble": {},
  "transaction_costs": {},
  "metadata": {}
}
```

## Candidate and Promotion Files
Optimizer run writes challenger bundle to:

- `config/decision_params.candidate.json`

Promotion is atomic local filesystem swap:

1. Archive current active bundle into `config/decision_params.history/`.
2. Replace `config/decision_params.active.json` with candidate file via atomic rename.

Rollback is also atomic and pointer-like:

- write selected history bundle back to `config/decision_params.active.json`
- archive previous active before swap

## Non-Optimizable Parameter Policy
Any parameter marked `safe_to_optimize=false` in
`optimizer/discovery/param_inventory.json` is never mutated by optimizer output.
