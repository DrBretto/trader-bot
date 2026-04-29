# Run Artifacts Contract

## Run Root
Each optimizer cycle writes local artifacts to:

- `runs/optimizer/<run_id>/...`

Run ID format:

- `run-<UTC_YYYYMMDDTHHMMSSZ>-seed<seed>-g<generations>-p<population>`

## Required Per-Run Files
- `run_manifest.json`
- `param_space_snapshot.json`
- `walk_forward_folds.json`
- `champion_metrics.json`
- `challenger_metrics.json`
- `gate_segment_metrics.json`
- `guardrail_results.json`
- `promotion_decision.json`
- `candidate_params_bundle.json`
- `generation_log.jsonl`
- `run.log`

## Global Files
- `runs/optimizer/index.json`
- `runs/optimizer/active_params_lineage.json`

## Dashboard-Consumed Mirrors
The optimizer mirrors static JSON for the dashboard UI into both:

- `dashboard/data/...`
- `frontend/public/data/...`

Required mirrored files:
- `optimizer_runs_index.json`
- `optimizer_runs/<run_id>.json`
- `active_params_lineage.json`

## Status Semantics
- `failed`: runtime error
- `completed`: run finished; no distinct challenger promotion
- `rejected_guardrails`: challenger failed hard constraints
- `rejected_objective`: guardrails passed but promotion deltas failed
- `promoted`: challenger promoted to active
