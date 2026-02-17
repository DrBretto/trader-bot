# Optimizer Runbook

## Overview
The optimizer is **offline-only** and runs as a one-shot local process on macOS. It never runs in AWS compute.

Canonical run command:

```bash
python -m optimizer.cli run --config config/optimizer.yaml
```

It reads historical daily artifacts from S3 and writes local run outputs under `runs/optimizer/`.

## Local One-Shot Run
From repo root:

```bash
.venv/bin/python -m optimizer.cli run --config config/optimizer.yaml
```

Useful companion commands:

```bash
.venv/bin/python -m optimizer.cli status --config config/optimizer.yaml
.venv/bin/python -m optimizer.cli evaluate --config config/optimizer.yaml
```

## launchd Scheduling (Weekly)
Default schedule: **Sunday 03:30 local time**.

Install:

```bash
./scripts/install_optimizer_launchd.sh
```

Uninstall:

```bash
./scripts/uninstall_optimizer_launchd.sh
```

Run immediately without waiting for schedule:

```bash
launchctl start com.traderbot.optimizer
```

launchd log files:

- `~/Library/Logs/traderbot/optimizer.log`
- `~/Library/Logs/traderbot/optimizer.error.log`

## Artifacts and Storage
Per-run artifacts:

- `runs/optimizer/<run_id>/run_manifest.json`
- `runs/optimizer/<run_id>/param_space_snapshot.json`
- `runs/optimizer/<run_id>/walk_forward_folds.json`
- `runs/optimizer/<run_id>/champion_metrics.json`
- `runs/optimizer/<run_id>/challenger_metrics.json`
- `runs/optimizer/<run_id>/gate_segment_metrics.json`
- `runs/optimizer/<run_id>/guardrail_results.json`
- `runs/optimizer/<run_id>/promotion_decision.json`
- `runs/optimizer/<run_id>/candidate_params_bundle.json`
- `runs/optimizer/<run_id>/generation_log.jsonl`
- `runs/optimizer/<run_id>/run.log`

Indexes and lineage:

- `runs/optimizer/index.json`
- `runs/optimizer/active_params_lineage.json`

Dashboard JSON mirrors:

- `dashboard/data/optimizer_runs_index.json`
- `dashboard/data/optimizer_runs/<run_id>.json`
- `dashboard/data/active_params_lineage.json`
- `frontend/public/data/optimizer_runs_index.json`
- `frontend/public/data/optimizer_runs/<run_id>.json`
- `frontend/public/data/active_params_lineage.json`

## Promotion and Rollback
Live params file consumed by runtime:

- `config/decision_params.active.json`

Candidate output file written by optimizer run:

- `config/decision_params.candidate.json`

Promotion swaps candidate to active atomically and archives prior active under:

- `config/decision_params.history/`

Manual promote command:

```bash
.venv/bin/python -m optimizer.cli promote --config config/optimizer.yaml
```

Rollback command:

```bash
.venv/bin/python -m optimizer.cli rollback --to <history_file> --config config/optimizer.yaml
```

`<history_file>` can be either:

- an absolute path
- a relative path
- a filename inside `config/decision_params.history/`
