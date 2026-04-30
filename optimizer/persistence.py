"""Run artifact persistence, index maintenance, and dashboard JSON mirrors."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from optimizer.config import OptimizerConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def build_run_id(config: OptimizerConfig, started_at: Optional[datetime] = None) -> str:
    ts = (started_at or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%SZ')
    return (
        f'run-{ts}-seed{config.random_seed}'
        f'-g{config.generations}-p{config.population_size}'
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding='utf-8')
    tmp.replace(path)


def write_json_list(path: Path, payload: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str))
            fh.write('\n')
    tmp.replace(path)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_run_dir(config: OptimizerConfig, run_id: str) -> Path:
    run_dir = config.run_root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_artifacts(
    run_dir: Path,
    artifacts: Dict[str, Dict[str, Any]],
    generation_log: List[Dict[str, Any]],
    run_logs: List[str],
) -> None:
    """Write the required per-run artifact set."""
    required_order = [
        'run_manifest.json',
        'param_space_snapshot.json',
        'walk_forward_folds.json',
        'champion_metrics.json',
        'challenger_metrics.json',
        'gate_segment_metrics.json',
        'guardrail_results.json',
        'promotion_decision.json',
        'candidate_params_bundle.json',
    ]

    for filename in required_order:
        payload = artifacts.get(filename, {})
        write_json(run_dir / filename, payload)

    write_jsonl(run_dir / 'generation_log.jsonl', generation_log)

    log_text = '\n'.join(run_logs) + ('\n' if run_logs else '')
    (run_dir / 'run.log').write_text(log_text, encoding='utf-8')


def update_run_index(
    config: OptimizerConfig,
    run_summary: Dict[str, Any],
    active_version: str,
) -> Dict[str, Any]:
    """Append run summary to global index (last 200 entries)."""
    index_path = config.run_root_path / 'index.json'
    index = read_json(index_path, default={'updated_at': utc_now_iso(), 'active_version': active_version, 'runs': []})

    runs = list(index.get('runs', []))
    runs.insert(0, run_summary)
    runs = runs[:200]

    index_payload = {
        'updated_at': utc_now_iso(),
        'active_version': active_version,
        'runs': runs,
    }
    write_json(index_path, index_payload)
    return index_payload


def append_lineage_event(
    config: OptimizerConfig,
    event: Dict[str, Any],
    active_version: str,
) -> Dict[str, Any]:
    """Append lineage event and update active version pointer."""
    lineage_path = config.run_root_path / 'active_params_lineage.json'
    lineage = read_json(
        lineage_path,
        default={'updated_at': utc_now_iso(), 'active_version': active_version, 'history': []},
    )

    history = list(lineage.get('history', []))
    history.insert(0, event)

    lineage_payload = {
        'updated_at': utc_now_iso(),
        'active_version': active_version,
        'history': history,
    }
    write_json(lineage_path, lineage_payload)
    return lineage_payload


def read_lineage(config: OptimizerConfig, active_version: str) -> Dict[str, Any]:
    lineage_path = config.run_root_path / 'active_params_lineage.json'
    return read_json(
        lineage_path,
        default={'updated_at': utc_now_iso(), 'active_version': active_version, 'history': []},
    )


def mirror_dashboard_artifacts(
    config: OptimizerConfig,
    run_id: str,
    index_payload: Dict[str, Any],
    lineage_payload: Dict[str, Any],
    run_detail_payload: Dict[str, Any],
) -> None:
    """Write dashboard-consumed static JSON into dashboard + frontend public paths."""
    target_roots = [config.dashboard_data_path, config.frontend_data_path]
    for root in target_roots:
        optimizer_runs_dir = root / 'optimizer_runs'
        optimizer_runs_dir.mkdir(parents=True, exist_ok=True)

        write_json(root / 'optimizer_runs_index.json', index_payload)
        write_json(root / 'active_params_lineage.json', lineage_payload)
        write_json(optimizer_runs_dir / f'{run_id}.json', run_detail_payload)


def update_rejection_streak(
    config: OptimizerConfig,
    run_summary: Dict[str, Any],
    challenger_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 4: maintain a persistent rejection counter.

    Increments on not-promoted runs (any decision != "promoted"), resets
    on any promotion. Returns the updated state including a flag for
    whether the alert threshold has been crossed *for the first time*
    on this update — so the caller can fire the SNS alert exactly once
    per streak-of-rejections.

    State file: runs/optimizer/rejection_streak.json. Lives alongside
    index.json and active_params_lineage.json.
    """
    streak_path = config.run_root_path / 'rejection_streak.json'
    state = read_json(
        streak_path,
        default={
            'consecutive_rejections': 0,
            'alert_threshold': config.rejection_streak_alert_threshold,
            'last_alert_run_id': None,
            'history': [],
        },
    )

    decision = str(run_summary.get('decision', 'not_promoted'))
    promoted = decision == 'promoted'

    if promoted:
        state['consecutive_rejections'] = 0
        state['last_alert_run_id'] = None
        new_alert_should_fire = False
    else:
        state['consecutive_rejections'] = int(state.get('consecutive_rejections', 0)) + 1
        # Fire alert if threshold reached AND we haven't already alerted
        # for this streak. Without the second condition the alert would
        # fire every weekly cycle once the streak is long enough.
        threshold = config.rejection_streak_alert_threshold
        already_alerted = state.get('last_alert_run_id') is not None
        new_alert_should_fire = (
            state['consecutive_rejections'] >= threshold and not already_alerted
        )
        if new_alert_should_fire:
            state['last_alert_run_id'] = run_summary.get('run_id')

    history = list(state.get('history', []))
    history.insert(0, {
        'run_id': run_summary.get('run_id'),
        'decision': decision,
        'timestamp': utc_now_iso(),
        'consecutive_rejections_after': state['consecutive_rejections'],
        'challenger_summary': challenger_metrics,
    })
    state['history'] = history[:30]
    state['updated_at'] = utc_now_iso()
    state['alert_threshold'] = config.rejection_streak_alert_threshold
    state['alert_should_fire'] = new_alert_should_fire

    write_json(streak_path, state)
    return state


def build_run_detail_payload(run_dir: Path, run_id: str) -> Dict[str, Any]:
    """Build consolidated per-run detail payload for UI."""

    def _load(name: str) -> Dict[str, Any]:
        path = run_dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding='utf-8'))

    detail = {
        'run_id': run_id,
        'run_manifest': _load('run_manifest.json'),
        'walk_forward_folds': _load('walk_forward_folds.json'),
        'champion_metrics': _load('champion_metrics.json'),
        'challenger_metrics': _load('challenger_metrics.json'),
        'gate_segment_metrics': _load('gate_segment_metrics.json'),
        'guardrail_results': _load('guardrail_results.json'),
        'promotion_decision': _load('promotion_decision.json'),
        'candidate_params_bundle': _load('candidate_params_bundle.json'),
    }

    generation_log_path = run_dir / 'generation_log.jsonl'
    generation_log: List[Dict[str, Any]] = []
    if generation_log_path.exists():
        for line in generation_log_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            generation_log.append(json.loads(line))
    detail['generation_log'] = generation_log

    run_log_path = run_dir / 'run.log'
    detail['run_log_path'] = str(run_log_path)
    return detail
