"""Candidate write and atomic promotion helpers for live decision params."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def _normalize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure full live-bundle schema for runtime loading."""
    return {
        'schema_version': str(bundle.get('schema_version', '1')),
        'version_id': bundle.get('version_id', 'opt-unknown'),
        'source_run_id': bundle.get('source_run_id'),
        'updated_at': bundle.get('updated_at', _utc_now()),
        'decision_params': bundle.get('decision_params', {}),
        'regime_compatibility': bundle.get('regime_compatibility', {}),
        'signals': bundle.get('signals', {}),
        'regime_fusion': bundle.get('regime_fusion', {}),
        'decision_engine': bundle.get('decision_engine', {}),
        'ensemble': bundle.get('ensemble', {}),
        'transaction_costs': bundle.get('transaction_costs', {}),
        'metadata': bundle.get('metadata', {}),
    }


def active_file_path(config_dir: Path) -> Path:
    return config_dir / 'decision_params.active.json'


def candidate_file_path(config_dir: Path) -> Path:
    return config_dir / 'decision_params.candidate.json'


def history_dir_path(config_dir: Path) -> Path:
    return config_dir / 'decision_params.history'


def load_active_bundle(config_dir: Path) -> Dict[str, Any]:
    active_path = active_file_path(config_dir)
    if not active_path.exists():
        raise FileNotFoundError(
            f'Missing live params file: {active_path}. '
            'Create config/decision_params.active.json before running optimizer.'
        )
    bundle = _normalize_bundle(_read_json(active_path))
    if not bundle.get('decision_params') or not bundle.get('regime_compatibility'):
        raise ValueError(
            f'Invalid active bundle at {active_path}: decision_params/regime_compatibility are required.'
        )
    return bundle


def write_candidate_bundle(config_dir: Path, bundle: Dict[str, Any]) -> Path:
    candidate_path = candidate_file_path(config_dir)
    normalized = _normalize_bundle(bundle)
    normalized['updated_at'] = _utc_now()
    _write_json_atomic(candidate_path, normalized)
    return candidate_path


def promote_candidate_bundle(config_dir: Path) -> Dict[str, Optional[str]]:
    """Atomically promote candidate bundle to active and archive previous active."""
    candidate_path = candidate_file_path(config_dir)
    active_path = active_file_path(config_dir)
    history_dir = history_dir_path(config_dir)

    if not candidate_path.exists():
        raise FileNotFoundError(
            f'Candidate bundle not found: {candidate_path}. '
            'Run optimizer first or write candidate manually.'
        )

    candidate = _normalize_bundle(_read_json(candidate_path))
    candidate['updated_at'] = _utc_now()

    previous_active = None
    previous_version = None
    if active_path.exists():
        previous_active = _normalize_bundle(_read_json(active_path))
        previous_version = str(previous_active.get('version_id', 'unknown'))

    if previous_active is not None:
        history_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{previous_version}.json"
        _write_json_atomic(history_dir / archive_name, previous_active)

    _write_json_atomic(candidate_path, candidate)
    candidate_path.replace(active_path)

    return {
        'from_version': previous_version,
        'to_version': str(candidate.get('version_id', 'opt-unknown')),
        'active_path': str(active_path),
    }
