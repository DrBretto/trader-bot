"""Rollback utilities for active decision params bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

from optimizer.promote import active_file_path, history_dir_path, load_active_bundle


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def resolve_history_target(config_dir: Path, target: str) -> Path:
    target_path = Path(target)
    if target_path.is_absolute() and target_path.exists():
        return target_path

    if target_path.exists():
        return target_path.resolve()

    history_path = history_dir_path(config_dir) / target
    if history_path.exists():
        return history_path

    raise FileNotFoundError(
        f'Rollback target not found: {target}. '
        f'Checked {target_path} and {history_path}.'
    )


def rollback_to(config_dir: Path, target: str) -> Dict[str, Optional[str]]:
    """Rollback active bundle to a historical file (atomic pointer swap)."""
    active_path = active_file_path(config_dir)
    history_dir = history_dir_path(config_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    if not active_path.exists():
        raise FileNotFoundError(f'Active bundle not found at {active_path}')

    current = load_active_bundle(config_dir)
    target_path = resolve_history_target(config_dir, target)
    target_bundle = _read_json(target_path)

    from_version = str(current.get('version_id', 'unknown'))
    to_version = str(target_bundle.get('version_id', 'unknown'))

    archive_name = f"{_utc_timestamp()}-rollback-from-{from_version}.json"
    _write_json_atomic(history_dir / archive_name, current)

    target_bundle['updated_at'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    _write_json_atomic(active_path, target_bundle)

    return {
        'from_version': from_version,
        'to_version': to_version,
        'active_path': str(active_path),
        'target_path': str(target_path),
    }
