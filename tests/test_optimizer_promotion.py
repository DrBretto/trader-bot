import json
from pathlib import Path

from optimizer.promote import load_active_bundle, promote_candidate_bundle, write_candidate_bundle
from optimizer.rollback import rollback_to


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _bundle(version: str) -> dict:
    return {
        'schema_version': '1',
        'version_id': version,
        'source_run_id': None,
        'updated_at': '2026-02-17T00:00:00Z',
        'decision_params': {'max_positions': 8},
        'regime_compatibility': {'risk_on_trend': {'equity': 1.0}},
        'signals': {},
        'regime_fusion': {},
        'decision_engine': {},
        'ensemble': {},
        'transaction_costs': {},
        'metadata': {},
    }


def test_promote_and_rollback(tmp_path: Path) -> None:
    config_dir = tmp_path / 'config'
    active = config_dir / 'decision_params.active.json'
    _write_json(active, _bundle('opt-current'))

    write_candidate_bundle(config_dir, _bundle('opt-candidate'))
    promote_result = promote_candidate_bundle(config_dir)

    assert promote_result['from_version'] == 'opt-current'
    assert promote_result['to_version'] == 'opt-candidate'
    assert load_active_bundle(config_dir)['version_id'] == 'opt-candidate'

    history_dir = config_dir / 'decision_params.history'
    history_files = list(history_dir.glob('*.json'))
    assert len(history_files) == 1

    rollback_result = rollback_to(config_dir, history_files[0].name)

    assert rollback_result['from_version'] == 'opt-candidate'
    assert rollback_result['to_version'] == 'opt-current'
    assert load_active_bundle(config_dir)['version_id'] == 'opt-current'
