"""Champion-challenger offline evolutionary optimization orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import random
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from optimizer.config import OptimizerConfig
from optimizer.data_access import load_optimizer_dataset
from optimizer.guardrails import evaluate_guardrails
from optimizer.param_space import (
    ParameterSpace,
    diff_genes,
    empirical_genome,
    is_calibration_only_diff,
    load_parameter_specs,
)
from optimizer.persistence import (
    append_lineage_event,
    build_run_detail_payload,
    build_run_id,
    ensure_run_dir,
    file_sha256,
    mirror_dashboard_artifacts,
    read_lineage,
    update_run_index,
    utc_now_iso,
    write_run_artifacts,
)
from optimizer.promote import load_active_bundle, promote_candidate_bundle, write_candidate_bundle
from optimizer.walk_forward import build_walk_forward_plan, evaluate_gate_segment, evaluate_walk_forward


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else 'unknown'
    except Exception:
        return 'unknown'


def _candidate_seed(base_seed: int, genome_id: str) -> int:
    try:
        offset = int(genome_id[:8], 16)
    except Exception:
        offset = 0
    return base_seed + offset


def _metrics_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    fold_metrics = payload['wf']['fold_metrics']
    return {
        'wf_objective': payload['wf']['wf_objective'],
        'wf_mean': payload['wf']['wf_mean'],
        'wf_stability_penalty': payload['wf']['wf_stability_penalty'],
        'fold_metrics': [metric.to_dict() for metric in fold_metrics],
        'guardrails': payload['guardrails'],
        'gate_metrics': payload['gate'].to_dict(),
    }


def _build_version_id(run_id: str, genome_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'opt-{ts}-{genome_id[:8]}-{run_id.split("-")[1]}'


def run_optimizer_cycle(config: OptimizerConfig) -> Dict[str, Any]:
    """Run one full champion-challenger optimization cycle."""
    started_dt = _now_utc()
    run_id = build_run_id(config, started_at=started_dt)
    run_dir = ensure_run_dir(config, run_id)

    logs: List[str] = []

    def log(message: str) -> None:
        line = f"[{utc_now_iso()}] {message}"
        print(line)
        logs.append(line)

    generation_log: List[Dict[str, Any]] = []
    status = 'failed'
    decision = 'not_promoted'

    active_version = 'unknown'
    champion_version_before = 'unknown'
    challenger_version = 'unknown'

    promotion_reason_codes: List[str] = []

    # Placeholders for artifacts even on failure paths.
    walk_forward_payload: Dict[str, Any] = {'folds': [], 'gate_segment': {'start': None, 'end': None, 'days': 0}}
    champion_metrics_payload: Dict[str, Any] = {}
    challenger_metrics_payload: Dict[str, Any] = {}
    gate_segment_metrics_payload: Dict[str, Any] = {}
    guardrail_results_payload: Dict[str, Any] = {'passed': False, 'checks': []}
    promotion_payload: Dict[str, Any] = {}
    candidate_bundle_payload: Dict[str, Any] = {}
    param_space_snapshot_payload: Dict[str, Any] = {'parameters': []}

    wf_delta = 0.0
    gate_delta = 0.0

    try:
        log('Loading active params bundle and discovery inventory')
        active_bundle = load_active_bundle(config.config_dir_path)
        champion_version_before = str(active_bundle.get('version_id', 'legacy'))
        active_version = champion_version_before

        specs = load_parameter_specs(config.inventory_file)
        if not specs:
            raise RuntimeError('No safe optimizable parameters discovered')

        param_space = ParameterSpace(specs)
        champion_genes = param_space.genes_from_bundle(active_bundle)
        champion_genome_id = param_space.genome_id(champion_genes)

        param_space_snapshot_payload = {
            'parameters': [
                {
                    'name': spec.name,
                    'type': spec.type,
                    'allowed_range': {'min': spec.min_value, 'max': spec.max_value},
                    'current_value': spec.current_value,
                    'component': spec.component,
                    'where_defined': spec.where_defined,
                }
                for spec in specs
            ]
        }

        log('Loading historical optimizer dataset from S3 daily artifacts')
        dataset = load_optimizer_dataset(config)

        log('Building deterministic walk-forward folds and promotion gate segment')
        plan = build_walk_forward_plan(
            dates=dataset.dates,
            train_days=config.train_days,
            test_days=config.test_days,
            step_days=config.step_days,
            gate_days=config.gate_days,
        )
        walk_forward_payload = plan.to_dict()

        eval_cache: Dict[str, Dict[str, Any]] = {}

        def evaluate_genes(genes: Dict[str, Any], include_gate: bool) -> Dict[str, Any]:
            genome_id = param_space.genome_id(genes)
            cached = eval_cache.get(genome_id)
            if cached is None:
                bundle = param_space.genes_to_bundle(genes, active_bundle)
                wf = evaluate_walk_forward(
                    dataset=dataset,
                    bundle=bundle,
                    plan=plan,
                    random_seed=_candidate_seed(config.random_seed, genome_id),
                    initial_capital=config.initial_portfolio_value,
                    max_workers=config.max_workers,
                )
                cached = {
                    'genome_id': genome_id,
                    'genes': genes,
                    'bundle': bundle,
                    'wf': wf,
                }
                eval_cache[genome_id] = cached

            if include_gate and 'gate' not in cached:
                gate = evaluate_gate_segment(
                    dataset=dataset,
                    bundle=cached['bundle'],
                    plan=plan,
                    random_seed=_candidate_seed(config.random_seed, genome_id),
                    initial_capital=config.initial_portfolio_value,
                )
                cached['gate'] = gate
                # Phase 3: classify the challenger as calibration-only iff
                # every gene that differs from the champion is tagged
                # parameter_class="normalization_constant" in the inventory.
                # Calibration-only challengers use the looser guardrail
                # path (trade-frequency gates dropped); mixed challengers
                # stay on the strict path.
                gene_diffs = diff_genes(champion_genes, genes)
                calibration_only = is_calibration_only_diff(gene_diffs, specs)
                cached['calibration_only'] = calibration_only
                cached['guardrails'] = evaluate_guardrails(
                    fold_metrics=cached['wf']['fold_metrics'],
                    gate_metrics=gate,
                    config=config.guardrails,
                    calibration_only=calibration_only,
                )

            return cached

        log('Evaluating champion baseline')
        champion_eval = evaluate_genes(champion_genes, include_gate=True)

        rng = random.Random(config.random_seed)
        population: List[Dict[str, Any]] = [champion_genes]

        # Empirical re-derivation candidate (Phase 2 of the
        # 2026-04-30 optimizer empirical-mutation packet). When the feature
        # flag is on, build a candidate genome where every parameter tagged
        # `parameter_class: normalization_constant` with an empirical_statistic
        # is set to the live-data statistic computed from the optimizer's
        # signal_row dataset. The candidate competes under the existing
        # fitness + guardrail pipeline; this is seeding, not replacement.
        empirical_summary: Dict[str, float] = {}
        if config.enable_empirical_mutation:
            signal_rows = [snapshot.signal_row for snapshot in dataset.snapshots]
            empirical_genes, empirical_summary = empirical_genome(
                specs=specs,
                signal_rows=signal_rows,
                base_genes=champion_genes,
            )
            if empirical_summary:
                log(
                    'Empirical-mutation candidate proposed for '
                    f'{len(empirical_summary)} normalization constants: '
                    + ', '.join(
                        f'{name}={value:.4f}' for name, value in empirical_summary.items()
                    )
                )
                population.append(empirical_genes)
            else:
                log('Empirical-mutation enabled but no candidates produced (insufficient data).')

        while len(population) < config.population_size:
            population.append(
                param_space.random_population_member(
                    rng=rng,
                    seed_genes=champion_genes,
                    mutation_scale=config.mutation_scale,
                )
            )

        best_challenger: Optional[Dict[str, Any]] = None
        started_monotonic = time.monotonic()
        timed_out = False

        for generation in range(1, config.generations + 1):
            elapsed_minutes = (time.monotonic() - started_monotonic) / 60.0
            if elapsed_minutes >= config.max_runtime_minutes:
                timed_out = True
                log(
                    'Max runtime reached; stopping evolution early at '
                    f'generation {generation - 1}.'
                )
                break

            scored: List[Tuple[float, str, Dict[str, Any], Dict[str, Any]]] = []
            for genes in population:
                evaluated = evaluate_genes(genes, include_gate=False)
                objective = _safe_float(evaluated['wf']['wf_objective'], float('-inf'))
                scored.append((objective, evaluated['genome_id'], genes, evaluated))

            scored.sort(key=lambda item: item[0], reverse=True)

            best_obj = scored[0][0]
            median_obj = scored[len(scored) // 2][0]
            worst_obj = scored[-1][0]
            unique_ids = {item[1] for item in scored}
            diversity = len(unique_ids) / max(1, len(scored))

            generation_log.append(
                {
                    'generation': generation,
                    'best_objective': best_obj,
                    'median_objective': median_obj,
                    'worst_objective': worst_obj,
                    'diversity': diversity,
                    'best_candidate_id': scored[0][1],
                }
            )

            for objective, genome_id, genes, evaluated in scored:
                if genome_id == champion_genome_id:
                    continue
                if best_challenger is None or objective > _safe_float(best_challenger['wf']['wf_objective'], float('-inf')):
                    best_challenger = {
                        'genome_id': genome_id,
                        'genes': genes,
                        'wf': evaluated['wf'],
                    }
                break

            elite_count = max(1, int(math.ceil(config.population_size * config.elite_fraction)))
            elites = scored[:elite_count]
            next_population: List[Dict[str, Any]] = [dict(item[2]) for item in elites]

            while len(next_population) < config.population_size:
                if len(elites) == 1:
                    base_genes = elites[0][2]
                    child = dict(base_genes)
                else:
                    parent_a, parent_b = rng.sample(elites, 2)
                    child = param_space.crossover(parent_a[2], parent_b[2], rng)
                child = param_space.mutate(
                    genes=child,
                    rng=rng,
                    mutation_rate=config.mutation_rate,
                    mutation_scale=config.mutation_scale,
                )
                next_population.append(child)

            population = next_population

        if best_challenger is None:
            best_challenger = {
                'genome_id': champion_genome_id,
                'genes': champion_genes,
                'wf': champion_eval['wf'],
            }
            log('No challenger exceeded champion in sampled population; using champion as fallback candidate.')

        log('Evaluating selected challenger on gate segment and guardrails')
        challenger_eval = evaluate_genes(best_challenger['genes'], include_gate=True)

        challenger_genome_id = challenger_eval['genome_id']
        challenger_version = _build_version_id(run_id, challenger_genome_id)

        candidate_bundle = dict(challenger_eval['bundle'])
        candidate_bundle['version_id'] = challenger_version
        candidate_bundle['source_run_id'] = run_id
        candidate_bundle['updated_at'] = utc_now_iso()
        candidate_bundle['metadata'] = {
            'promotion_type': 'challenger_promotion_attempt',
            'safe_param_count': len(specs),
            'wf_objective': challenger_eval['wf']['wf_objective'],
            'gate_score': challenger_eval['gate'].fold_score,
            'parent_version': champion_version_before,
            'timed_out': timed_out,
        }

        write_candidate_bundle(config.config_dir_path, candidate_bundle)

        champion_metrics_payload = _metrics_payload(champion_eval)
        challenger_metrics_payload = _metrics_payload(challenger_eval)

        champion_gate = champion_eval['gate']
        challenger_gate = challenger_eval['gate']

        gate_segment_metrics_payload = {
            'champion_gate_score': champion_gate.fold_score,
            'challenger_gate_score': challenger_gate.fold_score,
            'champion_gate_max_drawdown': champion_gate.max_drawdown,
            'challenger_gate_max_drawdown': challenger_gate.max_drawdown,
            'gate_guardrails': challenger_eval['guardrails'],
        }

        guardrail_results_payload = challenger_eval['guardrails']

        wf_delta = _safe_float(challenger_eval['wf']['wf_objective']) - _safe_float(champion_eval['wf']['wf_objective'])
        gate_delta = _safe_float(challenger_gate.fold_score) - _safe_float(champion_gate.fold_score)
        gate_drawdown_ok = (
            challenger_gate.max_drawdown
            >= champion_gate.max_drawdown - config.promotion_deltas.gate_drawdown_tolerance
        )

        guardrails_passed = bool(challenger_eval['guardrails'].get('passed', False))
        wf_improved = wf_delta >= config.promotion_deltas.wf_objective_delta
        gate_improved = gate_delta >= config.promotion_deltas.gate_score_delta

        if guardrails_passed:
            promotion_reason_codes.append('guardrails_passed')
        if wf_improved:
            promotion_reason_codes.append('wf_objective_improved')
        if gate_improved:
            promotion_reason_codes.append('gate_score_improved')
        if gate_drawdown_ok:
            promotion_reason_codes.append('gate_drawdown_ok')

        promoted_at: Optional[str] = None

        if challenger_genome_id == champion_genome_id:
            status = 'completed'
            decision = 'not_promoted'
            promotion_reason_codes.append('no_distinct_challenger')
        elif not guardrails_passed:
            status = 'rejected_guardrails'
            decision = 'not_promoted'
        elif wf_improved and gate_improved and gate_drawdown_ok:
            promote_result = promote_candidate_bundle(config.config_dir_path)
            active_version = str(promote_result['to_version'])
            status = 'promoted'
            decision = 'promoted'
            promoted_at = utc_now_iso()
            append_lineage_event(
                config=config,
                event={
                    'event_type': 'promotion',
                    'timestamp': promoted_at,
                    'from_version': promote_result.get('from_version'),
                    'to_version': promote_result.get('to_version'),
                    'run_id': run_id,
                    'reason': 'promotion passed gates',
                    'operator': 'optimizer',
                },
                active_version=active_version,
            )
        else:
            status = 'rejected_objective'
            decision = 'not_promoted'

        promotion_payload = {
            'run_id': run_id,
            'decision': decision,
            'reason_codes': promotion_reason_codes,
            'champion_version_before': champion_version_before,
            'challenger_version': challenger_version,
            'champion_wf_objective': champion_eval['wf']['wf_objective'],
            'challenger_wf_objective': challenger_eval['wf']['wf_objective'],
            'champion_gate_score': champion_gate.fold_score,
            'challenger_gate_score': challenger_gate.fold_score,
            'promoted_at': promoted_at,
        }

        candidate_bundle_payload = {
            'version_id': challenger_version,
            'decision_params': candidate_bundle.get('decision_params', {}),
            'regime_compatibility': candidate_bundle.get('regime_compatibility', {}),
            'signals': candidate_bundle.get('signals', {}),
            'regime_fusion': candidate_bundle.get('regime_fusion', {}),
            'decision_engine': candidate_bundle.get('decision_engine', {}),
            'ensemble': candidate_bundle.get('ensemble', {}),
            'transaction_costs': candidate_bundle.get('transaction_costs', {}),
            'source_run_id': run_id,
            'safe_param_names': [spec.name for spec in specs],
            'param_diffs': diff_genes(champion_genes, best_challenger['genes']),
        }

    except Exception as exc:
        status = 'failed'
        decision = 'not_promoted'
        log(f'Optimizer run failed: {exc}')
        promotion_reason_codes.append('runtime_error')
        promotion_payload = {
            'run_id': run_id,
            'decision': decision,
            'reason_codes': promotion_reason_codes,
            'champion_version_before': champion_version_before,
            'challenger_version': challenger_version,
            'champion_wf_objective': None,
            'challenger_wf_objective': None,
            'champion_gate_score': None,
            'challenger_gate_score': None,
            'promoted_at': None,
            'error': str(exc),
        }

    finished_dt = _now_utc()

    run_manifest = {
        'run_id': run_id,
        'started_at': _iso(started_dt),
        'finished_at': _iso(finished_dt),
        'status': status,
        'seed': config.random_seed,
        'population': config.population_size,
        'generations': config.generations,
        'max_days': config.max_days,
        'train_days': config.train_days,
        'test_days': config.test_days,
        'step_days': config.step_days,
        'gate_days': config.gate_days,
        'bucket': config.bucket,
        'region': config.region,
        'git_commit': _git_commit(),
        'param_inventory_hash': file_sha256(config.inventory_file),
        'pipeline_map_hash': file_sha256(config.pipeline_map_file),
        'data_inventory_hash': file_sha256(config.data_inventory_file),
        'max_runtime_minutes': config.max_runtime_minutes,
        'max_workers': config.max_workers,
    }

    artifacts = {
        'run_manifest.json': run_manifest,
        'param_space_snapshot.json': param_space_snapshot_payload,
        'walk_forward_folds.json': walk_forward_payload,
        'champion_metrics.json': champion_metrics_payload,
        'challenger_metrics.json': challenger_metrics_payload,
        'gate_segment_metrics.json': gate_segment_metrics_payload,
        'guardrail_results.json': guardrail_results_payload,
        'promotion_decision.json': promotion_payload,
        'candidate_params_bundle.json': candidate_bundle_payload,
    }

    write_run_artifacts(
        run_dir=run_dir,
        artifacts=artifacts,
        generation_log=generation_log,
        run_logs=logs,
    )

    run_summary = {
        'run_id': run_id,
        'started_at': run_manifest['started_at'],
        'finished_at': run_manifest['finished_at'],
        'status': status,
        'decision': decision,
        'champion_version_before': champion_version_before,
        'challenger_version': challenger_version,
        'wf_delta': wf_delta,
        'gate_delta': gate_delta,
    }

    index_payload = update_run_index(
        config=config,
        run_summary=run_summary,
        active_version=active_version,
    )

    lineage_payload = read_lineage(config=config, active_version=active_version)

    run_detail_payload = build_run_detail_payload(run_dir, run_id)
    mirror_dashboard_artifacts(
        config=config,
        run_id=run_id,
        index_payload=index_payload,
        lineage_payload=lineage_payload,
        run_detail_payload=run_detail_payload,
    )

    return {
        'run_id': run_id,
        'status': status,
        'decision': decision,
        'run_dir': str(run_dir),
        'active_version': active_version,
        'index_path': str(config.run_root_path / 'index.json'),
        'lineage_path': str(config.run_root_path / 'active_params_lineage.json'),
    }
