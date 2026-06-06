"""Configuration loading for the local offline optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class GuardrailConfig:
    max_drawdown: float = -0.25
    max_cost_ratio: float = 0.015
    min_total_round_trips: int = 20
    min_gate_round_trips: int = 5
    min_gate_win_rate: float = 0.45
    min_fold_annualized_return: float = -0.20


@dataclass
class PromotionDeltaConfig:
    wf_objective_delta: float = 0.01
    gate_score_delta: float = 0.01
    gate_drawdown_tolerance: float = 0.02


@dataclass
class PathConfig:
    run_root: str = 'runs/optimizer'
    dashboard_data_dir: str = 'dashboard/data'
    frontend_data_dir: str = 'frontend/public/data'
    config_dir: str = 'config'


@dataclass
class OptimizerConfig:
    bucket: str = 'investment-system-data'
    region: str = 'us-east-1'
    inventory_path: str = 'optimizer/discovery/param_inventory.json'
    data_inventory_path: str = 'optimizer/discovery/data_inventory.json'
    pipeline_map_path: str = 'optimizer/discovery/pipeline_map.json'

    max_days: int = 900
    # Holdout ceiling: when set (YYYY-MM-DD), the optimizer dataset DROPS all
    # daily dates >= this value, so no fold/gate/selection can touch the holdout.
    # Closes the leak where gate_dates = most-recent dates = the held-out window.
    holdout_start: str = ''
    train_days: int = 252
    test_days: int = 63
    step_days: int = 63
    gate_days: int = 63

    population_size: int = 40
    generations: int = 30
    elite_fraction: float = 0.2
    mutation_rate: float = 0.15
    mutation_scale: float = 0.2

    max_runtime_minutes: int = 120
    max_workers: int = 1
    random_seed: int = 20260217

    initial_portfolio_value: float = 100000.0

    # Phase 2: empirical-mutation feature flag. Default off so the
    # behavior is identical to pre-2026-04-30 until the operator
    # promotes via config (or via --empirical-mutation-debug for the
    # verification gate dry-run). See
    # docs/plans/2026-04-30-optimizer-empirical-mutation-RETURN.md.
    enable_empirical_mutation: bool = False

    # Phase 4: persistent-rejection alert threshold. After N consecutive
    # cycles where every challenger was rejected, fire an SNS alert. Reset
    # the counter on any promotion. Default 3 per packet.
    rejection_streak_alert_threshold: int = 3

    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    promotion_deltas: PromotionDeltaConfig = field(default_factory=PromotionDeltaConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    repo_root: Path = field(default_factory=lambda: Path.cwd())

    def resolved_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()

    @property
    def run_root_path(self) -> Path:
        return self.resolved_path(self.paths.run_root)

    @property
    def dashboard_data_path(self) -> Path:
        return self.resolved_path(self.paths.dashboard_data_dir)

    @property
    def frontend_data_path(self) -> Path:
        return self.resolved_path(self.paths.frontend_data_dir)

    @property
    def config_dir_path(self) -> Path:
        return self.resolved_path(self.paths.config_dir)

    @property
    def inventory_file(self) -> Path:
        return self.resolved_path(self.inventory_path)

    @property
    def data_inventory_file(self) -> Path:
        return self.resolved_path(self.data_inventory_path)

    @property
    def pipeline_map_file(self) -> Path:
        return self.resolved_path(self.pipeline_map_path)

    def to_effective_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['repo_root'] = str(self.repo_root)
        return data


def _read_config_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding='utf-8')

    # YAML is a superset of JSON; prefer zero-dependency parse first.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise ValueError(
                f"Unable to parse {path}. Provide JSON-compatible YAML or install PyYAML."
            ) from exc
        parsed = yaml.safe_load(text)  # type: ignore

    if not isinstance(parsed, dict):
        raise ValueError(f'Config file must contain an object at root: {path}')
    return parsed


def _read_nested(data: Dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key, default)
    return value if value is not None else default


def load_optimizer_config(config_path: str, repo_root: Path | None = None) -> OptimizerConfig:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        base = repo_root or Path.cwd()
        config_file = (base / config_file).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f'Optimizer config not found: {config_file}')

    raw = _read_config_file(config_file)

    guardrails_raw = raw.get('guardrails', {}) or {}
    deltas_raw = raw.get('promotion_deltas', {}) or {}
    paths_raw = raw.get('paths', {}) or {}

    cfg = OptimizerConfig(
        bucket=str(_read_nested(raw, 'bucket', OptimizerConfig.bucket)),
        region=str(_read_nested(raw, 'region', OptimizerConfig.region)),
        inventory_path=str(_read_nested(raw, 'inventory_path', OptimizerConfig.inventory_path)),
        data_inventory_path=str(_read_nested(raw, 'data_inventory_path', OptimizerConfig.data_inventory_path)),
        pipeline_map_path=str(_read_nested(raw, 'pipeline_map_path', OptimizerConfig.pipeline_map_path)),
        max_days=int(_read_nested(raw, 'max_days', OptimizerConfig.max_days)),
        holdout_start=str(_read_nested(raw, 'holdout_start', OptimizerConfig.holdout_start)),
        train_days=int(_read_nested(raw, 'train_days', OptimizerConfig.train_days)),
        test_days=int(_read_nested(raw, 'test_days', OptimizerConfig.test_days)),
        step_days=int(_read_nested(raw, 'step_days', OptimizerConfig.step_days)),
        gate_days=int(_read_nested(raw, 'gate_days', OptimizerConfig.gate_days)),
        population_size=int(_read_nested(raw, 'population_size', OptimizerConfig.population_size)),
        generations=int(_read_nested(raw, 'generations', OptimizerConfig.generations)),
        elite_fraction=float(_read_nested(raw, 'elite_fraction', OptimizerConfig.elite_fraction)),
        mutation_rate=float(_read_nested(raw, 'mutation_rate', OptimizerConfig.mutation_rate)),
        mutation_scale=float(_read_nested(raw, 'mutation_scale', OptimizerConfig.mutation_scale)),
        max_runtime_minutes=int(_read_nested(raw, 'max_runtime_minutes', OptimizerConfig.max_runtime_minutes)),
        max_workers=max(1, int(_read_nested(raw, 'max_workers', OptimizerConfig.max_workers))),
        random_seed=int(_read_nested(raw, 'random_seed', OptimizerConfig.random_seed)),
        initial_portfolio_value=float(_read_nested(raw, 'initial_portfolio_value', OptimizerConfig.initial_portfolio_value)),
        enable_empirical_mutation=bool(_read_nested(raw, 'enable_empirical_mutation', OptimizerConfig.enable_empirical_mutation)),
        rejection_streak_alert_threshold=int(_read_nested(raw, 'rejection_streak_alert_threshold', OptimizerConfig.rejection_streak_alert_threshold)),
        guardrails=GuardrailConfig(
            max_drawdown=float(_read_nested(guardrails_raw, 'max_drawdown', GuardrailConfig.max_drawdown)),
            max_cost_ratio=float(_read_nested(guardrails_raw, 'max_cost_ratio', GuardrailConfig.max_cost_ratio)),
            min_total_round_trips=int(_read_nested(guardrails_raw, 'min_total_round_trips', GuardrailConfig.min_total_round_trips)),
            min_gate_round_trips=int(_read_nested(guardrails_raw, 'min_gate_round_trips', GuardrailConfig.min_gate_round_trips)),
            min_gate_win_rate=float(_read_nested(guardrails_raw, 'min_gate_win_rate', GuardrailConfig.min_gate_win_rate)),
            min_fold_annualized_return=float(_read_nested(guardrails_raw, 'min_fold_annualized_return', GuardrailConfig.min_fold_annualized_return)),
        ),
        promotion_deltas=PromotionDeltaConfig(
            wf_objective_delta=float(_read_nested(deltas_raw, 'wf_objective_delta', PromotionDeltaConfig.wf_objective_delta)),
            gate_score_delta=float(_read_nested(deltas_raw, 'gate_score_delta', PromotionDeltaConfig.gate_score_delta)),
            gate_drawdown_tolerance=float(_read_nested(deltas_raw, 'gate_drawdown_tolerance', PromotionDeltaConfig.gate_drawdown_tolerance)),
        ),
        paths=PathConfig(
            run_root=str(_read_nested(paths_raw, 'run_root', PathConfig.run_root)),
            dashboard_data_dir=str(_read_nested(paths_raw, 'dashboard_data_dir', PathConfig.dashboard_data_dir)),
            frontend_data_dir=str(_read_nested(paths_raw, 'frontend_data_dir', PathConfig.frontend_data_dir)),
            config_dir=str(_read_nested(paths_raw, 'config_dir', PathConfig.config_dir)),
        ),
        repo_root=(repo_root or config_file.parent.parent).resolve(),
    )

    if cfg.gate_days <= 0:
        raise ValueError('gate_days must be > 0')
    if cfg.train_days <= 0 or cfg.test_days <= 0 or cfg.step_days <= 0:
        raise ValueError('train_days/test_days/step_days must be > 0')
    if cfg.population_size < 2:
        raise ValueError('population_size must be >= 2')

    return cfg
