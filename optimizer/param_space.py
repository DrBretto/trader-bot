"""Parameter-space extraction and genome transforms from discovery inventory."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ParameterSpec:
    name: str
    type: str
    min_value: float
    max_value: float
    current_value: Any
    component: str
    where_defined: str

    def clamp(self, value: Any) -> Any:
        if self.type == 'bool':
            return bool(value)
        if self.type == 'int':
            clipped = max(self.min_value, min(self.max_value, _safe_float(value, self.min_value)))
            return int(round(clipped))
        clipped = max(self.min_value, min(self.max_value, _safe_float(value, self.min_value)))
        return float(clipped)

    def sample(self, rng: random.Random) -> Any:
        if self.type == 'bool':
            return bool(rng.randint(0, 1))
        if self.type == 'int':
            lo = int(round(self.min_value))
            hi = int(round(self.max_value))
            if hi < lo:
                hi = lo
            return int(rng.randint(lo, hi))
        return float(rng.uniform(self.min_value, self.max_value))


def _infer_bounds(param: Dict[str, Any]) -> Tuple[float, float]:
    allowed = param.get('allowed_range')
    current = param.get('current_value')

    if isinstance(allowed, dict) and 'min' in allowed and 'max' in allowed:
        return _safe_float(allowed['min']), _safe_float(allowed['max'])

    if param.get('type') == 'bool':
        return 0.0, 1.0

    curr = _safe_float(current, 0.0)
    if param.get('type') == 'int':
        span = max(1.0, abs(curr))
        return max(0.0, curr - span), curr + span

    # Conservative fallback for floats
    span = max(0.1, abs(curr) * 0.5)
    return curr - span, curr + span


def _is_supported(name: str) -> bool:
    supported_prefixes = (
        'decision_params.',
        'regime_compatibility.',
        'regime_fusion.',
        'decision_engine.',
        'transaction_costs.',
        'ensemble.',
        'macro_credit.',
        'vol_uncertainty.',
        'fragility.',
        'entropy_shift.',
    )
    return name.startswith(supported_prefixes)


def load_parameter_specs(inventory_path: Path) -> List[ParameterSpec]:
    payload = json.loads(inventory_path.read_text(encoding='utf-8'))
    specs: List[ParameterSpec] = []

    for raw in payload.get('parameters', []):
        if not raw.get('safe_to_optimize', False):
            continue
        if not raw.get('used_live', False):
            continue

        name = str(raw.get('name', ''))
        if not _is_supported(name):
            continue

        min_value, max_value = _infer_bounds(raw)
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        spec = ParameterSpec(
            name=name,
            type=str(raw.get('type', 'float')),
            min_value=min_value,
            max_value=max_value,
            current_value=raw.get('current_value'),
            component=str(raw.get('component', 'unknown')),
            where_defined=str(raw.get('where_defined', 'unknown')),
        )
        specs.append(spec)

    specs.sort(key=lambda item: item.name)
    return specs


def _set_nested(root: Dict[str, Any], keys: Iterable[str], value: Any) -> None:
    keys = list(keys)
    node = root
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _get_nested(root: Dict[str, Any], keys: Iterable[str]) -> Any:
    node: Any = root
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _gene_to_path(gene_name: str) -> List[str]:
    if gene_name.startswith('decision_params.'):
        return ['decision_params', *gene_name.split('.')[1:]]
    if gene_name.startswith('regime_compatibility.'):
        return ['regime_compatibility', *gene_name.split('.')[1:]]
    if gene_name.startswith('regime_fusion.'):
        return ['regime_fusion', *gene_name.split('.')[1:]]
    if gene_name.startswith('decision_engine.'):
        return ['decision_engine', *gene_name.split('.')[1:]]
    if gene_name.startswith('ensemble.'):
        return ['ensemble', *gene_name.split('.')[1:]]
    if gene_name.startswith('transaction_costs.'):
        return ['transaction_costs', *gene_name.split('.')[1:]]
    if gene_name.startswith('macro_credit.'):
        return ['signals', 'macro_credit', *gene_name.split('.')[1:]]
    if gene_name.startswith('vol_uncertainty.'):
        return ['signals', 'vol_uncertainty', *gene_name.split('.')[1:]]
    if gene_name.startswith('fragility.'):
        return ['signals', 'fragility', *gene_name.split('.')[1:]]
    if gene_name.startswith('entropy_shift.'):
        return ['signals', 'entropy_shift', *gene_name.split('.')[1:]]
    return ['metadata', 'unsupported', gene_name]


class ParameterSpace:
    """Genome utilities backed by discovery parameter inventory."""

    def __init__(self, specs: List[ParameterSpec]):
        self.specs = specs
        self._spec_by_name = {spec.name: spec for spec in specs}

    def default_genes(self) -> Dict[str, Any]:
        genes: Dict[str, Any] = {}
        for spec in self.specs:
            genes[spec.name] = spec.clamp(spec.current_value)
        return genes

    def genes_from_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        genes = self.default_genes()
        for spec in self.specs:
            value = _get_nested(bundle, _gene_to_path(spec.name))
            if value is not None:
                genes[spec.name] = spec.clamp(value)
        return genes

    def genes_to_bundle(self, genes: Dict[str, Any], base_bundle: Dict[str, Any]) -> Dict[str, Any]:
        bundle = deepcopy(base_bundle)
        for spec in self.specs:
            if spec.name not in genes:
                continue
            _set_nested(bundle, _gene_to_path(spec.name), spec.clamp(genes[spec.name]))
        return bundle

    def random_population_member(
        self,
        rng: random.Random,
        seed_genes: Dict[str, Any],
        mutation_scale: float,
    ) -> Dict[str, Any]:
        child = dict(seed_genes)
        for spec in self.specs:
            base = spec.clamp(child.get(spec.name, spec.current_value))
            if spec.type == 'bool':
                if rng.random() < 0.1:
                    child[spec.name] = not bool(base)
                continue

            span = (spec.max_value - spec.min_value) * max(0.01, mutation_scale)
            if spec.type == 'int':
                delta = int(round(rng.uniform(-span, span)))
                child[spec.name] = spec.clamp(int(base) + delta)
            else:
                delta = rng.uniform(-span, span)
                child[spec.name] = spec.clamp(_safe_float(base) + delta)
        return child

    def crossover(self, a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        child: Dict[str, Any] = {}
        for spec in self.specs:
            source = a if rng.random() < 0.5 else b
            child[spec.name] = spec.clamp(source.get(spec.name, spec.current_value))
        return child

    def mutate(
        self,
        genes: Dict[str, Any],
        rng: random.Random,
        mutation_rate: float,
        mutation_scale: float,
    ) -> Dict[str, Any]:
        child = dict(genes)
        for spec in self.specs:
            if rng.random() > mutation_rate:
                continue
            if spec.type == 'bool':
                child[spec.name] = not bool(child.get(spec.name, False))
                continue
            span = (spec.max_value - spec.min_value) * max(0.01, mutation_scale)
            if spec.type == 'int':
                delta = int(round(rng.uniform(-span, span)))
                child[spec.name] = spec.clamp(int(child.get(spec.name, spec.current_value)) + delta)
            else:
                delta = rng.uniform(-span, span)
                child[spec.name] = spec.clamp(_safe_float(child.get(spec.name, spec.current_value)) + delta)
        return child

    @staticmethod
    def genome_id(genes: Dict[str, Any]) -> str:
        encoded = json.dumps(genes, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]


def diff_genes(base: Dict[str, Any], other: Dict[str, Any]) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    for key in sorted(set(base.keys()) | set(other.keys())):
        if base.get(key) == other.get(key):
            continue
        diffs.append({
            'name': key,
            'from': base.get(key),
            'to': other.get(key),
        })
    return diffs
