"""
PolicyGenome: Genetic representation of trading policy parameters.

Each genome encodes a complete trading strategy with parameters for:
- Position sizing
- Entry/exit thresholds
- Regime-specific behavior
- Risk management
"""

import json
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib


@dataclass
class GeneRange:
    """Defines the valid range and mutation parameters for a gene."""
    min_val: float
    max_val: float
    mutation_sigma: float  # Standard deviation for Gaussian mutation
    is_integer: bool = False

    def clip(self, value: float) -> float:
        """Clip value to valid range."""
        clipped = max(self.min_val, min(self.max_val, value))
        return int(round(clipped)) if self.is_integer else clipped

    def random(self) -> float:
        """Generate random value in range."""
        value = random.uniform(self.min_val, self.max_val)
        return int(round(value)) if self.is_integer else value

    def mutate(self, value: float) -> float:
        """Apply Gaussian mutation to value."""
        mutated = value + random.gauss(0, self.mutation_sigma)
        return self.clip(mutated)


# Gene definitions with ranges and mutation parameters
GENE_DEFINITIONS: Dict[str, GeneRange] = {
    # Position sizing
    'max_positions': GeneRange(3, 15, 1.5, is_integer=True),
    'max_position_weight': GeneRange(0.05, 0.30, 0.03),
    'min_order_dollars': GeneRange(100, 500, 50, is_integer=True),

    # Entry thresholds
    'buy_score_threshold': GeneRange(0.40, 0.85, 0.05),
    'min_health_buy': GeneRange(0.40, 0.80, 0.05),
    'momentum_weight': GeneRange(0.0, 1.0, 0.1),
    'mean_reversion_weight': GeneRange(0.0, 1.0, 0.1),

    # Exit thresholds
    'trailing_stop_base': GeneRange(0.05, 0.20, 0.02),
    'health_collapse_threshold': GeneRange(0.20, 0.50, 0.05),
    'profit_take_threshold': GeneRange(0.15, 0.50, 0.05),
    'max_holding_days': GeneRange(10, 90, 10, is_integer=True),

    # Regime multipliers (how aggressive to be in each regime)
    'regime_calm_uptrend': GeneRange(0.5, 1.5, 0.1),
    'regime_risk_on_trend': GeneRange(0.5, 1.5, 0.1),
    'regime_choppy': GeneRange(0.3, 1.2, 0.1),
    'regime_risk_off_trend': GeneRange(0.1, 0.8, 0.1),
    'regime_high_vol_panic': GeneRange(0.0, 0.5, 0.1),

    # Volatility adjustments
    'vol_low_multiplier': GeneRange(0.8, 1.3, 0.1),
    'vol_med_multiplier': GeneRange(0.6, 1.1, 0.1),
    'vol_high_multiplier': GeneRange(0.3, 0.9, 0.1),

    # Diversification
    'sector_max_weight': GeneRange(0.20, 0.50, 0.05),
    'correlation_penalty': GeneRange(0.0, 0.5, 0.05),
}


@dataclass
class PolicyGenome:
    """
    Genetic representation of a trading policy.

    Attributes:
        genes: Dictionary of gene name -> value
        generation: Which generation this genome was created in
        parent_ids: IDs of parent genomes (for crossover tracking)
        fitness: Evaluated fitness score (None if not yet evaluated)
        fitness_components: Breakdown of fitness components
        created_at: Timestamp of creation
    """
    genes: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness: Optional[float] = None
    fitness_components: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Initialize missing genes with random values."""
        for gene_name, gene_range in GENE_DEFINITIONS.items():
            if gene_name not in self.genes:
                self.genes[gene_name] = gene_range.random()

    @property
    def id(self) -> str:
        """Generate unique ID based on gene values."""
        gene_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.md5(gene_str.encode()).hexdigest()[:12]

    @classmethod
    def random(cls, generation: int = 0) -> 'PolicyGenome':
        """Create a genome with random genes."""
        genes = {name: gene_range.random() for name, gene_range in GENE_DEFINITIONS.items()}
        return cls(genes=genes, generation=generation)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyGenome':
        """Create genome from dictionary."""
        return cls(
            genes=data.get('genes', {}),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            fitness=data.get('fitness'),
            fitness_components=data.get('fitness_components', {}),
            created_at=data.get('created_at', datetime.now().isoformat())
        )

    def to_dict(self) -> Dict:
        """Convert genome to dictionary."""
        return {
            'id': self.id,
            'genes': self.genes,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'fitness': self.fitness,
            'fitness_components': self.fitness_components,
            'created_at': self.created_at
        }

    def mutate(self, mutation_rate: float = 0.1) -> 'PolicyGenome':
        """
        Create mutated copy of this genome.

        Args:
            mutation_rate: Probability of each gene mutating

        Returns:
            New PolicyGenome with mutated genes
        """
        new_genes = {}
        for gene_name, value in self.genes.items():
            if random.random() < mutation_rate:
                gene_range = GENE_DEFINITIONS.get(gene_name)
                if gene_range:
                    new_genes[gene_name] = gene_range.mutate(value)
                else:
                    new_genes[gene_name] = value
            else:
                new_genes[gene_name] = value

        return PolicyGenome(
            genes=new_genes,
            generation=self.generation + 1,
            parent_ids=[self.id]
        )

    @staticmethod
    def crossover(parent1: 'PolicyGenome', parent2: 'PolicyGenome') -> Tuple['PolicyGenome', 'PolicyGenome']:
        """
        Create two offspring through crossover of two parent genomes.

        Uses uniform crossover - each gene randomly selected from either parent.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Tuple of two child genomes
        """
        child1_genes = {}
        child2_genes = {}

        for gene_name in GENE_DEFINITIONS.keys():
            if random.random() < 0.5:
                child1_genes[gene_name] = parent1.genes.get(gene_name, 0)
                child2_genes[gene_name] = parent2.genes.get(gene_name, 0)
            else:
                child1_genes[gene_name] = parent2.genes.get(gene_name, 0)
                child2_genes[gene_name] = parent1.genes.get(gene_name, 0)

        generation = max(parent1.generation, parent2.generation) + 1
        parent_ids = [parent1.id, parent2.id]

        return (
            PolicyGenome(genes=child1_genes, generation=generation, parent_ids=parent_ids),
            PolicyGenome(genes=child2_genes, generation=generation, parent_ids=parent_ids)
        )

    def to_decision_params(self) -> Dict:
        """
        Convert genome to decision_params.json format.

        Returns:
            Dictionary compatible with decision engine configuration
        """
        return {
            'max_positions': int(self.genes['max_positions']),
            'max_position_weight': self.genes['max_position_weight'],
            'buy_score_threshold': self.genes['buy_score_threshold'],
            'min_health_buy': self.genes['min_health_buy'],
            'trailing_stop_base': self.genes['trailing_stop_base'],
            'min_order_dollars': int(self.genes['min_order_dollars']),
            'health_collapse_threshold': self.genes['health_collapse_threshold'],
            'profit_take_threshold': self.genes['profit_take_threshold'],
            'max_holding_days': int(self.genes['max_holding_days']),
            'momentum_weight': self.genes['momentum_weight'],
            'mean_reversion_weight': self.genes['mean_reversion_weight'],
            'sector_max_weight': self.genes['sector_max_weight'],
            'correlation_penalty': self.genes['correlation_penalty'],
        }

    def to_regime_compatibility(self) -> Dict[str, Dict[str, float]]:
        """
        Convert genome to regime_compatibility.json format.

        Returns:
            Dictionary mapping regimes to multipliers
        """
        return {
            'calm_uptrend': {
                'position_multiplier': self.genes['regime_calm_uptrend'],
                'vol_low': self.genes['vol_low_multiplier'],
                'vol_med': self.genes['vol_med_multiplier'],
                'vol_high': self.genes['vol_high_multiplier'],
            },
            'risk_on_trend': {
                'position_multiplier': self.genes['regime_risk_on_trend'],
                'vol_low': self.genes['vol_low_multiplier'],
                'vol_med': self.genes['vol_med_multiplier'],
                'vol_high': self.genes['vol_high_multiplier'],
            },
            'choppy': {
                'position_multiplier': self.genes['regime_choppy'],
                'vol_low': self.genes['vol_low_multiplier'] * 0.9,
                'vol_med': self.genes['vol_med_multiplier'] * 0.8,
                'vol_high': self.genes['vol_high_multiplier'] * 0.7,
            },
            'risk_off_trend': {
                'position_multiplier': self.genes['regime_risk_off_trend'],
                'vol_low': self.genes['vol_low_multiplier'] * 0.7,
                'vol_med': self.genes['vol_med_multiplier'] * 0.6,
                'vol_high': self.genes['vol_high_multiplier'] * 0.5,
            },
            'high_vol_panic': {
                'position_multiplier': self.genes['regime_high_vol_panic'],
                'vol_low': self.genes['vol_low_multiplier'] * 0.5,
                'vol_med': self.genes['vol_med_multiplier'] * 0.3,
                'vol_high': self.genes['vol_high_multiplier'] * 0.2,
            },
        }

    def distance(self, other: 'PolicyGenome') -> float:
        """
        Calculate genetic distance to another genome.

        Uses normalized Euclidean distance.

        Args:
            other: Another PolicyGenome

        Returns:
            Distance value (0 = identical, higher = more different)
        """
        distances = []
        for gene_name, gene_range in GENE_DEFINITIONS.items():
            v1 = self.genes.get(gene_name, 0)
            v2 = other.genes.get(gene_name, 0)
            # Normalize to [0, 1] range
            range_size = gene_range.max_val - gene_range.min_val
            if range_size > 0:
                normalized_diff = abs(v1 - v2) / range_size
                distances.append(normalized_diff ** 2)

        return np.sqrt(np.mean(distances)) if distances else 0.0

    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return f"PolicyGenome(id={self.id}, gen={self.generation}, fitness={fitness_str})"
