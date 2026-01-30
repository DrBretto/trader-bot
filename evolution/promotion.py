"""
Template promotion system for deploying evolved policies.

Promotes top-performing genomes to become active trading templates,
with validation, versioning, and rollback capabilities.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from evolution.genome import PolicyGenome


@dataclass
class PromotionRecord:
    """Record of a template promotion."""
    genome_id: str
    version: str
    promoted_at: str
    fitness: float
    fitness_components: Dict[str, float]
    generation: int
    status: str  # 'active', 'retired', 'rolled_back'
    notes: str


class TemplatePromoter:
    """
    Manages promotion of evolved genomes to active trading templates.

    Features:
    - Validation before promotion
    - Version tracking
    - Rollback capability
    - Promotion history
    """

    def __init__(
        self,
        min_fitness_threshold: float = 0.3,
        min_sharpe_threshold: float = 0.5,
        max_drawdown_threshold: float = 0.25,
        min_trades_threshold: int = 10
    ):
        """
        Initialize promoter with validation thresholds.

        Args:
            min_fitness_threshold: Minimum overall fitness to promote
            min_sharpe_threshold: Minimum Sharpe ratio component
            max_drawdown_threshold: Maximum acceptable drawdown
            min_trades_threshold: Minimum trades in backtest
        """
        self.min_fitness_threshold = min_fitness_threshold
        self.min_sharpe_threshold = min_sharpe_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_trades_threshold = min_trades_threshold

        self.active_template: Optional[PolicyGenome] = None
        self.promotion_history: List[PromotionRecord] = []
        self.version_counter = 0

    def validate(self, genome: PolicyGenome) -> tuple[bool, List[str]]:
        """
        Validate genome meets promotion criteria.

        Args:
            genome: PolicyGenome to validate

        Returns:
            Tuple of (is_valid, list of validation issues)
        """
        issues = []

        if genome.fitness is None:
            issues.append("Genome has not been evaluated")
            return False, issues

        if genome.fitness < self.min_fitness_threshold:
            issues.append(f"Fitness {genome.fitness:.3f} below threshold {self.min_fitness_threshold}")

        components = genome.fitness_components

        sharpe = components.get('sharpe', 0)
        if sharpe < self.min_sharpe_threshold:
            issues.append(f"Sharpe {sharpe:.3f} below threshold {self.min_sharpe_threshold}")

        # Note: drawdown is typically negative or stored as absolute value
        # Assuming max_drawdown is stored as positive value in backtest
        calmar = components.get('calmar', 0)
        if calmar < 0:
            issues.append(f"Negative Calmar ratio {calmar:.3f}")

        return len(issues) == 0, issues

    def promote(
        self,
        genome: PolicyGenome,
        force: bool = False,
        notes: str = ""
    ) -> Optional[PromotionRecord]:
        """
        Promote genome to active template.

        Args:
            genome: PolicyGenome to promote
            force: Skip validation if True
            notes: Optional notes about this promotion

        Returns:
            PromotionRecord if successful, None otherwise
        """
        # Validate unless forced
        if not force:
            is_valid, issues = self.validate(genome)
            if not is_valid:
                print(f"Promotion rejected: {issues}")
                return None

        # Retire current active template
        if self.active_template:
            self._retire_current()

        # Create new version
        self.version_counter += 1
        version = f"v{self.version_counter}.{datetime.now().strftime('%Y%m%d')}"

        # Create promotion record
        record = PromotionRecord(
            genome_id=genome.id,
            version=version,
            promoted_at=datetime.now().isoformat(),
            fitness=genome.fitness or 0,
            fitness_components=genome.fitness_components.copy(),
            generation=genome.generation,
            status='active',
            notes=notes
        )

        self.promotion_history.append(record)
        self.active_template = genome

        print(f"Promoted genome {genome.id} as {version}")
        return record

    def _retire_current(self):
        """Retire the current active template."""
        if not self.active_template:
            return

        # Find and update the active record
        for record in reversed(self.promotion_history):
            if record.genome_id == self.active_template.id and record.status == 'active':
                record.status = 'retired'
                break

    def rollback(self, version: Optional[str] = None) -> bool:
        """
        Rollback to a previous template version.

        Args:
            version: Specific version to rollback to, or previous if None

        Returns:
            True if rollback successful
        """
        # Find the record to rollback to
        target_record = None

        if version:
            for record in self.promotion_history:
                if record.version == version:
                    target_record = record
                    break
        else:
            # Find the most recent retired record
            for record in reversed(self.promotion_history):
                if record.status == 'retired':
                    target_record = record
                    break

        if not target_record:
            print("No suitable version found for rollback")
            return False

        # Mark current as rolled back
        for record in self.promotion_history:
            if record.status == 'active':
                record.status = 'rolled_back'

        # Reactivate target
        target_record.status = 'active'

        print(f"Rolled back to {target_record.version}")
        return True

    def get_active_config(self) -> Optional[Dict]:
        """
        Get the active template as configuration files.

        Returns:
            Dictionary with 'decision_params' and 'regime_compatibility'
        """
        if not self.active_template:
            return None

        return {
            'decision_params': self.active_template.to_decision_params(),
            'regime_compatibility': self.active_template.to_regime_compatibility(),
            'metadata': {
                'genome_id': self.active_template.id,
                'generation': self.active_template.generation,
                'fitness': self.active_template.fitness,
                'promoted_at': datetime.now().isoformat()
            }
        }

    def export_to_s3_format(self) -> Dict[str, str]:
        """
        Export active template in S3-compatible format.

        Returns:
            Dictionary mapping S3 keys to JSON content
        """
        if not self.active_template:
            return {}

        config = self.get_active_config()
        if not config:
            return {}

        return {
            'config/decision_params.json': json.dumps(config['decision_params'], indent=2),
            'config/regime_compatibility.json': json.dumps(config['regime_compatibility'], indent=2),
            'config/template_metadata.json': json.dumps(config['metadata'], indent=2)
        }

    def save_state(self) -> Dict:
        """Save promoter state for persistence."""
        return {
            'version_counter': self.version_counter,
            'active_template': self.active_template.to_dict() if self.active_template else None,
            'promotion_history': [
                {
                    'genome_id': r.genome_id,
                    'version': r.version,
                    'promoted_at': r.promoted_at,
                    'fitness': r.fitness,
                    'fitness_components': r.fitness_components,
                    'generation': r.generation,
                    'status': r.status,
                    'notes': r.notes
                }
                for r in self.promotion_history
            ],
            'thresholds': {
                'min_fitness': self.min_fitness_threshold,
                'min_sharpe': self.min_sharpe_threshold,
                'max_drawdown': self.max_drawdown_threshold,
                'min_trades': self.min_trades_threshold
            }
        }

    def load_state(self, state: Dict):
        """Load promoter state from saved data."""
        self.version_counter = state.get('version_counter', 0)

        active_data = state.get('active_template')
        if active_data:
            self.active_template = PolicyGenome.from_dict(active_data)

        self.promotion_history = [
            PromotionRecord(**r) for r in state.get('promotion_history', [])
        ]

        thresholds = state.get('thresholds', {})
        self.min_fitness_threshold = thresholds.get('min_fitness', self.min_fitness_threshold)
        self.min_sharpe_threshold = thresholds.get('min_sharpe', self.min_sharpe_threshold)
        self.max_drawdown_threshold = thresholds.get('max_drawdown', self.max_drawdown_threshold)
        self.min_trades_threshold = thresholds.get('min_trades', self.min_trades_threshold)

    def get_promotion_summary(self) -> Dict:
        """Get summary of promotion history."""
        return {
            'total_promotions': len(self.promotion_history),
            'active_version': next(
                (r.version for r in self.promotion_history if r.status == 'active'),
                None
            ),
            'active_genome_id': self.active_template.id if self.active_template else None,
            'active_fitness': self.active_template.fitness if self.active_template else None,
            'history': [
                {
                    'version': r.version,
                    'status': r.status,
                    'fitness': r.fitness,
                    'promoted_at': r.promoted_at
                }
                for r in self.promotion_history[-10:]  # Last 10 promotions
            ]
        }
