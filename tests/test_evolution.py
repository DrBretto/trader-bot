"""Tests for evolutionary search module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from evolution.genome import PolicyGenome, GENE_DEFINITIONS
from evolution.fitness import FitnessEvaluator, BacktestResult
from evolution.genetic import GeneticAlgorithm
from evolution.promotion import TemplatePromoter


class TestPolicyGenome:
    """Tests for PolicyGenome class."""

    def test_random_genome_has_all_genes(self):
        """Test that random genome contains all defined genes."""
        genome = PolicyGenome.random()
        for gene_name in GENE_DEFINITIONS.keys():
            assert gene_name in genome.genes

    def test_genes_within_range(self):
        """Test that all genes are within valid ranges."""
        genome = PolicyGenome.random()
        for gene_name, value in genome.genes.items():
            gene_range = GENE_DEFINITIONS[gene_name]
            assert gene_range.min_val <= value <= gene_range.max_val

    def test_genome_id_is_deterministic(self):
        """Test that genome ID is based on genes."""
        genome1 = PolicyGenome(genes={'max_positions': 5})
        genome2 = PolicyGenome(genes={'max_positions': 5})
        # IDs should be same for same genes (after post_init fills others)
        # Actually they'll differ because post_init adds random genes
        # Let's test that same full genes give same ID
        genome3 = PolicyGenome.from_dict(genome1.to_dict())
        genome3.genes = genome1.genes.copy()
        assert genome1.id == PolicyGenome(genes=genome1.genes.copy()).id

    def test_mutation_changes_genes(self):
        """Test that mutation modifies some genes."""
        genome = PolicyGenome.random()
        mutated = genome.mutate(mutation_rate=1.0)  # 100% mutation rate

        # At least some genes should change
        changes = sum(1 for k in genome.genes if genome.genes[k] != mutated.genes[k])
        assert changes > 0

    def test_mutation_stays_in_range(self):
        """Test that mutated genes stay within valid ranges."""
        genome = PolicyGenome.random()
        for _ in range(10):
            mutated = genome.mutate(mutation_rate=0.5)
            for gene_name, value in mutated.genes.items():
                gene_range = GENE_DEFINITIONS[gene_name]
                assert gene_range.min_val <= value <= gene_range.max_val

    def test_crossover_produces_two_children(self):
        """Test that crossover produces two distinct children."""
        parent1 = PolicyGenome.random()
        parent2 = PolicyGenome.random()

        child1, child2 = PolicyGenome.crossover(parent1, parent2)

        assert child1.id != child2.id
        assert child1.generation == parent1.generation + 1
        assert parent1.id in child1.parent_ids
        assert parent2.id in child1.parent_ids

    def test_crossover_inherits_from_parents(self):
        """Test that children inherit genes from both parents."""
        parent1 = PolicyGenome.random()
        parent2 = PolicyGenome.random()

        child1, child2 = PolicyGenome.crossover(parent1, parent2)

        # Each gene should come from one parent
        for gene_name in GENE_DEFINITIONS.keys():
            p1_val = parent1.genes[gene_name]
            p2_val = parent2.genes[gene_name]
            c1_val = child1.genes[gene_name]
            c2_val = child2.genes[gene_name]

            assert c1_val == p1_val or c1_val == p2_val
            assert c2_val == p1_val or c2_val == p2_val

    def test_to_decision_params_format(self):
        """Test that decision params are in expected format."""
        genome = PolicyGenome.random()
        params = genome.to_decision_params()

        assert 'max_positions' in params
        assert 'buy_score_threshold' in params
        assert 'trailing_stop_base' in params
        assert isinstance(params['max_positions'], int)

    def test_to_regime_compatibility_format(self):
        """Test that regime compatibility is in expected format."""
        genome = PolicyGenome.random()
        compat = genome.to_regime_compatibility()

        assert 'calm_uptrend' in compat
        assert 'risk_on_trend' in compat
        assert 'high_vol_panic' in compat
        assert 'position_multiplier' in compat['calm_uptrend']

    def test_distance_same_genome_is_zero(self):
        """Test that distance to self is zero."""
        genome = PolicyGenome.random()
        assert genome.distance(genome) == 0

    def test_distance_different_genomes_positive(self):
        """Test that distance between different genomes is positive."""
        genome1 = PolicyGenome.random()
        genome2 = PolicyGenome.random()
        assert genome1.distance(genome2) > 0

    def test_serialization_roundtrip(self):
        """Test that genome survives serialization."""
        genome = PolicyGenome.random()
        genome.fitness = 0.75
        genome.fitness_components = {'sharpe': 1.2}

        data = genome.to_dict()
        restored = PolicyGenome.from_dict(data)

        assert restored.genes == genome.genes
        assert restored.fitness == genome.fitness
        assert restored.generation == genome.generation


class TestFitnessEvaluator:
    """Tests for FitnessEvaluator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_days = 100
        n_symbols = 5
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        symbols = [f'SYM{i}' for i in range(n_symbols)]

        # Prices
        prices_data = []
        for symbol in symbols:
            base_price = 100
            prices = [base_price]
            for _ in range(n_days - 1):
                prices.append(prices[-1] * (1 + np.random.randn() * 0.02))
            for i, date in enumerate(dates):
                prices_data.append({'date': date, 'symbol': symbol, 'close': prices[i]})
        prices_df = pd.DataFrame(prices_data)

        # Features
        features_data = []
        for symbol in symbols:
            for date in dates:
                features_data.append({
                    'date': date,
                    'symbol': symbol,
                    'health_score': np.random.uniform(0.4, 0.9),
                    'vol_bucket': np.random.choice(['low', 'med', 'high']),
                    'return_21d': np.random.randn() * 0.05,
                    'return_63d': np.random.randn() * 0.10
                })
        features_df = pd.DataFrame(features_data)

        # Context
        regimes = ['calm_uptrend', 'risk_on_trend', 'choppy']
        context_data = [{'date': date, 'regime': np.random.choice(regimes)} for date in dates]
        context_df = pd.DataFrame(context_data)

        return prices_df, features_df, context_df

    def test_evaluator_initializes(self, sample_data):
        """Test that evaluator initializes with sample data."""
        prices_df, features_df, context_df = sample_data
        evaluator = FitnessEvaluator(prices_df, features_df, context_df)
        assert len(evaluator.dates) > 0
        assert len(evaluator.symbols) > 0

    def test_backtest_returns_result(self, sample_data):
        """Test that backtest produces a result."""
        prices_df, features_df, context_df = sample_data
        evaluator = FitnessEvaluator(prices_df, features_df, context_df)
        genome = PolicyGenome.random()

        result = evaluator.backtest(genome)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert result.total_return is not None

    def test_evaluate_sets_fitness(self, sample_data):
        """Test that evaluate sets genome fitness."""
        prices_df, features_df, context_df = sample_data
        evaluator = FitnessEvaluator(prices_df, features_df, context_df)
        genome = PolicyGenome.random()

        assert genome.fitness is None
        evaluator.evaluate(genome)
        assert genome.fitness is not None

    def test_evaluate_sets_components(self, sample_data):
        """Test that evaluate sets fitness components."""
        prices_df, features_df, context_df = sample_data
        evaluator = FitnessEvaluator(prices_df, features_df, context_df)
        genome = PolicyGenome.random()

        evaluator.evaluate(genome)

        assert 'sharpe' in genome.fitness_components
        assert 'calmar' in genome.fitness_components
        assert 'win_rate' in genome.fitness_components


class TestGeneticAlgorithm:
    """Tests for GeneticAlgorithm class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with minimal data."""
        n_days = 50
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        symbols = ['A', 'B', 'C']

        prices_data = []
        for symbol in symbols:
            prices = 100 * np.cumprod(1 + np.random.randn(n_days) * 0.02)
            for i, date in enumerate(dates):
                prices_data.append({'date': date, 'symbol': symbol, 'close': prices[i]})
        prices_df = pd.DataFrame(prices_data)

        features_data = []
        for symbol in symbols:
            for date in dates:
                features_data.append({
                    'date': date, 'symbol': symbol,
                    'health_score': 0.6, 'vol_bucket': 'med',
                    'return_21d': 0.02, 'return_63d': 0.05
                })
        features_df = pd.DataFrame(features_data)

        context_df = pd.DataFrame([{'date': d, 'regime': 'calm_uptrend'} for d in dates])

        return FitnessEvaluator(prices_df, features_df, context_df)

    def test_initialize_population(self, evaluator):
        """Test population initialization."""
        ga = GeneticAlgorithm(evaluator, population_size=10)
        ga.initialize_population()

        assert len(ga.population) == 10
        assert all(g.fitness is not None for g in ga.population)

    def test_evolve_improves_fitness(self, evaluator):
        """Test that evolution tends to improve fitness."""
        ga = GeneticAlgorithm(evaluator, population_size=10)
        ga.initialize_population()

        initial_best = ga.best_genome.fitness

        ga.evolve(generations=5)

        # Best fitness should not decrease
        assert ga.best_genome.fitness >= initial_best

    def test_get_top_genomes(self, evaluator):
        """Test getting top genomes."""
        ga = GeneticAlgorithm(evaluator, population_size=10)
        ga.initialize_population()

        top = ga.get_top_genomes(3)

        assert len(top) == 3
        # Should be sorted by fitness
        assert top[0].fitness >= top[1].fitness >= top[2].fitness

    def test_save_load_state(self, evaluator):
        """Test state persistence."""
        ga = GeneticAlgorithm(evaluator, population_size=10)
        ga.initialize_population()
        ga.evolve(generations=2)

        state = ga.save_state()

        ga2 = GeneticAlgorithm(evaluator, population_size=10)
        ga2.load_state(state)

        assert ga2.generation == ga.generation
        assert len(ga2.population) == len(ga.population)


class TestTemplatePromoter:
    """Tests for TemplatePromoter class."""

    def test_validate_unevaluated_genome(self):
        """Test that unevaluated genomes fail validation."""
        promoter = TemplatePromoter()
        genome = PolicyGenome.random()

        is_valid, issues = promoter.validate(genome)

        assert not is_valid
        assert len(issues) > 0

    def test_validate_good_genome(self):
        """Test that good genomes pass validation."""
        promoter = TemplatePromoter(min_fitness_threshold=0.1, min_sharpe_threshold=0.1)
        genome = PolicyGenome.random()
        genome.fitness = 0.5
        genome.fitness_components = {'sharpe': 0.8, 'calmar': 1.0}

        is_valid, issues = promoter.validate(genome)

        assert is_valid
        assert len(issues) == 0

    def test_promote_valid_genome(self):
        """Test promoting a valid genome."""
        promoter = TemplatePromoter(min_fitness_threshold=0.1, min_sharpe_threshold=0.1)
        genome = PolicyGenome.random()
        genome.fitness = 0.5
        genome.fitness_components = {'sharpe': 0.8, 'calmar': 1.0}

        record = promoter.promote(genome)

        assert record is not None
        assert promoter.active_template == genome
        assert record.status == 'active'

    def test_promote_rejects_invalid(self):
        """Test that invalid genomes are rejected."""
        promoter = TemplatePromoter(min_fitness_threshold=0.9)
        genome = PolicyGenome.random()
        genome.fitness = 0.5
        genome.fitness_components = {'sharpe': 0.8}

        record = promoter.promote(genome)

        assert record is None
        assert promoter.active_template is None

    def test_force_promote_bypasses_validation(self):
        """Test that force=True bypasses validation."""
        promoter = TemplatePromoter(min_fitness_threshold=0.9)
        genome = PolicyGenome.random()
        genome.fitness = 0.1
        genome.fitness_components = {}

        record = promoter.promote(genome, force=True)

        assert record is not None
        assert promoter.active_template == genome

    def test_get_active_config(self):
        """Test getting active config."""
        promoter = TemplatePromoter()
        genome = PolicyGenome.random()
        genome.fitness = 0.5
        genome.fitness_components = {}

        promoter.promote(genome, force=True)
        config = promoter.get_active_config()

        assert config is not None
        assert 'decision_params' in config
        assert 'regime_compatibility' in config
        assert 'metadata' in config

    def test_promotion_history_tracking(self):
        """Test that promotion history is tracked."""
        promoter = TemplatePromoter()

        for i in range(3):
            genome = PolicyGenome.random()
            genome.fitness = 0.5 + i * 0.1
            genome.fitness_components = {}
            promoter.promote(genome, force=True)

        assert len(promoter.promotion_history) == 3
        # Only last should be active
        active_count = sum(1 for r in promoter.promotion_history if r.status == 'active')
        assert active_count == 1

    def test_save_load_state(self):
        """Test state persistence."""
        promoter = TemplatePromoter()
        genome = PolicyGenome.random()
        genome.fitness = 0.5
        genome.fitness_components = {}
        promoter.promote(genome, force=True)

        state = promoter.save_state()

        promoter2 = TemplatePromoter()
        promoter2.load_state(state)

        assert promoter2.active_template is not None
        assert len(promoter2.promotion_history) == 1
