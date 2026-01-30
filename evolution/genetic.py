"""
Genetic Algorithm for evolving trading policies.

Implements selection, crossover, mutation, and population management
for optimizing PolicyGenome instances.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime

from evolution.genome import PolicyGenome
from evolution.fitness import FitnessEvaluator


@dataclass
class EvolutionStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    fitness_std: float
    population_diversity: float
    best_genome_id: str
    timestamp: str


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving trading policies.

    Features:
    - Tournament selection
    - Uniform crossover
    - Gaussian mutation
    - Elitism (preserve top performers)
    - Diversity maintenance
    """

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        population_size: int = 50,
        elite_count: int = 5,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        diversity_threshold: float = 0.1
    ):
        """
        Initialize genetic algorithm.

        Args:
            evaluator: FitnessEvaluator for scoring genomes
            population_size: Number of genomes in population
            elite_count: Number of top genomes to preserve unchanged
            tournament_size: Number of genomes in tournament selection
            crossover_rate: Probability of crossover vs cloning
            mutation_rate: Probability of each gene mutating
            diversity_threshold: Minimum genetic distance for diversity
        """
        self.evaluator = evaluator
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.diversity_threshold = diversity_threshold

        self.population: List[PolicyGenome] = []
        self.generation = 0
        self.history: List[EvolutionStats] = []
        self.best_genome: Optional[PolicyGenome] = None

    def initialize_population(self, seed_genomes: Optional[List[PolicyGenome]] = None):
        """
        Initialize population with random or seeded genomes.

        Args:
            seed_genomes: Optional list of genomes to include in initial population
        """
        self.population = []
        self.generation = 0

        # Add seed genomes if provided
        if seed_genomes:
            for genome in seed_genomes[:self.population_size]:
                self.population.append(genome)

        # Fill remaining with random genomes
        while len(self.population) < self.population_size:
            self.population.append(PolicyGenome.random(generation=0))

        # Evaluate initial population
        self.evaluator.evaluate_population(self.population)
        self._update_best()
        self._record_stats()

    def evolve(self, generations: int = 50, callback: Optional[Callable] = None) -> PolicyGenome:
        """
        Run evolution for specified number of generations.

        Args:
            generations: Number of generations to evolve
            callback: Optional callback(generation, stats) called each generation

        Returns:
            Best genome found
        """
        for _ in range(generations):
            self.generation += 1
            self._evolve_generation()

            stats = self.history[-1]
            if callback:
                callback(self.generation, stats)

            # Early stopping if fitness plateaus
            if len(self.history) >= 10:
                recent_best = [s.best_fitness for s in self.history[-10:]]
                if max(recent_best) - min(recent_best) < 0.001:
                    print(f"Fitness plateau detected at generation {self.generation}")
                    break

        return self.best_genome

    def _evolve_generation(self):
        """Evolve population by one generation."""
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness or float('-inf'), reverse=True)

        new_population = []

        # Elitism: preserve top performers
        for genome in self.population[:self.elite_count]:
            new_population.append(genome)

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                # Ensure parents are different enough
                attempts = 0
                while parent1.distance(parent2) < self.diversity_threshold and attempts < 5:
                    parent2 = self._tournament_select()
                    attempts += 1

                child1, child2 = PolicyGenome.crossover(parent1, parent2)

                # Mutate children
                child1 = child1.mutate(self.mutation_rate)
                child2 = child2.mutate(self.mutation_rate)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # Clone and mutate
                parent = self._tournament_select()
                child = parent.mutate(self.mutation_rate * 1.5)  # Higher mutation for clones
                new_population.append(child)

        # Maintain diversity - replace similar genomes
        new_population = self._maintain_diversity(new_population)

        # Update population
        self.population = new_population[:self.population_size]

        # Evaluate new genomes
        self.evaluator.evaluate_population(self.population)

        # Update best and record stats
        self._update_best()
        self._record_stats()

    def _tournament_select(self) -> PolicyGenome:
        """Select genome using tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        tournament.sort(key=lambda g: g.fitness or float('-inf'), reverse=True)
        return tournament[0]

    def _maintain_diversity(self, population: List[PolicyGenome]) -> List[PolicyGenome]:
        """
        Maintain genetic diversity by replacing too-similar genomes.

        Args:
            population: List of genomes

        Returns:
            Diversified population
        """
        diverse_population = []

        for genome in population:
            is_diverse = True
            for existing in diverse_population:
                if genome.distance(existing) < self.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_population.append(genome)
            elif len(diverse_population) < self.population_size:
                # Replace with random genome
                diverse_population.append(PolicyGenome.random(generation=self.generation))

        return diverse_population

    def _update_best(self):
        """Update best genome found so far."""
        for genome in self.population:
            if genome.fitness is not None:
                if self.best_genome is None or genome.fitness > (self.best_genome.fitness or float('-inf')):
                    self.best_genome = genome

    def _record_stats(self):
        """Record statistics for current generation."""
        fitnesses = [g.fitness for g in self.population if g.fitness is not None]

        if not fitnesses:
            return

        # Calculate diversity
        distances = []
        for i, g1 in enumerate(self.population):
            for g2 in self.population[i+1:]:
                distances.append(g1.distance(g2))
        diversity = np.mean(distances) if distances else 0

        stats = EvolutionStats(
            generation=self.generation,
            best_fitness=max(fitnesses),
            avg_fitness=np.mean(fitnesses),
            worst_fitness=min(fitnesses),
            fitness_std=np.std(fitnesses),
            population_diversity=diversity,
            best_genome_id=self.best_genome.id if self.best_genome else '',
            timestamp=datetime.now().isoformat()
        )

        self.history.append(stats)

    def get_top_genomes(self, n: int = 5) -> List[PolicyGenome]:
        """
        Get top n genomes by fitness.

        Args:
            n: Number of genomes to return

        Returns:
            List of top genomes
        """
        sorted_pop = sorted(
            self.population,
            key=lambda g: g.fitness or float('-inf'),
            reverse=True
        )
        return sorted_pop[:n]

    def save_state(self) -> Dict:
        """
        Save algorithm state for persistence.

        Returns:
            Dictionary with full state
        """
        return {
            'generation': self.generation,
            'population': [g.to_dict() for g in self.population],
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'history': [
                {
                    'generation': s.generation,
                    'best_fitness': s.best_fitness,
                    'avg_fitness': s.avg_fitness,
                    'worst_fitness': s.worst_fitness,
                    'fitness_std': s.fitness_std,
                    'population_diversity': s.population_diversity,
                    'best_genome_id': s.best_genome_id,
                    'timestamp': s.timestamp
                }
                for s in self.history
            ],
            'config': {
                'population_size': self.population_size,
                'elite_count': self.elite_count,
                'tournament_size': self.tournament_size,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'diversity_threshold': self.diversity_threshold
            }
        }

    def load_state(self, state: Dict):
        """
        Load algorithm state from saved data.

        Args:
            state: Dictionary with saved state
        """
        self.generation = state.get('generation', 0)
        self.population = [
            PolicyGenome.from_dict(g) for g in state.get('population', [])
        ]

        best_data = state.get('best_genome')
        if best_data:
            self.best_genome = PolicyGenome.from_dict(best_data)

        self.history = [
            EvolutionStats(**s) for s in state.get('history', [])
        ]

        config = state.get('config', {})
        self.population_size = config.get('population_size', self.population_size)
        self.elite_count = config.get('elite_count', self.elite_count)
        self.tournament_size = config.get('tournament_size', self.tournament_size)
        self.crossover_rate = config.get('crossover_rate', self.crossover_rate)
        self.mutation_rate = config.get('mutation_rate', self.mutation_rate)
        self.diversity_threshold = config.get('diversity_threshold', self.diversity_threshold)
