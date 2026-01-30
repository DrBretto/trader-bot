"""
Main evolution orchestrator.

Runs the evolutionary search process:
1. Load historical data from S3
2. Initialize or resume evolution
3. Run evolution for specified generations
4. Promote best genome if it passes validation
5. Save state and upload results
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd

from evolution.genome import PolicyGenome
from evolution.fitness import FitnessEvaluator
from evolution.genetic import GeneticAlgorithm
from evolution.promotion import TemplatePromoter


def load_data_from_s3(bucket: str, max_days: int = 365) -> tuple:
    """
    Load historical data from S3 for backtesting.

    Args:
        bucket: S3 bucket name
        max_days: Maximum days of history to load

    Returns:
        Tuple of (prices_df, features_df, context_df)
    """
    from src.utils.s3_client import S3Client

    s3 = S3Client(bucket)

    # Load aggregated historical data
    prices_df = s3.read_parquet('historical/prices.parquet')
    features_df = s3.read_parquet('historical/features.parquet')
    context_df = s3.read_parquet('historical/context.parquet')

    # Filter to max_days
    if len(prices_df) > 0:
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=max_days)
        prices_df = prices_df[prices_df['date'] >= cutoff_date]
        features_df = features_df[features_df['date'] >= cutoff_date]
        context_df = context_df[context_df['date'] >= cutoff_date]

    return prices_df, features_df, context_df


def run_evolution(
    bucket: str,
    population_size: int = 50,
    generations: int = 50,
    max_days: int = 365,
    resume: bool = False,
    auto_promote: bool = True,
    output_dir: str = '/tmp/evolution'
) -> dict:
    """
    Run evolution process.

    Args:
        bucket: S3 bucket name
        population_size: Size of genome population
        generations: Number of generations to evolve
        max_days: Days of historical data to use
        resume: Whether to resume from saved state
        auto_promote: Automatically promote best genome if valid
        output_dir: Local directory for outputs

    Returns:
        Dictionary with results
    """
    from src.utils.s3_client import S3Client

    os.makedirs(output_dir, exist_ok=True)
    s3 = S3Client(bucket)

    print(f"Starting evolution run at {datetime.now()}")
    print(f"Population: {population_size}, Generations: {generations}")

    # Load data
    print("Loading historical data...")
    prices_df, features_df, context_df = load_data_from_s3(bucket, max_days)

    if len(prices_df) == 0:
        raise ValueError("No historical data available for evolution")

    print(f"Loaded {len(prices_df)} price records, {len(features_df)} feature records")

    # Initialize evaluator
    evaluator = FitnessEvaluator(
        prices_df=prices_df,
        features_df=features_df,
        context_df=context_df
    )

    # Initialize or resume GA
    ga = GeneticAlgorithm(
        evaluator=evaluator,
        population_size=population_size
    )

    # Resume from saved state if requested
    seed_genomes = None
    if resume:
        try:
            state = s3.read_json('evolution/ga_state.json')
            if state:
                ga.load_state(state)
                print(f"Resumed from generation {ga.generation}")
        except Exception as e:
            print(f"Could not resume: {e}, starting fresh")

    # Initialize population
    if ga.generation == 0:
        # Try to load current decision params as seed
        try:
            current_params = s3.read_json('config/decision_params.json')
            if current_params:
                seed_genome = PolicyGenome(genes={})
                # Map current params to genes
                for key, value in current_params.items():
                    if key in seed_genome.genes:
                        seed_genome.genes[key] = value
                seed_genomes = [seed_genome]
                print("Seeded population with current config")
        except:
            pass

        ga.initialize_population(seed_genomes)

    # Evolution callback
    def on_generation(gen: int, stats):
        print(f"Gen {gen}: best={stats.best_fitness:.4f}, avg={stats.avg_fitness:.4f}, "
              f"diversity={stats.population_diversity:.3f}")

    # Run evolution
    print(f"\nRunning evolution for {generations} generations...")
    best_genome = ga.evolve(generations=generations, callback=on_generation)

    print(f"\nEvolution complete!")
    print(f"Best genome: {best_genome}")
    print(f"Fitness components: {best_genome.fitness_components}")

    # Save GA state
    ga_state = ga.save_state()
    state_path = os.path.join(output_dir, 'ga_state.json')
    with open(state_path, 'w') as f:
        json.dump(ga_state, f, indent=2)
    s3.write_json(ga_state, 'evolution/ga_state.json')
    print(f"Saved GA state to S3")

    # Save top genomes
    top_genomes = ga.get_top_genomes(10)
    top_data = [g.to_dict() for g in top_genomes]
    s3.write_json(top_data, 'evolution/top_genomes.json')

    # Promotion
    promoter = TemplatePromoter()

    # Load promoter state
    try:
        promoter_state = s3.read_json('evolution/promoter_state.json')
        if promoter_state:
            promoter.load_state(promoter_state)
    except:
        pass

    # Validate and potentially promote
    is_valid, issues = promoter.validate(best_genome)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if issues:
        print(f"Issues: {issues}")

    promotion_record = None
    if auto_promote and is_valid:
        promotion_record = promoter.promote(
            best_genome,
            notes=f"Auto-promoted after generation {ga.generation}"
        )

        if promotion_record:
            # Export and upload new config
            config_exports = promoter.export_to_s3_format()
            for key, content in config_exports.items():
                s3.write_json(json.loads(content), key)
            print("Uploaded new config to S3")

    # Save promoter state
    promoter_state = promoter.save_state()
    s3.write_json(promoter_state, 'evolution/promoter_state.json')

    # Generate report
    report = {
        'run_date': datetime.now().isoformat(),
        'generations_run': generations,
        'final_generation': ga.generation,
        'population_size': population_size,
        'data_days': max_days,
        'best_genome': best_genome.to_dict(),
        'validation_passed': is_valid,
        'validation_issues': issues,
        'promoted': promotion_record is not None,
        'promotion_version': promotion_record.version if promotion_record else None,
        'evolution_history': [
            {
                'generation': s.generation,
                'best_fitness': s.best_fitness,
                'avg_fitness': s.avg_fitness
            }
            for s in ga.history
        ]
    }

    report_path = os.path.join(output_dir, 'evolution_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    s3.write_json(report, f'evolution/reports/{datetime.now().strftime("%Y%m%d")}.json')

    return report


def create_dummy_data(n_days: int = 200, n_symbols: int = 20):
    """Create dummy data for testing evolution locally."""
    import numpy as np

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    symbols = [f'SYM{i}' for i in range(n_symbols)]

    # Prices
    prices_data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        returns = np.random.randn(n_days) * 0.02
        prices = base_price * np.cumprod(1 + returns)
        for i, date in enumerate(dates):
            prices_data.append({
                'date': date,
                'symbol': symbol,
                'close': prices[i]
            })
    prices_df = pd.DataFrame(prices_data)

    # Features
    features_data = []
    for symbol in symbols:
        for date in dates:
            features_data.append({
                'date': date,
                'symbol': symbol,
                'health_score': np.random.uniform(0.3, 0.9),
                'vol_bucket': np.random.choice(['low', 'med', 'high']),
                'return_21d': np.random.randn() * 0.05,
                'return_63d': np.random.randn() * 0.10
            })
    features_df = pd.DataFrame(features_data)

    # Context
    regimes = ['calm_uptrend', 'risk_on_trend', 'choppy', 'risk_off_trend', 'high_vol_panic']
    context_data = []
    for date in dates:
        context_data.append({
            'date': date,
            'regime': np.random.choice(regimes, p=[0.3, 0.3, 0.2, 0.15, 0.05])
        })
    context_df = pd.DataFrame(context_data)

    return prices_df, features_df, context_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evolutionary search')
    parser.add_argument('--bucket', type=str, default='investment-system-data')
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--max-days', type=int, default=365)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-promote', action='store_true')
    parser.add_argument('--local-test', action='store_true', help='Use dummy data for local testing')
    parser.add_argument('--output-dir', type=str, default='/tmp/evolution')
    args = parser.parse_args()

    if args.local_test:
        # Run with dummy data
        print("Running local test with dummy data...")

        prices_df, features_df, context_df = create_dummy_data()

        evaluator = FitnessEvaluator(
            prices_df=prices_df,
            features_df=features_df,
            context_df=context_df
        )

        ga = GeneticAlgorithm(
            evaluator=evaluator,
            population_size=args.population
        )

        ga.initialize_population()

        def on_gen(gen, stats):
            print(f"Gen {gen}: best={stats.best_fitness:.4f}, avg={stats.avg_fitness:.4f}")

        best = ga.evolve(generations=args.generations, callback=on_gen)
        print(f"\nBest genome: {best}")
        print(f"Decision params: {json.dumps(best.to_decision_params(), indent=2)}")
    else:
        report = run_evolution(
            bucket=args.bucket,
            population_size=args.population,
            generations=args.generations,
            max_days=args.max_days,
            resume=args.resume,
            auto_promote=not args.no_promote,
            output_dir=args.output_dir
        )

        print(f"\nEvolution report saved")
        print(f"Best fitness: {report['best_genome']['fitness']}")
        print(f"Promoted: {report['promoted']}")
