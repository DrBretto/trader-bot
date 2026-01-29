"""Data validation step for the daily pipeline."""

import pandas as pd
from typing import Dict, Any, List

from src.utils.data_validation import validate_price_data, validate_fred_data


def run(
    prices_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run data validation for prices and macro data.

    Args:
        prices_df: Price data from ingestion
        fred_df: FRED data from ingestion
        config: Configuration with universe and validation params

    Returns:
        Validation results including degraded_mode flag
    """
    # Extract universe symbols from config
    if isinstance(config.get('universe'), pd.DataFrame):
        universe = config['universe']['symbol'].tolist()
    elif isinstance(config.get('universe'), list):
        universe = [u['symbol'] if isinstance(u, dict) else u for u in config['universe']]
    else:
        universe = []

    # Required FRED series
    required_fred = ['DGS2', 'DGS10', 'VIXCLS', 'DCOILWTICO']

    print("Validating price data...")
    price_validation = validate_price_data(
        prices_df,
        universe,
        max_missing_pct=0.10,  # Allow up to 10% missing
        max_staleness_days=3
    )

    print("Validating FRED data...")
    fred_validation = validate_fred_data(
        fred_df,
        required_fred,
        max_staleness_days=5
    )

    # Aggregate validation results
    all_issues = price_validation['issues'] + fred_validation['issues']

    # Determine if we should run in degraded mode
    degraded_mode = (
        price_validation.get('degraded_mode', False) or
        fred_validation.get('degraded_mode', False)
    )

    # Critical failures that should halt the pipeline
    critical_failure = False
    if price_validation['coverage'] < 0.50:
        critical_failure = True
        all_issues.append("CRITICAL: Less than 50% price coverage")

    if len(price_validation.get('missing_critical', [])) > 3:
        critical_failure = True
        all_issues.append("CRITICAL: Too many critical symbols missing")

    result = {
        'valid': len(all_issues) == 0 and not critical_failure,
        'degraded_mode': degraded_mode,
        'critical_failure': critical_failure,
        'issues': all_issues,
        'price_coverage': price_validation['coverage'],
        'latest_price_date': price_validation.get('latest_date'),
        'missing_symbols': price_validation.get('missing_symbols', []),
        'missing_critical': price_validation.get('missing_critical', []),
        'fred_missing_series': fred_validation.get('missing_series', []),
        'warnings': []
    }

    # Add warnings for non-critical issues
    if degraded_mode and not critical_failure:
        result['warnings'].append("Running in degraded mode due to data quality issues")

    if price_validation['coverage'] < 0.90:
        result['warnings'].append(f"Price coverage is {price_validation['coverage']:.1%}")

    # Log validation summary
    if result['valid']:
        print("Data validation: PASSED")
    elif result['critical_failure']:
        print(f"Data validation: CRITICAL FAILURE - {all_issues}")
    else:
        print(f"Data validation: DEGRADED - {len(all_issues)} issues")

    return result
