"""
Merge historical context data with GDELT sentiment data.

Combines:
- training/data/historical_context.parquet (price/FRED features)
- training/data/historical_gdelt.parquet (GDELT sentiment)

Output: training/data/historical_combined.parquet
"""

import argparse
import pandas as pd
import numpy as np


def merge_historical_data(
    context_path: str = 'training/data/historical_context.parquet',
    gdelt_path: str = 'training/data/historical_gdelt.parquet',
    output_path: str = 'training/data/historical_combined.parquet'
) -> pd.DataFrame:
    """
    Merge historical context with GDELT data.

    Args:
        context_path: Path to historical context parquet
        gdelt_path: Path to GDELT parquet
        output_path: Path for combined output

    Returns:
        Combined DataFrame
    """
    print("Loading historical data...")

    # Load context data
    context_df = pd.read_parquet(context_path)
    context_df['date'] = pd.to_datetime(context_df['date'])
    print(f"  Context: {len(context_df)} rows, {context_df['date'].min()} to {context_df['date'].max()}")

    # Load GDELT data
    gdelt_df = pd.read_parquet(gdelt_path)
    gdelt_df['date'] = pd.to_datetime(gdelt_df['date'])
    print(f"  GDELT: {len(gdelt_df)} rows, {gdelt_df['date'].min()} to {gdelt_df['date'].max()}")

    # Merge on date
    print("Merging...")
    merged = context_df.merge(
        gdelt_df[['date', 'gdelt_doc_count', 'gdelt_avg_tone', 'gdelt_tone_std', 'gdelt_neg_tone_share']],
        on='date',
        how='left',
        suffixes=('', '_new')
    )

    # Update gdelt_avg_tone with new values (drop old placeholder column if exists)
    if 'gdelt_avg_tone_new' in merged.columns:
        merged['gdelt_avg_tone'] = merged['gdelt_avg_tone_new']
        merged = merged.drop(columns=['gdelt_avg_tone_new'])

    # Fill any remaining NaN GDELT values with 0 (for dates not in GDELT data)
    gdelt_cols = ['gdelt_doc_count', 'gdelt_avg_tone', 'gdelt_tone_std', 'gdelt_neg_tone_share']
    for col in gdelt_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    # Summary
    print()
    print("Merged data summary:")
    print(f"  Total rows: {len(merged)}")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    print()
    print("GDELT coverage:")
    gdelt_available = (merged['gdelt_doc_count'] > 0).sum()
    print(f"  Days with GDELT data: {gdelt_available} ({gdelt_available/len(merged)*100:.1f}%)")
    print(f"  Avg tone: {merged.loc[merged['gdelt_doc_count'] > 0, 'gdelt_avg_tone'].mean():.3f}")
    print()

    # Feature completeness
    print("Feature completeness:")
    for col in merged.columns:
        if col == 'date':
            continue
        null_pct = merged[col].isna().sum() / len(merged) * 100
        zero_pct = (merged[col] == 0).sum() / len(merged) * 100
        print(f"  {col:25} nulls: {null_pct:5.1f}%  zeros: {zero_pct:5.1f}%")

    # Save
    merged.to_parquet(output_path, index=False)
    print()
    print(f"Saved to {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Merge historical data with GDELT')
    parser.add_argument('--context', default='training/data/historical_context.parquet')
    parser.add_argument('--gdelt', default='training/data/historical_gdelt.parquet')
    parser.add_argument('--output', default='training/data/historical_combined.parquet')

    args = parser.parse_args()

    merge_historical_data(
        context_path=args.context,
        gdelt_path=args.gdelt,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
