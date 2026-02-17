"""Historical artifact loading for offline optimizer replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from optimizer.config import OptimizerConfig
from src.utils.s3_client import S3Client


@dataclass
class DailySnapshot:
    date: str
    next_date: str
    features_df: pd.DataFrame
    inference: Dict[str, Any]
    signal_row: Dict[str, Any]
    next_prices_df: pd.DataFrame


@dataclass
class OptimizerDataset:
    snapshots: List[DailySnapshot]
    universe_df: pd.DataFrame

    @property
    def dates(self) -> List[str]:
        return [snapshot.date for snapshot in self.snapshots]

    def by_date(self) -> Dict[str, DailySnapshot]:
        return {snapshot.date: snapshot for snapshot in self.snapshots}



def _latest_signal_row(signal_df: pd.DataFrame, date_str: str) -> Optional[Dict[str, Any]]:
    if len(signal_df) == 0:
        return None
    row = signal_df.iloc[-1].to_dict()
    row['date'] = date_str
    return row


def load_optimizer_dataset(config: OptimizerConfig) -> OptimizerDataset:
    """Load date-aligned daily artifacts required for deterministic replay."""
    s3 = S3Client(config.bucket, config.region)
    daily_dates = s3.list_daily_dates(max_days=max(config.max_days + 10, config.max_days))
    if len(daily_dates) < 2:
        raise RuntimeError('Insufficient daily artifacts in S3 for optimizer replay')

    snapshots: List[DailySnapshot] = []

    for idx, date_str in enumerate(daily_dates[:-1]):
        next_date = daily_dates[idx + 1]

        features_df = s3.read_parquet(f'daily/{date_str}/features.parquet')
        inference = s3.read_json(f'daily/{date_str}/inference.json')
        signal_df = s3.read_parquet(f'daily/{date_str}/signals.parquet')
        next_prices_df = s3.read_parquet(f'daily/{next_date}/prices.parquet')

        signal_row = _latest_signal_row(signal_df, date_str)

        if len(features_df) == 0:
            continue
        if not inference or not isinstance(inference, dict):
            continue
        if signal_row is None:
            continue
        if len(next_prices_df) == 0:
            continue

        if 'date' not in inference:
            inference['date'] = date_str

        snapshots.append(
            DailySnapshot(
                date=date_str,
                next_date=next_date,
                features_df=features_df,
                inference=inference,
                signal_row=signal_row,
                next_prices_df=next_prices_df,
            )
        )

    if len(snapshots) < (config.train_days + config.test_days + config.gate_days):
        raise RuntimeError(
            'Insufficient aligned snapshots for configured train/test/gate windows '
            f'(have={len(snapshots)})'
        )

    snapshots = snapshots[-config.max_days:]

    universe_df = s3.read_csv('config/universe.csv')
    if len(universe_df) == 0:
        universe_df = pd.DataFrame()

    return OptimizerDataset(
        snapshots=snapshots,
        universe_df=universe_df,
    )


def read_discovery_inputs(config: OptimizerConfig) -> Dict[str, Any]:
    """Load discovery snapshots to persist in run metadata."""

    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json_load(path)

    return {
        'pipeline_map': _read_json(config.pipeline_map_file),
        'data_inventory': _read_json(config.data_inventory_file),
        'param_inventory': _read_json(config.inventory_file),
    }


def json_load(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding='utf-8'))
