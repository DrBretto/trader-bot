import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "training" / "utils" / "data_loader.py"
SPEC = importlib.util.spec_from_file_location("training_data_loader", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
S3DataDownloader = MODULE.S3DataDownloader


class StubDownloader(S3DataDownloader):
    def __init__(self):
        pass

    def list_daily_artifacts(self, prefix="daily/", days=365):
        return ["2026-03-27", "2026-03-28", "2026-04-01"]

    def download_parquet(self, key: str, use_cache: bool = True) -> pd.DataFrame:
        if key.endswith("prices.parquet"):
            return pd.DataFrame(
                [{"symbol": "SPY", "close": 500.0, "volume": 1000}]
            )

        if key.endswith("features.parquet"):
            return pd.DataFrame(
                [{"symbol": "SPY", "health_score": 0.8, "vol_bucket": "med"}]
            )

        if key == "daily/2026-03-27/context.parquet":
            return pd.DataFrame(
                [{
                    "date": "2026-03-27",
                    "spy_return_1d": -0.01,
                    "rate_2y": 3.96,
                    "regime": "panic",
                }]
            )

        if key == "daily/2026-03-28/context.parquet":
            return pd.DataFrame(
                [{
                    "date": "2026-03-28",
                    "spy_return_1d": -0.02,
                    "rate_2y": 3.90,
                    "rate_3m": 3.70,
                    "yield_slope_10y_3m": 0.69,
                    "vvix_value": 110.0,
                }]
            )

        if key == "daily/2026-04-01/context.parquet":
            return pd.DataFrame(
                [{
                    "date": "2026-04-01",
                    "spy_return_1d": 0.01,
                    "rate_2y": 3.82,
                    "rate_3m": 3.71,
                    "yield_slope_10y_3m": 0.64,
                    "vvix_value": 90.0,
                    "vix_term_slope": -0.12,
                    "vixy_return_5d": 0.02,
                }]
            )

        return pd.DataFrame()


def test_build_historical_dataset_normalizes_mixed_context_schemas():
    downloader = StubDownloader()

    prices_df, features_df, context_df = downloader.build_historical_dataset(max_days=10)

    assert len(prices_df) == 3
    assert len(features_df) == 3
    assert len(context_df) == 3
    assert {
        "date",
        "spy_return_1d",
        "rate_2y",
        "regime",
        "rate_3m",
        "yield_slope_10y_3m",
        "vvix_value",
        "vix_term_slope",
        "vixy_return_5d",
    }.issubset(context_df.columns)
    assert pd.isna(context_df.loc[0, "rate_3m"])
    assert pd.isna(context_df.loc[1, "vix_term_slope"])
    assert context_df.loc[2, "vix_term_slope"] == -0.12
