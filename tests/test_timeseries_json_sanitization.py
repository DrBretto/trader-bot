"""Guardrail: timeseries JSON must not contain NaN/Infinity tokens."""

import json
import math

import pandas as pd

from src.steps import publish_artifacts


class FakeS3:
    """Minimal capture stub."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.json_writes = {}

    def write_json(self, obj, key: str) -> bool:
        # Mimic the real write_json with allow_nan=False
        body = json.dumps(obj, indent=2, default=str, allow_nan=False)
        self.json_writes[key] = body
        return True

    def read_parquet(self, key: str) -> pd.DataFrame:
        # Return a DataFrame with NaN to simulate the real scenario
        return pd.DataFrame([
            {
                'date': '2025-08-04',
                'final_regime_label': 'calm_uptrend',
                'override_reason': float('nan'),  # The known defect
                'macro_credit_score': 0.5,
                'vol_uncertainty_score': 0.3,
            },
            {
                'date': '2025-08-05',
                'final_regime_label': 'calm_uptrend',
                'override_reason': None,
                'macro_credit_score': 0.6,
                'vol_uncertainty_score': float('nan'),  # Another NaN
            },
        ])

    def write_parquet(self, df, key: str) -> bool:
        return True


def test_timeseries_json_contains_no_nan(monkeypatch):
    """Verify that pandas NaN values are sanitized to null before JSON write."""
    fake_s3 = FakeS3("test-bucket")
    monkeypatch.setattr(publish_artifacts, "S3Client", lambda b: fake_s3)
    monkeypatch.setattr(
        publish_artifacts,
        "build_dashboard_data",
        lambda *a, **kw: {"metrics": {}},
    )

    publish_artifacts.run(
        bucket="test-bucket",
        run_date="2025-08-06",
        prices_df=pd.DataFrame(),
        context_df=pd.DataFrame({'spy_return_1d': [0.01]}),
        features_df=pd.DataFrame(),
        inference_output={"regime": {"label": "calm_uptrend", "probs": {}}},
        llm_risks={},
        decisions={"expert_metrics": {}},
        portfolio_state={"portfolio_value": 100000, "cash": 100000, "holdings": []},
        trades=[],
        weather={},
        validation={},
        expert_signals={"macro_credit": {}, "vol_uncertainty": {}, "fragility": {}, "entropy_shift": {}},
    )

    ts_key = "dashboard/data/timeseries.json"
    assert ts_key in fake_s3.json_writes, "timeseries.json was not written"

    ts_body = fake_s3.json_writes[ts_key]

    # Must be parseable by standard JSON (no NaN/Infinity tokens)
    parsed = json.loads(ts_body)
    assert isinstance(parsed, list)
    assert len(parsed) > 0

    # Walk every value and confirm no float NaN survived
    for row in parsed:
        for key, val in row.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"NaN found in row {row['date']}, field {key}"
                assert not math.isinf(val), f"Inf found in row {row['date']}, field {key}"
