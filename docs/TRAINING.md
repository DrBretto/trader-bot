# Training Guide

This document covers how to train the ML models for the investment system.

## Overview

The system has two ML models:
1. **Regime Model** (GRU/Transformer) - Classifies market conditions into 5 regimes
2. **Health Model** (Autoencoder/VAE) - Scores asset health from 0-1

Training happens locally on MacBook and uploads models to S3.

## Prerequisites

```bash
# Install training dependencies (in project venv)
cd /Users/drbretto/Desktop/Projects/trader-bot
source .venv/bin/activate
pip install -r requirements-training.txt
```

Required packages:
- torch==2.1.2
- scikit-learn==1.3.2
- pandas, numpy, boto3, tqdm

## Quick Start

### Train Both Models

```bash
# Full training pipeline (downloads data from S3, trains, uploads)
python training/train.py --bucket investment-system-data --region us-east-1

# Local testing (skip S3 upload)
python training/train.py --bucket investment-system-data --skip-upload
```

### Train Individual Models

```bash
# Regime model only
python training/train_regime.py --data /path/to/context.parquet --output models/regime.pkl

# Health model only
python training/train_health.py --data /path/to/features.parquet --output models/health.pkl
```

## Training Options

### Main Orchestrator (`train.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--bucket` | investment-system-data | S3 bucket name |
| `--region` | us-east-1 | AWS region |
| `--max-days` | 365 | Days of historical data to use |
| `--regime-type` | gru | Model type: 'gru' or 'transformer' |
| `--health-type` | autoencoder | Model type: 'autoencoder' or 'vae' |
| `--epochs` | 100 | Max training epochs |
| `--skip-upload` | false | Skip S3 upload (local testing) |
| `--output-dir` | /tmp/models | Local output directory |

### Regime Model (`train_regime.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | - | Path to context parquet file |
| `--output` | models/regime_model.pkl | Output path |
| `--model-type` | gru | 'gru' or 'transformer' |
| `--epochs` | 100 | Max epochs |
| `--batch-size` | 32 | Batch size |

### Health Model (`train_health.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | - | Path to features parquet file |
| `--output` | models/health_model.pkl | Output path |
| `--model-type` | autoencoder | 'autoencoder' or 'vae' |
| `--epochs` | 100 | Max epochs |
| `--batch-size` | 64 | Batch size |

## Data Requirements

### Context Data (for Regime Model)

DataFrame with columns:
- `date` - Date
- `spy_return_1d`, `spy_return_21d` - SPY returns
- `spy_vol_21d` - SPY volatility
- `yield_slope` - 10Y - 2Y yield
- `credit_spread_proxy` - HYG/IEF ratio change
- `vixy_return_21d` - VIXY returns
- `risk_off_proxy` - TLT/SPY ratio change
- `gdelt_avg_tone` - GDELT sentiment

### Features Data (for Health Model)

DataFrame with columns:
- `date` - Date
- `symbol` - Asset symbol
- `return_1d`, `return_5d`, `return_21d`, `return_63d` - Returns
- `vol_21d`, `vol_63d` - Volatility
- `drawdown_21d`, `drawdown_63d` - Drawdowns
- `rel_strength_21d`, `rel_strength_63d` - Relative strength vs SPY

## Model Outputs

### Regime Model Output
```python
{
    'regime_label': 'risk_on_trend',  # One of 5 regimes
    'regime_probs': {
        'calm_uptrend': 0.1,
        'risk_on_trend': 0.7,
        'risk_off_trend': 0.05,
        'choppy': 0.1,
        'high_vol_panic': 0.05
    },
    'regime_embedding': [0.1, -0.2, ...]  # 8-dim vector
}
```

### Health Model Output
```python
{
    'health_score': 0.75,      # 0-1 score
    'vol_bucket': 'med',       # 'low', 'med', 'high'
    'behavior': 'momentum',    # 'momentum', 'mean_reversion', 'mixed'
    'latent': [0.1, -0.2, ...] # 16-dim vector
}
```

## S3 Artifacts

After training, models are uploaded to:
```
s3://investment-system-data/
├── models/
│   ├── latest.json           # Pointer to current versions
│   ├── regime_v20260130.pkl  # Regime model (versioned by date)
│   └── health_v20260130.pkl  # Health model (versioned by date)
```

### latest.json Format
```json
{
    "version": "20260130",
    "timestamp": "2026-01-30T10:00:00",
    "regime_model": "models/regime_v20260130.pkl",
    "health_model": "models/health_v20260130.pkl",
    "regime_metrics": {
        "test_accuracy": 0.85,
        "test_f1_macro": 0.78,
        "model_type": "gru"
    },
    "health_metrics": {
        "test_recon_mse": 0.02,
        "test_health_correlation": 0.82,
        "model_type": "autoencoder"
    }
}
```

## Lambda Integration

The Lambda function automatically:
1. Checks `models/latest.json` on startup
2. Loads trained models if available
3. Falls back to baseline models if:
   - No trained models exist
   - PyTorch not available in Lambda
   - Model loading fails

No Lambda redeployment needed after training - just upload new models.

## Running Tests

```bash
# Requires PyTorch
source .venv/bin/activate
pip install pytest torch

# Run ML model tests
python -m pytest tests/test_ml_models.py -v
```

## Monthly Training Schedule

Training is designed to run monthly via launchd (Phase 5). Manual runs:

```bash
# Run full training pipeline
python training/train.py --bucket investment-system-data --region us-east-1
```

Expected duration: 1-2 hours depending on data size and hardware.

## Troubleshooting

### "No module named 'torch'"
Install PyTorch: `pip install torch==2.1.2`

### "No historical data found"
Ensure daily pipeline has been running to generate data in S3.

### "CUDA out of memory"
Reduce batch size: `--batch-size 16`

### Model not loading in Lambda
Lambda uses baseline fallback. Check CloudWatch logs for errors.
To use trained models in Lambda, PyTorch must be in Lambda layer (increases size significantly).
