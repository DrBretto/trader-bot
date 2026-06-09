# Investment System

Autonomous daily trading **simulation** with ML-powered market regime detection and asset health scoring.

## Current Status

- **Pure simulation** — There is no broker. Each night the pipeline decides trade intents; the next morning it *simulates* the fill at the real market OPEN, and that simulated fill becomes the new forward point on the champion line.
- **Continuous forward line** — The displayed champion line advances one trading day per trading day, automatically. Tonight's analysis settles this morning's provisional point to the real close.
- **Staged challenger** — The optimizer identified `macro_downgrade_threshold = -0.75` (from default -0.50) as a candidate improvement. This is staged only — not promoted to active parameters.
- **Active params** — `opt-bootstrap` bundle. No parameter changes have been made since the initial deployment.

## Features

- **Daily Pipeline** - Automated data ingestion, feature engineering, and simulated trade execution
- **ML Models** - GRU/Transformer for regime classification, Autoencoder/VAE for health scoring
- **LLM Integration** - Claude Haiku (Bedrock) risk assessment and market weather reports
- **React Dashboard** - Portfolio metrics, charts, evidence surfaces, and market analysis
- **Evolutionary Optimization** - Genetic algorithm for policy parameter tuning
- **Simulated Execution** - Trade intents decided overnight are filled in simulation at the next market open; no broker is involved

## Architecture

```
Daily (10 PM ET):
  EventBridge → Lambda → [Ingest → Features → Inference → Decisions] → S3

Monthly (1st, 2 AM):
  launchd → train.py → Models uploaded to S3

Dashboard:
  S3 Static Site → React App → Reads from S3 artifacts
```

## Quick Start

### Prerequisites

- Python 3.11+
- AWS CLI configured
- API keys: OpenAI, FRED (free tiers available)

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-training.txt  # For ML training

# Run tests
pytest tests/ -v
```

### Deploy to AWS

See [docs/DEPLOY.md](docs/DEPLOY.md) for full deployment instructions.

```bash
# Quick deploy
./infrastructure/s3_setup.sh investment-system-data us-east-1
./infrastructure/secrets_setup.sh us-east-1
./infrastructure/lambda_deploy.sh investment-system-daily-pipeline investment-system-data us-east-1
./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline investment-system-data us-east-1
```

### Run Dashboard Locally

```bash
cd frontend
npm install
npm run dev
# Visit http://localhost:5173
```

## Project Structure

```
├── src/                    # Lambda function code
│   ├── handler.py          # Main Lambda entry point
│   ├── steps/              # Pipeline steps
│   ├── models/             # Baseline + trained model loaders
│   └── utils/              # S3 client, logging
├── training/               # ML training code
│   ├── models/             # GRU, Transformer, Autoencoder, VAE
│   ├── utils/              # Data loaders, metrics
│   ├── train.py            # Training orchestrator
│   └── train_*.py          # Individual model training
├── evolution/              # Evolutionary optimization
│   ├── genome.py           # PolicyGenome class
│   ├── fitness.py          # Backtest-based fitness
│   ├── genetic.py          # Genetic algorithm
│   └── promotion.py        # Template promotion
├── frontend/               # React dashboard
│   ├── src/components/     # UI components
│   └── src/types/          # TypeScript interfaces
├── automation/             # Operational scripts
│   ├── run_training.sh     # Monthly training wrapper
│   ├── check_costs.py      # AWS cost monitoring
│   └── check_pipeline.py   # Pipeline health check
├── infrastructure/         # AWS deployment scripts
├── tests/                  # Test suites
└── docs/                   # Documentation
```

## Documentation

- [Deployment Guide](docs/DEPLOY.md) - AWS deployment procedures
- [Training Guide](docs/TRAINING.md) - ML model training
- [Frontend Guide](docs/FRONTEND.md) - Dashboard development
- [Operations Guide](docs/OPERATIONS.md) - Day-to-day operations
- [Architecture Plan](docs/PLAN.md) - Implementation plan and status

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_ml_models.py -v      # ML model tests (20)
pytest tests/test_evolution.py -v       # Evolution tests (28)
pytest tests/test_baseline_models.py -v # Baseline tests
```

Current test count changes over time; run `pytest tests/ -v` for the latest totals.

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `S3_BUCKET` | S3 bucket name |
| `AWS_REGION` | AWS region |
| `INVESTMENT_SLACK_WEBHOOK` | Slack webhook for alerts |
| `INVESTMENT_ALERT_EMAIL` | Email for SES alerts |

### AWS Secrets

Stored in Secrets Manager:
- `investment-system/openai-key`
- `investment-system/fred-key`
- `investment-system/alphavantage-key`

## Cost Estimate

| Service | Monthly |
|---------|---------|
| Lambda | ~$6 |
| S3 | ~$1 |
| Secrets Manager | ~$1 |
| CloudWatch | ~$1 |
| **Total** | **~$9** |

## License

Private project - not for distribution.
