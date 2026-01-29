# Investment System - Implementation Plan

## Goals

Build a fully autonomous daily investment decision system that:
- Ingests market data from free APIs (Stooq, FRED, GDELT)
- Generates buy/sell signals using AI models + qualitative risk assessment
- Executes paper trades with full transparency
- Displays performance in an impressive React dashboard
- Runs completely hands-off with monthly model retraining

## Constraints

- Daily resolution only (no intraday)
- Paper trading only (no real money)
- AWS cost < $20/month
- Local training (MacBook, monthly, automated)
- Free data sources only

## Architecture

```
Daily (10pm ET):
  EventBridge → Lambda → [Ingest → Features → Inference → Decisions → LLM] → S3 Artifacts

Monthly (1st, 2am):
  launchd → train.py → Models uploaded to S3

Anytime:
  User visits dashboard → S3 static site → Reads artifacts → Shows charts
```

## Milestones

### Phase 1: Core Pipeline ✅ COMPLETE
- [x] Project structure and config files
- [x] Data ingestion (Stooq, FRED, GDELT)
- [x] Data validation
- [x] Feature engineering
- [x] Baseline regime model (rule-based)
- [x] Baseline health model (rank-based)
- [x] Decision engine
- [x] Paper trader
- [x] LLM integration (risk + weather)
- [x] Lambda handler
- [x] AWS infrastructure scripts
- [x] Unit tests (44 passing)
- [ ] Deploy to AWS and test end-to-end

### Phase 2: ML Models (Not Started)
- [ ] Regime model (Transformer/GRU)
- [ ] Health model (Autoencoder)
- [ ] Training pipeline
- [ ] Model versioning

### Phase 3: Frontend Dashboard (Not Started)
- [ ] React + Vite setup
- [ ] Hero metrics
- [ ] Equity curve chart
- [ ] Drawdown chart
- [ ] Monthly returns heatmap
- [ ] Weather report
- [ ] Portfolio/candidates tables

### Phase 4: Evolutionary Search (Not Started)
- [ ] PolicyGenome class
- [ ] Fitness evaluation
- [ ] Genetic algorithm
- [ ] Template promotion

### Phase 5: Automation (Not Started)
- [ ] launchd for monthly training
- [ ] Cost monitoring
- [ ] Error alerting
- [ ] Documentation

## Non-Goals

- Real money trading (paper only)
- Intraday trading (daily resolution only)
- Complex derivative strategies
- Multi-account management

---

## Deployment Instructions

### Prerequisites
- AWS CLI configured with credentials
- Python 3.11+
- API keys: OpenAI, FRED, Alpha Vantage (all have free tiers)

### Deploy Phase 1

```bash
# 1. Set up S3 bucket
./infrastructure/s3_setup.sh investment-system-data us-east-1

# 2. Configure API keys in Secrets Manager
./infrastructure/secrets_setup.sh us-east-1

# 3. Deploy Lambda function
./infrastructure/lambda_deploy.sh investment-system-daily-pipeline investment-system-data us-east-1

# 4. Set up EventBridge schedule
./infrastructure/eventbridge_setup.sh investment-system-daily-pipeline us-east-1

# 5. Test manually
aws lambda invoke \
  --function-name investment-system-daily-pipeline \
  --payload '{"bucket": "investment-system-data", "source": "manual"}' \
  /tmp/response.json && cat /tmp/response.json
```

## Deviations

**Directory Rename**: Changed `lambda/` to `src/` because `lambda` is a Python reserved keyword, causing import failures. Documented in `docs/DEVIATIONS.md`.
