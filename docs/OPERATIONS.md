# Operations Guide

Day-to-day operations for the investment system.

## Dashboard Data

The dashboard’s equity curve, drawdown series, and monthly returns are **built from daily artifacts** when the optional `portfolio/equity_history.json` (and related) files are missing. Each pipeline run writes `dashboard/dashboard.json` and, when building from daily, scans `daily/<date>/portfolio_state.json` to produce the chart series. No separate aggregation job is required.

---

## Daily Pipeline

The system runs automatically at 10 PM ET on weeknights via AWS EventBridge.

### What It Does

1. **Ingest** - Fetches prices (Stooq), economic data (FRED), sentiment (GDELT)
2. **Features** - Computes returns, volatility, drawdowns, relative strength
3. **Inference** - Runs regime classification and health scoring models
4. **LLM Risk** - GPT-4 reviews top candidates for qualitative risks
5. **Decisions** - Generates buy/sell signals based on scores and regime
6. **Trading** - Executes paper trades, updates portfolio state
7. **Weather** - Generates market weather report
8. **Publish** - Uploads all artifacts to S3

### Monitoring

Check pipeline status:
```bash
python automation/check_pipeline.py
```

View CloudWatch logs:
```bash
aws logs tail /aws/lambda/investment-system-daily-pipeline --follow
```

Check latest run:
```bash
aws s3 cp s3://investment-system-data/daily/latest.json - | jq
```

### Manual Trigger

```bash
aws lambda invoke \
  --function-name investment-system-daily-pipeline \
  --payload '{"bucket": "investment-system-data", "source": "manual"}' \
  --invocation-type Event \
  /tmp/response.json
```

---

## Monthly Training

Runs automatically on the 1st of each month at 2 AM via macOS launchd.

### What It Does

1. **ML Training** - Trains regime (GRU) and health (Autoencoder) models
2. **Evolution** - Runs genetic algorithm to optimize trading parameters
3. **Promotion** - Validates and promotes best evolved policy
4. **Upload** - Uploads new models and config to S3

### Install launchd Job

**Edit the plist** so the path in `ProgramArguments` points to your repo’s `automation/run_training.sh` (e.g. `~/path/to/trader-bot/automation/run_training.sh`). Then:

```bash
./automation/install_launchd.sh
```

### Manual Training

Training and evolution both load historical data by **building from daily artifacts** in S3 (`daily/<date>/*`). No separate `historical/*` upload is required.

```bash
# Full training (requires PyTorch)
source .venv/bin/activate
python training/train.py --bucket investment-system-data --region us-east-1

# Evolution only (uses same build-from-daily data as training)
python evolution/evolve.py --bucket investment-system-data --region us-east-1 --generations 25
```

### Check Training Logs

```bash
tail -f /tmp/investment-training.log
```

---

## Cost Management

### Check Current Costs

```bash
python automation/check_costs.py --budget 20
```

### Expected Costs

| Service | Monthly Cost |
|---------|-------------|
| Lambda | ~$6 |
| S3 | ~$1 |
| Secrets Manager | ~$1 |
| CloudWatch | ~$1 |
| **Total** | **~$9** |

### Cost Alerts

Alerts trigger when costs exceed 80% of budget ($16 by default).

Configure alert destinations:
```bash
export INVESTMENT_SLACK_WEBHOOK="https://hooks.slack.com/..."
export INVESTMENT_ALERT_EMAIL="your@email.com"
```

---

## Troubleshooting

### Pipeline Didn't Run

1. Check EventBridge rule is enabled:
   ```bash
   aws events describe-rule --name investment-system-daily-trigger
   ```

2. Check Lambda function exists:
   ```bash
   aws lambda get-function --function-name investment-system-daily-pipeline
   ```

3. Check CloudWatch for errors:
   ```bash
   aws logs filter-log-events \
     --log-group-name /aws/lambda/investment-system-daily-pipeline \
     --filter-pattern ERROR
   ```

### Data Quality Issues

The pipeline runs in "degraded mode" if data quality is poor:
- Missing prices for critical symbols
- Stale FRED data
- GDELT unavailable

Check validation results:
```bash
aws s3 cp s3://investment-system-data/daily/$(date +%Y-%m-%d)/run_report.json - | jq .validation
```

### Model Not Loading

Lambda falls back to baseline models if:
- PyTorch not in Lambda layer
- Model file corrupted
- Models not uploaded

Check model status:
```bash
aws s3 cp s3://investment-system-data/models/latest.json - | jq
```

### Training Failures

Check training logs:
```bash
cat /tmp/investment-training.log
cat /tmp/investment-training.error.log
```

Common issues:
- Insufficient historical data (need 30+ days)
- PyTorch not installed in venv
- S3 permissions

---

## Backup & Recovery

### Portfolio State

Portfolio state is stored in:
- `s3://bucket/portfolio/current_state.json`
- `s3://bucket/portfolio/trades_history.jsonl`

To restore from backup:
```bash
aws s3 cp s3://investment-system-data/portfolio/current_state.json /tmp/
# Edit if needed
aws s3 cp /tmp/current_state.json s3://investment-system-data/portfolio/current_state.json
```

### Config Files

Config stored in:
- `s3://bucket/config/universe.csv`
- `s3://bucket/config/decision_params.json`
- `s3://bucket/config/regime_compatibility.json`

---

## Useful Commands

```bash
# Check portfolio value
aws s3 cp s3://investment-system-data/daily/latest.json - | jq .portfolio_value

# List recent daily runs
aws s3 ls s3://investment-system-data/daily/ | tail -10

# Check regime
aws s3 cp s3://investment-system-data/daily/latest.json - | jq .regime

# View weather report
aws s3 cp s3://investment-system-data/daily/$(date +%Y-%m-%d)/weather_blurb.json - | jq

# Check current holdings
aws s3 cp s3://investment-system-data/portfolio/current_state.json - | jq .holdings

# View recent trades
aws s3 cp s3://investment-system-data/portfolio/trades_history.jsonl - | tail -5
```
