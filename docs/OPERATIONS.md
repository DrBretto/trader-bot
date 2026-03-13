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
6. **Trading** - Executes simulated or broker-routed trades, updates portfolio state
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

## Broker Execution Modes

The system supports three execution modes, controlled by `BROKER_MODE` env var or `broker_mode` in config:

| Mode | Description | Default |
|------|-------------|---------|
| `simulated` | Paper trading via `paper_trader` (no broker) | Yes |
| `alpaca_paper` | Alpaca paper trading (fractional/notional) | No |
| `alpaca_live` | Alpaca live trading (fractional/notional) | No |

### Safety Controls

- **Kill switch**: `BROKER_TRADING_ENABLED` must be explicitly set to `true` for non-simulated modes. Default is `false`.
- **Max order cap**: Per-order notional cap (default $5,000). Set via `broker.max_order_notional` in config.
- **Symbol allowlist**: Optional. Set via `broker.symbol_allowlist` in config.
- **Idempotent orders**: Deterministic `client_order_id` prevents duplicate submissions.

### Enabling Paper Mode

```bash
# Set env vars (for Lambda, use environment configuration)
export BROKER_MODE=alpaca_paper
export BROKER_TRADING_ENABLED=true

# Ensure Alpaca paper secrets are in Secrets Manager:
#   investment-system/alpaca-paper-key-id
#   investment-system/alpaca-paper-secret-key

# Smoke test first
python scripts/alpaca_paper_smoke_test.py --account-check
python scripts/alpaca_paper_smoke_test.py --place-order --close-after
```

### Rollback to Simulated Mode

```bash
# Option 1: Remove env var (defaults to simulated)
unset BROKER_MODE

# Option 2: Explicitly set
export BROKER_MODE=simulated
```

### Live Mode (After Paper Validation)

1. Complete paper trading validation for multiple sessions
2. Set up live API keys in Secrets Manager
3. Switch mode and enable:
   ```bash
   export BROKER_MODE=alpaca_live
   export BROKER_TRADING_ENABLED=true
   ```
4. Start with very small `max_order_notional` and narrow `symbol_allowlist`

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

## Cutover Continuity Bridge

When switching from simulated to broker execution, the portfolio value may jump (e.g. fresh Alpaca paper account at $100k vs $103k simulated). This causes a false loss in dashboard metrics.

The bridge script patches `external_cashflow` on the cutover-day `portfolio_state.json` to neutralize the discontinuity in return calculations.

### When to Use

- After switching `BROKER_MODE` from `simulated` to `alpaca_paper` (or `alpaca_live`)
- When the dashboard shows a sudden drop/jump on the cutover day

### Commands

```bash
# Dry-run (prints plan, writes artifact, no S3 mutation)
python scripts/bridge_cutover_continuity.py --cutover-date 2026-03-12

# Apply (patches portfolio_state.json in S3)
python scripts/bridge_cutover_continuity.py --cutover-date 2026-03-12 --apply

# With specific AWS profile
python scripts/bridge_cutover_continuity.py --cutover-date 2026-03-12 --apply --profile your-aws-profile
```

`--profile` is optional. Omit it when using IAM role credentials (Lambda/EC2) or pre-set AWS env credentials.

### Cautions

- Only run once per cutover. The script is idempotent (safe to re-run), but review the output.
- This does NOT change broker cash or positions — it only adjusts the accounting math.
- Dashboard equity/value are continuity-adjusted after bridge so historical performance remains comparable; raw broker value is still emitted as `metrics.broker_total_value`.
- After applying, re-run the morning execution or dashboard rebuild to see updated metrics.

---

## Bootstrapping Alpaca to Simulated Portfolio

After cutover, the broker account has no positions. This script places notional buy orders on Alpaca to recreate the simulated portfolio's allocation.

### Commands

```bash
# Dry-run (default)
python scripts/bootstrap_alpaca_from_sim_state.py --source-date 2026-03-11

# Apply (submits orders)
python scripts/bootstrap_alpaca_from_sim_state.py --source-date 2026-03-11 --apply

# With custom caps
python scripts/bootstrap_alpaca_from_sim_state.py --source-date 2026-03-11 --apply \
  --max-per-order 3000 --max-total 80000

# With symbol filter
python scripts/bootstrap_alpaca_from_sim_state.py --source-date 2026-03-11 --apply \
  --symbol-allowlist SPY GLD XLE
```

`--profile` is optional for this script as well; use it only when you intentionally need a specific local profile.

### Warnings

- **Market drift/slippage**: Prices may have moved since the source date. Weights are approximate.
- **Partial fills**: Some orders may partially fill or be rejected. Check the result artifact.
- **Idempotent**: Re-running skips symbols where existing position already meets target weight.
- After bootstrap, run the morning execution to reconcile and rebuild dashboard artifacts.

---

## Backup & Recovery

### Portfolio State

Portfolio state is stored per run date under `daily/<date>/`:
- `s3://bucket/daily/latest.json` (pointer to the most recent run)
- `s3://bucket/daily/<date>/portfolio_state.json`
- `s3://bucket/daily/<date>/trades.jsonl`
- `s3://bucket/daily/<date>/morning_execution.json` (morning phase report, when applicable)

To restore from backup:
```bash
LATEST_DATE=$(aws s3 cp s3://investment-system-data/daily/latest.json - | jq -r .date)
aws s3 cp "s3://investment-system-data/daily/${LATEST_DATE}/portfolio_state.json" /tmp/portfolio_state.json
# Edit if needed, then upload to a date-specific key:
aws s3 cp /tmp/portfolio_state.json "s3://investment-system-data/daily/${LATEST_DATE}/portfolio_state.json"
```

### Config Files

Config stored in:
- `s3://bucket/config/universe.csv`
- `s3://bucket/config/decision_params.active.json` (canonical live bundle)
- `s3://bucket/config/decision_params.json` (legacy/reference)
- `s3://bucket/config/regime_compatibility.json` (legacy/reference)

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
LATEST_DATE=$(aws s3 cp s3://investment-system-data/daily/latest.json - | jq -r .date)
aws s3 cp "s3://investment-system-data/daily/${LATEST_DATE}/portfolio_state.json" - | jq .holdings

# View recent trades
aws s3 cp "s3://investment-system-data/daily/${LATEST_DATE}/trades.jsonl" - | tail -5
```
