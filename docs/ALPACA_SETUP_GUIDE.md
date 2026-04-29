# Alpaca Setup Guide (Paper First, Live Later)

Last verified: March 12, 2026

This guide is for your workflow: fully automated trading, fractional-capable sizing, and paper validation before real funds.

## 1. What You Should Set Up First
- Alpaca paper account API keys
- Local secret storage for paper keys
- Broker connectivity smoke test against Alpaca paper endpoint
- End-to-end morning execution in paper mode only

Do not enable live mode yet.

## 2. Account Setup Checklist
1. Log into Alpaca dashboard and confirm your paper account is visible.
2. Create/regenerate paper API keys in the paper environment.
3. Store keys securely (never in git, never in chat logs).
4. Verify API connectivity before integrating with scheduler or automation.

## 3. Paper API Validation (Immediate)
Use your paper keys against the paper endpoint:

```bash
curl -sS https://paper-api.alpaca.markets/v2/account \
  -H "APCA-API-KEY-ID: <PAPER_KEY_ID>" \
  -H "APCA-API-SECRET-KEY: <PAPER_SECRET_KEY>" | jq '{status, account_number, cash, buying_power}'
```

Expected: JSON with active account fields and no auth error.

Optional tiny fractional/notional paper order test:

```bash
curl -sS -X POST https://paper-api.alpaca.markets/v2/orders \
  -H "APCA-API-KEY-ID: <PAPER_KEY_ID>" \
  -H "APCA-API-SECRET-KEY: <PAPER_SECRET_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"SPY","notional":"1","side":"buy","type":"market","time_in_force":"day"}' | jq
```

## 4. Repo Secret Strategy (Recommended)
Use AWS Secrets Manager entries (names can be adjusted, keep consistent with code):
- `investment-system/alpaca-paper-key-id`
- `investment-system/alpaca-paper-secret-key`
- `investment-system/alpaca-live-key-id`
- `investment-system/alpaca-live-secret-key`

Keep live keys unset until paper validation is stable.

## 5. Fractional Trading Notes (Why this matters for your small starting capital)
- Alpaca supports fractional trading for eligible US-listed stocks/ETFs.
- You can submit notional-dollar market orders (as low as $1).
- Fractional orders use `time_in_force=day`.

This is why the code handoff explicitly removes whole-share flooring for broker mode.

## 6. Fees and Funding Reality
- Alpaca states no minimum deposit for users.
- Alpaca states no fee for ACH deposits/withdrawals.
- Alpaca generally does not charge trade commissions for retail flow.
- Regulatory fees still apply (e.g., SEC/TAF on sells), deducted from proceeds.

## 7. Live Readiness Later (Do this only after paper passes)
1. Complete/confirm live account approval in dashboard.
2. Fund via ACH or wire (ACH is standard).
3. Generate live API keys.
4. Run smoke test in account-check mode first.
5. Enable live mode with very small order caps.
6. Start with tiny notional orders and monitor reconciled fills.

## 8. Safety Rules
- Never commit key material.
- Keep a global trading kill switch in config.
- Keep per-order notional caps until long enough paper burn-in is complete.
- Keep symbol allowlist narrow for first live sessions.

## Official Sources
- Fractional trading docs: https://docs.alpaca.markets/docs/fractional-trading
- Paper trading docs: https://docs.alpaca.markets/docs/paper-trading
- Trading API auth/base URL: https://docs.alpaca.markets/v1.1/docs/authentication-1
- Account plans/status model: https://docs.alpaca.markets/docs/account-plans
- Minimum deposit support: https://alpaca.markets/support/alpaca-minimum-deposit
- ACH funding support: https://alpaca.markets/support/domestic-user-fund-account
- ACH transfer fee support: https://alpaca.markets/support/alpaca-brokerage-fees
- Commission support: https://alpaca.markets/support/commission-clearing-fees
- Regulatory fees support: https://alpaca.markets/support/regulatory-fees
- Disclosures: https://alpaca.markets/disclosures

