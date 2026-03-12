#!/bin/bash
# Secrets Manager setup script for the investment system

set -e

REGION="${1:-us-east-1}"

echo "Setting up Secrets Manager secrets in $REGION"
echo ""
echo "This script will prompt you for API keys."
echo "Leave blank to skip a secret."
echo ""

# OpenAI API Key
read -p "Enter OpenAI API key (or press Enter to skip): " OPENAI_KEY
if [ -n "$OPENAI_KEY" ]; then
    echo "Creating OpenAI secret..."
    aws secretsmanager create-secret \
        --name "investment-system/openai-key" \
        --description "OpenAI API key for LLM risk checks and weather blurbs" \
        --secret-string "{\"api_key\": \"$OPENAI_KEY\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/openai-key" \
        --secret-string "{\"api_key\": \"$OPENAI_KEY\"}" \
        --region "$REGION"
    echo "OpenAI secret created/updated"
fi

# FRED API Key
read -p "Enter FRED API key (or press Enter to skip): " FRED_KEY
if [ -n "$FRED_KEY" ]; then
    echo "Creating FRED secret..."
    aws secretsmanager create-secret \
        --name "investment-system/fred-key" \
        --description "FRED API key for macroeconomic data" \
        --secret-string "{\"api_key\": \"$FRED_KEY\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/fred-key" \
        --secret-string "{\"api_key\": \"$FRED_KEY\"}" \
        --region "$REGION"
    echo "FRED secret created/updated"
fi

# Alpha Vantage API Key
read -p "Enter Alpha Vantage API key (or press Enter to skip): " AV_KEY
if [ -n "$AV_KEY" ]; then
    echo "Creating Alpha Vantage secret..."
    aws secretsmanager create-secret \
        --name "investment-system/alphavantage-key" \
        --description "Alpha Vantage API key for fallback price data" \
        --secret-string "{\"api_key\": \"$AV_KEY\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/alphavantage-key" \
        --secret-string "{\"api_key\": \"$AV_KEY\"}" \
        --region "$REGION"
    echo "Alpha Vantage secret created/updated"
fi

# Alpaca Paper Trading Key ID
read -p "Enter Alpaca Paper API Key ID (or press Enter to skip): " ALPACA_PAPER_KEY
if [ -n "$ALPACA_PAPER_KEY" ]; then
    echo "Creating Alpaca Paper Key ID secret..."
    aws secretsmanager create-secret \
        --name "investment-system/alpaca-paper-key-id" \
        --description "Alpaca paper trading API key ID" \
        --secret-string "{\"key\": \"$ALPACA_PAPER_KEY\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/alpaca-paper-key-id" \
        --secret-string "{\"key\": \"$ALPACA_PAPER_KEY\"}" \
        --region "$REGION"
    echo "Alpaca Paper Key ID secret created/updated"
fi

# Alpaca Paper Trading Secret Key
read -p "Enter Alpaca Paper API Secret Key (or press Enter to skip): " ALPACA_PAPER_SECRET
if [ -n "$ALPACA_PAPER_SECRET" ]; then
    echo "Creating Alpaca Paper Secret Key secret..."
    aws secretsmanager create-secret \
        --name "investment-system/alpaca-paper-secret-key" \
        --description "Alpaca paper trading API secret key" \
        --secret-string "{\"key\": \"$ALPACA_PAPER_SECRET\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/alpaca-paper-secret-key" \
        --secret-string "{\"key\": \"$ALPACA_PAPER_SECRET\"}" \
        --region "$REGION"
    echo "Alpaca Paper Secret Key secret created/updated"
fi

# Alpaca Live Trading Key ID (optional — set up only after paper validation)
read -p "Enter Alpaca Live API Key ID (or press Enter to skip): " ALPACA_LIVE_KEY
if [ -n "$ALPACA_LIVE_KEY" ]; then
    echo "Creating Alpaca Live Key ID secret..."
    aws secretsmanager create-secret \
        --name "investment-system/alpaca-live-key-id" \
        --description "Alpaca live trading API key ID" \
        --secret-string "{\"key\": \"$ALPACA_LIVE_KEY\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/alpaca-live-key-id" \
        --secret-string "{\"key\": \"$ALPACA_LIVE_KEY\"}" \
        --region "$REGION"
    echo "Alpaca Live Key ID secret created/updated"
fi

# Alpaca Live Trading Secret Key (optional)
read -p "Enter Alpaca Live API Secret Key (or press Enter to skip): " ALPACA_LIVE_SECRET
if [ -n "$ALPACA_LIVE_SECRET" ]; then
    echo "Creating Alpaca Live Secret Key secret..."
    aws secretsmanager create-secret \
        --name "investment-system/alpaca-live-secret-key" \
        --description "Alpaca live trading API secret key" \
        --secret-string "{\"key\": \"$ALPACA_LIVE_SECRET\"}" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "investment-system/alpaca-live-secret-key" \
        --secret-string "{\"key\": \"$ALPACA_LIVE_SECRET\"}" \
        --region "$REGION"
    echo "Alpaca Live Secret Key secret created/updated"
fi

echo ""
echo "Secrets setup complete!"
echo ""
echo "To get a free FRED API key, visit: https://fred.stlouisfed.org/docs/api/api_key.html"
echo "To get a free Alpha Vantage key, visit: https://www.alphavantage.co/support/#api-key"
echo "For Alpaca API keys, visit: https://app.alpaca.markets/paper/dashboard/overview"
