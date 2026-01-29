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

echo ""
echo "Secrets setup complete!"
echo ""
echo "To get a free FRED API key, visit: https://fred.stlouisfed.org/docs/api/api_key.html"
echo "To get a free Alpha Vantage key, visit: https://www.alphavantage.co/support/#api-key"
