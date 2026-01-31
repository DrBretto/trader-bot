"""LLM weather blurb generation for daily market summary using Claude Haiku via Bedrock."""

import json
import os
from typing import Dict, Any, List

import boto3

# Try OpenAI as fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


WEATHER_BLURB_PROMPT = """You are a portfolio manager writing a daily market brief.

Date: {date}
Regime: {regime}

Portfolio State:
- Total value: ${portfolio_value:,.0f}
- Cash: ${cash:,.0f} ({cash_pct:.1%})
- Positions: {num_positions}
- Day return: {day_return:+.2%}

Market Context:
- SPY: {spy_return:.2%} (21d: {spy_21d:.2%})
- VIX proxy: {vixy_return:.2%}
- Treasury 10y: {rate_10y:.2%}
- Credit spread: {credit_spread:.2%}

Today's Actions:
- Buys: {buys_summary}
- Sells: {sells_summary}

Task:
Write a brief, professional daily update (80-140 words) explaining:
1. Current market conditions (regime)
2. Why actions were taken
3. Risk outlook

Then provide 3 bullet takeaways (<15 words each).

Respond with JSON only (no markdown):
{{
  "headline": "12 word headline",
  "blurb": "80-140 word narrative",
  "takeaways": ["bullet 1", "bullet 2", "bullet 3"]
}}"""


def generate_fallback_weather(
    snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a simple fallback weather report without LLM."""
    regime = snapshot.get('regime', 'risk_on_trend')
    day_return = snapshot.get('day_return', 0)

    # Simple regime descriptions
    regime_headlines = {
        'calm_uptrend': 'Markets Steady in Low-Vol Uptrend',
        'risk_on_trend': 'Risk-On Momentum Continues',
        'risk_off_trend': 'Caution Mode as Markets Pull Back',
        'choppy': 'Sideways Action with Elevated Volatility',
        'high_vol_panic': 'Defensive Posture Amid Market Stress'
    }

    headline = regime_headlines.get(regime, 'Daily Portfolio Update')

    # Simple blurb
    if day_return > 0.01:
        outlook = "Portfolio gained ground today."
    elif day_return < -0.01:
        outlook = "Portfolio faced headwinds today."
    else:
        outlook = "Portfolio held steady today."

    blurb = f"{outlook} Operating in {regime.replace('_', ' ')} regime. System continues to monitor conditions and adjust positions as needed."

    takeaways = [
        f"Regime: {regime.replace('_', ' ').title()}",
        f"Day return: {day_return:+.2%}",
        "Risk managed per protocol"
    ]

    return {
        'headline': headline,
        'blurb': blurb,
        'takeaways': takeaways
    }


def call_llm_weather_bedrock(
    snapshot: Dict[str, Any],
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """Generate daily weather report using Claude Haiku via Amazon Bedrock."""
    client = boto3.client('bedrock-runtime', region_name=region)

    prompt = WEATHER_BLURB_PROMPT.format(**snapshot)

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "system": "You are a portfolio manager. Always respond with valid JSON only, no markdown.",
            "messages": [{"role": "user", "content": prompt}]
        })

        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            accept="application/json",
            body=body
        )

        response_body = json.loads(response['body'].read())
        result_text = response_body['content'][0]['text']

        # Clean up response (remove markdown code blocks if present)
        if result_text.startswith('```'):
            lines = result_text.split('\n')
            result_text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        result = json.loads(result_text)

        if not all(k in result for k in ['headline', 'blurb', 'takeaways']):
            print("LLM weather (Bedrock): Missing required keys, using fallback")
            return generate_fallback_weather(snapshot)

        return result

    except json.JSONDecodeError as e:
        print(f"LLM weather (Bedrock): JSON parse error: {e}")
        return generate_fallback_weather(snapshot)
    except Exception as e:
        print(f"LLM weather (Bedrock) failed: {e}")
        return generate_fallback_weather(snapshot)


def call_llm_weather_openai(
    snapshot: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """Generate daily weather report using OpenAI (fallback)."""
    client = OpenAI(api_key=api_key)

    prompt = WEATHER_BLURB_PROMPT.format(**snapshot)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a portfolio manager. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        result_text = response.choices[0].message.content

        if result_text.startswith('```'):
            result_text = result_text.split('\n', 1)[1]
        if result_text.endswith('```'):
            result_text = result_text.rsplit('\n', 1)[0]

        result = json.loads(result_text)

        if not all(k in result for k in ['headline', 'blurb', 'takeaways']):
            return generate_fallback_weather(snapshot)

        return result

    except Exception as e:
        print(f"LLM weather (OpenAI) failed: {e}")
        return generate_fallback_weather(snapshot)


def call_llm_weather_blurb(
    snapshot: Dict[str, Any],
    api_key: str = '',
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """
    Generate daily weather report using LLM.

    Uses Claude Haiku via Amazon Bedrock (preferred), falls back to OpenAI.

    Args:
        snapshot: Dict with portfolio and market data
        api_key: OpenAI API key (fallback only)
        region: AWS region for Bedrock

    Returns:
        {'headline': str, 'blurb': str, 'takeaways': list[str]}
    """
    # Try Bedrock first (uses AWS credentials, no API key needed)
    result = call_llm_weather_bedrock(snapshot, region)
    if result and 'headline' in result:
        return result

    # Fall back to OpenAI if Bedrock fails and we have an API key
    if OPENAI_AVAILABLE and api_key:
        return call_llm_weather_openai(snapshot, api_key)

    print("No LLM available for weather blurb")
    return generate_fallback_weather(snapshot)


def run(
    inference_output: Dict[str, Any],
    decisions: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    context_df: 'pd.DataFrame',
    openai_key: str = '',
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """
    Generate daily weather blurb.

    Uses Claude Haiku via Amazon Bedrock (preferred), falls back to OpenAI.

    Args:
        inference_output: Output from run_inference
        decisions: Output from decision_engine
        portfolio_state: Current portfolio state
        context_df: Market context
        openai_key: OpenAI API key (fallback)
        region: AWS region for Bedrock

    Returns:
        Weather blurb dict
    """
    import pandas as pd

    print("Generating weather blurb (Bedrock/Haiku)...")

    # Build snapshot for prompt
    regime = inference_output.get('regime', {}).get('label', 'risk_on_trend')
    run_date = inference_output.get('date', pd.Timestamp.now().strftime('%Y-%m-%d'))

    # Portfolio metrics
    portfolio_value = portfolio_state.get('portfolio_value', 100000)
    cash = portfolio_state.get('cash', 100000)
    cash_pct = cash / portfolio_value if portfolio_value > 0 else 1.0
    num_positions = len(portfolio_state.get('holdings', []))

    # Calculate day return (would need previous day's value in production)
    day_return = 0.0  # Placeholder

    # Market context
    spy_return = 0
    spy_21d = 0
    vixy_return = 0
    rate_10y = 0
    credit_spread = 0

    if len(context_df) > 0:
        ctx = context_df.iloc[0]
        spy_return = ctx.get('spy_return_1d', 0) or 0
        spy_21d = ctx.get('spy_return_21d', 0) or 0
        vixy_return = ctx.get('vixy_return_21d', 0) or 0
        rate_10y = ctx.get('rate_10y', 0) or 0
        credit_spread = ctx.get('credit_spread_proxy', 0) or 0

    # Actions summary
    actions = decisions.get('actions', [])
    buys = [a for a in actions if a['action'] == 'BUY']
    sells = [a for a in actions if a['action'] == 'SELL']

    if buys:
        buys_summary = ', '.join(f"{a['symbol']} ({a.get('shares', 0)} shares)" for a in buys[:3])
        if len(buys) > 3:
            buys_summary += f" (+{len(buys) - 3} more)"
    else:
        buys_summary = "None"

    if sells:
        sells_summary = ', '.join(f"{a['symbol']} ({a.get('reason', 'N/A')})" for a in sells[:3])
        if len(sells) > 3:
            sells_summary += f" (+{len(sells) - 3} more)"
    else:
        sells_summary = "None"

    snapshot = {
        'date': run_date,
        'regime': regime,
        'portfolio_value': portfolio_value,
        'cash': cash,
        'cash_pct': cash_pct,
        'num_positions': num_positions,
        'day_return': day_return,
        'spy_return': spy_return,
        'spy_21d': spy_21d,
        'vixy_return': vixy_return,
        'rate_10y': rate_10y,
        'credit_spread': credit_spread,
        'buys_summary': buys_summary,
        'sells_summary': sells_summary
    }

    # Generate weather blurb (tries Bedrock first, falls back to OpenAI)
    weather = call_llm_weather_blurb(snapshot, openai_key, region)

    print(f"  Headline: {weather.get('headline', 'N/A')}")

    return {
        'date': run_date,
        **weather
    }
