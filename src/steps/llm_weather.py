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

{portfolio_block}
Market Context:
- SPY: {spy_return:.2%} (21d: {spy_21d:.2%})
- VIX proxy: {vixy_return:.2%}
- Treasury 10y: {rate_10y:.2%}
- Credit spread: {credit_spread:.2%}
{expert_context}
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


def _build_portfolio_block(canon_metrics: Dict[str, Any]) -> str:
    """Portfolio posture for the prompt — CANON line only (PKT-TB-001).

    The internal intent-sizing sim book must never steer the published
    narrative. When canon metrics are unavailable (e.g. dashboard held),
    the block is omitted rather than substituted.
    """
    if not canon_metrics or canon_metrics.get('total_value') is None:
        return "Portfolio State: (canon metrics unavailable this run — do not invent portfolio figures)\n"
    return (
        "Portfolio State (canon line):\n"
        f"- Total value: ${canon_metrics['total_value']:,.0f}\n"
        f"- Cash: {canon_metrics.get('cash_pct', 0.0):.1%}\n"
        f"- Positions: {canon_metrics.get('num_positions', 0)}\n"
    )


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
    region: str = 'us-east-1',
    expert_signals: Dict[str, Any] = None,
    canon_metrics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate daily weather blurb.

    Uses Claude Haiku via Amazon Bedrock (preferred), falls back to OpenAI.

    Args:
        inference_output: Output from run_inference
        decisions: Output from decision_engine
        portfolio_state: Internal sim state (kept for signature compatibility;
            its book values are NOT used in the prompt — PKT-TB-001)
        context_df: Market context
        openai_key: OpenAI API key (fallback)
        region: AWS region for Bedrock
        expert_signals: Expert signal outputs for context (optional)
        canon_metrics: Canon-line posture for the prompt's portfolio block
            (total_value/cash_pct/num_positions); block omitted when None

    Returns:
        Weather blurb dict
    """
    import pandas as pd

    print("Generating weather blurb (Bedrock/Haiku)...")

    # Use fused regime if available
    expert_metrics = decisions.get('expert_metrics', {})
    regime = expert_metrics.get(
        'final_regime_label',
        inference_output.get('regime', {}).get('label', 'risk_on_trend')
    )
    run_date = inference_output.get('date', pd.Timestamp.now().strftime('%Y-%m-%d'))

    # Portfolio posture: canon line only (PKT-TB-001) — the internal sim
    # book's values are never fed to the prompt.
    portfolio_block = _build_portfolio_block(canon_metrics or {})

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

    # Build expert signal context string. Each numeric signal is annotated
    # with its scale + direction so the LLM does not silently hallucinate
    # narrative semantics (e.g. reading raw "Fragility: 0.99" as "low
    # fragility" when 0.99 is in fact max-pegged danger on a 0..1 scale).
    expert_context = ''
    if expert_signals is not None:
        macro = expert_signals.get('macro_credit', {})
        vol = expert_signals.get('vol_uncertainty', {})
        frag = expert_signals.get('fragility', {})
        ent = expert_signals.get('entropy_shift', {})
        throttle = expert_metrics.get('risk_throttle_factor', 0.0)

        def _frag_label(s: float) -> str:
            if s >= 0.85: return "EXTREMELY HIGH (cross-asset correlation pegged; risk-off cue)"
            if s >= 0.70: return "HIGH (elevated cross-asset coupling)"
            if s >= 0.50: return "ELEVATED"
            if s >= 0.30: return "MODERATE"
            return "LOW (decoupled assets)"

        def _vol_label(s: float, regime: str) -> str:
            base = f"vol-regime={regime}"
            if s >= 0.80: return f"HIGH UNCERTAINTY ({base})"
            if s >= 0.60: return f"ELEVATED ({base})"
            if s >= 0.40: return f"MODERATE ({base})"
            return f"LOW ({base})"

        def _entropy_label(s: float, flag: bool) -> str:
            if flag: return "REGIME-SHIFT FLAG ACTIVE"
            if s >= 0.80: return "HIGH (sustained shift pressure)"
            if s >= 0.60: return "ELEVATED"
            return "stable"

        def _macro_label(s: float) -> str:
            # Score is in [-1, +1]; positive = supportive, negative = restrictive
            if s >= 0.30: return "SUPPORTIVE"
            if s >= -0.30: return "NEUTRAL"
            return "RESTRICTIVE"

        frag_score = frag.get('fragility_score', 0.5)
        vol_score = vol.get('vol_uncertainty_score', 0.5)
        ent_score = ent.get('entropy_score', 0.5)
        macro_score = macro.get('macro_credit_score', 0.0)

        expert_context = (
            f"\nExpert Signals (scale 0..1 unless noted; HIGH = risk-off cue):\n"
            f"- Macro/Credit: {macro_score:+.2f} on [-1,+1] -> {_macro_label(macro_score)} "
            f"(yield slope 10y-3m: {macro.get('yield_slope_10y_3m', 0):.2f}%)\n"
            f"- Vol Uncertainty: {vol_score:.2f} -> {_vol_label(vol_score, vol.get('vol_regime_label', 'calm'))}\n"
            f"- Fragility: {frag_score:.2f} -> {_frag_label(frag_score)}\n"
            f"- Entropy Shift: {ent_score:.2f} -> {_entropy_label(ent_score, ent.get('entropy_shift_flag', False))}\n"
            f"- Risk Throttle factor: {throttle:.0%}\n"
            f"\nIMPORTANT: ground your narrative in the LABELED interpretations above.\n"
            f"Do NOT describe a HIGH-fragility regime as 'calm', 'stable', or 'low fragility'.\n"
            f"Do NOT describe a REGIME-SHIFT-FLAG-ACTIVE day as 'steady'.\n"
            f"If signals disagree with the regime label, name the tension explicitly.\n"
        )

    snapshot = {
        'date': run_date,
        'regime': regime,
        'portfolio_block': portfolio_block,
        'day_return': day_return,
        'spy_return': spy_return,
        'spy_21d': spy_21d,
        'vixy_return': vixy_return,
        'rate_10y': rate_10y,
        'credit_spread': credit_spread,
        'expert_context': expert_context,
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
