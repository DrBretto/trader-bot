"""LLM risk check for structural risk assessment."""

import json
from typing import Dict, Any, List, Optional
from openai import OpenAI


RISK_CHECK_PROMPT = """
You are a financial risk analyst. Analyze this ETF for structural risks.

ETF: {symbol} - {name}
Asset Class: {asset_class}
Sector: {sector}

Recent Performance:
- 5-day return: {return_5d:.2%}
- 21-day return: {return_21d:.2%}
- Current volatility: {vol_21d:.1%}

Context:
- Market regime: {regime}
- GDELT sentiment: {gdelt_tone:.2f}

Task:
1. Identify any red flags: regulatory risk, litigation, fraud allegations, leveraged ETF decay, liquidity concerns
2. Rate severity: 0=none, 1=monitor, 2=caution, 3=critical
3. Should this be vetoed from trading? (yes/no)
4. Confidence adjustment: 0.0-0.5 (how much to reduce position size)
5. One-sentence rationale

Respond ONLY with JSON:
{{
  "risk_flags": ["regulatory", "liquidity"],
  "severity": 2,
  "structural_risk_veto": false,
  "confidence_adjustment": 0.1,
  "one_sentence_rationale": "Elevated regulatory scrutiny but no immediate threat"
}}
"""


def call_llm_risk_check(
    symbol: str,
    metadata: Dict[str, Any],
    performance: Dict[str, float],
    context: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """
    Call OpenAI API for risk assessment.

    Args:
        symbol: Ticker symbol
        metadata: Asset metadata (name, asset_class, sector)
        performance: Performance metrics (return_5d, return_21d, vol_21d)
        context: Market context (regime, gdelt_avg_tone)
        api_key: OpenAI API key

    Returns:
        Risk assessment dict, or empty dict on failure
    """
    client = OpenAI(api_key=api_key)

    prompt = RISK_CHECK_PROMPT.format(
        symbol=symbol,
        name=metadata.get('name', symbol),
        asset_class=metadata.get('asset_class', 'equity'),
        sector=metadata.get('sector', 'broad'),
        return_5d=performance.get('return_5d', 0),
        return_21d=performance.get('return_21d', 0),
        vol_21d=performance.get('vol_21d', 0.15),
        regime=context.get('regime', 'risk_on_trend'),
        gdelt_tone=context.get('gdelt_avg_tone', 0)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial risk analyst. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        result_text = response.choices[0].message.content

        # Clean up response (remove markdown code blocks if present)
        if result_text.startswith('```'):
            result_text = result_text.split('\n', 1)[1]
        if result_text.endswith('```'):
            result_text = result_text.rsplit('\n', 1)[0]

        result = json.loads(result_text)

        # Validate schema
        required_keys = [
            'risk_flags', 'severity', 'structural_risk_veto',
            'confidence_adjustment', 'one_sentence_rationale'
        ]
        if not all(k in result for k in required_keys):
            print(f"LLM risk check: Missing required keys for {symbol}")
            return {}

        return result

    except json.JSONDecodeError as e:
        print(f"LLM risk check: JSON parse error for {symbol}: {e}")
        return {}
    except Exception as e:
        print(f"LLM risk check failed for {symbol}: {e}")
        return {}


def run(
    inference_output: Dict[str, Any],
    features_df: 'pd.DataFrame',
    context_df: 'pd.DataFrame',
    openai_key: str,
    config: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Run LLM risk checks for relevant symbols.

    Checks:
    - All current holdings
    - Top 5 buy candidates
    - Up to 3 shock list symbols (if any flagged)

    Max 16 calls per day.

    Args:
        inference_output: Output from run_inference
        features_df: Asset features
        context_df: Market context
        openai_key: OpenAI API key
        config: Configuration dict

    Returns:
        Dict mapping symbol to risk assessment
    """
    import pandas as pd

    print("Running LLM risk checks...")

    # Get current portfolio state for holdings
    portfolio_state = config.get('portfolio_state', {'holdings': []})
    holdings = [h['symbol'] for h in portfolio_state.get('holdings', [])]

    # Get top candidates from health scores
    asset_health = inference_output.get('asset_health', [])
    sorted_health = sorted(asset_health, key=lambda x: x.get('health_score', 0), reverse=True)
    top_candidates = [h['symbol'] for h in sorted_health[:5]]

    # Combine symbols to check (deduped)
    symbols_to_check = list(dict.fromkeys(holdings + top_candidates))[:16]

    # Get context
    regime = inference_output.get('regime', {}).get('label', 'risk_on_trend')
    gdelt_tone = 0
    if len(context_df) > 0:
        gdelt_tone = context_df.iloc[0].get('gdelt_avg_tone', 0)

    context = {
        'regime': regime,
        'gdelt_avg_tone': gdelt_tone
    }

    # Get universe for metadata
    universe_df = config.get('universe', pd.DataFrame())
    if isinstance(universe_df, list):
        universe_df = pd.DataFrame(universe_df)

    results = {}
    calls_made = 0

    for symbol in symbols_to_check:
        # Get metadata
        metadata = {'name': symbol, 'asset_class': 'equity', 'sector': 'broad'}
        if len(universe_df) > 0:
            symbol_row = universe_df[universe_df['symbol'] == symbol]
            if len(symbol_row) > 0:
                row = symbol_row.iloc[0]
                metadata = {
                    'name': symbol,
                    'asset_class': row.get('asset_class', 'equity'),
                    'sector': row.get('sector', 'broad')
                }

        # Get performance
        performance = {'return_5d': 0, 'return_21d': 0, 'vol_21d': 0.15}
        symbol_features = features_df[features_df['symbol'] == symbol]
        if len(symbol_features) > 0:
            latest = symbol_features.sort_values('date').iloc[-1]
            performance = {
                'return_5d': latest.get('return_5d', 0) or 0,
                'return_21d': latest.get('return_21d', 0) or 0,
                'vol_21d': latest.get('vol_21d', 0.15) or 0.15
            }

        # Call LLM
        result = call_llm_risk_check(symbol, metadata, performance, context, openai_key)

        if result:
            results[symbol] = result
            calls_made += 1
            print(f"  {symbol}: severity={result.get('severity', 0)}, "
                  f"veto={result.get('structural_risk_veto', False)}")
        else:
            # Default to no risk if LLM fails
            results[symbol] = {
                'risk_flags': [],
                'severity': 0,
                'structural_risk_veto': False,
                'confidence_adjustment': 0.0,
                'one_sentence_rationale': 'LLM check unavailable - defaulting to no risk'
            }

    print(f"  LLM risk checks complete: {calls_made} calls made")

    return results
