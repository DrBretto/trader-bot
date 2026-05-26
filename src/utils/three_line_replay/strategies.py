"""Strategy modifications for the three-line replay extension.

Each strategy is a callable that gets a ``StrategyContext`` and may:
- Mutate the per-variant config before decision_engine.run.
- Inject extra BUY intents (e.g., top-up rebalances) post-decision.

Constraints (no look-ahead): strategies read only state observable at
the decision moment (current portfolio, today's regime/PSM/fragility,
panic-streak counter, per-position entry_psm captured at open). They
must not read future prices, future regime labels, or final outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class StrategyContext:
    inputs_date: str
    portfolio: Any
    variant_config: Dict[str, Any]
    expert_signals: Dict[str, Any]
    expert_metrics: Dict[str, Any]
    decisions: Dict[str, Any]
    panic_streak: int
    last_regime: Optional[str]
    features_df: Any
    inference: Dict[str, Any]
    llm_risks: Dict[str, Any]


@dataclass
class Strategy:
    name: str
    description: str
    pre_decision: Optional[Callable[[StrategyContext], None]] = None
    post_decision: Optional[Callable[[StrategyContext, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None
    params: Dict[str, Any] = field(default_factory=dict)


def topup_on_psm_rise(
    trigger_ratio: float = 1.2,
    fraction: float = 1.0,
    min_dollars: float = 250.0,
    allowed_regimes: Tuple[str, ...] = ('risk_on_trend', 'calm_uptrend'),
) -> Strategy:
    """Top up existing positions when current PSM rises >= trigger_ratio
    above entry-time PSM AND current regime is in allowed_regimes."""

    def post(ctx: StrategyContext, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        em = ctx.expert_metrics or {}
        current_psm = em.get('position_size_modifier')
        if current_psm is None:
            return intents
        regime = em.get('final_regime_label') or ctx.decisions.get('regime')
        if regime not in allowed_regimes:
            return intents
        params = ctx.variant_config.get('decision_params', {})
        max_position_weight = float(params.get('max_position_weight', 0.20))
        portfolio_value = ctx.decisions.get('_portfolio_value_passed', 0.0)
        if portfolio_value <= 0:
            return intents

        symbols_with_intents = {a['symbol'] for a in intents}
        new_intents = list(intents)
        for p in ctx.portfolio.positions:
            if p.symbol in symbols_with_intents:
                continue
            entry_psm = getattr(p, 'entry_psm', None)
            if entry_psm is None or entry_psm <= 0:
                continue
            ratio = current_psm / entry_psm
            if ratio < trigger_ratio:
                continue
            sym_features = ctx.features_df[ctx.features_df['symbol'] == p.symbol]
            if len(sym_features) == 0:
                continue
            current_close = float(sym_features.sort_values('date')['close'].iloc[-1])
            current_holding_value = p.shares * current_close
            current_target = portfolio_value * max_position_weight * current_psm
            gap = current_target - current_holding_value
            top_up_dollars = gap * fraction
            if top_up_dollars < min_dollars:
                continue
            new_intents.append({
                'action': 'BUY',
                'symbol': p.symbol,
                'shares': int(top_up_dollars / current_close) if current_close > 0 else 0,
                'price': current_close,
                'dollars': top_up_dollars,
                'reason': f'TOPUP_PSM_{ratio:.2f}_RATIO_{regime}',
                'asset_class': p.asset_class,
                'sector': p.sector,
                'leverage_flag': p.leverage_flag,
                '_topup_origin': True,
            })
            p.entry_psm = current_psm
        return new_intents

    return Strategy(
        name=f'topup_psm_{trigger_ratio:.1f}_frac{fraction:.1f}',
        description=f'Top-up at PSM ratio >= {trigger_ratio}, fraction {fraction}',
        post_decision=post,
        params={'trigger_ratio': trigger_ratio, 'fraction': fraction, 'min_dollars': min_dollars,
                'allowed_regimes': allowed_regimes},
    )


def extend_fragility_relax(
    extra_regimes: Tuple[str, ...] = ('choppy',),
    min_confidence: float = 0.50,
) -> Strategy:
    """Extend the F-7 fragility-relax to also fire in `extra_regimes`."""

    def pre(ctx: StrategyContext) -> None:
        rf = ctx.variant_config.setdefault('regime_fusion_overrides', {})
        existing = list(rf.get('fragility_relax_regimes', ('risk_on_trend', 'calm_uptrend')))
        for r in extra_regimes:
            if r not in existing:
                existing.append(r)
        rf['fragility_relax_regimes'] = tuple(existing)
        rf['fragility_relax_in_risk_on'] = True
        rf['fragility_relax_confidence'] = min_confidence

    return Strategy(
        name=f'extend_relax_{"_".join(extra_regimes)}_conf{min_confidence:.2f}',
        description=f'Extend fragility relax to {extra_regimes} at confidence>={min_confidence}',
        pre_decision=pre,
        params={'extra_regimes': extra_regimes, 'min_confidence': min_confidence},
    )


def compose(strategies: List[Strategy], name: str) -> Strategy:
    def pre(ctx: StrategyContext) -> None:
        for s in strategies:
            if s.pre_decision is not None:
                s.pre_decision(ctx)

    def post(ctx: StrategyContext, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = intents
        for s in strategies:
            if s.post_decision is not None:
                out = s.post_decision(ctx, out)
        return out

    return Strategy(
        name=name,
        description=' + '.join(s.name for s in strategies),
        pre_decision=pre if any(s.pre_decision for s in strategies) else None,
        post_decision=post if any(s.post_decision for s in strategies) else None,
    )
