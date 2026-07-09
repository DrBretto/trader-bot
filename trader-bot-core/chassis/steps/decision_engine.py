"""Decision engine for buy/sell/hold decisions."""

import math
from typing import Dict, Any, List, Optional

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_nested(overrides: Dict[str, Any], path: str, default: float) -> float:
    node: Any = overrides
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return _safe_float(node, default)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _apply_ensemble_overrides(
    regime_data: Dict[str, Any],
    ensemble_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute ensemble regime fields when optimizer overrides are provided."""
    if not ensemble_overrides:
        return regime_data

    gru = regime_data.get('gru_prediction', {})
    transformer = regime_data.get('transformer_prediction', {})
    gru_probs_raw = gru.get('probs', {})
    transformer_probs_raw = transformer.get('probs', {})

    if not isinstance(gru_probs_raw, dict) or not isinstance(transformer_probs_raw, dict):
        return regime_data
    if not gru_probs_raw or not transformer_probs_raw:
        return regime_data

    labels = sorted(set(gru_probs_raw.keys()) | set(transformer_probs_raw.keys()))
    if not labels:
        return regime_data

    gru_weight = _safe_float(ensemble_overrides.get('gru_weight'), 0.5)
    transformer_weight = _safe_float(ensemble_overrides.get('transformer_weight'), 0.5)
    weight_total = gru_weight + transformer_weight
    if weight_total <= 0:
        gru_weight, transformer_weight = 0.5, 0.5
        weight_total = 1.0
    gru_weight /= weight_total
    transformer_weight /= weight_total

    gru_vec = [max(_safe_float(gru_probs_raw.get(label), 0.0), 0.0) for label in labels]
    transformer_vec = [max(_safe_float(transformer_probs_raw.get(label), 0.0), 0.0) for label in labels]
    ensemble_vec = [
        gru_weight * g + transformer_weight * t
        for g, t in zip(gru_vec, transformer_vec)
    ]

    ensemble_sum = sum(ensemble_vec)
    if ensemble_sum > 0:
        ensemble_vec = [v / ensemble_sum for v in ensemble_vec]

    probs = {label: ensemble_vec[idx] for idx, label in enumerate(labels)}
    best_idx = max(range(len(labels)), key=lambda idx: ensemble_vec[idx])
    label = labels[best_idx]
    confidence = ensemble_vec[best_idx]

    dot = sum(g * t for g, t in zip(gru_vec, transformer_vec))
    norm_gru = math.sqrt(sum(g * g for g in gru_vec))
    norm_transformer = math.sqrt(sum(t * t for t in transformer_vec))
    if norm_gru > 0 and norm_transformer > 0:
        disagreement = 1.0 - (dot / (norm_gru * norm_transformer))
    else:
        disagreement = _safe_float(regime_data.get('disagreement'), 0.0)
    disagreement = _clip(disagreement, 0.0, 1.0)

    base_floor = _safe_float(ensemble_overrides.get('multiplier', {}).get('base_floor'), 0.5)
    base_span = _safe_float(ensemble_overrides.get('multiplier', {}).get('base_span'), 0.5)
    disagreement_threshold = _safe_float(ensemble_overrides.get('disagreement_threshold'), 0.3)
    penalty_scale = _safe_float(
        ensemble_overrides.get('multiplier', {}).get('disagreement_penalty_scale'),
        0.5,
    )
    clip_min = _safe_float(ensemble_overrides.get('multiplier', {}).get('clip_min'), 0.5)
    clip_max = _safe_float(ensemble_overrides.get('multiplier', {}).get('clip_max'), 1.0)

    position_multiplier = base_floor + base_span * confidence
    if disagreement > disagreement_threshold and disagreement_threshold < 1.0:
        penalty_ratio = (disagreement - disagreement_threshold) / (1.0 - disagreement_threshold)
        position_multiplier *= (1.0 - penalty_scale * penalty_ratio)
    position_multiplier = _clip(position_multiplier, clip_min, clip_max)

    updated = dict(regime_data)
    updated['label'] = label
    updated['probs'] = probs
    updated['confidence'] = confidence
    updated['disagreement'] = disagreement
    updated['agreement'] = 1.0 - disagreement
    updated['position_size_multiplier'] = position_multiplier
    return updated


def load_regime_compatibility(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Load regime compatibility multipliers from config."""
    return config.get('regime_compatibility', {
        'risk_on_trend': {'equity': 1.10, 'bond': 0.95},
        'risk_off_trend': {'equity': 0.85, 'bond': 1.10},
        'high_vol_panic': {'equity': 0.50, 'bond': 1.20},
        'choppy': {'equity': 0.95, 'bond': 1.05},
        'calm_uptrend': {'equity': 1.10, 'bond': 0.95}
    })


def score_candidates(
    asset_health: List[Dict],
    features_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    regime_label: str,
    regime_compat: Dict[str, Dict[str, float]],
    ranking_scores: Optional[Dict[str, float]] = None,
    ranking_blend: float = 0.0,
) -> pd.DataFrame:
    """
    Score all eligible candidates for buying.

    Returns:
        DataFrame with: symbol, score, health_score, vol_bucket, reason_code
    """
    # Convert asset_health to DataFrame
    health_df = pd.DataFrame(asset_health)

    if len(health_df) == 0:
        return pd.DataFrame()

    # Get latest features
    if len(features_df) > 0:
        latest_date = features_df['date'].max()
        latest_features = features_df[features_df['date'] == latest_date].copy()
    else:
        latest_features = pd.DataFrame()

    # Merge with universe metadata
    if len(universe_df) > 0:
        merged = health_df.merge(universe_df, on='symbol', how='left')
    else:
        merged = health_df.copy()
        merged['asset_class'] = 'equity'
        merged['sector'] = 'broad'
        merged['eligible'] = 1
        merged['leverage_flag'] = 0

    # Filter to eligible only
    merged = merged[merged['eligible'] == 1]

    # Ensure required columns exist (may be missing after merge if no matches)
    for col, default in [('asset_class', 'equity'), ('sector', 'broad'), ('leverage_flag', 0)]:
        if col not in merged.columns:
            merged[col] = default

    # Get regime multiplier for each asset
    def get_multiplier(row):
        sector_key = row.get('sector', '')
        asset_key = row.get('asset_class', 'equity')

        compat = regime_compat.get(regime_label, {})
        mult = compat.get(sector_key, compat.get(asset_key, 1.0))
        return mult

    merged['regime_multiplier'] = merged.apply(get_multiplier, axis=1)

    # Base score: blend health score with ranking score if available
    merged['base_score'] = merged['health_score']
    if ranking_scores and ranking_blend > 0:
        merged['ranking_score'] = merged['symbol'].map(
            lambda s: ranking_scores.get(s, 0.5)
        )
        blend = min(max(ranking_blend, 0.0), 1.0)
        merged['base_score'] = (
            (1.0 - blend) * merged['health_score'] +
            blend * merged['ranking_score']
        )

    # Apply regime multiplier
    merged['final_score'] = (merged['base_score'] * merged['regime_multiplier']).clip(0, 1)

    merged['reason_code'] = 'SCORED'

    return merged[['symbol', 'final_score', 'health_score', 'vol_bucket', 'reason_code',
                   'asset_class', 'sector', 'leverage_flag']]


def filter_buy_candidates(
    scored_df: pd.DataFrame,
    current_holdings: List[str],
    params: Dict[str, Any],
    regime_label: str,
    llm_risk_flags: Dict[str, Dict],
    high_vol_exception_score: float = 0.80,
    ablation: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply buy filters.

    Returns:
        DataFrame of buy-eligible candidates
    """
    if len(scored_df) == 0:
        return pd.DataFrame()

    # Replay-ablation flags (PKT-TB-004); all default OFF = no behavior change.
    ablation = ablation or {}

    candidates = scored_df.copy()

    # Filter: not already held
    candidates = candidates[~candidates['symbol'].isin(current_holdings)]

    # Filter: score threshold (regime-conditional). The flat threshold sizes the
    # bar the same in a risk-on rally as in a panic, so the book never leans in
    # when the coast clears (the 2026-05 cash-during-rally failure). A
    # `buy_score_threshold_by_regime` map (regime -> threshold) lets the bar fall
    # in benign regimes and rise in stressed ones; absent it, the flat value is
    # used (no behavior change).
    bst_by_regime = params.get('buy_score_threshold_by_regime') or {}
    buy_thresh = bst_by_regime.get(regime_label, params.get('buy_score_threshold', 0.65))
    candidates = candidates[candidates['final_score'] >= buy_thresh]

    # Filter: health threshold (regime-conditional, same pattern)
    mhb_by_regime = params.get('min_health_buy_by_regime') or {}
    min_health = mhb_by_regime.get(regime_label, params.get('min_health_buy', 0.60))
    candidates = candidates[candidates['health_score'] >= min_health]

    # Filter: vol bucket (high vol allowed only in calm_uptrend with exceptional score)
    if len(candidates) > 0 and not ablation.get('disable_vol_bucket_filter'):
        def check_vol(row):
            if row['vol_bucket'] == 'high':
                return regime_label == 'calm_uptrend' and row['final_score'] > high_vol_exception_score
            return True
        candidates = candidates[candidates.apply(check_vol, axis=1)]

    # Filter: LLM veto
    if len(candidates) > 0 and not ablation.get('disable_llm_buy_veto'):
        def check_llm_veto(row):
            symbol = row['symbol']
            if symbol in llm_risk_flags:
                return not llm_risk_flags[symbol].get('structural_risk_veto', False)
            return True
        candidates = candidates[candidates.apply(check_llm_veto, axis=1)]

    # Filter: panic mode (only bonds/commodities/defensives)
    if (regime_label == 'high_vol_panic' and len(candidates) > 0
            and not ablation.get('disable_panic_buy_filter')):
        candidates = candidates[candidates['asset_class'].isin(['bond', 'commodity'])]

    # Sort by score descending
    candidates = candidates.sort_values('final_score', ascending=False)

    return candidates


# --- Correlated-cluster groupings for max_sector_weight enforcement (spec 9.1) ---
# universe.csv `sector` labels are granular (52 labels / 64 symbols), so a 35%
# cap on raw labels never binds on a correlated bet spread across several labels
# (the 2026-06-05 growth book was sector_tech + style_growth + theme_innovation +
# industry_biotech — four labels, none individually >35%, ~65% aggregate). The
# spec intends max_sector_weight to cap aggregate CORRELATED exposure, so raw
# labels are coarsened into economic risk clusters. Empirically (Apr-2025..Jun-
# 2026 daily returns) within-cluster avg pairwise correlation ~0.6 for equity
# clusters / ~0.94 rates, vs growth-vs-defensive cross-correlation ~0.32 — the
# clusters are real, not arbitrary. Override per-deployment via config['sector_clusters'].
DEFAULT_SECTOR_CLUSTERS = {
    'sector_tech': 'growth_tech', 'industry_semis': 'growth_tech',
    'style_growth': 'growth_tech', 'theme_innovation': 'growth_tech',
    'industry_biotech': 'growth_tech', 'factor_momentum': 'growth_tech',
    'broad': 'broad_equity', 'global': 'broad_equity',
    'sector_utilities': 'defensive_equity', 'sector_cons_staples': 'defensive_equity',
    'sector_healthcare': 'defensive_equity', 'factor_minvol': 'defensive_equity',
    'factor_dividend': 'defensive_equity', 'factor_dividend_growth': 'defensive_equity',
    'factor_quality': 'defensive_equity', 'sector_reit': 'defensive_equity',
    'sector_financials': 'cyclical_equity', 'sector_energy': 'cyclical_equity',
    'sector_industrials': 'cyclical_equity', 'sector_materials': 'cyclical_equity',
    'sector_cons_disc': 'cyclical_equity', 'sector_comm': 'cyclical_equity',
    'industry_regional_banks': 'cyclical_equity', 'industry_transport': 'cyclical_equity',
    'industry_retail': 'cyclical_equity', 'industry_aerospace_defense': 'cyclical_equity',
    'factor_value': 'cyclical_equity', 'style_value': 'cyclical_equity',
    'country_brazil': 'intl_equity', 'country_china': 'intl_equity',
    'country_india': 'intl_equity', 'country_japan': 'intl_equity',
    'international_dev': 'intl_equity', 'international_em': 'intl_equity',
    'region_europe': 'intl_equity',
    'treas_long': 'rates', 'treas_intermediate': 'rates', 'treas_short': 'rates',
    'treas_tips': 'rates', 'aggregate': 'rates', 'muni': 'rates',
    'credit_high_yield': 'credit', 'credit_investment_grade': 'credit',
    'gold': 'commodity', 'silver': 'commodity', 'oil': 'commodity',
    'natural_gas': 'commodity', 'broad_commodities': 'commodity',
    'usd': 'fx', 'eur': 'fx', 'volatility': 'vol',
}


def _cluster_of(symbol: str, sector_by_symbol: Dict[str, str],
                cluster_map: Dict[str, str]) -> str:
    """Map a symbol to its correlated risk cluster. Unknown labels self-cluster
    (cap at the raw level) so an unmapped symbol is never silently un-capped."""
    raw = sector_by_symbol.get(symbol)
    if raw is None:
        return f"_sym_{symbol}"
    return cluster_map.get(raw, raw)


def evaluate_holdings(
    portfolio_state: Dict[str, Any],
    asset_health: List[Dict],
    prices_df: pd.DataFrame,
    params: Dict[str, Any],
    regime_label: str,
    llm_risk_flags: Dict[str, Dict],
    ablation: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Evaluate each holding for SELL or REDUCE signals.

    Returns:
        List of dicts: [{'symbol': 'SPY', 'action': 'SELL', 'reason': 'STOP_HIT'}, ...]
    """
    actions = []

    # Replay-ablation flags (PKT-TB-004); all default OFF = no behavior change.
    ablation = ablation or {}

    health_map = {h['symbol']: h for h in asset_health}

    dust_value_threshold = params.get('dust_value_threshold', 1.00)

    # As-of date for calendar-based triggers (PKT-TB-004). The legacy
    # leveraged hold cap measures days_held against wall-clock now(), which is
    # correct in production but wrong in historical replays (every replayed
    # holding looks months old). New calendar triggers below use the price
    # data's own latest date so they behave identically in both contexts.
    asof_date = None
    if len(prices_df) > 0 and 'date' in prices_df.columns:
        asof_date = pd.to_datetime(prices_df['date'].max())

    # Structural-decay hold cap (PKT-TB-004 Risk Architect VIXY candidate).
    # Volatility ETPs (VIXY) bleed structurally via futures roll: median 21d
    # buy-and-hold return ~-9% with only ~26% of windows positive, so any hold
    # beyond the initial spike-capture window is negative-sum. The leveraged
    # hold cap does not cover them (leverage_flag=0) and health/stop triggers
    # lag the bleed. Default absent = OFF = production behavior unchanged.
    vol_decay_cfg = params.get('vol_decay_constraints') or {}
    vol_decay_max_days = int(vol_decay_cfg.get('max_hold_days', 0) or 0)
    vol_decay_sectors = set(vol_decay_cfg.get('sectors', ['volatility']))

    for holding in portfolio_state.get('holdings', []):
        symbol = holding['symbol']
        entry_price = holding['entry_price']
        peak_price = holding.get('peak_price', entry_price)
        shares = holding['shares']

        # Get current price
        symbol_prices = prices_df[prices_df['symbol'] == symbol]
        if len(symbol_prices) == 0:
            continue

        current_price = symbol_prices.sort_values('date')['close'].iloc[-1]

        # Skip dust positions — too small to trade on any broker
        if abs(shares * current_price) < dust_value_threshold:
            continue

        # Update peak price
        if current_price > peak_price:
            peak_price = current_price
            holding['peak_price'] = peak_price

        # Get current health
        current_health = health_map.get(symbol, {}).get('health_score', 0.5)
        sell_health_threshold = params.get('sell_health_threshold', 0.35)

        # Track peak health per holding (mirrors peak_price). Persisted on the
        # holding dict so the night-phase publish round-trips it for the REDUCE
        # HEALTH_DROP trigger below.
        peak_health = holding.get('peak_health')
        if peak_health is None or current_health > peak_health:
            peak_health = current_health
        holding['peak_health'] = peak_health

        # Persistence gate: track consecutive days at/below the sell-health
        # threshold per holding. Required because a single-day health dip is
        # noisy and was historically liquidating profitable positions on a
        # one-day reading; the config has carried `sell_health_days: 3` since
        # the optimizer's bootstrap but the gate was never wired. The counter
        # is mutated on the holding dict so the night-phase publish persists
        # it in portfolio_state.json for the next run.
        sell_health_days_required = max(int(params.get('sell_health_days', 3) or 0), 1)
        consecutive_below = int(holding.get('consecutive_below_health_days', 0) or 0)
        if current_health <= sell_health_threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0
        holding['consecutive_below_health_days'] = consecutive_below

        # Days held
        entry_date = pd.to_datetime(holding.get('entry_date', pd.Timestamp.now()))
        days_held = (pd.Timestamp.now() - entry_date).days

        # Trailing stop
        is_leveraged = holding.get('leverage_flag', 0) == 1
        stop_pct = params.get('trailing_stop_leveraged' if is_leveraged else 'trailing_stop_base', 0.10)
        trailing_stop_price = peak_price * (1 - stop_pct)

        # CHECK SELL TRIGGERS

        # 1. Stop hit
        if current_price <= trailing_stop_price:
            actions.append({
                'symbol': symbol,
                'action': 'SELL',
                'reason': 'STOP_HIT',
                'shares': shares,
                'price': current_price,
                'details': f'Price {current_price:.2f} <= stop {trailing_stop_price:.2f}'
            })
            continue

        # 2. Health collapse — only fire once health has been at/below threshold
        # for sell_health_days consecutive runs. Below-threshold-but-not-yet-
        # persistent holdings stay open and continue to accrue the counter.
        if (current_health <= sell_health_threshold
                and consecutive_below >= sell_health_days_required):
            actions.append({
                'symbol': symbol,
                'action': 'SELL',
                'reason': 'HEALTH_COLLAPSE',
                'shares': shares,
                'price': current_price,
                'details': (
                    f'Health {current_health:.2f} <= {sell_health_threshold:.2f} '
                    f'for {consecutive_below} day(s) (gate={sell_health_days_required})'
                ),
            })
            continue

        # 3. Panic mode (asset not allowed)
        if regime_label == 'high_vol_panic' and not ablation.get('disable_panic_force_sell'):
            asset_class = holding.get('asset_class', 'equity')
            if asset_class not in ['bond', 'commodity']:
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'REGIME_PANIC',
                    'shares': shares,
                    'price': current_price,
                    'details': f'Asset class {asset_class} not allowed in panic'
                })
                continue

        # 4. LLM structural risk veto
        if symbol in llm_risk_flags and not ablation.get('disable_llm_sell_veto'):
            if llm_risk_flags[symbol].get('structural_risk_veto', False):
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'LLM_VETO',
                    'shares': shares,
                    'price': current_price,
                    'details': llm_risk_flags[symbol].get('one_sentence_rationale', '')
                })
                continue

        # 5. Leveraged max hold days
        if is_leveraged:
            max_days = params.get('leveraged_constraints', {}).get('max_hold_days', 10)
            if days_held >= max_days:
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'LEVERAGE_HOLD_CAP',
                    'shares': shares,
                    'price': current_price,
                    'details': f'Held {days_held} days >= max {max_days}'
                })
                continue

        # 5b. Structural-decay hold cap (PKT-TB-004 VIXY candidate, default OFF).
        # Calendar days vs the price data's as-of date (replay-consistent),
        # NOT wall-clock now().
        if vol_decay_max_days and holding.get('sector') in vol_decay_sectors:
            days_held_asof = days_held
            if asof_date is not None:
                days_held_asof = (asof_date - entry_date).days
            if days_held_asof >= vol_decay_max_days:
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'VOL_DECAY_HOLD_CAP',
                    'shares': shares,
                    'price': current_price,
                    'details': (
                        f'Decay-class sector {holding.get("sector")} held '
                        f'{days_held_asof} days >= max {vol_decay_max_days}'
                    ),
                })
                continue

        # CHECK REDUCE TRIGGERS (spec 9.5 — half-size trims, never coded until
        # 2026-06-06). Evaluated only if no SELL trigger fired above.

        # 6. Health deterioration: trim when health has fallen reduce_health_drop
        #    below its peak — a slow-deterioration signal that price-based stops
        #    miss. Persistence gate (default 1 = spec-faithful) is configurable;
        #    set reduce_health_drop_days higher to match the HEALTH_COLLAPSE
        #    noise-hardening if single-day proves too trim-happy.
        reduce_drop = params.get('reduce_health_drop')
        if reduce_drop:
            below_drop = int(holding.get('consecutive_health_drop_days', 0) or 0)
            if (peak_health - current_health) >= float(reduce_drop):
                below_drop += 1
            else:
                below_drop = 0
            holding['consecutive_health_drop_days'] = below_drop
            drop_days_required = max(int(params.get('reduce_health_drop_days', 1) or 0), 1)
            if below_drop >= drop_days_required:
                # Re-arm: reset the trailing health peak to the current level so
                # the trigger fires once per fresh reduce_health_drop decline
                # rather than every day a position sits below its peak.
                holding['peak_health'] = current_health
                holding['consecutive_health_drop_days'] = 0
                actions.append({
                    'symbol': symbol,
                    'action': 'REDUCE',
                    'reason': 'HEALTH_DROP',
                    'shares': shares,
                    'price': current_price,
                    'details': (
                        f'Health {current_health:.2f} <= peak {peak_health:.2f} '
                        f'- {reduce_drop} for {below_drop} day(s)'
                    ),
                })
                continue
        else:
            holding['consecutive_health_drop_days'] = 0

        # 7. Regime shift: trim positions opened in a benign regime once the
        #    regime degrades to choppy/risk_off. Requires entry_regime captured
        #    at buy time; positions without it (legacy/seed) skip this trigger.
        entry_regime = holding.get('entry_regime')
        if (regime_label in ('choppy', 'risk_off_trend')
                and entry_regime in ('calm_uptrend', 'risk_on_trend')
                and not ablation.get('disable_reduce_regime_shift')):
            # Fire ONCE per degradation (spec: "once the regime degrades"), not
            # every day the regime stays bad. Stamp the holding's regime to the
            # current (degraded) label so it won't re-trim until the position is
            # re-established in a benign regime and degrades again.
            holding['entry_regime'] = regime_label
            actions.append({
                'symbol': symbol,
                'action': 'REDUCE',
                'reason': 'REGIME_SHIFT',
                'shares': shares,
                'price': current_price,
                'details': f'Regime {entry_regime} -> {regime_label}',
            })
            continue

    return actions


def compute_exposure_trims(
    portfolio_state: Dict[str, Any],
    pending_actions: List[Dict[str, Any]],
    prices_df: pd.DataFrame,
    portfolio_value: float,
    target_exposure: float,
    regime_label: str,
    params: Dict[str, Any],
    trim_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """PKT-TB-004 Risk Architect candidate: book-level exposure trim.

    Today the throttle stack (position_size_modifier x risk_throttle ->
    effective_exposure_multiplier published by regime fusion) gates only NEW
    buys; an already-deployed book ignores the regime view entirely (the canon
    book sat ~100% invested with target_gross_exposure ~0.27). This is the
    SELL-side complement: when current gross exposure exceeds the fusion
    target by more than `trigger_gap`, emit exact-share REDUCE intents
    pro-rata across holdings to bring the book back to target +
    `hysteresis_gap`.

    No-churn properties:
    - trims TO target + hysteresis_gap (never below target), so a re-fire
      requires the gap to re-open by at least trigger_gap - hysteresis_gap;
    - optional `persistence_days` requires the breach to persist N
      consecutive runs (counter carried on portfolio_state, same pattern as
      the per-holding health counters);
    - per-order minimum `min_trim_dollars` suppresses dust trims.

    Interaction with existing sells: holdings with a pending SELL are
    excluded from both the gross calculation and the trim; pending REDUCEs
    are counted at their post-reduce size. Config lives at
    decision_engine_overrides['exposure_trim']; absent/enabled=False =
    production behavior unchanged.
    """
    cfg = trim_cfg or {}
    if not cfg.get('enabled') or portfolio_value <= 0:
        return []
    skip_regimes = set(cfg.get('skip_regimes', []) or [])
    if regime_label in skip_regimes:
        return []

    trigger_gap = _safe_float(cfg.get('trigger_gap'), 0.10)
    hysteresis_gap = min(_safe_float(cfg.get('hysteresis_gap'), 0.05), trigger_gap)
    min_trim = _safe_float(
        cfg.get('min_trim_dollars'),
        _safe_float(params.get('min_order_dollars'), 250.0),
    )
    persistence_days = max(int(cfg.get('persistence_days', 1) or 1), 1)
    target_floor = _safe_float(cfg.get('target_floor'), 0.0)
    target = _clip(_safe_float(target_exposure, 1.0), target_floor, 1.0)

    pending_sells = {
        a.get('symbol') for a in pending_actions
        if str(a.get('action', '')).upper() == 'SELL'
    }
    pending_reduces = {
        a.get('symbol') for a in pending_actions
        if str(a.get('action', '')).upper() == 'REDUCE'
    }

    positions = []
    gross = 0.0
    for holding in portfolio_state.get('holdings', []):
        symbol = holding.get('symbol')
        if symbol in pending_sells:
            continue
        symbol_prices = prices_df[prices_df['symbol'] == symbol]
        if len(symbol_prices) == 0:
            continue
        price = float(symbol_prices.sort_values('date')['close'].iloc[-1])
        shares = _safe_float(holding.get('shares'), 0.0)
        if symbol in pending_reduces:
            # paper_trader executes legacy REDUCE as shares // 2 of the
            # intent's share count (= the full holding for trigger-emitted
            # reduces); count the post-reduce remainder.
            shares = shares - int(shares) // 2
        market_value = shares * price
        if market_value <= 0 or price <= 0:
            continue
        gross += market_value
        positions.append((symbol, shares, price, market_value))

    current_ratio = gross / portfolio_value

    counter_key = 'exposure_trim_consecutive_over'
    over = current_ratio > target + trigger_gap
    counter = int(portfolio_state.get(counter_key, 0) or 0)
    counter = counter + 1 if over else 0
    portfolio_state[counter_key] = counter
    if not over or counter < persistence_days:
        return []

    trim_to = target + hysteresis_gap
    total_trim = (current_ratio - trim_to) * portfolio_value
    if total_trim < min_trim or gross <= 0:
        return []

    actions: List[Dict[str, Any]] = []
    for symbol, shares, price, market_value in positions:
        trim_dollars = total_trim * market_value / gross
        if trim_dollars < min_trim:
            continue
        trim_shares = int(trim_dollars / price)
        if trim_shares <= 0:
            continue
        trim_shares = min(trim_shares, int(shares))
        if trim_shares <= 0:
            continue
        actions.append({
            'symbol': symbol,
            'action': 'REDUCE',
            'reason': 'EXPOSURE_TRIM',
            'shares': trim_shares,
            # Exact-share trim: paper_trader honors reduce_shares verbatim
            # instead of the legacy halving.
            'reduce_shares': trim_shares,
            'price': price,
            'details': (
                f'Gross {current_ratio:.2f} > target {target:.2f} '
                f'+ {trigger_gap:.2f}; trim to {trim_to:.2f}'
            ),
        })

    if actions:
        # Re-arm: a fresh breach is required before the next trim.
        portfolio_state[counter_key] = 0
    return actions


def compute_position_size(
    symbol: str,
    portfolio_value: float,
    current_price: float,
    vol_bucket: str,
    regime_label: str,
    params: Dict[str, Any],
    llm_confidence_adj: float = 0.0,
    ensemble_multiplier: float = 1.0,
    position_size_modifier: float = 1.0,
    risk_throttle_factor: float = 0.0,
    decision_engine_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute target shares and dollars for a position.

    Args:
        symbol: Ticker symbol
        portfolio_value: Total portfolio value
        current_price: Current asset price
        vol_bucket: Volatility bucket (low/med/high)
        regime_label: Current market regime
        params: Decision parameters
        llm_confidence_adj: LLM-based position reduction (0-0.5)
        ensemble_multiplier: Ensemble model sizing multiplier (0.5-1.0)
            - Reduced when models disagree on regime
        position_size_modifier: Expert signal sizing modifier (0.25-1.0)
            - Incorporates fragility, entropy, panic overrides
        risk_throttle_factor: Risk throttle from expert signals (0.0-1.0)
            - Higher = more throttling applied to position

    Returns:
        {'shares': int|float, 'dollars': float, 'final_weight': float}
        shares is int in simulated mode (backward compat), float otherwise.
    """
    target_weight = params.get('max_position_weight', 0.20)
    base_dollars = portfolio_value * target_weight

    # Adjust by volatility bucket
    overrides = decision_engine_overrides or {}
    vol_adj_map = {
        'low': _get_nested(overrides, 'position_size.vol_adj.low', 1.10),
        'med': _get_nested(overrides, 'position_size.vol_adj.med', 1.0),
        'high': _get_nested(overrides, 'position_size.vol_adj.high', 0.80),
    }
    vol_adj = vol_adj_map.get(vol_bucket, 1.0)

    # Adjust by regime
    regime_adj = {
        'calm_uptrend': _get_nested(overrides, 'position_size.regime_adj.calm_uptrend', 1.10),
        'risk_on_trend': _get_nested(overrides, 'position_size.regime_adj.risk_on_trend', 1.10),
        'choppy': _get_nested(overrides, 'position_size.regime_adj.choppy', 0.90),
        'risk_off_trend': _get_nested(overrides, 'position_size.regime_adj.risk_off_trend', 0.80),
        'high_vol_panic': _get_nested(overrides, 'position_size.regime_adj.high_vol_panic', 0.50),
    }.get(regime_label, 1.0)

    # Adjust by LLM confidence
    llm_adj = 1.0 - llm_confidence_adj

    # Adjust by ensemble model agreement (reduces size when models disagree).
    #
    # Caller-pattern note (the F-2030-A finding from 2026-04-30 audit was
    # a false positive — see `docs/plans/2026-04-30-ensemble-double-fix-RETURN.md`):
    # `regime_fusion.decide_regime_v3` folds `ensemble_multiplier` into
    # `position_size_modifier` at `src/signals/regime_fusion.py:262`, so
    # a naive call to `compute_position_size` with both arguments non-neutral
    # would double-apply the multiplier. The v3 production callers defend
    # against this by passing `ensemble_multiplier=1.0` whenever
    # `expert_signals is not None` (see `:798` and `:828`); the v2 callers
    # pass `position_size_modifier=1.0` (its default). Either way, the
    # multiplier is applied exactly once.
    #
    # The gate below (`ensemble_multiplier_already_applied`, default False)
    # is defense-in-depth for a future caller that forgets the v3 1.0 swap.
    # When True, `ensemble_adj` is forced to 1.0 regardless of the passed
    # `ensemble_multiplier`, making the function robust to incorrect call
    # patterns. Promoting the gate via config is a no-op under all current
    # callers and is safe; it ships as documentation more than as a fix.
    ensemble_already_applied = bool(_get_nested(
        overrides, 'position_size.ensemble_multiplier_already_applied', False
    ))
    ensemble_adj = 1.0 if ensemble_already_applied else ensemble_multiplier

    # Expert signal adjustments. `position_size_modifier` is the canonical
    # full-chain output of `decide_regime_v3` (fragility cap, entropy gate,
    # panic override, ensemble multiplier — all included). The v2 / legacy
    # callers leave it at the default 1.0.
    expert_adj = position_size_modifier

    # Risk throttle: higher throttle = smaller positions
    throttle_scale = _get_nested(overrides, 'position_size.throttle_scale', 0.5)
    throttle_adj = 1.0 - (risk_throttle_factor * throttle_scale)

    # Final target
    adjusted_dollars = (base_dollars * vol_adj * regime_adj * llm_adj
                        * ensemble_adj
                        * expert_adj * throttle_adj)

    # Check minimum (use adjusted_dollars before share conversion)
    min_order = params.get('min_order_dollars', 250)
    if adjusted_dollars < min_order:
        return {'shares': 0, 'dollars': 0, 'final_weight': 0.0}

    # Shares — keep dollars as canonical; shares is derived
    if current_price > 0:
        shares = int(adjusted_dollars / current_price)
    else:
        shares = 0

    # Re-check after int floor (a high-priced stock could round to 0)
    actual_dollars = shares * current_price if shares > 0 else adjusted_dollars
    if shares == 0 and current_price > 0:
        # Whole-share floor dropped below min — record the dollar intent
        # so broker-mode callers can still use notional ordering
        actual_dollars = adjusted_dollars

    final_weight = actual_dollars / portfolio_value if portfolio_value > 0 else 0

    return {
        'shares': shares,
        'dollars': round(actual_dollars, 2),
        'final_weight': final_weight,
        'ensemble_multiplier': ensemble_multiplier
    }


def _build_watchlist(
    scored_df: pd.DataFrame,
    current_holdings: List[str],
    features_df: pd.DataFrame,
    portfolio_value: float,
    regime_label: str,
    params: Dict[str, Any],
    ensemble_multiplier: float,
    position_size_modifier: float,
    risk_throttle_factor: float,
    target_count: int = 10,
    decision_engine_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build a ranked watchlist of top-scored candidates for the dashboard.

    Returns all scored symbols (excluding current holdings), sorted by score,
    enriched with return data, behavior classification, and suggested size.
    """
    if len(scored_df) == 0:
        return []

    # Exclude current holdings
    watchlist_df = scored_df[~scored_df['symbol'].isin(current_holdings)].copy()
    watchlist_df = watchlist_df.sort_values('final_score', ascending=False).head(target_count)

    if len(watchlist_df) == 0:
        return []

    # Get return data from features
    latest_features = pd.DataFrame()
    if len(features_df) > 0:
        latest_date = features_df['date'].max()
        latest_features = features_df[features_df['date'] == latest_date]

    result = []
    for _, row in watchlist_df.iterrows():
        symbol = row['symbol']

        # Get returns from features
        return_21d = 0.0
        return_63d = 0.0
        if len(latest_features) > 0:
            sym_feat = latest_features[latest_features['symbol'] == symbol]
            if len(sym_feat) > 0:
                return_21d = float(sym_feat.iloc[0].get('return_21d', 0) or 0)
                return_63d = float(sym_feat.iloc[0].get('return_63d', 0) or 0)

        # Classify behavior from returns
        if return_21d > 0.02:
            behavior = 'momentum'
        elif return_21d < -0.02:
            behavior = 'mean_reversion'
        else:
            behavior = 'mixed'

        # Get current price for sizing
        symbol_prices = features_df[features_df['symbol'] == symbol] if len(features_df) > 0 else pd.DataFrame()
        current_price = 0.0
        if len(symbol_prices) > 0:
            current_price = float(symbol_prices.sort_values('date')['close'].iloc[-1])

        # Compute suggested size
        suggested_size = 0.0
        if current_price > 0:
            position = compute_position_size(
                symbol, portfolio_value, current_price,
                row['vol_bucket'], regime_label, params,
                0.0, ensemble_multiplier, position_size_modifier, risk_throttle_factor,
                decision_engine_overrides=decision_engine_overrides,
            )
            suggested_size = position['dollars']

        result.append({
            'symbol': symbol,
            'score': float(row['final_score']),
            'health_score': float(row['health_score']),
            'vol_bucket': row['vol_bucket'],
            'behavior': behavior,
            'return_21d': return_21d,
            'return_63d': return_63d,
            'suggested_size': suggested_size,
        })

    return result


def run(
    inference_output: Dict[str, Any],
    llm_risks: Dict[str, Dict],
    features_df: pd.DataFrame,
    config: Dict[str, Any],
    validation: Dict[str, Any],
    expert_signals: Optional[Dict[str, Any]] = None,
    ranking_scores: Optional[Dict[str, float]] = None,
    ranking_blend: float = 0.0,
) -> Dict[str, Any]:
    """
    Run decision engine to generate buy/sell actions.

    Args:
        inference_output: Output from run_inference
        llm_risks: LLM risk flags per symbol
        features_df: Asset features
        config: Configuration dict
        validation: Data validation results
        expert_signals: Expert signal outputs (None = use legacy path)

    Returns:
        Decisions dict with actions list
    """
    print("Running decision engine...")

    params = config.get('decision_params', {})
    regime_compat = config.get('regime_compatibility', {})
    decision_engine_overrides = config.get('decision_engine_overrides', {})
    # Replay-ablation flags (PKT-TB-004): layer toggles for counterfactual
    # attribution replays. Absent/empty = production behavior, unchanged.
    ablation = decision_engine_overrides.get('ablation', {}) or {}
    regime_fusion_overrides = config.get('regime_fusion_overrides')
    ensemble_overrides = config.get('ensemble_overrides')
    universe_df = config.get('universe', pd.DataFrame())

    if isinstance(universe_df, list):
        universe_df = pd.DataFrame(universe_df)

    run_input = dict(inference_output)
    run_input['regime'] = _apply_ensemble_overrides(
        dict(inference_output.get('regime', {})),
        ensemble_overrides,
    )

    asset_health = run_input['asset_health']
    run_date = run_input['date']

    # Get ensemble metrics
    ensemble_regime_label = run_input['regime']['label']
    ensemble_multiplier = run_input['regime'].get('position_size_multiplier', 1.0)
    regime_disagreement = run_input['regime'].get('disagreement', 0.0)

    # Apply regime fusion v3 if expert signals available
    position_size_modifier = 1.0
    risk_throttle_factor = 0.0
    expert_metrics = {}

    if expert_signals is not None:
        from chassis.signals.regime_fusion import decide_regime_v3

        macro = expert_signals.get('macro_credit', {})
        vol = expert_signals.get('vol_uncertainty', {})
        frag = expert_signals.get('fragility', {})
        ent = expert_signals.get('entropy_shift', {})

        fusion = decide_regime_v3(
            ensemble_regime_label=ensemble_regime_label,
            trend_risk_on_prob=run_input['regime'].get('probs', {}).get('risk_on_trend', 0.0),
            panic_prob=run_input['regime'].get('probs', {}).get('high_vol_panic', 0.0),
            ensemble_disagreement=regime_disagreement,
            ensemble_multiplier=ensemble_multiplier,
            macro_credit_score=macro.get('macro_credit_score', 0.0),
            vol_uncertainty_score=vol.get('vol_uncertainty_score', 0.5),
            vol_regime_label=vol.get('vol_regime_label', 'calm'),
            fragility_score=frag.get('fragility_score', 0.5),
            entropy_score=ent.get('entropy_score', 0.5),
            entropy_shift_flag=ent.get('entropy_shift_flag', False),
            params=regime_fusion_overrides,
        )

        regime_label = fusion['final_regime_label']
        position_size_modifier = fusion['position_size_modifier']
        risk_throttle_factor = fusion['risk_throttle_factor']

        expert_metrics = {
            'final_regime_label': fusion['final_regime_label'],
            'regime_confidence': fusion['regime_confidence'],
            'position_size_modifier': position_size_modifier,
            'risk_throttle_factor': risk_throttle_factor,
            'override_reason': fusion.get('override_reason'),
            'effective_exposure_multiplier': fusion.get('effective_exposure_multiplier', 1.0),
            'target_gross_exposure': fusion.get('target_gross_exposure', 1.0),
            'throttle_mapping': fusion.get('throttle_mapping'),
            'fusion_rules': fusion.get('fusion_rules', []),
            'macro_credit_score': macro.get('macro_credit_score', 0.0),
            'vol_uncertainty_score': vol.get('vol_uncertainty_score', 0.5),
            'vol_regime_label': vol.get('vol_regime_label', 'calm'),
            'fragility_score': frag.get('fragility_score', 0.5),
            'entropy_shift_flag': ent.get('entropy_shift_flag', False),
        }

        if fusion.get('override_reason'):
            print(f"  Regime fusion: {ensemble_regime_label} -> {regime_label} ({fusion['override_reason']})")
        print(f"  Position size modifier: {position_size_modifier:.2f}, Risk throttle: {risk_throttle_factor:.2f}")
    else:
        # Legacy path: use ensemble label directly
        regime_label = ensemble_regime_label

    if regime_disagreement > 0.3:
        print(f"  Ensemble disagreement: {regime_disagreement:.2f} - reducing position sizes by {(1 - ensemble_multiplier) * 100:.0f}%")

    # Load current portfolio state
    portfolio_state = config.get('portfolio_state', {
        'cash': params.get('initial_portfolio_value', 100000),
        'holdings': [],
        'portfolio_value': params.get('initial_portfolio_value', 100000)
    })

    current_holdings = [h['symbol'] for h in portfolio_state.get('holdings', [])]
    portfolio_value = portfolio_state.get('portfolio_value', 100000)

    actions = []

    # 1. Evaluate existing holdings for sells
    sell_actions = evaluate_holdings(
        portfolio_state,
        asset_health,
        features_df,
        params,
        regime_label,
        llm_risks,
        ablation=ablation,
    )

    for action in sell_actions:
        actions.append({
            'action': action['action'],
            'symbol': action['symbol'],
            'shares': action['shares'],
            'price': action['price'],
            'dollars': action['shares'] * action['price'],
            'reason': action['reason'],
            'details': action.get('details', '')
        })

    # 1b. Book-level exposure trim (PKT-TB-004 Risk Architect candidate).
    # Config absent / enabled=False = production behavior unchanged. Requires
    # the v3 fusion target (expert_metrics); the legacy path has no target.
    trim_cfg = decision_engine_overrides.get('exposure_trim') or {}
    trim_actions: List[Dict[str, Any]] = []
    if trim_cfg.get('enabled') and expert_metrics:
        trim_actions = compute_exposure_trims(
            portfolio_state=portfolio_state,
            pending_actions=actions,
            prices_df=features_df,
            portfolio_value=portfolio_value,
            target_exposure=expert_metrics.get('effective_exposure_multiplier', 1.0),
            regime_label=regime_label,
            params=params,
            trim_cfg=trim_cfg,
        )
        for action in trim_actions:
            actions.append({
                'action': action['action'],
                'symbol': action['symbol'],
                'shares': action['shares'],
                'reduce_shares': action['reduce_shares'],
                'price': action['price'],
                'dollars': action['shares'] * action['price'],
                'reason': action['reason'],
                'details': action.get('details', ''),
            })
        if trim_actions:
            print(f"  Exposure trim: {len(trim_actions)} pro-rata REDUCE intents "
                  f"(target {expert_metrics.get('effective_exposure_multiplier', 1.0):.2f})")

    # 2. Score and filter buy candidates
    scored = score_candidates(
        asset_health,
        features_df,
        universe_df,
        regime_label,
        regime_compat,
        ranking_scores=ranking_scores,
        ranking_blend=ranking_blend,
    )

    buy_candidates = filter_buy_candidates(
        scored,
        current_holdings,
        params,
        regime_label,
        llm_risks,
        high_vol_exception_score=_get_nested(
            decision_engine_overrides,
            'high_vol_bucket_exception_score',
            0.80,
        ),
        ablation=ablation,
    )

    # 3. Generate buy orders (respect max positions and cash reserve)
    num_holdings = len(current_holdings) - len([a for a in sell_actions if a['action'] == 'SELL'])
    max_positions = params.get('max_positions', 8)
    available_slots = max_positions - num_holdings

    # Calculate available cash
    min_cash_pct = params.get('min_cash_reserve_by_regime', {}).get(regime_label, 0.10)
    min_cash = portfolio_value * min_cash_pct
    available_cash = portfolio_state.get('cash', portfolio_value) - min_cash

    # Add proceeds from sells
    for action in sell_actions:
        if action['action'] == 'SELL':
            available_cash += action.get('shares', 0) * action.get('price', 0)

    # --- Intended portfolio risk controls (spec 9.1) ---
    # (a) max_sector_weight: cap aggregate weight per CORRELATED cluster. Seeded
    #     from current holdings (minus pending sells) so the cap accounts for
    #     existing exposure, not just new buys. Orders are CLAMPED to the
    #     remaining cluster headroom rather than rejected, preserving the
    #     diversification intent without starving the book.
    # (b) gross-exposure ceiling: cap total deployed capital at
    #     portfolio_value * effective_exposure_multiplier (the published
    #     target_gross_exposure that the engine previously ignored). Off unless
    #     decision_engine.enforce_gross_exposure_cap is set, because the
    #     per-position throttle is already applied in compute_position_size and
    #     this is the additional BOOK-LEVEL ceiling.
    sector_by_symbol = {}
    if len(universe_df) > 0 and 'sector' in universe_df.columns:
        sector_by_symbol = dict(zip(universe_df['symbol'], universe_df['sector']))
    # Default: cap per spec-literal raw `sector` label. The correlated-cluster
    # grouping (DEFAULT_SECTOR_CLUSTERS) is a STRICTER preventative option that
    # over-constrains returns and double-counts with the REDUCE layer (which
    # already trims correlated concentration dynamically). Opt in via
    # config['sector_clusters'] = decision_engine.DEFAULT_SECTOR_CLUSTERS only if
    # a hard preventative cap on correlated exposure is explicitly wanted.
    cluster_map = config.get('sector_clusters', {})
    max_sector_weight = params.get('max_sector_weight')  # None -> cap disabled
    min_order = params.get('min_order_dollars', 250)

    def _cur_price(sym):
        sp = features_df[features_df['symbol'] == sym]
        if len(sp) == 0:
            return None
        return float(sp.sort_values('date')['close'].iloc[-1])

    sold_syms = {a['symbol'] for a in sell_actions if a['action'] == 'SELL'}
    cluster_dollars: Dict[str, float] = {}
    deployed_dollars = 0.0
    for h in portfolio_state.get('holdings', []):
        if h['symbol'] in sold_syms:
            continue
        p = _cur_price(h['symbol'])
        if p is None:
            continue
        mv = h.get('shares', 0) * p
        deployed_dollars += mv
        cl = _cluster_of(h['symbol'], sector_by_symbol, cluster_map)
        cluster_dollars[cl] = cluster_dollars.get(cl, 0.0) + mv

    # Exposure trims free up book-level and cluster headroom
    for t in trim_actions:
        t_dollars = t['shares'] * t['price']
        deployed_dollars -= t_dollars
        cl = _cluster_of(t['symbol'], sector_by_symbol, cluster_map)
        cluster_dollars[cl] = cluster_dollars.get(cl, 0.0) - t_dollars

    enforce_gross = bool(_get_nested(
        decision_engine_overrides, 'enforce_gross_exposure_cap', False))
    eff_exp_mult = float(expert_metrics.get('effective_exposure_multiplier', 1.0) or 1.0)
    gross_cap = portfolio_value * eff_exp_mult if enforce_gross else float('inf')

    # When the book is being trimmed back toward target, buying the same
    # night is incoherent (and would churn the trim proceeds straight back
    # into the market). Default True; tunable for the replay battery.
    if trim_actions and trim_cfg.get('block_buys_when_trimming', True):
        available_slots = 0

    buy_count = 0
    for _, candidate in buy_candidates.iterrows():
        if buy_count >= available_slots:
            break

        symbol = candidate['symbol']

        # Get current price
        symbol_prices = features_df[features_df['symbol'] == symbol]
        if len(symbol_prices) == 0:
            continue

        current_price = symbol_prices.sort_values('date')['close'].iloc[-1]

        # Get LLM confidence adjustment
        if ablation.get('disable_llm_size_adj'):
            llm_conf_adj = 0.0
        else:
            llm_conf_adj = llm_risks.get(symbol, {}).get('confidence_adjustment', 0.0)

        # Compute position size (includes ensemble + expert signal adjustments)
        position = compute_position_size(
            symbol,
            portfolio_value,
            current_price,
            candidate['vol_bucket'],
            regime_label,
            params,
            llm_conf_adj,
            ensemble_multiplier if expert_signals is None else 1.0,
            position_size_modifier,
            risk_throttle_factor,
            decision_engine_overrides=decision_engine_overrides,
        )

        dollars = position['dollars']
        if position['shares'] <= 0 or dollars <= 0:
            continue

        # Clamp by gross-exposure ceiling
        if enforce_gross:
            gross_head = gross_cap - deployed_dollars
            if gross_head <= 0:
                continue
            dollars = min(dollars, gross_head)

        # Clamp by correlated-cluster concentration cap
        cluster = _cluster_of(symbol, sector_by_symbol, cluster_map)
        if max_sector_weight:
            cluster_head = portfolio_value * float(max_sector_weight) - cluster_dollars.get(cluster, 0.0)
            if cluster_head <= 0:
                continue
            dollars = min(dollars, cluster_head)

        # Re-derive shares after any clamp; drop sub-min / unaffordable orders
        if dollars < min_order or dollars > available_cash:
            continue
        shares = int(dollars / current_price) if current_price > 0 else 0
        if shares <= 0:
            continue
        dollars = shares * current_price
        if dollars < min_order or dollars > available_cash:
            continue

        actions.append({
            'action': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': current_price,
            'dollars': dollars,
            'weight': dollars / portfolio_value if portfolio_value > 0 else 0,
            'reason': f"SCORE_{candidate['final_score']:.2f}_HEALTH_{candidate['health_score']:.2f}",
            'score': candidate['final_score'],
            'health': candidate['health_score'],
            'vol_bucket': candidate['vol_bucket']
        })

        available_cash -= dollars
        deployed_dollars += dollars
        cluster_dollars[cluster] = cluster_dollars.get(cluster, 0.0) + dollars
        buy_count += 1

    print(f"  Generated {len(actions)} actions: "
          f"{len([a for a in actions if a['action'] == 'BUY'])} buys, "
          f"{len([a for a in actions if a['action'] == 'SELL'])} sells")

    # 4. Build watchlist of top candidates for dashboard
    watchlist = _build_watchlist(
        scored, current_holdings, features_df, portfolio_value,
        regime_label, params, ensemble_multiplier if expert_signals is None else 1.0,
        position_size_modifier, risk_throttle_factor,
        target_count=max(
            len(current_holdings),
            int(_get_nested(decision_engine_overrides, 'watchlist.min_target_count', 8)),
        ),
        decision_engine_overrides=decision_engine_overrides,
    )

    result = {
        'date': run_date,
        'regime': regime_label,
        'actions': actions,
        'buy_candidates': watchlist,
        'pass_filters': {
            'price_coverage': validation.get('price_coverage', 0),
            'context_freshness': 1 if not validation.get('degraded_mode', False) else 0,
            'degraded_mode': validation.get('degraded_mode', False)
        },
        'ensemble_metrics': {
            'disagreement': regime_disagreement,
            'position_size_multiplier': ensemble_multiplier,
            'confidence': run_input['regime'].get('confidence', 1.0)
        }
    }

    if expert_metrics:
        result['expert_metrics'] = expert_metrics

    return result
