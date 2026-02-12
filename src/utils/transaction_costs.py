"""Simulate realistic transaction costs (bid-ask spread + slippage) for paper trading."""

import random
from typing import Tuple

# Half-spread in basis points by sector/asset_class.
# These are conservative estimates for ETF trading during regular market hours.
_SPREAD_BPS = {
    # Highly liquid broad-market ETFs
    'broad': 1.0,
    'global': 2.0,
    # Bonds
    'treas_long': 2.0,
    'treas_intermediate': 1.5,
    'treas_short': 1.0,
    'treas_tips': 2.5,
    'credit_investment_grade': 2.5,
    'credit_high_yield': 3.0,
    'aggregate': 2.0,
    'muni': 3.0,
    # Sector ETFs
    'sector_tech': 2.0,
    'sector_financials': 2.5,
    'sector_energy': 3.0,
    'sector_healthcare': 2.5,
    'sector_industrials': 3.0,
    'sector_cons_disc': 3.0,
    'sector_cons_staples': 3.0,
    'sector_utilities': 3.0,
    'sector_materials': 3.0,
    'sector_comm': 3.0,
    'sector_reit': 3.0,
    # Industry / niche
    'industry_semis': 2.5,
    'industry_biotech': 4.0,
    'industry_regional_banks': 5.0,
    'industry_retail': 5.0,
    'industry_aerospace_defense': 4.0,
    'industry_transport': 4.0,
    # Factor / style
    'factor_momentum': 3.0,
    'factor_quality': 3.0,
    'factor_value': 3.0,
    'factor_minvol': 3.0,
    'factor_dividend_growth': 3.0,
    'factor_dividend': 3.0,
    'style_growth': 2.5,
    'style_value': 2.5,
    # International
    'international_dev': 3.0,
    'international_em': 5.0,
    'country_japan': 5.0,
    'country_brazil': 6.0,
    'country_china': 6.0,
    'country_india': 6.0,
    'region_europe': 4.0,
    'theme_innovation': 5.0,
    # Commodities
    'gold': 2.0,
    'silver': 3.0,
    'broad_commodities': 4.0,
    'oil': 4.0,
    'natural_gas': 8.0,
    # FX
    'usd': 3.0,
    'eur': 3.0,
    # Volatility
    'volatility': 8.0,
}

# Fallback by asset_class when sector not found
_ASSET_CLASS_DEFAULTS = {
    'equity': 3.0,
    'bond': 2.5,
    'commodity': 4.0,
    'fx': 3.0,
    'vol': 8.0,
}

# Random slippage range (basis points, uniform).  Applied additively.
_SLIPPAGE_RANGE_BPS = 2.0


def get_half_spread_bps(sector: str, asset_class: str = 'equity') -> float:
    """Look up the half-spread in basis points for a given sector/asset_class."""
    if sector in _SPREAD_BPS:
        return _SPREAD_BPS[sector]
    return _ASSET_CLASS_DEFAULTS.get(asset_class, 3.0)


def apply_transaction_costs(
    price: float,
    action: str,
    sector: str = 'broad',
    asset_class: str = 'equity',
) -> Tuple[float, float]:
    """Apply bid-ask spread and slippage to a trade price.

    Args:
        price: Raw market price.
        action: 'BUY' or 'SELL' (or 'REDUCE', treated as SELL).
        sector: Sector from universe.csv.
        asset_class: Asset class from universe.csv.

    Returns:
        (fill_price, total_cost_bps) where total_cost_bps is the signed
        cost in basis points (always positive = cost to trader).
    """
    half_spread_bps = get_half_spread_bps(sector, asset_class)

    # Random slippage: uniform in [-range, +range], but biased against trader
    # (adds to cost on average by using abs for the adverse component)
    slippage_bps = random.uniform(-_SLIPPAGE_RANGE_BPS, _SLIPPAGE_RANGE_BPS)

    if action == 'BUY':
        # Buyer pays more: spread + slippage
        total_bps = half_spread_bps + slippage_bps
        fill_price = price * (1 + total_bps / 10000)
    else:
        # Seller receives less: spread - slippage (slippage can help or hurt)
        total_bps = half_spread_bps - slippage_bps
        fill_price = price * (1 - total_bps / 10000)

    # Cost to trader is always the absolute deviation from market price
    actual_cost_bps = abs(fill_price - price) / price * 10000

    return fill_price, actual_cost_bps
