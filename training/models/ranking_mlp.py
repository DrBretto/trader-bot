"""Simple MLP for cross-sectional asset ranking.

Takes per-asset feature vectors and predicts relative forward return rank.
This is the baseline ranking head — intentionally simple to establish whether
cross-sectional ranking adds value over health-score-only selection.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

RANKING_FEATURES = [
    'return_1d', 'return_5d', 'return_21d', 'return_63d',
    'vol_21d', 'vol_63d', 'drawdown_63d', 'trend_63d',
    'rel_strength_21d', 'rel_strength_63d',
]


class RankingMLP(nn.Module):
    """Simple feed-forward network for cross-sectional ranking."""

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (batch, 1) scores in [0, 1]."""
        return self.net(x)

    def predict_scores(
        self,
        features: Dict[str, float],
        normalization: Dict[str, Dict[str, float]],
        regime_label: str = '',
        asset_class: str = '',
        sector: str = '',
    ) -> float:
        """Predict ranking score for a single asset from a feature dict."""
        self.eval()
        vals = []
        for feat in RANKING_FEATURES:
            raw = features.get(feat, 0.0) or 0.0
            mean = normalization.get(feat, {}).get('mean', 0.0)
            std = normalization.get(feat, {}).get('std', 1.0)
            if std < 1e-8:
                std = 1.0
            vals.append((raw - mean) / std)

        # Add regime one-hot if model was trained with regime conditioning
        regime_classes = ['calm_uptrend', 'risk_on_trend', 'risk_off_trend', 'choppy', 'high_vol_panic']
        if f'regime_{regime_classes[0]}' in normalization:
            for rc in regime_classes:
                vals.append(1.0 if regime_label == rc else 0.0)

        # Add asset-class one-hot if model was trained with structural features
        ac_classes = ['equity', 'bond', 'commodity', 'fx', 'vol']
        if 'ac_equity' in normalization:
            for ac in ac_classes:
                vals.append(1.0 if asset_class == ac else 0.0)

        # Add sector-group one-hot
        sg_classes = [
            'eq_broad', 'eq_sector', 'eq_intl', 'eq_factor',
            'eq_industry', 'eq_style', 'eq_theme',
            'bond_treas', 'bond_credit', 'bond_other',
            'commodity', 'fx', 'vol',
        ]
        if 'sg_eq_broad' in normalization:
            _sector_to_group = {
                'broad': 'eq_broad', 'global': 'eq_broad',
                'volatility': 'vol', 'usd': 'fx', 'eur': 'fx',
                'gold': 'commodity', 'silver': 'commodity',
                'broad_commodities': 'commodity', 'oil': 'commodity',
                'natural_gas': 'commodity',
                'aggregate': 'bond_other', 'muni': 'bond_other',
            }
            sg = _sector_to_group.get(sector, '')
            if not sg:
                if sector.startswith('sector_'):
                    sg = 'eq_sector'
                elif sector.startswith('international_') or sector.startswith('country_') or sector.startswith('region_'):
                    sg = 'eq_intl'
                elif sector.startswith('factor_'):
                    sg = 'eq_factor'
                elif sector.startswith('industry_'):
                    sg = 'eq_industry'
                elif sector.startswith('style_'):
                    sg = 'eq_style'
                elif sector.startswith('theme_'):
                    sg = 'eq_theme'
                elif sector.startswith('treas_'):
                    sg = 'bond_treas'
                elif sector.startswith('credit_'):
                    sg = 'bond_credit'
                else:
                    sg = 'eq_broad'
            for s in sg_classes:
                vals.append(1.0 if sg == s else 0.0)

        with torch.no_grad():
            x = torch.tensor([vals], dtype=torch.float32)
            score = self.net(x).item()
        return score
