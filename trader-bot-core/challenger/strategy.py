"""Strategy container for the ported ORB-1 tilt (first-class port of the two
dataclasses ``tilt_adapter`` consumes from
``src/utils/three_line_replay/strategies.py`` — VERBATIM).

``tilt.make_tilt_strategy`` returns a ``Strategy`` whose ``post_decision`` takes a
``StrategyContext`` and the incumbent's intents and returns the tilted intents.
The unrelated helper strategies (topup / fragility-relax / compose) in the legacy
module are NOT carried — the challenger uses only the tilt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


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
