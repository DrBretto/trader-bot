"""Native two-stage brain engine (PKT-TB-008).

Public surface:

    select(f, theta_sel)                    -> SelectionResult   (Stage 1)
    allocate(sel, f, theta_size, portfolio, lot_policy) -> AllocationResult (Stage 2)
    run_engine(...)                         -> EngineOutput      (full pipeline)
    held_symbols_invariant(...)             -> bool              (nightly self-check gate)

Contracts: ForecastBundle (f), SelectionParams (theta_sel), SizingParams
(theta_size), LotPolicy, PortfolioState, SelectionResult, AllocationResult.
"""
from __future__ import annotations

from .allocation import allocate
from .contracts import (
    AllocationResult,
    ForecastBundle,
    LotPolicy,
    PortfolioState,
    SelectionParams,
    SelectionResult,
    SizingParams,
    assert_no_dollar_surface,
)
from .engine import (
    EngineOutput,
    build_trade_intents,
    default_theta_size_grid,
    held_symbols_invariant,
    run_engine,
)
from .parity import ParityController, ParityLedger, build_parity_ledger_entry
from .selection import select

__all__ = [
    "select",
    "allocate",
    "run_engine",
    "held_symbols_invariant",
    "default_theta_size_grid",
    "build_trade_intents",
    "EngineOutput",
    "ForecastBundle",
    "SelectionParams",
    "SizingParams",
    "LotPolicy",
    "PortfolioState",
    "SelectionResult",
    "AllocationResult",
    "ParityController",
    "ParityLedger",
    "build_parity_ledger_entry",
    "assert_no_dollar_surface",
]
