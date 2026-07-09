"""The native two-stage brain engine: Select then Allocate, one native pipeline.

``run_engine(...)`` runs Stage-1 selection and Stage-2 allocation, emits
``trade_intents.json`` in the exact schema the morning executor already consumes
(zero morning-path change), runs the closed-loop parity controller, appends the
nightly parity residual ledger, and exposes the invariant self-check gate that
the nightly run uses (and that PKT-TB-009 proves exhaustively).

This module is the engine's existence (PKT-TB-008). Wiring it into the live
nightly Lambda and branding its line "New Brain" is gated to PKT-TB-009 green +
PKT-TB-012 -- this module brands nothing.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from .allocation import allocate
from .contracts import (
    AllocationResult,
    ForecastBundle,
    LotPolicy,
    PortfolioState,
    SelectionParams,
    SelectionResult,
    SizingParams,
)
from .parity import ParityController, ParityLedger
from .selection import select

EXPIRES_AFTER_DAYS = 3


@dataclass(frozen=True)
class EngineOutput:
    trade_intents: dict          # canonical schema, ready to write to daily/<D>/trade_intents.json
    selection: SelectionResult
    allocation: AllocationResult
    parity_ledger_entry: dict
    corrected_gross_frac: float


def build_trade_intents(
    f: ForecastBundle,
    sel: SelectionResult,
    alloc: AllocationResult,
    now_iso: Optional[str] = None,
) -> dict:
    """Assemble the canonical ``trade_intents.json`` payload.

    Keys mirror ``src/handler.py`` exactly: generated_date / generated_timestamp /
    regime / actions / buy_candidates / expert_metrics / expires_after_days.
    """
    buy_candidates = [
        {
            "symbol": sym,
            "score": float(f.mu_M1.get(sym, 0.0)),
            "health": float(f.health.get(sym, 0.0)),
            "tier": sel.tier.get(sym, ""),
            "w_target": float(sel.w_target.get(sym, 0.0)),
            "vol_bucket": f.vol_bucket.get(sym, "med"),
        }
        for sym in sel.ordered
    ]
    return {
        "generated_date": f.date,
        "generated_timestamp": now_iso if now_iso is not None else "",
        "regime": f.regime_label,
        "actions": alloc.intents,
        "buy_candidates": buy_candidates,
        "expert_metrics": {
            "engine": "native_two_stage",
            "theta_sel_hash": sel.theta_sel_hash,
            "kappa": alloc.kappa,
            "held_symbols": sorted(alloc.held_symbols),
            "lot_infeasible": sorted(alloc.lot_infeasible),
            "parity": dict(alloc.parity_record),
        },
        "expires_after_days": EXPIRES_AFTER_DAYS,
    }


def run_engine(
    f: ForecastBundle,
    theta_sel: SelectionParams,
    theta_size: SizingParams,
    portfolio: PortfolioState,
    parity_ledger: Optional[ParityLedger] = None,
    now_iso: Optional[str] = None,
    incumbent_intents: Optional[dict] = None,
    out_dir: Optional[Path] = None,
) -> EngineOutput:
    """Run the full pipeline for one decision date.

    If ``out_dir`` is given, writes ``trade_intents.json`` (and, when provided,
    ``trade_intents.incumbent.json`` as the ledger reference) under it.
    """
    # Stage 1 -- selection (no dollar read; enforced inside select()).
    sel = select(f, theta_sel)

    # Closed-loop parity correction on the book-level gross target.
    controller = ParityController(gain=theta_size.parity_gain)
    realized_prev = parity_ledger.last_realized_gross_frac() if parity_ledger else None
    regime_mult = float(theta_size.regime_exposure_multiplier.get(f.regime_label, 1.0))
    target_star = float(theta_size.gross_target) * regime_mult
    corrected_gross = controller.corrected_target(target_star, realized_prev)

    # Stage 2 -- allocation within the fixed set.
    alloc = allocate(
        sel, f, theta_size, portfolio, theta_sel.lot_policy,
        parity_target_gross=corrected_gross,
    )

    # Nightly parity residual ledger (built even without a persistent ledger).
    if parity_ledger is not None:
        parity_entry = parity_ledger.append(alloc.parity_record)
    else:
        from .parity import build_parity_ledger_entry
        parity_entry = build_parity_ledger_entry(alloc.parity_record)

    trade_intents = build_trade_intents(f, sel, alloc, now_iso=now_iso)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "trade_intents.json").open("w", encoding="utf-8") as fh:
            json.dump(trade_intents, fh, indent=1, sort_keys=False)
        if incumbent_intents is not None:
            with (out_dir / "trade_intents.incumbent.json").open("w", encoding="utf-8") as fh:
                json.dump(incumbent_intents, fh, indent=1, sort_keys=False)

    return EngineOutput(
        trade_intents=trade_intents,
        selection=sel,
        allocation=alloc,
        parity_ledger_entry=parity_entry,
        corrected_gross_frac=corrected_gross,
    )


# --------------------------------------------------------------- invariant gate
def default_theta_size_grid() -> List[SizingParams]:
    """A small Cartesian theta_size grid for the nightly self-check gate.

    PKT-TB-009 proves the invariant exhaustively (>=30 fixtures x the full grid);
    this is the lightweight gate the nightly run calls before any forward write.
    """
    grid: List[SizingParams] = []
    for mpw, mcw, gt, crp in itertools.product(
        (0.05, 0.10, 0.20, 0.30, 0.50),
        (0.10, 0.35, 1.0),
        (0.5, 1.0, 1.25),
        (0.0, 0.10, 0.50),
    ):
        grid.append(SizingParams(
            gross_target=gt,
            max_position_weight=mpw,
            max_cluster_weight=mcw,
            cash_reserve_pct=crp,
        ))
    return grid


def held_symbols_invariant(
    f: ForecastBundle,
    theta_sel: SelectionParams,
    portfolio: PortfolioState,
    theta_size_grid: Optional[Sequence[SizingParams]] = None,
) -> bool:
    """True iff held_symbols is constant across the sizing grid for fixed (f, theta_sel).

    This is the nightly self-check gate. A False result must ABORT the night
    (never degrade silently) per the design's abort-never-degrade rule.
    """
    grid = list(theta_size_grid) if theta_size_grid is not None else default_theta_size_grid()
    sel = select(f, theta_sel)
    reference: Optional[frozenset] = None
    for theta_size in grid:
        alloc = allocate(sel, f, theta_size, portfolio, theta_sel.lot_policy)
        if reference is None:
            reference = alloc.held_symbols
        elif alloc.held_symbols != reference:
            return False
    return True
