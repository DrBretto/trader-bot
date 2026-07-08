"""Stage 2 -- ALLOCATION: size / scale within the fixed set.

``allocate(sel, f, theta_size, portfolio, lot_policy, parity_target=None) -> AllocationResult``

Allocation may scale toward cash but may **never drop a selected name** for cash,
cap, or budget reasons. The only legitimate set-shrink is ``min_order``
lot-infeasibility, decided against a theta_size-INDEPENDENT reference notional
(``lot_policy``), so the lot-infeasibility set is byte-identical across the whole
sizing grid. Caps are a feasibility projection (waterfill WITHIN S); cash/gross
is reconciled by a single uniform down-scale ``kappa`` that shrinks every
position proportionally and drops none. There is no shared cash counter and no
ranked-walk truncation site here -- that is the incumbent bug, designed out.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional

from .contracts import (
    AllocationResult,
    ForecastBundle,
    LotPolicy,
    PortfolioState,
    SelectionResult,
    SizingParams,
)

_PROJ_MAX_ITER = 64


def _waterfill(
    targets: Dict[str, float],
    cap: float,
    members: List[str],
) -> Dict[str, float]:
    """Clip each member to ``cap``; redistribute freed mass to *uncapped members
    only*, in proportion to their current allocation. Mass never leaves the set
    -- residual that cannot be absorbed (all members capped) stays as cash.
    """
    if cap <= 0 or not members:
        return dict(targets)
    alloc = dict(targets)
    for _ in range(_PROJ_MAX_ITER):
        over = {s: alloc[s] - cap for s in members if alloc[s] > cap + 1e-9}
        if not over:
            break
        freed = sum(over.values())
        for s in over:
            alloc[s] = cap
        uncapped = [s for s in members if alloc[s] < cap - 1e-9]
        base = sum(alloc[s] for s in uncapped)
        if not uncapped or base <= 0:
            break  # nowhere to put it -> residual becomes cash, set unchanged
        for s in uncapped:
            alloc[s] += freed * (alloc[s] / base)
    return alloc


def _cluster_waterfill(
    targets: Dict[str, float],
    cluster_cap: float,
    cluster_of: Mapping[str, str],
    members: List[str],
) -> Dict[str, float]:
    """Project per-cluster concentration <= cluster_cap, redistributing freed mass
    to uncapped clusters' members only. Stays within S.
    """
    if cluster_cap <= 0 or not members:
        return dict(targets)
    alloc = dict(targets)
    for _ in range(_PROJ_MAX_ITER):
        cluster_dollars: Dict[str, float] = {}
        for s in members:
            cluster_dollars.setdefault(cluster_of.get(s, s), 0.0)
            cluster_dollars[cluster_of.get(s, s)] += alloc[s]
        over_clusters = {c: d - cluster_cap for c, d in cluster_dollars.items() if d > cluster_cap + 1e-9}
        if not over_clusters:
            break
        freed = 0.0
        for c, excess in over_clusters.items():
            c_members = [s for s in members if cluster_of.get(s, s) == c]
            base = sum(alloc[s] for s in c_members)
            if base <= 0:
                continue
            for s in c_members:
                cut = excess * (alloc[s] / base)
                alloc[s] -= cut
                freed += cut
        # redistribute to members of under-cap clusters, proportionally, within S
        under = [
            s for s in members
            if cluster_dollars.get(cluster_of.get(s, s), 0.0) < cluster_cap - 1e-9
        ]
        base = sum(alloc[s] for s in under)
        if not under or base <= 0:
            break
        for s in under:
            alloc[s] += freed * (alloc[s] / base)
    return alloc


def _reconcile_action(symbol: str, target_shares: int, current_shares: int, price: float,
                      w_target: float, mu: float, health: float, vol_bucket: str) -> Optional[dict]:
    """Diff target vs current holding into a canonical morning-executor action.

    Schema matches what ``src/steps/morning_executor.py`` consumes byte-for-byte
    (action/symbol/shares/price/dollars + the BUY display fields).
    """
    delta = target_shares - current_shares
    if delta > 0:
        shares = delta
        return {
            "action": "BUY",
            "symbol": symbol,
            "shares": int(shares),
            "price": float(price),
            "dollars": float(round(shares * price, 2)),
            "weight": float(w_target),
            "reason": f"M1_{mu:.3f}_HEALTH_{health:.2f}",
            "score": float(mu),
            "health": float(health),
            "vol_bucket": vol_bucket,
        }
    if delta < 0:
        shares = -delta
        return {
            "action": "REDUCE",
            "symbol": symbol,
            "shares": int(shares),
            "reduce_shares": int(shares),
            "price": float(price),
            "dollars": float(round(shares * price, 2)),
            "reason": "ALLOC_DOWNSCALE",
        }
    # delta == 0 -> HOLD: still emit so set membership is explicit on the surface
    return {
        "action": "HOLD",
        "symbol": symbol,
        "shares": int(current_shares),
        "price": float(price),
        "dollars": float(round(current_shares * price, 2)),
        "reason": "ALLOC_HOLD",
    }


def allocate(
    sel: SelectionResult,
    f: ForecastBundle,
    theta_size: SizingParams,
    portfolio: PortfolioState,
    lot_policy: LotPolicy,
    parity_target_gross: Optional[float] = None,
) -> AllocationResult:
    """Size within the fixed set. Drops no name except logged lot-infeasibility."""
    members = list(sel.ordered)

    # --- the ONLY legitimate set-shrink: lot-infeasibility vs a theta_size-
    #     INDEPENDENT reference notional (hoisted to Stage-1). Decided BEFORE any
    #     theta_size-dependent scaling, so the set is byte-identical across the grid.
    lot_infeasible = frozenset(
        s for s in members if not lot_policy.is_lot_feasible(sel.w_target.get(s, 0.0))
    )
    held = [s for s in members if s not in lot_infeasible]
    held_set = frozenset(held)

    nav = float(portfolio.nav)

    # (1) book-level gross via a single scalar G = gross_target * regime_multiplier,
    #     optionally corrected by the closed-loop ex-post parity controller.
    regime_mult = float(theta_size.regime_exposure_multiplier.get(f.regime_label, 1.0))
    target_gross_frac = float(theta_size.gross_target) * regime_mult
    if parity_target_gross is not None:
        target_gross_frac = float(parity_target_gross)
    gross_budget = target_gross_frac * nav

    # (2) d_target over held names (re-normalise weights across held set only)
    w_sum = sum(sel.w_target.get(s, 0.0) for s in held)
    d_target = {
        s: (gross_budget * sel.w_target.get(s, 0.0) / w_sum if w_sum > 0 else 0.0)
        for s in held
    }

    # (3) caps as a feasibility projection (waterfill WITHIN held set)
    d_capped = _waterfill(d_target, theta_size.max_position_weight * nav, held)
    d_capped = _cluster_waterfill(
        d_capped, theta_size.max_cluster_weight * nav, portfolio.cluster_of, held
    )

    # (4) cash/gross reconciliation by a single uniform down-scale kappa in (0,1].
    #     Shrinks every position proportionally; drops NO name.
    investable = nav * (1.0 - float(theta_size.cash_reserve_pct))
    gross = sum(d_capped.values())
    kappa = 1.0
    if gross > investable and gross > 0:
        kappa = investable / gross
    d_final = {s: kappa * d_capped[s] for s in held}

    # integer-lot solve. Membership is fixed (held_set); shares are 'how much'.
    intents: List[dict] = []
    realized_gross = 0.0
    realized_beta_num = 0.0
    realized_sigma_num = 0.0
    for s in held:
        price = float(portfolio.marks.get(s, 0.0))
        target_shares = int(d_final[s] / price) if price > 0 else 0
        current_shares = int(portfolio.positions.get(s, 0))
        action = _reconcile_action(
            s,
            target_shares,
            current_shares,
            price,
            sel.w_target.get(s, 0.0),
            float(f.mu_M1.get(s, 0.0)),
            float(f.health.get(s, 0.0)),
            f.vol_bucket.get(s, "med"),
        )
        if action is not None:
            intents.append(action)
        held_dollars = target_shares * price
        realized_gross += held_dollars
        realized_beta_num += held_dollars * float(portfolio.beta.get(s, 1.0))
        realized_sigma_num += held_dollars * float(portfolio.sigma.get(s, 0.0))

    # exits: names currently held but no longer in the held set -> SELL to flat
    for s, shares in portfolio.positions.items():
        if s in held_set or int(shares) <= 0:
            continue
        price = float(portfolio.marks.get(s, 0.0))
        intents.append({
            "action": "SELL",
            "symbol": s,
            "shares": int(shares),
            "price": price,
            "dollars": float(round(shares * price, 2)),
            "reason": "NOT_SELECTED",
        })

    realized_gross_frac = realized_gross / nav if nav > 0 else 0.0
    lot_residual = sum(d_final[s] for s in held) - realized_gross  # un-lotted cash
    parity_record = {
        "date": f.date,
        "target_gross_frac": target_gross_frac,
        "realized_gross_frac": realized_gross_frac,
        "realized_beta": (realized_beta_num / realized_gross) if realized_gross > 0 else 0.0,
        "realized_sigma": (realized_sigma_num / realized_gross) if realized_gross > 0 else 0.0,
        "kappa": kappa,
        "lot_residual_dollars": float(round(lot_residual, 2)),
        "lot_infeasible": sorted(lot_infeasible),
        "n_held": len(held),
    }

    return AllocationResult(
        intents=intents,
        held_symbols=held_set,
        lot_infeasible=lot_infeasible,
        kappa=kappa,
        parity_record=parity_record,
    )
