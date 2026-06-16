"""Frozen data contracts for the native two-stage brain.

The three input contracts encode the architectural spine (DESIGN_DOSSIER §2.1):

  f        -- ForecastBundle : the forecast inputs Stage-1 selection reads.
              NONE of its fields is a dollar / cash / NAV / cluster / cap value.
  theta_sel-- SelectionParams: hand-frozen, content-hashed selection params.
              Dimensionless / structural only -- no book-dollar value.
  theta_size- SizingParams   : ALL sizing parameters. Stage-2 only.

The split is not cosmetic: ``select(f, theta_sel)`` has no parameter through
which any sizing or book-dollar value can enter, so the selected set is
literally independent of ``theta_size`` by construction. ``assert_no_dollar_surface``
turns that into an enforced, testable invariant.

LotPolicy is the one min-order notional the design hoists to Stage-1: it is
frozen-by-hand and **theta_size-independent**, so the lot-infeasibility set it
produces in Stage-2 is byte-identical across every sizing grid point (C5).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

# Tokens that may never appear in a Stage-1 input field name. A book-dollar /
# cash / NAV / cluster / cap value entering Select would re-couple selection to
# sizing -- exactly the entanglement DISC-TRADER-BOT-SELECTION-SIZING-ENTANGLEMENT
# designs out. ``assert_no_dollar_surface`` enforces this at call time.
_FORBIDDEN_SELECTION_FIELD_TOKENS = (
    "dollar",
    "cash",
    "nav",
    "cluster",
    "cap",
    "gross",
    "position_weight",
    "notional",
    "min_order",
    "reserve",
)


def _canonical_hash(payload: object) -> str:
    """Stable sha256 over a JSON-canonicalised payload (sorted keys)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- f
@dataclass(frozen=True)
class ForecastBundle:
    """``f`` -- the forecast inputs Stage-1 reads. No dollar value lives here.

    Field shapes mirror the frozen organ-input contract used by the PKT-TB-007
    prototype (``organ_inputs_007.v1``): ``mu_M1`` is the raw per-symbol M1
    score, ``event_block`` is the M4 hard-exclusion flag (derived upstream from
    M4 exceedance probability), ``eligible`` folds in universe membership /
    tradability (coverage, ADV floor, leverage) so no raw dollar ADV is read here.
    """

    date: str
    mu_M1: Mapping[str, float]          # PRIMARY ordering key (raw per-symbol M1 score)
    health: Mapping[str, float]         # health_score in [0,1]; binary eligibility gate
    regime_label: str                   # one of the 5 regime labels; admissibility gate
    event_block: Mapping[str, bool]     # M4 hard exclusion flag
    idio_vol: Mapping[str, float]       # tiebreak only (idiosyncratic vol; not a book dollar)
    eligible: Mapping[str, bool]        # universe membership / tradability (sizing-independent)
    asset_class: Mapping[str, str]      # for regime admissibility
    vol_bucket: Mapping[str, str] = field(default_factory=dict)  # 'low'|'med'|'high' (display)

    def symbols(self) -> Tuple[str, ...]:
        return tuple(self.mu_M1.keys())


# ------------------------------------------------------------------- lot policy
@dataclass(frozen=True)
class LotPolicy:
    """theta_size-INDEPENDENT min-order notional, hoisted to Stage-1 (C5).

    A selected name is lot-infeasible iff, at the *reference* NAV (a frozen
    constant, NOT the live book NAV), its target weight cannot clear ``min_order``.
    Because both ``reference_nav`` and ``min_order`` are frozen and independent of
    ``theta_size``, the lot-infeasibility partition is a function of
    ``(f, theta_sel, lot_policy)`` only -- byte-identical across the sizing grid.
    """

    min_order: float = 250.0
    reference_nav: float = 100_000.0

    def reference_notional(self, w_target: float) -> float:
        return w_target * self.reference_nav

    def is_lot_feasible(self, w_target: float) -> bool:
        return self.reference_notional(w_target) >= self.min_order


# ------------------------------------------------------------------- theta_sel
@dataclass(frozen=True)
class SelectionParams:
    """``theta_sel`` -- hand-frozen, content-hashed. Dimensionless / structural.

    No field is a book-dollar / cash / NAV / cluster / cap value. ``lot_policy``
    is carried for provenance and content-hashing (it is frozen-by-hand at the
    same time as theta_sel) but is **never passed into ``select``** -- it is
    consumed only by Stage-2 allocation. This keeps Select's read surface clean.
    """

    N: int = 10
    h_min: float = 0.60                 # health binary eligibility gate
    core_fraction: float = 0.55         # fraction of N that is Core (top of mu_M1 rank)
    # regime -> admissible asset classes; a regime absent from the map admits all.
    regime_admissibility: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    # tiebreak after mu_M1 desc: lower idio_vol first, then symbol asc (frozen).
    tiebreak: Tuple[str, ...] = ("mu_M1_desc", "idio_vol_asc", "symbol_asc")
    lot_policy: LotPolicy = field(default_factory=LotPolicy)

    def content_hash(self) -> str:
        payload = asdict(self)
        # regime_admissibility values may be tuples -> already JSON-stable via default=str
        return _canonical_hash(payload)


# ------------------------------------------------------------------ theta_size
@dataclass(frozen=True)
class SizingParams:
    """``theta_size`` -- ALL sizing parameters. Stage-2 only; never reaches Select."""

    gross_target: float = 1.0
    max_position_weight: float = 0.20
    max_cluster_weight: float = 0.35
    cash_reserve_pct: float = 0.10
    # regime -> book-level gross multiplier (single scalar, never a per-name list-walk)
    regime_exposure_multiplier: Mapping[str, float] = field(default_factory=dict)
    parity_gain: float = 0.5            # closed-loop ex-post parity controller gain

    def content_hash(self) -> str:
        return _canonical_hash(asdict(self))


# --------------------------------------------------------------- portfolio state
@dataclass(frozen=True)
class PortfolioState:
    """Live book state -- read ONLY by Stage-2 allocation."""

    nav: float
    cash: float
    positions: Mapping[str, int]        # current shares held
    marks: Mapping[str, float]          # price per symbol
    cluster_of: Mapping[str, str] = field(default_factory=dict)
    beta: Mapping[str, float] = field(default_factory=dict)
    sigma: Mapping[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------- stage outputs
@dataclass(frozen=True)
class SelectionResult:
    """Immutable Stage-1 output -- the invariant subject."""

    selected_set: FrozenSet[str]
    ordered: Tuple[str, ...]
    tier: Mapping[str, str]             # symbol -> 'core' | 'satellite'
    w_target: Mapping[str, float]       # dimensionless, sum == 1 over selected_set
    forecast_meta: Mapping[str, object]
    theta_sel_hash: str


@dataclass(frozen=True)
class AllocationResult:
    """Immutable Stage-2 output."""

    intents: List[dict]                 # canonical morning-executor action dicts
    held_symbols: FrozenSet[str]        # selected_set minus the lot-infeasible set
    lot_infeasible: FrozenSet[str]      # theta_size-INDEPENDENT shrink set (logged distinctly)
    kappa: float                        # uniform cash/gross down-scale in (0,1]
    parity_record: Mapping[str, object]


def assert_no_dollar_surface(*contracts: object) -> None:
    """Raise if any contract exposes a book-dollar / cash / NAV / cluster / cap field.

    Called inside ``select`` so the 'Select reads no dollar value' guarantee is
    enforced at runtime, not merely documented. Mappings (``f`` fields) are
    inspected by their *field names*, not their values: a forecast vector keyed
    by symbol is fine; a field literally named ``nav``/``cash``/... is not.
    """
    for contract in contracts:
        if contract is None:
            continue
        try:
            fields = asdict(contract)
        except TypeError:  # not a dataclass instance
            continue
        for name in fields:
            lowered = name.lower()
            for token in _FORBIDDEN_SELECTION_FIELD_TOKENS:
                if token in lowered:
                    raise AssertionError(
                        f"Stage-1 selection input {type(contract).__name__}.{name} "
                        f"exposes a forbidden sizing/book-dollar surface ('{token}'). "
                        "Selection must read no dollar/cash/NAV/cluster/cap value."
                    )
