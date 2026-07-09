"""Closed-loop ex-post parity controller + nightly residual ledger (the F8 fix).

F8 lesson: an open-loop ex-ante (gross, beta, sigma) projection still leaked
~4.4pp of realized gross, one-signed, because path divergence compounded. The
native brain removes the dominant leak structurally (one book, no incumbent arm
to diverge from); this module adds the second defence: a *closed-loop*
proportional correction that pulls persistent drift back instead of re-leaking
one-signed, plus a nightly ledger that records realized-vs-target so future
leaks are measured ex-post, never assumed away.

Controller law (per night t):

    target_t = clip( target* + (target* - realized_{t-1}) * gain , lo, hi )

A single scalar correction on the book-level gross target -- never a per-name
list-walk (that would re-enter the entanglement). ``realized_{t-1}`` is the
prior night's realized gross fraction from the parity ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ParityController:
    gain: float = 0.5
    lo: float = 0.0
    hi: float = 2.0

    def corrected_target(self, target_star: float, realized_prev: Optional[float]) -> float:
        """Proportional ex-post correction. First night (no prior) is uncorrected."""
        if realized_prev is None:
            return _clip(target_star, self.lo, self.hi)
        corrected = target_star + (target_star - realized_prev) * self.gain
        return _clip(corrected, self.lo, self.hi)


def build_parity_ledger_entry(parity_record: Mapping[str, object]) -> dict:
    """One nightly row: realized (gross, beta, sigma) vs target + lot residual.

    This is the F8 disclosure form -- the realized ex-post parity series that
    LIVE_PREREG pre-registers as a forward read.
    """
    target = float(parity_record.get("target_gross_frac", 0.0))
    realized = float(parity_record.get("realized_gross_frac", 0.0))
    return {
        "date": parity_record.get("date"),
        "target_gross_frac": round(target, 6),
        "realized_gross_frac": round(realized, 6),
        "gross_residual_frac": round(realized - target, 6),
        "realized_beta": round(float(parity_record.get("realized_beta", 0.0)), 6),
        "realized_sigma": round(float(parity_record.get("realized_sigma", 0.0)), 6),
        "kappa": round(float(parity_record.get("kappa", 1.0)), 6),
        "lot_residual_dollars": float(parity_record.get("lot_residual_dollars", 0.0)),
        "lot_infeasible": list(parity_record.get("lot_infeasible", [])),
        "n_held": int(parity_record.get("n_held", 0)),
    }


class ParityLedger:
    """Append-only nightly parity residual ledger (newline-delimited JSON)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self.entries: List[dict] = []

    def last_realized_gross_frac(self) -> Optional[float]:
        """Prior night's realized gross fraction -- the controller feedback term."""
        if not self.entries:
            if self.path is not None and self.path.exists():
                self._load()
            if not self.entries:
                return None
        return float(self.entries[-1]["realized_gross_frac"])

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            self.entries = [json.loads(line) for line in fh if line.strip()]

    def append(self, parity_record: Mapping[str, object]) -> dict:
        entry = build_parity_ledger_entry(parity_record)
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry
