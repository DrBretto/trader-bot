"""PKT-TB-007 — ORB-1 genome (BUILD_SPEC_007 §4.1; wave-1 expression subset).

12-gene genome. Genes exist ONLY for shipped organs (organ_trust keys define the
shipped directional/damp roster the adapter listens to). B0 = the zero point:
tilt_gain 0.0 => the adapter is structurally neutral (returns the chassis
intents object unchanged — verified by B0-EXPR).

FIXED, never genes (§4.1): T_max, parity bounds/eps_sigma, tiers/masks,
ballast 0, VIXY exclusion. Those constants live in tilt_adapter.py.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# gene -> (lo, hi, B0)   (§4.1 table, ranges in natural units)
GENE_RANGES = {
    "tilt_gain": (0.0, 1.0, 0.0),
    "conviction_temp": (0.5, 2.0, 1.0),
    "dead_zone": (0.0, 0.3, 0.1),
    "disp_gain": (0.0, 2.0, 0.5),
    "cap_core": (0.0, 0.025, 0.0125),
    "cap_conditional": (0.0, 0.0125, 0.00625),
    "defensive_fraction": (0.0, 0.5, 0.25),
    "event_damp_strength": (0.0, 1.0, 0.0),
}
ORGAN_TRUST_RANGE = (-2.0, 2.0, 0.0)
ALLOWED_ORGANS = ("M1", "M2", "M4", "M5")
DIRECTIONAL_ORGANS = ("M1", "M2", "M5")


@dataclass
class Genome007:
    """organ_trust: logit listen-weights, key present iff the organ ships."""
    organ_trust: Dict[str, float] = field(default_factory=dict)
    tilt_gain: float = 0.0
    conviction_temp: float = 1.0
    dead_zone: float = 0.1
    disp_gain: float = 0.5
    cap_core: float = 0.0125
    cap_conditional: float = 0.00625
    defensive_fraction: float = 0.25
    event_damp_strength: float = 0.0

    def __post_init__(self):
        for k, v in self.organ_trust.items():
            if k not in ALLOWED_ORGANS:
                raise ValueError(f"unknown organ {k!r} in organ_trust")
            lo, hi, _ = ORGAN_TRUST_RANGE
            if not (lo <= float(v) <= hi):
                raise ValueError(f"organ_trust[{k}]={v} outside [{lo},{hi}]")
        for g, (lo, hi, _) in GENE_RANGES.items():
            v = float(getattr(self, g))
            if not (lo <= v <= hi):
                raise ValueError(f"gene {g}={v} outside [{lo},{hi}]")

    # ------------------------------------------------------------- construction
    @classmethod
    def b0(cls, organs: tuple = ()) -> "Genome007":
        """The zero point. tilt_gain 0 => structurally neutral regardless of
        which organs ship."""
        return cls(organ_trust={o: ORGAN_TRUST_RANGE[2] for o in organs})

    @classmethod
    def from_dict(cls, d: dict) -> "Genome007":
        d = dict(d)
        trust = {k: float(v) for k, v in (d.pop("organ_trust", {}) or {}).items()}
        kwargs = {g: float(d[g]) for g in GENE_RANGES if g in d}
        unknown = set(d) - set(GENE_RANGES)
        if unknown:
            raise ValueError(f"unknown genes {sorted(unknown)}")
        return cls(organ_trust=trust, **kwargs)

    @classmethod
    def from_json(cls, path: Path) -> "Genome007":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------- export
    def to_dict(self) -> dict:
        out = {g: float(getattr(self, g)) for g in GENE_RANGES}
        out["organ_trust"] = {k: float(v) for k, v in sorted(self.organ_trust.items())}
        return out

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True)
                              .encode()).hexdigest()[:12]

    @property
    def shipped_directional(self) -> list:
        return [o for o in DIRECTIONAL_ORGANS if o in self.organ_trust]

    @property
    def is_b0(self) -> bool:
        ref = Genome007.b0(tuple(self.organ_trust))
        return self.to_dict() == ref.to_dict()
