"""Tier 1 unit-fixture — the regime tilt deterministically re-ranks mu.

Data source: the real ``config/regime_compatibility.json``. Predicate: under
``choppy``, a bond with mu near-equal to an equity is boosted ABOVE it, because
the real deployed multipliers (bond 1.05 > equity 0.95) tilt the selection score
``mu × regime_score_mult``. The multipliers are read from the real config — not
planted. Salvages ``test_regime_chassis`` as the deterministic unit half of the
regime canary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import select  # noqa: E402
from engine.contracts import ForecastBundle, SelectionParams  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "regime_compatibility.json"


@pytest.mark.unit
def test_regime_rerank_invariant():
    table = json.loads(_CONFIG.read_text())
    choppy = table["choppy"]
    m_eq = float(choppy["equity"])
    m_bond = float(choppy["bond"])
    # the real config must actually tilt bond above equity for this test to mean
    # anything (guards against a silently-inert chassis).
    assert m_bond > m_eq, f"real config does not boost bond over equity under choppy ({m_bond} vs {m_eq})"

    syms = ["EQ", "BOND"]
    mu = {"EQ": 0.500, "BOND": 0.500}          # near-equal raw mu (bond a touch lower)
    mu["BOND"] = 0.495                          # equity slightly HIGHER pre-tilt...
    f = ForecastBundle(
        date="2026-07-02",
        mu_M1=mu,
        health={s: 0.9 for s in syms},
        regime_label="choppy",
        event_block={s: False for s in syms},
        idio_vol={"EQ": 0.10, "BOND": 0.10},
        eligible={s: True for s in syms},
        asset_class={"EQ": "equity", "BOND": "bond"},
        vol_bucket={s: "med" for s in syms},
        regime_score_mult={"EQ": m_eq, "BOND": m_bond},
    )
    # pre-tilt, EQ (0.500) > BOND (0.495). Post-tilt: EQ 0.500*0.95=0.475 <
    # BOND 0.495*1.05=0.520 — the regime chassis must FLIP the order.
    res = select(f, SelectionParams(N=2, h_min=0.6))
    order = list(res.ordered)
    assert order.index("BOND") < order.index("EQ"), (
        f"regime tilt did NOT re-rank: order {order} — chassis inert (bond should "
        f"lead equity once the choppy multipliers are applied)")
