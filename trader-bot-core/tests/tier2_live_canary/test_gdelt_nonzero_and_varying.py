"""Tier 2 live-canary — the brain's GDELT features are nonzero AND move
day-over-day.

The old suite's ONLY GDELT tests asserted zeros as correct (``gdelt_doc_count ==
0``), so a dead GDELT feed was the pass condition. This canary reads the brain's
real GDELT feature panel (the forward path's ``store/gdelt_cache/
gdelt_features.parquet``, rebuilt each run by ``inference.gdelt_forward``) and
asserts: (1) the latest day carries at least one nonzero G-feature, and (2) the
latest day's feature vector differs from the prior day's — a frozen/dead GDELT
feed makes both fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_FEATURES = Path(__file__).resolve().parents[2] / "store" / "gdelt_cache" / "gdelt_features.parquet"


def _g_columns(df):
    return [c for c in df.columns
            if c.startswith(("g1_", "g2_", "g3_", "g4_", "g5_"))
            and str(df[c].dtype).startswith(("float", "int"))]


@pytest.mark.live_canary
def test_gdelt_nonzero_and_varying():
    if not _FEATURES.exists():
        pytest.fail(f"brain GDELT feature panel absent: {_FEATURES} — GDELT ingest never ran")
    import pandas as pd
    df = pd.read_parquet(_FEATURES)
    datecol = "gdelt_date" if "gdelt_date" in df.columns else df.columns[0]
    df = df.sort_values(datecol).reset_index(drop=True)
    assert len(df) >= 2, f"GDELT panel has <2 days ({len(df)}) — cannot judge variation"

    gcols = _g_columns(df)
    assert gcols, "GDELT panel carries no G-feature columns — schema broken"

    latest = df.iloc[-1][gcols].astype(float)
    prior = df.iloc[-2][gcols].astype(float)

    # (1) nonzero: a dead feed is all-zero.
    nonzero = int((latest.abs() > 1e-12).sum())
    assert nonzero >= 1, (
        f"every GDELT feature is zero on the latest day ({df.iloc[-1][datecol]}) "
        f"— the dead-feed failure the old suite certified green")

    # (2) varying: a frozen feed is byte-identical day-over-day.
    changed = int((latest.sub(prior).abs() > 1e-12).sum())
    assert changed >= 1, (
        f"GDELT feature vector byte-identical {df.iloc[-2][datecol]} -> "
        f"{df.iloc[-1][datecol]} — frozen feed")
