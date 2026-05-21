"""Regression tests for the optimized-canon promotion inside three_line_replay.

If the mobile chart, mobile sticky bar, mobile hero metrics, or desktop
hero metrics ever silently fall back to the old hybrid line again, these
tests fail. They pin two contracts the extender must keep stable:

1. `equity_curve[i].value` == `equity_curve[i].optimized_value` on every row
   where the champion replay has a value. `value` is what the mobile chart
   and `metrics.total_value` derivations read.
2. `equity_curve[i].hybrid_value` is preserved on every row where the
   hybrid replay has a value, distinct from `value` on rows where the
   champion diverges. Desktop chart and tooltip read `hybrid_value`.

Plus the metadata contract:
- `metrics.canon_source` == 'optimized_champion'
- `timeline_correction.main_line_label` == 'Portfolio (optimized champion)'
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.three_line_replay import extender


HYBRID_MAP = {
    "2026-03-12": 102422.19,
    "2026-04-07": 102672.00,
    "2026-04-08": 103864.86,
    "2026-05-05": 107235.83,
    "2026-05-08": 108420.63,
}
PRE_MAP = {
    "2026-03-12": 102422.19,
    "2026-04-07": 102513.24,
    "2026-04-08": 103912.84,
    "2026-05-05": 107047.10,
    "2026-05-08": 108240.54,
}
CHAMP_MAP = {
    "2026-03-12": 102422.19,
    "2026-04-07": 102832.60,
    "2026-04-08": 104317.08,
    "2026-05-05": 111356.43,
    "2026-05-08": 113000.79,
}


def _fake_run_variant(cache, cfg, strategy, trading_dates, universe):
    """Return one of the three pinned variant results based on identity."""
    label = getattr(cfg, "label", None)
    if strategy is not None:
        m = CHAMP_MAP
        actions = [
            {"date": "2026-04-07", "symbol": "DBC", "action": "BUY", "shares": 74, "price": 29.47, "reason": "regime"},
            {"date": "2026-04-09", "symbol": "DBC", "action": "BUY", "shares": 20, "price": 29.85, "reason": "topup"},
        ]
        holdings = [
            {"symbol": "DBC", "shares": 94, "entry_price": 29.47, "close_price": 30.32, "entry_date": "2026-04-07"},
        ]
        final_value = m["2026-05-08"]
    elif label == "pre_hybrid":
        m = PRE_MAP
        actions = []
        holdings = []
        final_value = m["2026-05-08"]
    else:
        m = HYBRID_MAP
        actions = [
            {"date": "2026-04-07", "symbol": "DBC", "action": "BUY", "shares": 74, "price": 29.47, "reason": "regime"},
        ]
        holdings = [
            {"symbol": "DBC", "shares": 74, "entry_price": 29.47, "close_price": 30.32, "entry_date": "2026-04-07"},
        ]
        final_value = m["2026-05-08"]
    return {
        "date_value_map": dict(m),
        "actions": actions,
        "final_holdings": holdings,
        "final_cash": 47761.19,
        "final_value": final_value,
        "final_date": "2026-05-08",
    }


def _fake_load_variant_configs(cache):
    class Cfg:
        label = "hybrid"
    class PreCfg:
        label = "pre_hybrid"
    return Cfg(), PreCfg()


class FakeCache:
    def __init__(self, *args, **kwargs):
        # Match S3Cache(s3_client, bucket=...) without binding to its signature.
        self.universe_df = pd.DataFrame([{"symbol": "DBC"}])

    def list_daily_dates(self):
        return list(HYBRID_MAP.keys())

    def get_csv(self, key):
        return self.universe_df


def _seed_dash() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for d in HYBRID_MAP.keys():
        rows.append({
            "date": d,
            "value": HYBRID_MAP[d],
            "raw_value": HYBRID_MAP[d],
            "benchmark": 95000.0,
            "cumulative_external_cashflow": 0.0,
        })
    return {
        "snapshot": {"id": "stub", "date": "2026-05-08", "phase": "morning", "timestamp": "stub"},
        "metrics": {"total_value": HYBRID_MAP["2026-05-08"], "timestamp": "stub"},
        "equity_curve": rows,
        "drawdowns": [],
        "monthly_returns": [],
        "trades": [],
        "trade_summary": {},
        "holdings": [],
        "round_trips": [],
    }


class TestOptimizedCanonPromotion:
    def _run(self):
        dash = _seed_dash()
        with patch.object(extender, "S3Cache", FakeCache), \
             patch.object(extender, "load_variant_configs", _fake_load_variant_configs), \
             patch.object(extender, "run_variant", _fake_run_variant):
            return extender.extend_dashboard(s3_client=None, dash=dash)

    def test_value_field_carries_optimized_champion(self):
        """equity_curve[i].value must equal champion value on every patched row.

        Mobile chart reads `point.value`. If this regression-fails, mobile
        renders the hybrid line again.
        """
        out = self._run()
        for row in out["equity_curve"]:
            d = row["date"]
            if d in CHAMP_MAP:
                assert row["value"] == CHAMP_MAP[d], (
                    f"equity_curve[{d}].value={row['value']} != champion {CHAMP_MAP[d]}; "
                    f"canon promotion regressed"
                )

    def test_hybrid_value_preserved_per_row(self):
        """equity_curve[i].hybrid_value must carry the hybrid replay value.

        Desktop chart's dotted comparison line reads `hybrid_value`. Tooltip
        also reads it. If this regression-fails, hybrid comparison disappears.
        """
        out = self._run()
        rows_with_hybrid = [r for r in out["equity_curve"] if "hybrid_value" in r]
        assert len(rows_with_hybrid) >= len(HYBRID_MAP), "hybrid_value missing from rows"
        for row in rows_with_hybrid:
            d = row["date"]
            if d in HYBRID_MAP:
                assert row["hybrid_value"] == HYBRID_MAP[d]

    def test_metrics_total_value_uses_optimized(self):
        """metrics.total_value must equal the optimized champion endpoint.

        Both desktop and mobile hero `TOTAL VALUE` read `metrics.total_value`.
        """
        out = self._run()
        assert out["metrics"]["total_value"] == CHAMP_MAP["2026-05-08"]

    def test_canon_source_flag_present(self):
        """metrics.canon_source must be 'optimized_champion'.

        This is the machine-readable canon stamp the post-deploy verifier
        and any future consumer (alerts, monitoring) can read.
        """
        out = self._run()
        assert out["metrics"]["canon_source"] == "optimized_champion"

    def test_timeline_correction_labels_promoted(self):
        """timeline_correction must label the canon as the optimized champion."""
        out = self._run()
        tc = out["timeline_correction"]
        assert tc["main_line_label"] == "Portfolio (optimized champion)"
        assert tc["canon_source"] == "optimized_champion"
        assert tc["comparison_line_field"] == "hybrid_value"

    def test_optimized_value_field_still_emitted(self):
        """`optimized_value` must remain populated for backwards compat.

        Desktop chart's `optimizedValue ?? correctedValue` fallback relies
        on it. Removing this field would break the desktop chart.
        """
        out = self._run()
        for row in out["equity_curve"]:
            if row["date"] in CHAMP_MAP:
                assert row.get("optimized_value") == CHAMP_MAP[row["date"]]
