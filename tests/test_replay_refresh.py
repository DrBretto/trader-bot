"""Regression locks for the TILT-canon promotion."""

import json

from app.shadow_publish import build_payload
from lines.line import build_line_view
from lines.replay_refresh import _cent_check, _grid_check, promoted_values


def test_native_replay_roles_are_swapped_for_promoted_storage():
    native = {
        "date": "2026-07-10",
        "value": 111_514.91,
        "benchmark": 106_398.45,
        "comparison": 114_966.55,
    }
    assert promoted_values(native) == {
        "value": 114_966.55,
        "benchmark": 106_398.45,
        "comparison": 111_514.91,
    }


def test_continuity_is_cent_exact_not_a_relative_tolerance():
    assert _cent_check("tilt", "2026-07-10", 114_966.552, 114_966.554)["ok"]
    assert not _cent_check("tilt", "2026-07-10", 114_966.55, 114_966.57)["ok"]


def test_existing_promoted_grid_cannot_skip_a_real_session():
    rows = [
        {"date": "2026-06-11"},
        {"date": "2026-06-12"},
        {"date": "2026-06-16"},
    ]
    check = _grid_check(
        rows,
        ["2026-06-12", "2026-06-15", "2026-06-16"],
        "2026-06-11",
    )
    assert not check["ok"]
    assert check["missing"] == ["2026-06-15"]


def test_line_metrics_follow_tilt_value_not_comparison():
    rows = [
        {"date": "2026-07-08", "value": 100.0, "benchmark": 100.0,
         "comparison": 100.0, "segment": "new_brain", "model_id": "x"},
        {"date": "2026-07-09", "value": 101.0, "benchmark": 99.0,
         "comparison": 50.0, "segment": "new_brain", "model_id": "x"},
        {"date": "2026-07-10", "value": 102.0, "benchmark": 98.0,
         "comparison": 200.0, "segment": "new_brain", "model_id": "x"},
    ]
    view = build_line_view(rows)
    assert view["line_metrics"]["total_value"] == 102.0
    assert view["line_metrics"]["max_drawdown"] == 0.0


def test_shadow_payload_mirrors_tilt_as_live_and_two_stage_as_shadow_a():
    rows = [
        {"date": "2026-06-11", "value": 100.0, "benchmark": 100.0,
         "comparison": 100.0},
        {"date": "2026-06-12", "value": 102.0, "benchmark": 101.0,
         "comparison": 99.0},
    ]
    cache = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    payload = build_payload({}, cache)
    assert payload["live_line"][-1] == ["2026-06-12", 102.0]
    assert payload["shadow_A"][-1] == ["2026-06-12", 99.0]
    assert "two-stage comparison" in payload["challenger_source"]
