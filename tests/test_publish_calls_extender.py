"""Durability guardrail for the three-line replay extension.

The corrected optimized-champion `value` line is produced by
`extend_dashboard` (src/utils/three_line_replay/extender.py). On 2026-06-06 we
discovered the production publish path had NO committed call site for it — the
deployed Lambda ran it only from working-tree code baked in via Dockerfile.lambda
`COPY src/`. A fresh checkout + rebuild would silently drop the extension and the
chart would fall back to the raw broker line.

These structural tests assert both publish functions call `extend_dashboard`, so
the regression cannot recur unnoticed: they survive `COPY src/` rebuilds and fail
CI the moment the call site is dropped.
"""
import inspect

from src.steps import publish_artifacts


def test_night_publish_wires_extender():
    src = inspect.getsource(publish_artifacts.run)
    assert "extend_dashboard(" in src, (
        "night publish (publish_artifacts.run) must call extend_dashboard — "
        "dropping it regresses the displayed line to the raw broker line"
    )


def test_morning_publish_wires_extender():
    src = inspect.getsource(publish_artifacts.publish_morning_artifacts)
    assert "extend_dashboard(" in src, (
        "morning publish (publish_morning_artifacts) must call extend_dashboard"
    )
