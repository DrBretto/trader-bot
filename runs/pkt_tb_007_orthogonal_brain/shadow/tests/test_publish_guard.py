"""PKT-TRADER-BOT-SHADOW-PUBLISH-SAFETY — the fail-loud never-overwrite guard.

Proves the shadow-publish path REFUSES to overwrite a currently-populated
challenger line with an empty or point-reduced one (the 2026-07-01 incident: a
frozen engine built an empty payload and wiped the dotted line off the
dashboard). A frozen/degraded run must ABORT+ALERT and leave the good line
intact — never publish over it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SHADOW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW))

import shadow_nightly as SN                                # noqa: E402


def _payload(n_a=3, n_u=3, n_live=3):
    """A minimal timeseries payload with the guarded series populated to n pts."""
    pt = ["2026-06-11", 100000.0]
    return {
        "shadow_A": [pt] * n_a, "shadow_B": [pt] * n_a,
        "shadow_I": [pt] * n_a, "shadow_R": [pt] * n_a,
        "shadow_F": [pt] * n_a, "shadow_E": [pt] * n_a,
        "shadow_U": [pt] * n_u,
        "live_line": [pt] * n_live,
    }


# --------------------------------------------------------------------------- #
# the pure guard predicate
# --------------------------------------------------------------------------- #
def test_populated_to_empty_is_refused():
    current = _payload(15, 15, 15)
    new = _payload(0, 0, 0)          # the exact incident: 15-pt line -> empty
    with pytest.raises(SN.DestructivePublishError):
        SN.assert_publish_not_destructive(current, new)


def test_populated_to_point_reduced_is_refused():
    current = _payload(15, 15, 15)
    new = _payload(13, 15, 15)       # fewer points on one series -> refused
    with pytest.raises(SN.DestructivePublishError):
        SN.assert_publish_not_destructive(current, new)


def test_forward_advance_is_allowed():
    current = _payload(15, 15, 15)
    new = _payload(16, 16, 16)       # the line advancing is fine
    SN.assert_publish_not_destructive(current, new)   # no raise


def test_same_length_is_allowed():
    current = _payload(15, 15, 15)
    SN.assert_publish_not_destructive(current, _payload(15, 15, 15))


def test_empty_current_has_nothing_to_protect():
    # first publish (or an already-wiped line) -> any write proceeds
    SN.assert_publish_not_destructive(_payload(0, 0, 0), _payload(15, 15, 15))
    SN.assert_publish_not_destructive(None, _payload(0, 0, 0))


# --------------------------------------------------------------------------- #
# the guard is WIRED INTO mirror_to_s3 BEFORE the S3 put (abort, don't publish)
# --------------------------------------------------------------------------- #
class _FrozenSource:
    """get_json returns the good populated line; put_json must never be reached
    when the incoming payload is destructive."""

    def __init__(self, current):
        self.current = current
        self.put_calls = []

    def get_json(self, key):
        return self.current

    def put_json(self, key, obj):
        self.put_calls.append(key)
        raise AssertionError("mirror_to_s3 published over the good line — the "
                             "guard failed to abort before put_json")


def test_mirror_to_s3_aborts_before_publishing_an_empty_payload():
    src = _FrozenSource(_payload(15, 15, 15))
    ctx = SimpleNamespace(source=src, logf=None)
    with pytest.raises(SN.DestructivePublishError):
        SN.mirror_to_s3(ctx, _payload(0, 0, 0))
    assert src.put_calls == [], "no S3 object may be written on a destructive run"
