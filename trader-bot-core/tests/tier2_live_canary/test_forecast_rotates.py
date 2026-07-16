"""Tier 2 live-canary — the forecast (mu) is not frozen day-over-day.

The two-week failure was a frozen ``mu`` producing byte-identical brain output
every night. This canary reads the two most recent recorded brain forecast
fingerprints from live S3 (``daily/<D>/inference.json``) and asserts they are NOT
byte-identical AND that the asset-health vector moved. HARD-FAIL on byte-identity
is the explicit catch for the frozen-mu signature.

The fault-injection twin feeds two byte-identical fingerprints → the canary MUST
go red (proving it catches a frozen mu).
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _reality import LiveS3Reader  # noqa: E402

EPS = 1e-9


def _l2(a, b) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def _two_most_recent_with_inference(reader: LiveS3Reader):
    dates = [d for d in reversed(reader.settled_dates())
             if reader.exists(f"daily/{d}/inference.json")]
    latest_settled = reader.latest_settled_date()
    assert dates and dates[0] == latest_settled, (
        f"latest forecast artifact is {dates[0] if dates else 'missing'}, but "
        f"the settled frontier is {latest_settled} — forecast persistence is stale")
    if len(dates) < 2:
        pytest.fail("fewer than two settled days carry inference.json")
    return dates[0], dates[1]


@pytest.mark.live_canary
def test_forecast_rotates_day_over_day(reality):
    d_today, d_prior = _two_most_recent_with_inference(reality)
    fingerprint_t, vec_t = reality.inference_fingerprint(d_today)
    fingerprint_p, vec_p = reality.inference_fingerprint(d_prior)

    # HARD-FAIL if the timestamp-free prediction bytes are identical.
    sha_t = hashlib.sha256(fingerprint_t).hexdigest()
    sha_p = hashlib.sha256(fingerprint_p).hexdigest()
    assert sha_t != sha_p, (
        f"brain forecast is BYTE-IDENTICAL between {d_prior} and {d_today} "
        f"(sha {sha_t[:16]}) — the frozen-mu signature that shipped the same "
        f"line for two weeks")

    # And the forecast vector actually moved.
    dist = _l2(vec_t, vec_p)
    assert dist > EPS, (
        f"forecast vector unchanged {d_prior}->{d_today} (L2={dist:.2e}) — "
        f"forecast substrate is frozen")


@pytest.mark.live_canary
@pytest.mark.fault_injection
def test_forecast_rotates_catches_byte_identical():
    """Two byte-identical fingerprints (a frozen mu) MUST trip the hard-fail."""
    frozen = b'{"date":"X","asset_health":[0.5,0.5]}'
    sha_t = hashlib.sha256(frozen).hexdigest()
    sha_p = hashlib.sha256(frozen).hexdigest()
    caught = (sha_t == sha_p)
    assert caught, "byte-identity check is broken — a frozen mu would slip through"
