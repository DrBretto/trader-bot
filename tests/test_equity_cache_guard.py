"""PKT-TRADER-BOT-SHADOW-PUBLISH-SAFETY — canon analogue of the publish guard.

By the same principle as the shadow-publish never-overwrite guard, the canon
cache projection (``equity_history.jsonl``) must never be overwritten with an
empty or point-reduced fold. A degraded/partial leaf listing (the canon analogue
of a frozen shadow run) must ABORT+ALERT and leave the populated line intact,
never silently wipe it.
"""
import pytest

from src.canon.equity_ledger import (
    CACHE_KEY,
    POINTS_PREFIX,
    DestructiveCacheError,
    EquityLedger,
)

# reuse the in-memory S3 stub + append helper the primitive suite already uses
from tests.test_equity_ledger import FakeS3, _append, _seed_three


def _cache_row_count(s3):
    body = s3.store.get(CACHE_KEY, b"")
    return sum(1 for line in body.decode().splitlines() if line.strip())


# --------------------------------------------------------------------------- #
# the pure never-shrink predicate
# --------------------------------------------------------------------------- #
def test_assert_cache_not_shrunk_refuses_empty_and_reduced():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)                          # populated cache: 3 rows
    assert _cache_row_count(s3) == 3

    with pytest.raises(DestructiveCacheError):
        led._assert_cache_not_shrunk(b"")     # populated -> empty
    with pytest.raises(DestructiveCacheError):
        led._assert_cache_not_shrunk(b'{"date":"x"}\n')   # populated -> 1 row


def test_assert_cache_not_shrunk_allows_growth_and_equal():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)
    led._assert_cache_not_shrunk(b'a\nb\nc\n')      # equal (3) — ok
    led._assert_cache_not_shrunk(b'a\nb\nc\nd\n')   # grown (4) — ok


def test_empty_cache_has_nothing_to_protect():
    s3 = FakeS3()
    led = EquityLedger(s3)                    # no cache yet
    led._assert_cache_not_shrunk(b"")         # nothing to protect — ok


# --------------------------------------------------------------------------- #
# WIRED: a degraded rebuild (partial leaf listing) is refused, cache untouched
# --------------------------------------------------------------------------- #
def test_rebuild_refuses_to_wipe_good_cache_on_partial_leaf_listing():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed_three(led)
    good_cache = s3.store[CACHE_KEY]
    assert _cache_row_count(s3) == 3

    # simulate a degraded read: two of the three point leaves vanish from the
    # listing (env fault), so the rebuilt fold would shrink 3 -> 1.
    point_keys = [k for k in list(s3.store) if k.startswith(POINTS_PREFIX)]
    for k in point_keys[1:]:
        del s3.store[k]

    with pytest.raises(DestructiveCacheError):
        led.rebuild(write=True)

    # the good cache was left byte-intact — nothing wiped.
    assert s3.store[CACHE_KEY] == good_cache
    assert _cache_row_count(s3) == 3
