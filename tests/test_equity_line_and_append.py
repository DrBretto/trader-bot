"""FP-08-3 line view + parity gate, and the append-one-point advance mechanism.

Runs against the same in-memory FakeS3 (IfNoneMatch write-once) the ledger unit
suite uses — no real S3, no production data.
"""
import json

import pytest

from src.canon.equity_ledger import CACHE_KEY, EquityLedger
from src.canon.equity_line import (
    build_line_view, load_line_view, read_cache_rows,
    assert_ledger_parity, assert_seed_parity_against_rendered, LedgerParityError,
)
from src.canon import equity_append


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class FakeS3:
    def __init__(self):
        self.store = {}
        self.daily = {}  # date -> portfolio_state dict

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        if IfNoneMatch == "*" and Key in self.store:
            raise Exception("PreconditionFailed: key exists")
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()
        return {}

    def get_object(self, Bucket, Key):
        if Key in self.store:
            return {"Body": _Body(self.store[Key])}
        if Key.startswith("daily/") and Key.endswith("/portfolio_state.json"):
            d = Key.split("/")[1]
            if d in self.daily:
                return {"Body": _Body(json.dumps(self.daily[d]).encode())}
        raise Exception("NoSuchKey")

    def list_objects_v2(self, Bucket, Prefix="", Delimiter=None, ContinuationToken=None):
        if Delimiter == "/" and Prefix == "daily/":
            return {"CommonPrefixes": [{"Prefix": f"daily/{d}/"} for d in sorted(self.daily)],
                    "IsTruncated": False}
        keys = sorted(k for k in self.store if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def _seed(led, rows):
    for d, v, b, c, seg in rows:
        led.append(date=d, value=v, benchmark=b, comparison=c, segment=seg,
                   model_id="FREEZE_ORB1@test", source="native_two_stage",
                   issued_by="op@example.com")


# --------------------------------------------------------------------------- #
# line view
# --------------------------------------------------------------------------- #
def test_line_view_reproduces_segments_and_metrics():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed(led, [
        ("2026-06-11", 114772.39, 100.0, None, "frozen_champion"),
        ("2026-06-12", 115000.0, 101.0, None, "incumbent"),
        ("2026-06-23", 117693.43, 107.0, 115874.68, "new_brain"),
    ])
    lv = load_line_view(s3)
    ec = lv["equity_curve"]
    assert [r["date"] for r in ec] == ["2026-06-11", "2026-06-12", "2026-06-23"]
    # segment markers reproduced from segment + value
    assert ec[0]["champion_frozen_value"] == 114772.39 and ec[0]["new_brain_value"] is None
    assert ec[1]["incumbent_value"] == 115000.0 and ec[1]["champion_frozen_value"] is None
    assert ec[-1]["new_brain_value"] == 117693.43 and ec[-1]["optimized_value"] == 117693.43
    assert lv["line_metrics"]["total_value"] == 117693.43
    assert lv["terminal"] == {"date": "2026-06-23", "value": 117693.43}


def test_parity_gate_holds_on_empty_and_on_pin_mismatch():
    s3 = FakeS3()
    empty = build_line_view([])
    with pytest.raises(LedgerParityError):
        assert_ledger_parity(empty)
    led = EquityLedger(s3)
    _seed(led, [("2026-06-24", 116090.23, 106.0, None, "new_brain")])
    lv = load_line_view(s3)
    assert_ledger_parity(lv, terminal_pin=116090.23)  # ok
    with pytest.raises(LedgerParityError):
        assert_ledger_parity(lv, terminal_pin=999999.99)  # drifted


def test_seed_parity_against_rendered_exact():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed(led, [("2026-06-24", 116090.23, 106.63, None, "new_brain")])
    lv = load_line_view(s3)
    rendered = [{"date": "2026-06-24", "value": 116090.23, "benchmark": 106.63}]
    assert_seed_parity_against_rendered(lv, rendered, terminal_pin=116090.23)
    with pytest.raises(LedgerParityError):
        assert_seed_parity_against_rendered(
            lv, [{"date": "2026-06-24", "value": 1.0, "benchmark": 106.63}],
            terminal_pin=116090.23)


# --------------------------------------------------------------------------- #
# append-one-point advance mechanism
# --------------------------------------------------------------------------- #
def test_append_advances_and_leaves_prior_byte_identical():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed(led, [
        ("2026-06-23", 117693.43, 107.0, None, "new_brain"),
        ("2026-06-24", 116090.23, 106.0, None, "new_brain"),
    ])
    # prior settled book marks (the RETURN ENGINE — raw ~$97k, never the line)
    s3.daily["2026-06-24"] = {"sim_book_value": 97000.0, "benchmark_value": 106.0}
    # snapshot all prior leaf bytes
    before = {k: v for k, v in s3.store.items() if k.startswith("canon/equity_ledger/points/")}

    # tonight: book up +1% (raw 97970), benchmark +0.5%
    portfolio_state = {"portfolio_value": 97970.0, "benchmark_value": 106.53}
    res = equity_append.append_settled_point_for_publish(
        s3, "2026-06-25", portfolio_state, issued_by="night@trader-bot")

    assert res["action"] == "append"
    # the displayed value is the STORED anchor (116090.23) x (1 + 1%) — NOT the raw 97970
    assert res["value"] == pytest.approx(116090.23 * 1.01, rel=1e-9)
    assert res["value"] != pytest.approx(97970.0)
    # every prior leaf is byte-identical (advance-without-revert)
    after = {k: v for k, v in s3.store.items() if k.startswith("canon/equity_ledger/points/")}
    for k, v in before.items():
        assert after[k] == v, f"prior leaf {k} changed — history reverted!"
    assert len(after) == len(before) + 1  # exactly one new leaf


def test_append_is_noop_at_frontier():
    s3 = FakeS3()
    led = EquityLedger(s3)
    _seed(led, [("2026-06-25", 116126.12, 107.0, None, "new_brain")])
    rows_before = read_cache_rows(s3)
    res = equity_append.append_settled_point_for_publish(
        s3, "2026-06-25", {"portfolio_value": 97000.0, "benchmark_value": 107.0})
    assert res["action"] == "noop"
    assert read_cache_rows(s3) == rows_before  # nothing changed


def test_append_skips_when_unseeded():
    s3 = FakeS3()
    res = equity_append.append_settled_point_for_publish(
        s3, "2026-06-25", {"portfolio_value": 97000.0})
    assert res["action"] == "skip"
