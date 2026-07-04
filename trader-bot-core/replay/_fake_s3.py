"""In-memory S3 stand-in for the KEEP-verbatim EquityLedger.

The equity ledger is S3-only in production (``src/canon/equity_ledger.py``). The
local governed invoke (and the test suite — ``tests/test_equity_ledger.py``'s
``FakeS3``) drives it against an in-memory dict that honours the two behaviours
the append-only / content-addressed guards depend on:

  * ``IfNoneMatch="*"`` write-once — a second put to an existing key raises a
    PreconditionFailed-shaped error (the idempotent-no-op path); and
  * ``list_objects_v2`` prefix listing over stored keys.

This is a stand-in for storage ONLY; it exercises the real ledger's append,
supersede, frontier gate, cache fold, and never-shrink guard unchanged.
"""
from __future__ import annotations

import io
from typing import Dict


class _PreconditionFailed(Exception):
    """Shaped so equity_ledger._put_leaf_write_once recognises 'already present'."""

    def __init__(self, msg: str = "PreconditionFailed"):
        super().__init__(msg)


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    def __init__(self):
        self.store: Dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        if IfNoneMatch == "*" and Key in self.store:
            raise _PreconditionFailed()
        self.store[Key] = Body if isinstance(Body, (bytes, bytearray)) else bytes(Body)
        return {"ETag": '"%d"' % len(self.store[Key])}

    def get_object(self, *, Bucket, Key):
        if Key not in self.store:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": _Body(self.store[Key])}

    def head_object(self, *, Bucket, Key):
        if Key not in self.store:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"ContentLength": len(self.store[Key])}

    def list_objects_v2(self, *, Bucket, Prefix="", ContinuationToken=None, Delimiter=None):
        keys = sorted(k for k in self.store if k.startswith(Prefix))
        if Delimiter:
            prefixes = set()
            contents = []
            for k in keys:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
                else:
                    contents.append({"Key": k})
            return {"CommonPrefixes": [{"Prefix": p} for p in sorted(prefixes)],
                    "Contents": contents, "IsTruncated": False}
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}
