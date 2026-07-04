"""lines/ — the append-only, content-addressed equity LEDGER (clean-core home).

The confirmed-correct KEEP-code kernel, relocated VERBATIM from ``src/canon/``
(import-path edits only, no logic rewrite — this is what keeps it the
confirmed-correct kernel):

  * ``ledger.py``  (was ``equity_ledger.py``) — append-only, content-addressed;
    ``_assert_cache_not_shrunk`` never-shrink guard; pure ``fold_cache``.
  * ``append.py``  (was ``equity_append.py``) — line = anchor x canon return;
    frontier no-op guard.
  * ``line.py``    (was ``equity_line.py``)   — read-side fold + parity gate.
  * ``seed.py``    (was ``equity_seed.py``)   — one-time seed from the live chart.

The displayed equity line is a STORED fact (one write-once leaf per trading day,
appended at the frontier, never recomputed), not a recomputed view — so the old
"history moves every other day" failure class is structurally impossible here.

Dependency-light (JSON + an S3 client) so it lives on the thin publish branch.
It still reuses the already-tested write-once / content-addressing primitives in
``src.utils.corrections`` (an external dependency, not in this relocation's scope);
the clean-spine invoke puts the repo root on ``sys.path`` so that import resolves.
"""
