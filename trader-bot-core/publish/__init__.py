"""publish/ — the non-destructive publish gates + the ONE challenger mechanism.

  * ``dashboard.py``  — the framework invariant-3 gates extracted from the
    1181-line ``src/steps/publish_artifacts.py``: parity-or-hold
    (``verify_ledger_or_hold``) + never-shrink (``assert_publish_not_shrunk``, the
    publish analogue of the ledger's ``_assert_cache_not_shrunk``) + the ledger-fold
    line build path (``build_line_surface``), composed as the non-destructive
    ``publish_line``. A degraded or empty publish is STRUCTURALLY unable to overwrite
    a populated line.
  * ``challenger.py`` — the ONE challenger mechanism (M1-conviction tilt), extracted
    verbatim from ``replay/driver.py`` so it has a single canonical home. The
    UNCONFIRMED second mechanism (``src/brain/tilt_live.py``) is deliberately NOT
    carried into the clean core.

Dependency-light so it lives on the thin, non-inference publish branch (JSON + an
S3 client; ``challenger.py`` adds only numpy, already a spine dep).
"""
