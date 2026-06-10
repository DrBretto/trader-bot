# data_cache.pkl (not committed — 70 MB)

Pinned replay dataset: 194 aligned snapshots (2025-08-04 → 2026-06-09) of
s3://investment-system-data/daily/ artifacts + per-date context rows.
Rebuild deterministically with:

    .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/build_cache.py

Provenance (source, range, built_at) is recorded in every manifest under
../manifests/. NOTE: a rebuild after S3 backfill/changes may not be
byte-identical to the original cache; manifests pin what was actually used.
