# GDELT seed store (frozen backfill aggregates)

`gdelt_cache/daily/<YYYYMMDD>.json` — the frozen GDELT daily aggregates
2015-02-18 .. **2026-06-10** produced by the PKT-TB-006 backfill. Read-only in
production (baked with the code; on Lambda this lives under the read-only
`/var/task`). The clean feed (`feeds/gdelt.py`) reads this seed and writes new
forward days to a **separate writable store** (`/tmp/gdelt_cache` on Lambda, or
`trader-bot-core/store/gdelt_cache/` locally) — never back into this seed.

## Not committed (git-ignored)

The bulk (`daily/`, `records/`, `manifest.jsonl`) is git-ignored — 97M of daily
aggregates, and the per-record `records/` extract is a 3.3G class LLM-funnel
artifact NOT used by the G1–G5 feature panel. The tracked artifacts are the code
(`feeds/gdelt.py`) and the dictionaries (`feeds/gdelt_dicts/`). Baking the seed
into a deploy image is a P9 (cutover) concern, out of scope for the
writable-store fix.

## Provenance / rebuild

Relocated from
`runs/pkt_tb_006_clean_sheet_brain/prototype/gdelt_cache/daily/` on 2026-07-03.
To rebuild from scratch:

    python -m trader_bot_core.feeds.gdelt forward --start 2015-02-18 --end 2026-06-10

To advance the forward store to the latest settled day (the nightly job):

    python -m trader_bot_core.feeds.gdelt forward     # seed_max+1 .. latest complete UTC day

To rebuild the feature panel (seed ∪ forward store):

    python -m trader_bot_core.feeds.gdelt build
