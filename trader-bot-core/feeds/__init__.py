"""feeds — external data sources coerced to the trader-bot-core contracts.

prices.py    Yahoo v8 chart raw endpoint PRIMARY; yfinance/AV/stooq fallbacks.
gdelt.py     GDELT daily aggregates + G1-G5 feature panel, WRITABLE store (RO-FS
             fix). This is THE gdelt source for the clean feature path. The legacy
             ``src/steps/ingest_gdelt.py`` (wrong-URL, cosmetic-only zeros —
             asserts ``gdelt_doc_count == 0``) is RETIRED from the clean path: it
             is never imported here and must not feed clean-folder features. It is
             not deleted (that is P9 cutover), it is only cut out of this path.
contracts.py The OHLCVBar dtype boundary shared with P2 (gdelt) and store.extend().
"""
