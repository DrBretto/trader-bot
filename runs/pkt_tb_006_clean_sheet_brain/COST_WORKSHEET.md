# COST_WORKSHEET — SYN-1 deployed shape + Phase C actuals (PKT-TB-006)

**Mandatory per packet** (envelope: ≤$10/mo target, hard fail ~$15/mo). Two sections:
the projected monthly worksheet for the DEPLOYED design (itemized, FA-audited arithmetic
from ATTACK_FEASIBILITY.md + BUILD_SPEC §14), and the ACTUAL one-time Phase C spend
recorded call-by-call in `prototype/bedrock_spend.jsonl`.

## 1. Deployed monthly worksheet (projected; marginal vs the account's ~$9/mo of record)

| Line | Arithmetic | Marginal $/mo | Absolute $/mo |
|---|---|---|---|
| Lambda night | 110→~175 s/night (+25 s GDELT 8 zips, +40 s LLM funnel+call, +5 s SYN-1 inference, +5 s CBOE/FRED/COT CSVs) × 22 nights × 2.94 GB ≈ 11.3k GB-s | +$0.06 | $6.20 (conservative carry; ≈$0.19 at list price) |
| Lambda morning | unchanged | +$0.00 | incl. |
| Bedrock Haiku (LLM organ) | measured: ~10.0k in + 1.26k out tokens/night (687-call actual mean $0.00418/call) × 22 × 1.5 headroom | +$0.14 | $0.14 |
| S3 storage + requests | +~25 MB steady-state artifacts + ~70 req/day | +$0.01 | $1.06 |
| ECR | no new heavy deps (torch already in image) | +$0.00 | $0.12 |
| Secrets / CloudWatch / SNS | unchanged | +$0.00 | $2.00 |
| Data transfer | inbound free | +$0.00 | $0.01 |
| **TOTAL** | | **≈ +$0.21/mo marginal** | **≈ $9.5/mo absolute** — inside the ≤$10 target |

Monthly training stays on the operator's Mac (launchd): $0 AWS. The full final-grade
training cycle measured in this run completed in **≈80 s of compute** on cached stores
(cold-cache first run of a month ≈ 35-50 min incl. data refresh) — far inside the 1-2 h
window. Contingency: pinned Haiku model retirement ⇒ artifacts frozen, live organ falls
back to gpt-4o-mini at comparable cents; IAM widening to a newer Haiku-class ID is a
follow-on (≈$0.55/mo — still in envelope), never assumed.

## 2. Phase C actual one-time spend (recorded)

| Item | Actual |
|---|---|
| Bedrock calls (pilot 10d + Tier-1 132d + Tier-2 532d + re-scores + Phase A verification) | **687 calls, 6.876M input tok, 0.868M output tok = $2.873** (hard cap $3.10; ledger `prototype/bedrock_spend.jsonl`) |
| GDELT backfills (top-up 124 d + deep 4,131 d, 2→4 files/day) | $0 cash (HTTP from data.gdeltproject.org; ~3.4 GB local cache, Mac disk) |
| CBOE / FRED / COT / OHLCV pulls | $0 cash (free APIs; existing FRED key) |
| Local storage footprint (prototype, deletable) | gdelt_cache 3.4 GB + store 709 MB + cache 60 MB |
| S3 / AWS resources created | **none** (read-only against existing bucket; no deploys) |
| Training compute | Mac CPU only; $0 |

**Reductions taken (also on the final line):** LLM Tier-2 backfill window reduced to
2024-08-15→2026-01-28 (latest contiguous window fitting the $3.10 cap; full 2024-01 start
projected ≈$3.9). CAST OOF seeds 3→2, deploy ensemble 5→3, 8-day gradient mini-batches
(loss semantics preserved) — wall-clock rungs from the pre-authorized shrink ladder.
