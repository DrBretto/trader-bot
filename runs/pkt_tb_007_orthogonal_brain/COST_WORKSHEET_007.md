# COST_WORKSHEET_007 — deployed shape + Phase C/D actuals (PKT-TB-007)

Envelope: ≤$10/mo target, hard fail ~$15/mo (packet-standard). The deployed shape is
**TB-006's worksheet carried forward** (≈$9.5/mo absolute, +$0.21/mo marginal vs the
account's ~$9/mo of record) **plus the ORB-1 tilt — verified to add nothing material**:

## 1. Deployed monthly worksheet (projected)

| Line | TB-007 change vs TB-006 shape | Marginal $/mo | Absolute $/mo |
|---|---|---|---|
| Lambda night | organ precompute (M1 CAST inference + M2 GBM + M3 HAR + M4 elastic net + M5 rule + disp/p_exceed assembly) writes one ~8 KB organ-inputs JSON; measured locally at seconds of CPU on cached stores — rides inside the existing ~175 s/night envelope (<+10 s) | +$0.00–0.01 | $6.20 (carry) |
| Tilt adapter at decision time | pure numpy on the precomputed file (rank-z, sigmoid, ≤16-name projection, ≤5 iterations) — milliseconds; **B0 shipped ⇒ the adapter returns intents unchanged in production** | +$0.00 | incl. |
| Bedrock Haiku | **no new calls** — LLM retired from the decision path; the $0.14/mo monitoring emission is the TB-006 line, unchanged | +$0.00 | $0.14 |
| S3 storage + requests | +~1 MB/mo organ-inputs JSONs + expression logs | +$0.00–0.01 | $1.06 |
| ECR / Secrets / CloudWatch / SNS / transfer | unchanged (no new deps: torch already in image) | +$0.00 | $2.12 + $0.12 |
| **TOTAL** | | **≈ +$0.21/mo marginal (TB-006 carry; tilt adds ≤$0.02)** | **≈ $9.5/mo absolute — inside the ≤$10 target** |

Monthly organ retraining stays on the operator's Mac (launchd): $0 AWS. Note the honest
production state: with B0 shipped the only deployed deltas are the measurement program
(organ files + expression logs); the tilt itself is structurally neutral until a future
packet certifies a fitness instrument.

## 2. This run's actual one-time spend (Phase C build + Phase D battery)

| Item | Actual |
|---|---|
| **Bedrock this run** | **$0 — confirmed from the spend ledger**: TB-006's `bedrock_spend.jsonl` cumulative stands at **$2.873** (= TB-006's closed worksheet figure, last entry date_scored 2026-01-28); TB-007 wrote no spend ledger and contains **no Bedrock client code** (repo grep: boto3 usage is S3-only — extend_cache.py, list_s3_daily.py, surrogate_007.py); the LLM falsifier consumed the TB-006 cached `llm_features.parquet` read-only |
| Phase D battery compute | **37.2 s wall-clock** for all 9 replay arms (separate OS processes) + ~5 s E2 subset reads + evidence assembly; Mac CPU only — **≈ $0** |
| Phase C compute (organs, gates, anchor, smoke EA) | Mac CPU only (heaviest: CAST 2-seed training, minutes); $0 AWS |
| GDELT / OHLCV / market data | $0 cash — TB-006 caches read-only; no new pulls |
| S3 / AWS resources created | **none** (offline disk-cache replays; s3_client=None) |
| Local storage footprint (deletable) | store/ ≈ 200 MB (panel, nightly_007, models_out_007) + runs_battery_007 ≈ 25 MB |

## 3. Reductions taken (printed on the final line)

- **PERM-DESC arm cut** — reported not-run (chair adjudication 3); its 1-day timebox was
  never opened; the pre-committed forfeit sentence prints in BAKEOFF §5.
- **EA production sequence not run** (no certified fitness instrument) — saves the
  ≤1,900-genome evaluation budget and the 6-rotation re-derivation cycle entirely;
  ea_cycles 0/3.
- **$0 new Bedrock** (cap was $0; held).
- Battery executed at 10/18 replay arms, 0/9 retrains — under every cap.
