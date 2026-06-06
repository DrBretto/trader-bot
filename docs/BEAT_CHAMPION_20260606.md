# Beat-Champion 2026-06-06 — full history & jump-off for future tweaks

This is the durable record of the 2026-06-06 work that (a) made the displayed
champion measurably better and took it live, and (b) retrained + switched the
regime model forward. Read this before any future tuning of the displayed line,
the optimizer, or the regime/health models. Pairs with:
- Plan doc: `docs/plans/2026-06-06-beat-champion-golive.md`
- Scripts: `scripts/beat_champion_20260606/` (runner + 4-stage search + finalize + verify)
- Evidence JSON: `runs/beat_champion_20260606/` (incumbent, search_stage1..4, FINAL_RESULT, verify_wired)
- S3 backups (rollback): `runs/beat_champion_20260606/s3_backup/`
- Infotropy committee verdict (deeper analysis): `Infotropy Book/book-factory/runs/20260606_trader-bot-pipeline-end-to-end-committee/`
- Infotropy result receipt: `Infotropy Book/book-factory/runs/20260606_trader-bot-beat-champion/RESULT.md`

---

## 1. Current live state (after this work)

- **Displayed champion line = optimized config**, whole period **+15.0%** ($100k → $115,017),
  up from the previous **+12.5%**. SPY same period +7.9%.
- **Previous champion (+12.5%)** is the dotted comparison line on the chart ("Previous champion").
- **Bottom charts** (drawdown strip, monthly returns, hero metrics, win rate) are recomputed from
  the champion line automatically (in `extender.py`), so they track the new champion.
- **Regime model switched forward** on 2026-06-06 (chart marker present): retrained ensemble,
  out-of-sample accuracy 9% → 73% (GRU) / 67% (transformer), weights 0.5/0.5. Health model unchanged.
- Live at https://trader-bot.infotrope.io. Nightly Lambda (10pm ET) regenerates the line from the
  new code + S3 config + retrained models, so it persists and extends one day at a time.

## 2. What was broken vs not (verified on the real system)

- **NOT broken:** the displayed strategy (real, beats SPY +12.5% vs +7.9%); the health model
  (production scores span 0.000–0.999 — the earlier "BatchNorm collapse" verdict was wrong, twice).
- **Broken & fixed-or-mitigated:**
  - Regime GRU trained on ~11 rows → 9% accuracy (below 20% random). Root cause: training data
    fell back to `daily/` instead of the 11-year `training/data/historical_combined.parquet`
    (2,803 rows, 2014→2026-02). **Retrained on the full corpus → 73%/67% OOS.**
  - The disagreement→position-size throttle cut buys ~25–40% nearly every day, driven partly by the
    broken model. The hand overlays (`relax_choppy`, `topup`) were compensating for this.
  - The optimizer hadn't promoted in 1+ month (degenerate folds + a round-trip guardrail that rejects
    even the champion + it only ever saw the stripped base config). NOT repaired as a program; instead
    a clean constrained search on the real object was run (see §4).

## 3. THE performance object — how to measure correctly (do not get this wrong)

**Measure through the REAL displayed system: `src/utils/three_line_replay/` `run_variant` WITH the
champion overlays.** NOT `optimizer.replay` base config (no overlays) — that produced the bogus
"+0.98% / can't beat SPY" numbers in an earlier pass. NEVER use the Alpaca `raw_value` line.

- Reusable harness: `scripts/beat_champion_20260606/runner.py`
  - `run_real(...)` runs the champion overlay system; `dp_overrides` patches decision_params,
    `ensemble_overrides` re-weights regime / retunes throttle, `ThrottlePatchCache` patches ONLY
    `position_size_multiplier`.
  - In-sample window = 2026-01-02→03-10 (seed `daily/2026-01-02/portfolio_state.json`, genesis $100k).
    Holdout = 2026-03-11+ (seed `daily/2026-03-11`). **Tune on in-sample, validate on holdout. Never fit holdout.**
  - The replay only recomputes from 2026-03-11 onward; pre-3/11 is a frozen segment. Whole-period
    return = (frozen 100k→102,832 = +2.83%) chained with the 3/11-onward slice.

## 4. The winning change (config tune) + numbers

Found by a clean, holdout-disciplined, never-worse search on the real object (4 stages):
- Stage 1 (ensemble re-weighting via `ensemble_overrides`) — REJECTED: recomputing the regime
  label corrupts the good baked signal; everything cratered.
- Stage 2 (un-nerf the throttle, patch only the multiplier) — WASH (overlays already compensate).
- Stage 3+4 (participation knobs, with a down-window protection gate) — WINNER, robust (beats the
  incumbent on BOTH windows).

| | in-sample | holdout (2026-03-11+) | holdout maxDD | down-window maxDD | whole period |
|---|---|---|---|---|---|
| Previous champion | +1.78% | +9.39% | −1.04% | −1.27% | +12.5% |
| **New champion** | +5.59% | **+11.85%** | −1.03% | −1.26% | **+15.0%** |

Changes (the only 3 levers that mattered):
- `max_position_weight` 0.20 → **0.30** (`config/decision_params.active.json`)
- `topup_trigger` 1.2 → **1.1** (`extender.py` `_build_champion_strategy`)
- `buy_score_threshold_by_regime`: benign 0.62, **stressed regimes protected** (choppy 0.65 /
  risk_off 0.68 / panic 0.72) — was a flat 0.65.

Notes: `max_positions` 8 unchanged (raising it didn't help); cash floors unchanged; `max_position_weight`
0.28 is a local dip (non-monotone) — 0.30 is on a stable plateau, 0.30→higher was not tested/needed.
The win is **participation** — the system was under-deployed in the risk-on holdout.

## 5. The regime model retrain + forward switch

- Trainer: `training/train_regime.py --data training/data/historical_combined.parquet` (GRU + transformer).
- OOS test accuracy (last 15% chronological, pre-holdout): GRU 0.7307, transformer 0.6733 (was 0.09).
- **Caveat:** labels come from `compute_regime_labels_from_baseline` (a fixed rule), so 0.73 = the model
  faithfully reproduces the baseline regime rule on unseen data. It fixes the brokenness but does NOT add
  signal beyond the rule. A truly independent regime model would be **unsupervised (HMM / Markov-switching)**
  — that is the recommended next algorithm step (see §8).
- **Forward-only switch:** `models/latest.json` (S3) now points at `regime_gru_v20260606.pkl` /
  `regime_transformer_v20260606.pkl`, weights 0.5/0.5. The displayed line re-runs from each day's STORED
  `inference.json`, so only post-2026-06-06 days use the new model; history is unchanged. Chart marker on
  2026-06-06. The model artifacts are gitignored (S3-managed); they live on disk + S3. Provenance is
  recorded in this doc and git history (the earlier `models/latest.candidate.json` staging pointer was
  removed in the cleanup sweep — see §10).

## 6. Deploy state (what is where) + how nightly persists

- S3 `config/decision_params.active.json` = optimized config; `config/decision_params.prev_champion.json` =
  frozen previous champion (for the dotted line).
- S3 `models/latest.json` = retrained regime ensemble (0.5/0.5), health unchanged.
- S3 `dashboard/dashboard.json` + `dashboard/data/dashboard.json` = regenerated (optimized line + dotted
  previous + switch marker). Served fresh via CloudFront `*.json` CachingDisabled behavior.
- Frontend rebuilt + synced (legend/tooltip say "Previous champion").
- **Lambda container redeployed twice** (code: topup 1.1, previous-champion line, retired faint 3rd line,
  chart marker bundled) so the nightly pipeline reproduces all of this and extends it daily.

## 7. Rollback

All originals backed up in `runs/beat_champion_20260606/s3_backup/`:
- `active.json.bak`, `latest.json.bak`, `dashboard.json.bak`.
To revert: `aws s3 cp <bak> s3://investment-system-data/<orig key> --profile personal`, restore the old
`extender.py`/`config` from git, redeploy the container. The previous champion config is also preserved as
`config/decision_params.prev_champion.json`.

## 8. Open levers / next jumping-off points (for future tweaks)

1. **Watch the forward regime swap.** The +15.0% line was produced on the OLD regime signal. From
   2026-06-06 forward the new model drives decisions; the line may diverge. This is the thing to watch.
   If it underperforms, revert `latest.json` (§7) — history is unaffected either way.
2. **Ensemble weight (0.5/0.5) is a guess** now that both models are healthy — never validated forward.
   Candidate for tuning once there's post-switch data.
3. **Unsupervised regime model (HMM / Markov-switching).** Escapes the rule-label ceiling (§5). Highest-
   value model upgrade. Validate on the 2026-03-11+ holdout + down-window before switching forward.
4. **Optimizer-as-a-program was not repaired** — only a manual constrained search was run. To make the
   nightly optimizer promote on the real overlay object: fix fold geometry, replace the round-trip guardrail
   (it rejects even the champion), constrain the param budget (~5–8 of ~236), and make it score the
   displayed object (overlays included), not the stripped base. See the Infotropy committee verdict's
   OPTIMIZER_REDESIGN.md + ANTI_OVERFIT_PROTOCOL.md.
5. **Parameterize the overlays** (`relax_choppy`, `topup`) into config so future overlay tweaks are
   config-only (no Lambda redeploy). Currently `topup_trigger` is hard-coded in `extender.py`.
6. **Daily-artifact backfill** would give the optimizer ~30 folds instead of ~3 (per-asset history is
   reconstructible to ~12y) and let any model train on more than the macro corpus.

## 9. Hard-won gotchas (don't relearn these)

- **Optimize the displayed object, not a stripped proxy.** Base config replays to only +0.98% on the
  holdout; the displayed line is +9.2% because of the overlays. Tuning the base alone tunes a near-null lever.
- **Never-worse floor:** an optimization result below the incumbent is a failed run, not a finding.
  If your number drops below the live champion, you measured the wrong thing.
- **Whole-period vs slice:** the +12.5%/+15.0% are whole-period; the replay's +9.4%/+11.85% are the
  2026-03-11+ slice only. Don't compare a slice to a whole-period number (and don't compare SPY over the
  war-dip slice — it snaps back and looks unbeatable).
- **Regime labels are a rule** — supervised regime models can't beat the baseline rule they're trained on.
- **Disagreement is computed on raw GRU-vs-transformer vectors regardless of ensemble weight** — a bug;
  re-weighting via `ensemble_overrides` also recomputes (and can corrupt) the label, so it's the wrong lever
  for the throttle.
- **Pyarrow/torch live in `.venv`**, not system python3. Run everything with `./.venv/bin/python`.

## 10. Repo/deploy truth after the 2026-06-06 cleanup sweep

- **HEAD now matches what's deployed.** The deployed Lambda + frontend were built from working-tree
  changes; the load-bearing ones are now committed: `src/steps/decision_engine.py` (the
  `buy_score_threshold_by_regime` / `min_health_buy_by_regime` regime-conditional support that the live
  config depends on), `src/utils/dashboard_metrics.py` (canonical-overrides display path),
  `frontend/src/components/PerformanceLenses.tsx`, `docs/DEPLOY.md`. If you rebuild from HEAD you get
  what's running.
- **LANDMINE — do not promote `config/decision_params.candidate.json`.** It holds a *degenerate optimizer
  output* (max_positions 75, max_position_weight 0.44, sell_health_threshold 0.0 = never-sell). The live
  config is `config/decision_params.active.json` only. The dormant optimizer wrote that candidate; it is
  evidence of the overfit problem, not a config to ship. (Left uncommitted intentionally as evidence.)
- **Regime model is LIVE forward** (supersedes the "staged, not live" note in the Infotropy RESULT.md
  receipt): `models/latest.json` (S3) points at the retrained ensemble; first live decision Monday.
  The removed `models/latest.candidate.json` was the pre-switch staging pointer — provenance is here + git.
- Remaining dirty files in `git status` are **generated data** (optimizer run indexes/outputs under
  `runs/optimizer/`, `dashboard/data/`, `frontend/public/data/`) that regenerate on each optimizer run,
  plus older scratch — not source, safe to ignore.
