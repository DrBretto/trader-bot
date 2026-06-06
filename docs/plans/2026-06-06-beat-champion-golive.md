# 2026-06-06 — Beat-champion config go-live + previous-champion dotted line

## Context
PKT-TRADER-BOT-OPTIMIZE-BEAT-CHAMPION-20260606 produced an optimized config that beats
the live champion on the real system, whole period +12.5% → +15.0% (out-of-sample
2026-03-11+ slice +9.39% → +11.85%, drawdown preserved, down-window protected).
Operator authorized going live: optimized = solid champion line; previous champion =
dotted comparison line; bottom charts follow the new champion. Continuity through the
2026-03 (Iran-war) window is intentionally kept (frozen pre-2026-03-11 segment unchanged).

The newly RETRAINED regime models are NOT part of this go-live (staged only); the live
line uses the existing regime signal + the participation config. This is deliberate so
the watched line does not move from a silent model swap.

## Plan
- [x] Optimized config in `config/decision_params.active.json` (max_position_weight 0.30,
      buy_score_threshold_by_regime with stressed regimes protected).
- [x] Overlay `topup_trigger` 1.2→1.1 in `extender.py` `_build_champion_strategy`.
- [ ] Snapshot the PREVIOUS champion config to `config/decision_params.prev_champion.json`
      (pre-optimization: mpw 0.20, flat 0.65 threshold) for the dotted comparison line.
- [ ] `extender.py`: feed the previous champion (prev config + old topup 1.2 overlay) into
      `hybrid_value` (the existing dotted comparison slot); stop emitting the old
      active-without-overlays line and `pre_hybrid_value`. Solid `value`/`optimized_value`
      = new optimized champion. Drawdowns/monthly/metrics already recompute from `value`.
- [ ] Frontend label: legend + tooltip "Hybrid (comparison)" → "Previous champion".
- [ ] Deploy: backup S3 active.json + dashboard.json; upload new active.json +
      prev_champion.json; regenerate dashboard.json via extender locally + upload; rebuild
      Lambda container (so nightly runs persist the new code); rebuild + sync frontend.
- [ ] Verify on https://trader-bot.infotrope.io.

## Execution Log
- Config + overlay edits done and validated on the real harness (+15.0% whole period).

## Follow-ups
- Wire the retrained regime models into the live inference path (regenerate daily
  inference) and re-confirm before relying on them — separate task.
- Optional: move `topup_trigger` into config so future overlay tweaks are config-only.
