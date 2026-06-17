# Retire the old models (PKT-TB-014)

## Context
D-AUTO-20260616 decision 5 + DESIGN_DOSSIER §7 row 014: now that the native two-stage New Brain is
the live path (PKT-TB-016), retire the old models. Three things: stop the champion-replay-as-live
path, retire the weekly optimizer (it tuned the retired champion config), and ensure the monthly
training job can never read/overwrite the frozen ORB-1 brain weights.

## Plan
- **Weekly optimizer** — unload `com.traderbot.optimizer` (launchctl) + remove its installed plist;
  block re-install by retiring `scripts/install_optimizer_launchd.sh`.
- **Champion-replay-as-live** — already retired: `extender.py` serves the champion ≤2026-06-11 from
  the byte-immutable static table `config/champion_freeze_20260611.json` and never recomputes it
  (`run_champion_replay` no longer called; forward dates `d > boundary` are the New Brain realized
  line). Verified, not re-edited.
- **Monthly-training ORB-1 isolation** — structural (the Batch image `Dockerfile.training` does not
  ship `runs/` or `brain/`, so the frozen weights are absent) + enforced (`training/frozen_brain_guard.py`
  called at the top of `train.py`, `evolve.py`, and `run_training_batch.sh`, which raises if any
  training output path overlaps the frozen tree).
- Verify with tests.

## Execution Log
- Weekly optimizer unloaded + plist removed (`scripts/uninstall_optimizer_launchd.sh`); installer
  retired (prints RETIRED, exits 1). launchd now: `com.traderbot.shadow` (kept — New Brain forward
  attribution) + `com.investment-system.monthly-training` (now ORB-1-isolated); optimizer gone.
- Added `training/frozen_brain_guard.py` + guard calls in train.py / evolve.py / run_training_batch.sh.
- Verified: `tests/test_frozen_brain_training_isolation.py` 6/6 green;
  `tests/test_three_line_replay_canon_promotion.py` 5/5 green (champion clamp).

## Follow-ups
- The frozen-brain guard ships in the Batch training image whenever it is next rebuilt; the active
  monthly-training launchd job runs `train.py` from the repo directly, so the guard is already live
  for it on commit. No deploy required for this retirement.
