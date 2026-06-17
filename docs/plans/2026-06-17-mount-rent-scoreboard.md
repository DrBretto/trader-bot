# Mount the "paying rent" scoreboard on the dashboard (PKT-TB-015)

## Context
PKT-TB-011 built `frontend/src/components/RentLedger.tsx` (the 5-rung leave-one-organ-out
"paying rent" scoreboard) and exported it, but `App.tsx`/mobile never imported or rendered it —
they only mount the old `SystemBrainPanel` (GRU/transformer/regime + "65/35 hybrid ranking MLP"
blend). So the operator sees the OLD brain section and the new rent scoreboard not at all.
Governed by PKT-TB-015-MOUNT-RENT-SCOREBOARD-ON-DASHBOARD-V1-20260617.

Live shadow data is currently `shadow_timeseries.v1` (no `organ_ledger`/`forecast_leg` yet); the
v2 ladder publication is PKT-TB-011's pipeline concern. This packet is the FRONTEND wiring +
honest empty/awaiting state, independent of the cloud deploy.

## Plan
- `RentLedger.tsx`: always render the 4 canonical rungs (regime / forecast / event / universe),
  filling from `organ_ledger` where present and showing a labeled "awaiting forward settled data"
  state per missing rung. Never return null, never a fabricated number. Synthesize the forecast-skill
  card from v1 `stats` (mean_ic/ic_t/n_weeks_ic) when `forecast_leg` is absent so the real certified
  IC still shows. Footer (additivity/FDR/divergence/m4) renders only when those stats exist.
- `App.tsx`: mount `RentLedger` as the PRIMARY brain section in `zone-2-right` (wired to
  `shadow.organ_ledger`/`forecast_leg`/`stats`), above a demoted SystemBrainPanel.
- `SystemBrainPanel.tsx`: demote — retitle "System Brain" → "Live model inputs"; remove the
  contradictory "65/35 hybrid ranking MLP / Weights fixed from optimizer" blend line + hybrid chip
  (the retired model picture the rent ladder contradicts); keep the still-valid regime ensemble +
  expert-signal monitor + regime history as honest INPUTS.
- `MobileDashboard.tsx`: add a primary "What's paying rent" section rendering `RentLedger`; demote
  the existing brain section to "Model inputs".
- Typecheck/build green; deploy frontend per docs/DEPLOY.md.

## Execution Log
- RentLedger rewritten: always renders the 4 canonical rungs, honest per-rung "awaiting" state,
  synthesized forecast-skill card from v1 stats, accruing note. Title is now the primary
  "System Brain — what's paying rent".
- App.tsx: RentLedger mounted in zone-2-right above the demoted SystemBrainPanel.
- SystemBrainPanel demoted to "Live model inputs"; removed the "65/35 hybrid ranking MLP / optimizer
  weights" blend line + hybrid chip; kept regime ensemble + expert-signal monitor + regime history.
- MobileDashboard: primary "what's paying rent" section added; old brain section relabeled "Model inputs".
- Build green (tsc + vite). Deployed (bundle index-BRDsGnrm.js) + CloudFront invalidated.
- Verified live (headless): scoreboard + 4 rungs render, honest awaiting states, forecast-skill card,
  one primary brain section + demoted "Live model inputs", zero console errors.

## Follow-ups
- The live shadow job still emits v1 (no organ_ledger); the v2 ladder + forecast_leg publication
  (shadow_nightly.py) needs to actually run forward — until then the scoreboard shows the honest
  awaiting state. Surfaced as an executor_concern (ties to the PKT-TB-016 gdelt read-only finding).
