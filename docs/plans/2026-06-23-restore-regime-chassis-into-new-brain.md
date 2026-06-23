# Restore the regime chassis into the New Brain (regime_compatibility)

## Context

On 2026-06-23 the live book dropped −3.37% vs SPY −1.27% — the biggest single-day
drop ever (`max_drawdown` −0.45% → −3.49% in one day). Diagnosis (verified against
Yahoo to the penny — prices are real, not a fetch bug):

- The book was ~51% concentrated in two correlated high-beta sleeves: the
  semis/tech complex (SMH+SOXX+XLK+ARKK ≈ $28k) and precious metals (GLD+SLV ≈ $17k).
  On a risk-off gap morning (semis −6/−7%, silver −5%) a concentrated book takes
  ~2.6× SPY's loss.
- **Root cause: the New Brain (native two-stage engine, PKT-TB-008/010/012) threw
  away the regime chassis.** `regime_admissibility = {}` and
  `regime_exposure_multiplier = {}` are both frozen EMPTY in `brain/FREEZE_ORB1.json`,
  and the engine references `config/regime_compatibility.json` **nowhere**. The
  shared chassis regime picker is computed and then discarded. The original
  `src/steps/decision_engine.py` (`score_candidates`, lines 111–195) multiplied each
  candidate's score by the per-(regime,sector) compatibility multiplier — in
  `high_vol_panic` it knocks tech ×0.4 / equity ×0.5 and lifts bonds ×1.2. That
  defensive rotation does not exist in the model now trading the book.
- The orthogonal-brain spec itself (`runs/pkt_tb_007_orthogonal_brain/designs/
  proposals/PROPOSAL_ORTHOGONALITY.md §0`) is explicit: the regime picker's
  "regime multiplier series" is a CHASSIS sense the new members must add value
  ON TOP OF (orthogonality gate |ρ|≤0.7 against it); "M1+M2 are the directional
  rank inputs to the chassis socket." The cutover replaced the chassis instead of
  plugging into it. That is the deviation this plan fixes.

Operator directive: "take the one thing that worked and add value to it — that was
the deal. If it didn't do that, do it." → restore `regime_compatibility` into the
New Brain pipeline, keeping the orthogonal M1 forecast on top.

Note: 06-23's regime was `risk_on_trend` (the gap was overnight while the regime
was still risk-on), so the regime tilt alone would NOT have caught this specific
day — in risk-on it leans into tech. The structural fixes are three, together:
(1) regime-aware selection, (2) regime-aware exposure cut, (3) a correlated-cluster
cap that actually binds (the granular sector taxonomy currently defeats the 35% cap).

## Plan

- [ ] `src/brain/engine/contracts.py`: add `regime_score_mult: Mapping[str,float]`
      to `ForecastBundle` (default `{}` → multiplier 1.0; field name passes
      `assert_no_dollar_surface` — no forbidden token). Dimensionless, regime×sector
      derived, constant across the θ_size grid → `held_symbols_invariant` preserved.
- [ ] `src/brain/engine/selection.py`: primary ordering key becomes
      `-(mu_M1[sym] * regime_score_mult.get(sym,1.0))` — faithful port of the
      original `final_score = base_score × regime_multiplier`, with mu_M1 (the
      orthogonal forecast) as the base. Record the mult in `forecast_meta`.
- [ ] `src/brain/forecast_adapter.py`: load `regime_compatibility` (from config),
      read `sector` from `universe_df`, compute `regime_score_mult` per symbol via
      the original lookup `compat.get(sector, compat.get(asset_class, 1.0))`; pass
      into the bundle. In `build_portfolio_state`, map granular sectors → correlation
      groups for `cluster_of` (semis/tech/innovation → `tech_growth`; gold/silver →
      `precious_metals`) so `max_cluster_weight` binds on the real complex.
- [ ] `src/brain/runtime.py`: thread `regime_compat = config.get('regime_compatibility')`
      into `build_forecast_bundle`.
- [ ] `config/correlation_groups.json`: the sector→group map (auditable, not inline).
- [ ] `brain/FREEZE_ORB1.json`: set `theta_size.regime_exposure_multiplier`
      (calm 1.0 / risk_on 1.0 / choppy 0.90 / risk_off 0.75 / panic 0.50);
      recompute `theta_size.content_hash`; recompute `engine.engine_sha` after the
      contracts/selection edits.
- [ ] Tests: update any pinned engine_sha / θ hashes / selection-set fixtures; add
      a regime-tilt rotation test (panic demotes tech, lifts defensives) and a
      cluster-bind test (tech_growth capped at 35%).
- [ ] `LIVE_PREREG.md`: record the re-pre-registration (engine + θ moved → §3
      no-mid-stream → current forward reads terminate, fresh prereg opens).
- [ ] Replay recent days incl. a risk_off/panic fixture; full suite + invariant gate.
- [ ] Deploy container Lambda per `docs/DEPLOY.md`; verify tonight's night run.

## Execution Log

- 2026-06-23: Diagnosed (see Context). Confirmed New Brain references
  regime_compatibility nowhere; original decision_engine applies it at
  score_candidates:173-174. Plan written.
- 2026-06-23: Implemented. contracts.py `ForecastBundle.regime_score_mult`;
  selection.py `_regime_adjusted_mu` primary ordering key + forecast_meta
  provenance; forecast_adapter.py `_regime_mult_for` (faithful get_multiplier
  port) + `_load_correlation_groups` + sector→group cluster_of; runtime.py threads
  `config['regime_compatibility']`. config/correlation_groups.json added.
- 2026-06-23: Re-froze brain/FREEZE_ORB1.json — engine_sha
  `0e1d22ec…`→`4ae7b79d…`, theta_size hash `4c4844ae…`→`9561c1d0…`,
  regime_exposure_multiplier {calm/risk_on 1.0, choppy 0.90, risk_off 0.75,
  panic 0.50}. theta_sel unchanged. brain.active.json + LIVE_PREREG.md re-prereg
  recorded (fresh forward start 2026-06-24). assert_cold_start green.
- 2026-06-23: Tests — 5 new (test_regime_chassis_restore.py) green; full suite
  391 pass / 1 xfail / 5 FAIL (the latter pre-existing PKT-TB-012 FakeS3/metrics
  debt — verified identical with my changes stashed). Invariant green on real
  06-23 bundle under risk_on AND panic.
- 2026-06-23: Real-data replay on real 06-23 M1 mu — risk_on boosts defense/
  equity (ITA→#1); forced panic halves every equity (×0.50), lifts oil (USO→#1)
  and bonds (×1.20), cuts gross to 0.50. Rotation confirmed.
- 2026-06-23: Container Lambda deploy launched (lambda_deploy_container.sh).

## Follow-ups

- Whether `max_cluster_weight` (0.35) should be tightened is a separate tuning call
  for the operator — this plan only makes the cap *bind* via correct grouping.
- Shadow rent ladder (the evaluation timeline) is being fixed in parallel
  (KeyError 'R' state migration) — separate task.
