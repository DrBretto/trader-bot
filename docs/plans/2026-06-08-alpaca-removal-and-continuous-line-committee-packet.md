# PKT-TRADER-BOT-ALPACA-REMOVAL-AND-CONTINUOUS-LINE-20260608

Packet Owner:
- A committee, instantiated in a fresh session in `/Users/drbretto/Desktop/Projects/trader-bot/`.

Date:
- 2026-06-08

Authority surface:
- `CLAUDE.md` (project workflow, plan-doc discipline)
- `docs/PLAN.md`, `docs/OPERATIONS.md`, `docs/DEPLOY.md` (current operational truth — to be amended, not blindly trusted)
- This packet (the diagnosis substrate below is pre-verified; the committee confirms before acting, but does not need to re-derive it from scratch)

Authority level:
- execution (plan first, then implement, then verify — all in the one packet run)

Discovery mode:
- bounded. The goal and the seven invariants in §2 are fixed. The committee verifies the substrate, root-causes the one open question (§5 Phase 0), implements, and verifies. New findings outside scope get logged in the RETURN doc for operator review, not silently absorbed.

---

## 1. What this packet exists to fix

This system is a **pure trading simulation**. There is no live brokerage and there must not be one. It is supposed to run a full cycle **every trading day, automatically**:

- **Night** (after close): run the algorithm on fresh market data → regime, signals, inference, decision engine → produce tomorrow's trade intents.
- **Next morning** (at open): **simulate** those intents at that morning's real prices → fills, holdings, cash update. That fill is the new forward point.

The dashboard's headline equity line (solid optimized-champion) and its dashed previous-champion comparison line should then **advance forward by one trading day, on their own, every day** — exactly as they do when the operator re-runs the simulation by hand. Today they do not advance reliably without manual intervention, and the operator has spent **weeks** repeatedly hand-running `scripts/reextend_dashboard_now.py` / `scripts/resimulate_mar11_forward_independent.py` to push the line forward. Meanwhile an **Alpaca paper broker is still wired into the live sequence** (`BROKER_MODE=alpaca_paper` on the deployed Lambda) — an entire dead code path the operator does not use and wants gone for good.

**Two outcomes, done right and for keeps:**
1. **Alpaca is removed from the system entirely** — code, config, env, secrets, infra, tests, docs, scripts. The simulated path is the only path.
2. **The forward line advances automatically and continuously, every trading day, with no manual re-simulation** — and when it can't advance honestly, the pipeline says so loudly instead of silently publishing a stale line. The manual re-sim scripts remain available as a deliberate operator override, not a necessity.

This app is going to be shown to people soon. It must be clean, correct, and self-sustaining. Precision is the requirement, not speed.

---

## 2. Hard invariants (load-bearing — violate any one and the packet has failed)

These are non-negotiable. The committee designs around them.

1. **THE PAST LINE IS SACRED.** The historical values of every line are accurate and must not change. The verification gate (§6) is a **byte-level diff** of the produced `dashboard.json` against the current live one: every `equity_curve` point, `drawdowns` point, `monthly_returns` entry, and metric **at every historical date must be identical**. Only the newest forward trading day(s) may differ. If any past value moves — even by a rounding digit — **STOP and report**. Do not "fix" the past. Do not let the line regress to raw/uncorrected values.

2. **THE DISPLAY DOES NOT CHANGE.** The frontend (`frontend/`) is not touched. All lines stay: solid optimized-champion (`value`/`optimized_value`), dashed previous-champion (`hybrid_value`), benchmark, the drawdown chart, monthly-returns heatmap, and all bottom panels. `dashboard.json`'s schema/shape is preserved exactly. The committee changes how the data is *produced and kept current*, never what is rendered.

3. **THE REPLAY METHODOLOGY IS NOT CHANGED.** The optimized-champion and previous-champion lines are computed by `src/utils/three_line_replay/` re-running each strategy over market history. That math is correct and stays. Removing Alpaca must not alter a single line value (it cannot — see §3, the broker-independence proof — but the diff in invariant 1 is what guarantees it).

4. **THE MANUAL RE-SIM STAYS.** `scripts/reextend_dashboard_now.py` and `scripts/resimulate_mar11_forward_independent.py` remain functional as an operator-invoked override. After a manual re-sim sets the line, the automated daily cycle must pick up from there and keep extending forward — they must not fight.

5. **PURE SIMULATION, EVERY TRADING DAY.** Night decides, morning simulates fills at morning prices. No Alpaca, no second account, no broker reconcile. The line advances one trading day per trading day, unattended. Weekends/holidays produce no new close and therefore no new point — and that must cause **no breakage and no manual repair**.

6. **FAIL LOUD, NEVER SILENT.** The current automated extender is "internally defensive" — on any internal failure it returns the dashboard **unchanged** and the pipeline publishes anyway, advancing nothing, with no alert. That silent-degradation behavior is a direct cause of the operator's manual-repair treadmill. The automated path must carry the same ABORT guard the manual script already has (verify the extension actually stamped and actually advanced) and must alert / fail visibly when it cannot.

7. **CLEANED UP.** No dead Alpaca scaffolding, no orphaned config keys, no stale docs claiming "paper-live active." When the committee is done, a reader of the repo sees a simulation system, not a half-decommissioned brokerage integration.

---

## 3. Confirmed substrate (pre-verified — confirm, don't re-derive)

The following was established by direct code reading and S3 inspection on 2026-06-08. Confirm each before relying on it; flag any drift.

### 3a. The displayed line is 100% broker-independent (the safety guarantee)
This is why removing Alpaca cannot move the past line:
- `src/utils/three_line_replay/extender.py` — `_market_return_extend()` is **defined but never called** (the only two references in the repo are its definition and a comment stating it is "intentionally not called"). The earlier version that scaled gap dates by broker `raw_value` (which leaked the $5,000 broker cap) was removed on 2026-05-16.
- `src/utils/three_line_replay/replay_engine.py` `run_variant()` reads **only** `daily/{date}/inference.json`, `features.parquet`, `signals.parquet`, and `prices.parquet` (market data + model outputs). It seeds the portfolio once from `daily/2026-03-11/portfolio_state.json` and forward-simulates on market OHLC. It **never** reads the live broker, the Alpaca cache, or `raw_value`.
- Fills are simulated at the next session's **open** (`quote['open']`); marks use OHLC **close**. Both from `prices.parquet`, not a broker.
- Gap/tail dates (weekends, missing-artifact days, the latest 1–2 unpriced days) are **flat-held** at the last simulated champion value (`extender.py` per-row patch loop, ~lines 219–249). No external series injected.
- Displayed **holdings** come from the champion replay's `final_holdings`, not the live `portfolio_state.json` (`extender.py` "Canon final state … champion replay" section).

**Implication:** Alpaca lives on a *separate* track (live order placement + `portfolio_state.json` bookkeeping + `alpaca_truth` post-hoc fill reconciliation). None of it feeds the lines. Cutting it is safe for the past — and the §6 diff proves it per-run.

### 3b. How the line advances forward (the mechanism that must become continuous)
- A date appears as an `equity_curve` **row** because `daily/{date}/portfolio_state.json` exists with a `portfolio_value` (`publish_artifacts.py:_build_equity_curve_from_daily`, ~lines 343–375).
- That row gets a **real champion value** only when the replay can price it. The replay uses a **2-day look-ahead** (`replay_engine.py` ~lines 514–539): to price decision date `D` it needs `daily/D/`{inference,features,signals} **and** `daily/{D+1}/prices.parquet` (the next session's OHLC that fills D's intents and marks them). Missing any critical artifact → date skipped → flat-held.
- The night phase **does** write the full analysis set every weekday (`prices/context/features/inference/llm_risk/signals/decisions/portfolio_state/trade_intents/weather` — confirmed present for 06-02 … 06-06) **and does publish the extended `dashboard.json`** (`publish_artifacts.run` step 13, ~lines 725–751), gated on `expert_signals != null` (`_can_publish_dashboard`). So the corpus and the publish call both exist.

### 3c. Where Alpaca is wired in (the removal map — confirm completeness, expand if needed)
- **Broker module:** `src/brokers/` — `alpaca.py` (AlpacaBroker), `router.py` (`BrokerMode`, `SimulatedBroker`, `resolve_broker_mode`, `is_trading_enabled`, `get_broker`), `base.py`, `__init__.py`.
- **Handler:** `src/handler.py` — broker import (line ~39); night-phase mode resolve + Alpaca secret fetch + broker reconcile (~lines 175–215); publish-broker construction (~505–520); morning-phase broker construction + execution (~608–689); midday-phase broker construction (~764–775); `config['broker']` bundle load (~line 104).
- **Morning executor:** `src/steps/morning_executor.py` — `_execute_via_broker`, `_await_broker_order_update`, `_wait_for_orders_to_settle`, `_reconcile_portfolio_from_broker[_with_retry]`, the `use_broker` branch in `run()` (the simulated branch via `paper_trader.execute_trade` is the one to keep).
- **Midday checker:** `src/steps/midday_checker.py` — calls into `morning_executor._execute_via_broker` / `_reconcile_portfolio_from_broker` (~lines 280, 361, 416).
- **Price ingestion:** `src/steps/ingest_prices.py` — `fetch_alpaca_daily()` (~165–192), Alpaca-priority in the daily fetch (~335), and the `broker=` snapshot-preferred path in `fetch_morning_quotes()` (~371–405). Morning quotes must fall back cleanly to the non-broker source (yfinance/Stooq).
- **Alpaca truth:** `src/utils/alpaca_truth.py` (whole file) and its use in `src/utils/dashboard_metrics.py` (post-hoc fill reconciliation — used for the metrics/ledger, **not** the line; removing it must not change historical metrics — the diff catches it).
- **Publish:** `src/steps/publish_artifacts.py` — `refresh_alpaca_orders_cache` import + calls (~lines 12, 558, 825), `broker=` params.
- **Config/env/secrets:** `.env.example` (BROKER_MODE, BROKER_TRADING_ENABLED, ALPACA_* keys); Secrets Manager `investment-system/alpaca-*`; `config/aws_config.json` secrets block; `router.py` `max_order_notional`/`symbol_allowlist`.
- **Infra (deployed Lambda env):** `BROKER_MODE=alpaca_paper`, `BROKER_TRADING_ENABLED=true` on `investment-system-daily-pipeline` (us-east-1, profile `personal`).
- **Scripts:** `scripts/bootstrap_alpaca_from_sim_state.py`, `scripts/alpaca_paper_smoke_test.py`.
- **Tests:** `tests/test_alpaca_broker.py`, `test_broker_router.py`, `test_morning_executor_broker.py`, `test_alpaca_truth.py`, `test_bootstrap_alpaca.py`, `test_broker_cutover_profile_guardrails.py`, broker-path assertions in `test_data_ingestion.py` / `test_continuity_repair_invariants.py` / `test_dashboard_metrics.py`.
- **Docs:** `docs/ALPACA_SETUP_GUIDE.md`, broker sections in `docs/OPERATIONS.md` / `docs/DEPLOY.md`, the "paper-live active" framing in `README.md` / `summary.md`, and the 2026-03-12 Alpaca plan docs in `docs/plans/`.

`alpaca_orders.json` and `execution_mode: "alpaca_paper"` values already baked into **historical** `dashboard.json` / trade records are **frozen history** — do not rewrite them (that would touch the past). New runs simply stop producing them.

---

## 4. Committee structure

The operator wants this **planned by committee first, then executed** — not one executor pattern-matching its way through. Convene the following roles. Phase 0 is planning + adversarial ratification; only after the plan is ratified does implementation begin.

- **Architect** — owns the Phase 0 root-cause and the implementation design; resolves the central design decision (§5 Phase 0, item C); writes the plan the others attack.
- **Excision surgeon** — owns Alpaca removal (§5 Phase 1); produces the exact edit list from §3c, confirms completeness with a fresh repo-wide sweep, and removes without touching the keep-path (`SimulatedBroker`/`paper_trader`).
- **Continuity engineer** — owns the automatic, self-verifying forward extension (§5 Phase 2).
- **Adversarial verifier** — owns the invariants in §2. Independently runs the §6 diff, tries to break the claim that the past is unchanged and that the line advances honestly, and has veto power over the stop condition. This role must be a *different* agent than the implementers.
- **Cleanup & docs steward** — owns §5 Phase 3.

Planning-first means: each role independently reads the relevant code, the Architect drafts the plan, the Adversarial verifier and the others red-team it (especially the central design decision and the invariants), and the committee converges on a ratified plan **written into the RETURN doc** before any edit lands. If the committee discovers the substrate in §3 is wrong, it stops and surfaces to the operator rather than proceeding on a false base.

---

## 5. Scope (phased)

### Phase 0 — Confirm substrate, root-cause continuity, ratify the plan (NO edits yet)
1. Confirm §3a, §3b, §3c against live code. Flag any drift.
2. **Root-cause, definitively, why the line does not advance on its own** and why the operator keeps re-running the manual scripts. The night path *is* coded to publish the extended dashboard (§3b), so the cause is **not** "night never publishes." Determine which of these (or what else) is the real, recurring mechanism, with evidence:
   - **Silent extender degradation (prime suspect).** `extend_dashboard` returns the dashboard unchanged on any internal failure and the pipeline publishes anyway (`extender.py` ~162–170; `publish_artifacts.py` ~726–751). The manual script `reextend_dashboard_now.py` exists *specifically* because it has an ABORT guard (`timeline_correction.version == "lambda-three-line-replay-v2-optimized-canon"` and `metrics.canon_source == "optimized_champion"`) that the automated path lacks. Identify what makes the extender silently no-op in the Lambda (S3Cache miss? replay unable to price the latest dates because of the 2-day look-ahead? an exception swallowed by the defensive wrapper?).
   - **Look-ahead lag + gaps.** The replay needs `daily/{D+1}/prices.parquet` to price `D`. Confirm whether the latest 1–2 days are perpetually flat-held for this structural reason, and whether a single failed/with-held night (or the weekend) compounds into a stuck tail that only a manual re-sim clears.
   - **Morning produces no priceable artifacts.** The morning phase writes `portfolio_state.json` + `morning_execution.json` but not a `prices.parquet`, so the replay cannot use the morning fills to advance a day until the next night. Decide whether this is why "it doesn't update in the morning like it's supposed to."
   - Rule each in or out with file:line and S3/CloudWatch evidence. Do not ship a guess.
3. **Central design decision (C):** specify the exact artifact flow that makes the line advance **one trading day per trading day, automatically**, consistent with the operator's model (night decides → morning simulates the fill → that fill is the new point) **without changing the replay methodology or any past value.** Options to weigh (committee picks and justifies in the RETURN doc): make the morning run emit the priceable artifact the replay needs for the just-executed day; and/or run+verify the extender on a cadence that guarantees forward progress as soon as the corpus permits; and/or close the look-ahead gap honestly. The constraint dominates the mechanism: if a choice risks moving the past, it is wrong.
4. Write the ratified plan into the RETURN doc. Get the Adversarial verifier's sign-off on the plan before Phase 1.

### Phase 1 — Excise Alpaca completely
- Remove every item in §3c. The system resolves to the simulated path with **no** broker abstraction to select — `simulated` is not a "mode," it is the only behavior. Prefer deleting the mode machinery over leaving a one-branch switch.
- Keep and harden the `paper_trader` execution path that the simulated branch already uses (`morning_executor.run()` non-`use_broker` branch; `paper_trader.execute_trade`).
- Morning-quote fetching falls back cleanly to the non-broker source; no `broker=` parameter survives.
- Remove Alpaca env/secrets from the deployed Lambda (drop `BROKER_MODE`/`BROKER_TRADING_ENABLED` or pin `simulated`; ensure no `investment-system/alpaca-*` secret is read).
- Delete Alpaca-only scripts and tests; retarget tests that asserted broker behavior to assert the simulated behavior instead. **Do not delete tests that protect the past or the simulation.**

### Phase 2 — Continuous, self-verifying forward extension
- Implement the design from Phase 0(C) so the night→morning cycle advances the line every trading day with **zero** manual steps.
- **Port the manual script's ABORT guard into the automated publish path** (invariant 6): after `extend_dashboard`, the pipeline verifies the extension actually stamped (`timeline_correction.version`, `canon_source == optimized_champion`) **and** that the line advanced when it should have (a new priceable trading day yielded a new champion point). If it didn't, the run must **not** silently publish a stale line — it surfaces (SNS/ops alert via the project's existing channel) and fails visibly.
- Preserve idempotency and the manual-override relationship (invariant 4): a manual re-sim sets the line; the next automated cycle continues forward from it without conflict.

### Phase 3 — Cleanup & docs
- Remove dead Alpaca scaffolding, orphaned config keys, the Alpaca plan docs (or mark them clearly superseded), and the "paper-live active" framing in `README.md`/`summary.md`/`docs/OPERATIONS.md`/`docs/DEPLOY.md`. The repo should read as a simulation system.
- Update `docs/PLAN.md`/`OPERATIONS.md` to describe the continuous simulated cycle and the (now optional) manual override.

### Phase 4 — Verify (see §6)

---

## 6. Verification gate (load-bearing)

The packet is **not** complete until all of the following pass and are recorded in the RETURN doc:

1. **PAST-UNCHANGED DIFF (the most important gate).** Produce a `dashboard.json` from the post-change code over the live corpus and diff it against the current live `dashboard.json`. Every historical `equity_curve.value`/`optimized_value`/`hybrid_value`, every `drawdowns` point, every `monthly_returns` entry, and every historical metric **must be byte-identical**. Only the newest forward trading day(s) may differ. Record the diff summary. **If any past value moved, the packet has failed — STOP and report; do not paper over it.**
2. **FORWARD-ADVANCE PROOF.** Demonstrate, against the live corpus, that running the night→morning cycle (no manual script) advances the line by exactly the expected new trading day — the point that the manual re-sim would have produced. Show the before/after tail.
3. **ALPACA-GONE PROOF.** A repo-wide sweep (`alpaca`, `Alpaca`, `ALPACA`, `BROKER_MODE`, `broker_mode`, `AlpacaBroker`, `mode_label`, `notional`) returns only: frozen historical data values in already-published artifacts, and superseded/dated docs. No live code path constructs or calls a broker. The deployed Lambda env carries no active Alpaca config. Record the sweep output.
4. **FAIL-LOUD PROOF.** Demonstrate that when the extender cannot honestly advance the line (simulate the failure), the automated path surfaces/fails instead of silently publishing a stale line.
5. **TESTS GREEN.** `pytest tests/ -v` passes after retargeting. No skipped tests hiding removed coverage.
6. **DISPLAY UNCHANGED.** Confirm `dashboard.json` schema/shape is preserved and the frontend was not modified.

---

## 7. Out of scope (do not do these)
- Changing the strategy, parameters, regime model, optimizer, or any line's *math*.
- Touching the frontend or any rendered element.
- Rewriting historical artifacts to scrub `execution_mode: "alpaca_paper"` from already-published trade records (that is the past — leave it).
- Introducing any new live brokerage or external trading API.
- Timed soak/stagger rollouts — ship the full change at once and verify in place.

---

## 8. Stop condition

All §6 gates pass, the committee's Adversarial verifier has signed off on the past-unchanged diff and the fail-loud proof, the deployed Lambda is Alpaca-free and running the continuous simulated cycle, and `docs/plans/2026-06-08-alpaca-removal-and-continuous-line-RETURN.md` documents the ratified plan, the definitive root cause, the diff summary, and the sweep output — ending with the literal final line:

```
TRADER-BOT ALPACA REMOVAL + CONTINUOUS LINE COMPLETE — PAST VERIFIED UNCHANGED — AWAITING OPERATOR REVIEW
```

If at any point the past line would move, or the committee cannot root-cause the continuity failure with evidence, **STOP and surface to the operator** — do not proceed on assumption. That assumption-and-proceed pattern is exactly what this packet exists to end.
