# RETURN — PKT-TRADER-BOT-ALPACA-REMOVAL-AND-CONTINUOUS-LINE-20260608

Status: **PHASE 0 COMPLETE — NOT YET RATIFIED FOR IMPLEMENTATION.**
The adversarial verifier **vetoed** the continuity design (Phase 2). Phase 1 (Alpaca excision) is ratified as past-safe. One operator-owned design fork is open (see §D). **No edits have landed.**

Committee: run by a 5-agent workflow (Excision surgeon, Broker-independence analyst, Root-cause investigator, Architect, Adversarial verifier — verifier a different agent than implementers, per §4). AWS was reachable (profile `personal`, us-east-1); live S3 + CloudWatch evidence was used.

---

## A. Substrate confirmation (§3) — CONFIRMED, with sharpenings

§3a (line is broker-independent) and §3b (advance mechanism) **confirmed against live code**, with one sharpening:

- The **replay machinery is genuinely broker-free**: `replay_engine.run_variant` reads only `daily/{date}/{inference,features,signals}` + `prices.parquet`, seeds once from `daily/2026-03-11/portfolio_state.json`, fills at next-session **open**, marks at **close**; `_market_return_extend` is defined-but-never-called (only refs are its definition + a comment + unit tests). Live proof: every `equity_curve` row ≥ 2026-03-12 has `value == optimized_value` (replay-sourced), diverging ~$18k from the broker `raw_value` (~$96.6k).
- **Sharpening (drift from §3a's "100% broker-independent" wording):** the line is broker-independent **only while the extender succeeds**. The *base* curve the extender overwrites IS broker-derived post-seam (`portfolio_state.json.portfolio_value` = broker equity under `BROKER_MODE=alpaca_paper`). On a silent extender no-op, that broker-tainted base would publish. **Removing Alpaca strictly hardens this** (degraded path falls back to pure-sim, not the broker book) — but it does not by itself close the no-op hole; the fail-loud guard is still required independently.

§3c (removal map) **substantially correct, with three corrections** (from an independent repo-wide sweep):
1. `midday_checker.py` is understated — it has **three** independent `use_broker = not isinstance(broker, SimulatedBroker)` gates (239,319,414) **plus** a `broker.check_account()` block (241-251), not just the 3 call sites §3c names.
2. **Deletion-coupling landmine:** `tests/test_broker_cutover_profile_guardrails.py` `FILES_TO_SCAN` asserts `scripts/bootstrap_alpaca_from_sim_state.py` **and four 2026-03-12 plan docs** exist (`path.exists()`). Deleting any (which §3c authorizes) FAILS the test unless `FILES_TO_SCAN` is retargeted **in the same change**.
3. §3c's claim that `config/aws_config.json` has an alpaca secrets block is **wrong** — it lists only openai/fred/alphavantage. The alpaca secret names are hardcoded in `handler.py`; no `aws_config.json` edit is needed. Also: only alpaca-**paper** secrets exist (no live keys).

**Confirmed FALSE POSITIVES not to touch:** every `traded_notional` hit (backtest engine metric across optimizer_guardrails / run_shadow_day / run_fusion_sweep / run_gate_ablation / constrained_tune), the `broker_total_value` metric (= `portfolio_value`, a dashboard.json schema field), and the extender's `broker_order_id`/`broker_status`/`execution_mode='optimized_champion_replay'` schema fields — these feed/describe the **sacred line** (invariants 1+2).

---

## B. DEFINITIVE ROOT CAUSE (the §5 Phase 0 open question)

**It is NOT silent extender degradation (the packet's prime suspect — RULED OUT).** Live S3 `dashboard.json` (LastModified 2026-06-08T13:48Z) carries the version + `canon_source=optimized_champion` stamps fresh; CloudWatch stream `14aa678` shows the morning run completed with **no** `three_line_replay extension FAILED` line. The defensive `try/except` is not being hit.

**Primary (B): the replay's structural 2-day look-ahead.** `replay_engine.py:514` `for i in range(len-2)`, `inputs_date=trading_dates[i+1]`, `prices_file_date=trading_dates[i+2]`, then `_ohlc_for_date(prices_df, inputs_date)` — i.e. it prices decision date D by reading **D's OHLC bar out of `daily/{D+1}/prices.parquet`**. That's load-bearing: the night run that produced D's decisions ran *before* D's session traded, so D's own bar only lands in the *next* night's ingest. Consequence: the newest 1–2 trading days **can never be priced** and are flat-held (`extender.py:231-236`). Live: displayed `value` is identical `115015.27` for 2026-06-05/06-06/06-08; true last simulated day = 06-05; CloudWatch logs `missing prices file 2026-06-08 ... NoSuchKey`.

**Contributing (C): the morning run emits no priceable artifact, and its bare prefix poisons the look-ahead.** `publish_morning_artifacts` writes `portfolio_state.json` but **no** `prices.parquet`. Because a `daily/{today}/` prefix exists once `portfolio_state.json` is written, it becomes `trading_dates[-1]` (priceless), pushing the last genuinely-priceable night day into the `trading_dates[-2]` slot where the replay needs today's still-absent prices to price it. So the morning fill (computed at real morning prices via `paper_trader.execute_trade`) is **invisible** to the displayed line, and the bare prefix actively **blocks** the prior night-complete day. This is why "it doesn't update in the morning."

**Net:** the line permanently trails the latest closed day by 1–2 days and never shows "today." The manual `reextend` script doesn't compute anything different — it just *looks* like it un-sticks the line.

**Sacred-past boundary:** every value ≤ 2026-06-05 plus the genuine 06-05 point must stay byte-identical; only the 06-05→06-06→06-08 flat-held region is in play for any fix.

---

## C. RATIFIED — Phase 1 (Alpaca excision) is PAST-SAFE and ready

The verifier refuted the "alpaca_truth removal moves a value" attack (`design_survives: true` — the extender recomputes trades/round_trips/metrics from the champion replay and overwrites those fields on success; alpaca_truth values only survive on the no-op path, which the byte-diff catches). The excision edit map (delete list, edit list, keep-path, test retargets, operator-gated infra/secret removal) is complete and recorded in the workflow output. Keep-path verified: `SimulatedBroker`, `paper_trader.execute_trade` (morning + midday branches), `_update_valuations_from_quotes`, ingest yfinance/Stooq morning fallback, `dashboard_metrics` canonical curve (minus 2 alpaca_truth lines), `historical_corrections`, replay/extender, and the manual re-sim scripts.

**Dominant risk (gated, not blocking):** the `dashboard_metrics.py` alpaca_truth removal (it computes the canonical curve = the past) must be proven value-neutral by the §6.1 byte-diff against the **live S3 dashboard.json** before it lands. If any past value moves → STOP.

---

## D. VETOED — Phase 2 (continuity) + the open operator fork

**Architect's Design Decision C (have the morning emit a provisional priced bar for the executed day so the replay prices it the same morning) was VETOED by the adversarial verifier:**
- It **moves the past**: the next night overwrites the morning-injected prices slot with D's real session close and **re-prices D** (`publish_artifacts.py:567-568`; `replay_engine.py:650` — the morning snapshot close ≠ the session close).
- A single-epoch §6.1 byte-diff would not catch a value that only moves on the *following* night; it needs a cross-epoch stability check.

**The underlying tension (the real fork):** the displayed champion line is a **close-marked replay**. A day D's close does not exist at D's morning, so the replay *cannot honestly* produce D's point in D's morning — any morning point is provisional and gets re-marked at the close. The operator's stated model ("the morning fill is the new forward point, every day") collides with invariant 3 (replay methodology / mark-at-close unchanged).

**Two honest directions:**
- **R1 (recommended): close-marked night-advance.** The line advances each night to the latest **closed, fully-priced** trading day; fix the morning look-ahead poisoning so the night reliably advances, and port the fail-loud ABORT guard (`reextend_dashboard_now.py:51-55`) into both publish sites so genuine stalls alert (`send_alert` → existing `investment-system-alerts` SNS topic) instead of silently flat-holding. Past-safe; ends the manual treadmill; advances one trading day per trading day — but it shows the latest **closed** day (effectively "yesterday"), not "today at open." The "morning advance" was the operator's model of the *mechanism*; what's actually wanted (reliable, automatic, no manual re-sim) is delivered honestly.
- **R2: provisional "today" point at morning.** Permitted by invariant 1 only if D stays "the newest forward day" when it changes — but it requires injecting a non-replay-close-marked value (the exact thing the architect's own rejected-options list warned against) and the verifier showed the corpus-injection form moves the past. Higher risk, likely violates invariant 3.

**Fail-loud guard (both directions, ratified design):** post-`extend_dashboard`, before the S3 write, verify (1) stamp present (version + `canon_source==optimized_champion`), (2) the line advanced when a genuinely-new priceable day exists, (3) **no false alarm** on weekends/holidays/structurally-unpriceable-latest-day (check the corpus's priceable frontier, not wall-clock). On failure: skip the write (preserve last-known-good), `send_alert`, set `success=False`. `send_alert` never raises, so the guard can't crash the pipeline.

---

## E. OPEN ITEMS before Phase 0 is fully ratified

1. **Operator fork (D):** confirm R1 (honest close-marked night-advance) vs. insistence on a provisional morning "today" point. This determines the entire Phase 2 mechanism.
2. After the fork is chosen, a **second committee round** finalizes + ratifies the concrete Phase 2 mechanism (the continuity engineer must prove against the live corpus/cadence that the chosen design reliably advances the line by exactly one, handling interior gaps and weekend `night_date`/`run_date` divergence), and the verifier signs off.
3. Operator-gated AWS actions, to sequence **after** code edits land:
   - Drop `BROKER_MODE`/`BROKER_TRADING_ENABLED` from the live Lambda env (required so the alpaca-gone sweep finds the active config clean) — but **record their exact current values** (`BROKER_MODE=alpaca_paper`, `BROKER_TRADING_ENABLED=true`) in DEPLOY.md for one-command restore.
   - **Operator decision (2026-06-08): PRESERVE the two Alpaca secrets** — do NOT delete `investment-system/alpaca-paper-key-id` / `alpaca-paper-secret-key` from Secrets Manager. Leave them in place (unread, ~$0.80/mo) as the saved copy in case Alpaca is revisited later. The §6.3 sweep is still satisfied: "no alpaca-* secret is *read*" by any live code path. Add a "How to restore Alpaca" note to DEPLOY.md pointing at the preserved secrets + the env-var values + this packet/RETURN as the removal record.

---

## F. ROUND 2 — Phase 2 continuity design RATIFIED (with one mandatory amendment)

Second committee round (continuity engineer → architect → adversarial verifier). The verifier's structured-output emit hit a harness bug; its full verdict was recovered from the transcript. **Design is sound; one blocking fix required, now folded in.**

**Operator's clarified model (authoritative):** night = run the algorithm → intents. Morning = risk re-check + simulate the fill at the real **open** (the trade; stays in the morning). Today's dot appears in the morning from the open fill and settles to its real value at the close — only today's dot moves. (Operator also confirmed: the replay DOES simulate the morning trade — fills at `quote['open']`, `replay_engine.py:415`, with the overnight-gap guard.)

**Ratified mechanism (past-safe):**
1. **Night** (labeled D, runs evening of D-1): unchanged. Writes `daily/D/prices.parquet` covering history through D-1. The close-marked replay frontier advances here, one closed day at a time — the fix below just stops it being blocked.
2. **Morning** (labeled T): write a **self-contained, single-date** `daily/{T}/morning_prices.parquet` from the quotes the morning already fetched (`open = quote['open']`, synthesized `close = quote['price']` = the morning last). A **separate key** — never the shared `prices.parquet`.
3. **Replay fallback**: after the normal loop, if the newest prefix T is structurally unpriceable (no real successor `prices.parquet` yet) AND `daily/{T}/morning_prices.parquet` exists, price **only T** from it. This draws today's dot in the morning.
4. **Finalization**: the next night writes the real `daily/{T+1}/prices.parquet` (T's true close); the standard loop re-prices T to settled and never consults the provisional again. **Only T's dot moves between epochs.**

**Why it answers the round-1 veto:** each past date prices from its own pre-existing dedicated successor file; `_ohlc_for_date` is an equality filter (added rows can't realign earlier dates); the provisional is a separate key consumed only as the newest-day fallback. Verifier: prior-day isolation **survives**.

**MANDATORY AMENDMENT (verifier's blocking fix — STUCK-PROVISIONAL):** if the settlement night never lands, today's provisional would freeze into permanent history (past moves) while the advance-check stays silent. **Add a provisional-staleness guard:** a frontier dot still carrying a provisional mark after one night cycle must settle to its real close or **alarm** (`send_alert` → `investment-system-alerts`). Mark provisional rows explicitly (e.g. a `provisional: true` flag on the row / a recorded provisional frontier date) so the guard can detect non-settlement. Phase 4 must prove this: force a stuck provisional → alarm + no silent freeze.

**Non-blocking fixes folded into Phase 2:**
1. `morning_executor.run()` discards the fetched quotes on the no-intents (`:540`) and stale-intents (`:562`) paths — return the real `morning_quotes` DataFrame on **all** paths so today's dot can draw even on a no-trade day.
2. `result['morning_prices']` is not threaded into `publish_morning_artifacts` — thread it through `handler.py:685-690`.
3. Held-symbol provisional marks: fall back to last close where a symbol lacks a fresh quote; implement + test.
4. The empty-folder fix must **preserve interior dates** in `trading_dates` (only neutralize the bare *newest* prefix, never drop historical priced prefixes).

**Verifier correction to §B/§D:** **midday does NOT republish `dashboard.json`** — `midday_checker.run()` writes only `portfolio_state.json` + its report. Only the night and morning paths publish the dashboard. (Earlier note that implied a midday dashboard overwrite is inaccurate.)

**Verification additions:** plus a **cross-epoch stability check** (build the dashboard at the morning epoch and again at the night epoch; assert only the newest day differs) and a **stuck-provisional fail-loud test** (no settlement night → alarm, no frozen-provisional history). Byte-diff baseline is always the **live S3** `dashboard.json`.

**Phase 0 is now RATIFIED** (Phase 1 past-safe; Phase 2 ratified-with-amendment). Proceeding to implementation. The §8 completion line stays absent until all §6 gates pass.

---

## G. IMPLEMENTATION + VERIFICATION (executed 2026-06-08)

Branch: `ai/alpaca-removal-continuous-line` (trader-bot is its own git repo).

### Commits
- **`b4290f8` — Phase 1: excise Alpaca.** Broker mode machinery removed entirely (not stubbed). Deleted `src/brokers/alpaca.py`, `src/utils/alpaca_truth.py`, the two Alpaca scripts, four Alpaca tests, `docs/ALPACA_SETUP_GUIDE.md`. Collapsed every execution path (`morning_executor`, `midday_checker`) to the simulated `paper_trader` keep-path; `ingest_prices` falls back to Stooq/yfinance; `handler` constructs no broker; `dashboard_metrics` lost only the 2-line alpaca_truth reconcile (sacred curve logic untouched); `publish_artifacts` lost the broker params/refresh. Tests retargeted (incl. the guardrail `FILES_TO_SCAN` landmine). Verified: 0 new test failures (the 6 pre-existing failures are identical on the base commit), sacred `dashboard_metrics.py` diff = the exact 3-line removal, no live broker symbol remains.
- **`cd4e961` — Phase 2: continuous line + fail-loud guard.** `run_variant` plan-refactor adds one optional provisional newest-day entry (priced from `daily/{T}/morning_prices.parquet`, settles next night); extender stamps `provisional_frontier`/`champion_frontier` (metadata only); morning writes the single-date provisional parquet (quotes it already fetched, now returned on all paths + threaded through handler); `_verify_extension_or_alarm` at both publish sites (stamp + forward-advance checks; holds last-known-good + SNS-alerts on failure; no false alarm on weekends/Mondays). 12 new guard unit tests.
- **Phase 3 (docs)** — strip "paper-live"/Alpaca framing to pure-simulation; superseded banners on the 2026-03-12 plan docs; "How to restore Alpaca" note in DEPLOY.md; stale-comment cleanups. (Separate commit.)

### §6 verification gates (all run against the LIVE S3 dashboard.json baseline, not the working tree)
1. **PAST-UNCHANGED BYTE-DIFF — PASS.** New `extend_dashboard` over the live corpus vs live `dashboard.json`: `equity_curve` value/optimized_value/hybrid_value/pre_hybrid_value/cashflow diffs = **0**; `drawdowns` = **0**; `monthly_returns` = **0**; sacred metrics = **0**. `champion_frontier=2026-06-05`, `provisional_frontier=None` (no morning_prices file exists yet → provisional correctly inert; Monday flat-held, unchanged). The replay refactor moves nothing in the past.
2. **CROSS-EPOCH ISOLATION (the round-1 veto's required gate) — PASS.** Staged real weekday 2026-06-04 as "today": morning epoch priced it provisionally (perturbed close → `115749.39`); night epoch settled it to the real close (`115507.61`, = the live historical value). **40 prior dates compared, 0 diffs — only the newest dot moved.** A provisional that later settles to a different value cannot move any prior day.
3. **ALPACA-GONE — PASS (code).** No live code path constructs or calls a broker; remaining tokens are the `SimulatedBroker` keep-path, the `BaseBroker` interface, and frozen-history comments (the 2026-03-12 cutover seam in `dashboard_metrics`/`historical_corrections`/`canonical_replay_anchor`, which stay). **Lambda-env removal + secret-preserve is operator-gated (pending — see §E/§3).**
4. **FAIL-LOUD — PASS (unit).** 12 guard tests: alarm on missing stamp; alarm on `champion_frontier < newest_priceable` (silent-stall / stuck-provisional regression); silent pass on weekend/no-new-day; `_newest_priceable_date` correctly skips bare prefixes / Mondays / structural-latest-day.
5. **TESTS GREEN — PASS (no regression).** `pytest tests/`: **291 passed, 6 failed**. The 6 are pre-existing stale-test-double / canonical-anchor-fixture-collision failures, proven identical on the base commit; **zero introduced by this work**. Not fixed here (unrelated; the 3 dashboard_metrics ones touch the canonical anchor — out of scope per No-Drift).
6. **DISPLAY UNCHANGED — PASS.** `frontend/` diff empty. `dashboard.json` schema preserved: no new `equity_curve` row fields (provisional info is `timeline_correction` metadata only); the newest dot simply now exists/settles like any live equity tip.

### REMAINING (operator-gated — NOT done unattended)
1. **Deploy** the rebuilt Lambda image (Phase 2 code) following `docs/DEPLOY.md`.
2. **Remove** `BROKER_MODE`/`BROKER_TRADING_ENABLED` from the `investment-system-daily-pipeline` Lambda env (values recorded for restore). **Preserve** the two `investment-system/alpaca-paper-*` secrets (operator decision).
3. **Merge** `ai/alpaca-removal-continuous-line` → main after the deploy verifies.

The §8 completion line remains absent until the deployed Lambda is Alpaca-free and running the continuous simulated cycle (post-deploy).
