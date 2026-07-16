"""Night forward pipeline — the clean-core path the thin router dispatches to.

ONE code path, history == forward (dossier Output 5): the same
``feeds→store→freshness_gate→features→forecast`` chain the replay keystone uses
(here behind ``decide.cutover.production_forecaster``) feeds the SAME
``engine→decide`` (``run_cutover``), then the settled leaf appends to the
content-addressed ledger and the line publishes non-destructively. Abort-never-
degrade: any operational failure returns ``ok=False`` → the incumbent intents are
retained, SNS fires, and the missed-run heartbeat is emitted with ``ok=False``.

This is a WIRING module (it composes clean-core module calls); the heavy compute
lives inside those modules — ``production_forecaster`` (feeds/store/forecast) and
``run_engine`` (selection/allocation). The router (``app.handler``) stays pure
dispatch. Production is pointed at this entrypoint at **P9** (cutover + observe a
full live night+morning); P8 ships it wired and import-verified but does NOT cut
prod over, so nothing here is invoked against prod during P8.

Portfolio-load + settled-mark seam: the first-class clean-core portfolio loader
(``store.portfolio.load_portfolio_state``) lands at P9 — the night path no longer
borrows the old ``src.steps.paper_trader`` chassis loader for this read.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Mapping

EXPIRES_AFTER_DAYS = 3

# A US session for day D is not SETTLED until after its close. Close is 16:00 ET
# (20:00 UTC in EDT / 21:00 UTC in EST); use a conservative 21:15 UTC buffer that
# covers both DST regimes AND the settle lag. The scheduled night fires at 03:00
# UTC (well past the prior day's settle) so it is never blocked; this only trips a
# night mis-invoked DURING market hours (e.g. a fall-through), which must NEVER
# produce an unsettled bar or append an intraday leaf (the frozen-then-locked-in
# corruption class).
_SETTLE_BUFFER_UTC = (21, 15)


def _is_settled_session(date_str: str) -> bool:
    """True iff day ``date_str``'s US market session has closed (settled)."""
    from chassis.utils.market_calendar import is_trading_session

    if not is_trading_session(date_str):
        return False
    now = datetime.now(timezone.utc)
    h, m = _SETTLE_BUFFER_UTC
    try:
        close = datetime.fromisoformat(date_str[:10]).replace(
            tzinfo=timezone.utc) + timedelta(hours=h, minutes=m)
    except Exception:  # noqa: BLE001 — a malformed date is treated as not-settled
        return False
    return now >= close


def _load_universe_df(config_dir):
    import pandas as pd  # pipeline glue (universe DataFrame), not compute
    return pd.read_csv(config_dir / "universe.csv")


def _persist_forecast_record(s3, settled: str,
                             record: Mapping[str, object]) -> dict:
    """Persist the exact successful forecast consumed by the decision engine."""
    if not isinstance(record, Mapping):
        raise RuntimeError("successful cutover returned no forecast record")
    payload = dict(record)
    if str(payload.get("date") or "") != settled:
        raise RuntimeError(
            f"forecast record date {payload.get('date')!r} != settled {settled}"
        )
    mu = payload.get("mu")
    if not isinstance(mu, Mapping) or not mu:
        raise RuntimeError("successful cutover returned an empty forecast mu")
    if not s3.write_json(payload, f"daily/{settled}/inference.json"):
        raise RuntimeError("inference.json write returned false")
    return payload


def _merge_night_pointer(latest: Mapping[str, object], settled: str,
                         trade_intents: Mapping[str, object],
                         timestamp: str) -> dict:
    """Advance night-owned fields without regressing a later morning snapshot."""
    out = dict(latest or {})
    current_date = str(out.get("date") or "")
    current_intents = str(out.get("intents_date") or "")

    # A historical retry must not point tomorrow's executor at older intents.
    if not current_intents or settled >= current_intents:
        out.update({
            "intents_date": settled,
            "regime": trade_intents.get("regime", "unknown"),
            "actions_count": len(trade_intents.get("actions", [])),
        })

    # Morning uses `date` as the operational snapshot date. If morning has
    # already advanced beyond this settled session, it owns date/phase/time.
    if not current_date or settled >= current_date:
        out.update({
            "date": settled,
            "phase": "night",
            "timestamp": timestamp,
        })

    out.pop("portfolio_value", None)
    out.pop("positions_count", None)
    return out


def run_night(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Run one forward night. Returns a Lambda-shaped response dict."""
    from chassis.utils.s3_client import S3Client
    from chassis.utils.sns_alerts import send_alert
    from store.portfolio import load_portfolio_state
    from decide.cutover import (run_cutover, load_brain_config,
                                production_forecaster, _regime_compat_path)
    from decide.freshness_gate import _latest_settled_trading_day
    from publish.dashboard import build_publish_surface, publish_line
    from monitors.canary_gate import run_post_pipeline_canaries
    from monitors.watchdog import emit_run_heartbeat

    s3 = S3Client(bucket, region)
    settled = event.get("run_date") or _latest_settled_trading_day()

    # HARD SETTLED-SESSION GUARD: the night produces the settled bar and appends a
    # canon leaf for ``settled``. If ``settled``'s session has NOT closed yet (a
    # night mis-invoked during market hours, e.g. an unrouted source falling through
    # to this default path), producing a live intraday bar and appending an
    # unsettled leaf would corrupt the line — and because the frontier append is
    # write-once, the real settled leaf could never land. Abort BEFORE any produce /
    # extend / append / publish. Nothing is written; the scheduled post-close night
    # advances the line correctly.
    if not _is_settled_session(settled):
        return {"statusCode": 409, "body": json.dumps({
            "status": "skipped", "phase": "night", "date": settled,
            "reason": (f"session {settled} not settled yet (market open / pre-close); "
                       "night refuses to produce an unsettled bar or append an "
                       "intraday leaf — no writes")})}

    config = dict(load_brain_config())
    config["mode"] = "live"
    universe_df = _load_universe_df(_regime_compat_path().parent)
    portfolio_state = load_portfolio_state(s3, as_of=settled)

    # ---- settled-bar PRODUCTION (CL-708001) — RESTORED. The clean night path is a
    #      pure CONSUMER of settled bars (production_forecaster's extend downloads
    #      daily/<D>/prices.parquet and splices it); the P9 cutover dropped the
    #      chassis writer (src/steps/publish_artifacts.py) that produced them, so the
    #      store froze at 07-02 and every night ABORTED stale. Produce the settled
    #      bar HERE (Yahoo v8 primary + fallbacks) so the store can advance. Never
    #      fabricates: a genuine feed outage writes nothing and the (untouched)
    #      freshness gate downstream aborts and surfaces it.
    from feeds.produce import produce_settled_prices
    try:
        produce_report = produce_settled_prices(settled, universe_df, s3, bucket=bucket)
    except Exception as e:  # noqa: BLE001 — production is best-effort; the freshness
        # gate is the safety net. A crash here must not bypass the gate: log it and
        # let the extend+gate run over whatever bars ARE in S3 (aborts if stale).
        produce_report = {"produced": False, "reason": f"producer crashed: {type(e).__name__}: {e}"}
    print(f"  [PRODUCE] settled-bar production: {json.dumps(produce_report, default=str)}")

    # ---- engine decision (feeds→store→freshness→features→forecast→engine→decide)
    incumbent = s3.read_json(f"daily/{settled}/trade_intents.json") or None
    res = run_cutover(
        date=settled, features_df=None, regime_label=None,
        universe_df=universe_df, portfolio_state=portfolio_state,
        forecaster=production_forecaster, config=config,
        incumbent_intents=incumbent,
    )

    if not res.ok:
        # abort-never-degrade: keep the incumbent, page, mark the heartbeat red.
        send_alert(
            subject="[TraderBot] CRITICAL: night ABORTED to incumbent",
            body=(f"The clean-spine night for {settled} returned ok=False and the "
                  f"incumbent intents were retained (nothing written).\n\n{res.reason}"),
            region=region)
        emit_run_heartbeat(region=region, ok=False)
        return {"statusCode": 500, "body": json.dumps(
            {"status": "aborted", "phase": "night", "date": settled,
             "reason": res.reason})}

    # Preserve the exact forecast that the successful engine run consumed. The
    # live rotation canary reads this artifact; recomputing or synthesizing it
    # after the decision would let the canary validate a different prediction.
    _persist_forecast_record(s3, settled, res.forecast_record)

    # ---- persist intents (EXACT schema the morning executor consumes) ----
    trade_intents = dict(res.trade_intents)
    if not trade_intents.get("generated_timestamp"):
        trade_intents["generated_timestamp"] = datetime.now().isoformat()
    if not trade_intents.get("expires_after_days"):
        trade_intents["expires_after_days"] = EXPIRES_AFTER_DAYS
    if not s3.write_json(trade_intents, f"daily/{settled}/trade_intents.json"):
        raise RuntimeError("trade_intents.json write returned false")

    selected_universe = list(
        res.selected_universe
        or trade_intents.get("expert_metrics", {}).get("held_symbols", [])
    )
    universe_fingerprint = {
        "date": settled,
        "selected_universe": selected_universe,
        "engine": res.engine,
        "generated_timestamp": trade_intents["generated_timestamp"],
        "source": "clean_core_night",
    }
    if not s3.write_json(
        universe_fingerprint,
        f"daily/{settled}/brain_selected_universe.json",
    ):
        raise RuntimeError("brain_selected_universe.json write returned false")

    # Operational pointers advance independently of the chart publish. Coupling
    # this pointer to a dashboard guard left the morning executor reading days-old
    # intents/state even though newer artifacts existed.
    latest = _merge_night_pointer(
        s3.read_json_strict("daily/latest.json") or {},
        settled,
        trade_intents,
        datetime.now().isoformat(),
    )
    if not s3.write_json(latest, "daily/latest.json"):
        raise RuntimeError("daily/latest.json write returned false")

    # The scheduled replay refresh owns BOTH model lines. The night must never turn
    # this internal sizing state into a displayed return ratio.
    append_report = {
        "action": "deferred_to_replay_refresh",
        "date": settled,
        "writer": "advance-challenger",
    }
    dashboard_data = build_publish_surface(s3.s3)
    publish_report = publish_line(dashboard_data, s3.s3, phase="night",
                                  run_date=settled)
    if publish_report.get("held"):
        raise RuntimeError(
            "night line re-publish held: " + str(publish_report.get("reason"))
        )

    # These in-band canaries validate the freshly-written chassis. Displayed-line
    # freshness is checked only after the 04:30 replay writer, never before it.
    canary = run_post_pipeline_canaries(tier="live", alert=True)
    # CL-708150: empty-but-critical config canary — fails LOUD if a load-bearing
    # table (regime_compatibility / theta_sel.regime_admissibility) is present-but-
    # empty (the silent-inert class that let the regime gate sit dead). Alerting,
    # non-abort (the engine already ran under the assert_regime_chassis_loaded gate).
    from monitors.config_canary import run_config_canary
    config_canary = run_config_canary(alert=True, config=config)
    post_ok = bool(canary.get("ok") and config_canary.get("ok"))
    emit_run_heartbeat(region=region, ok=post_ok)
    if not post_ok:
        raise RuntimeError(
            "night post-pipeline checks failed: "
            f"canary_ok={canary.get('ok')} config_canary_ok={config_canary.get('ok')}"
        )

    return {"statusCode": 200, "body": json.dumps({
        "status": "success", "phase": "night", "date": settled,
        "intents_count": len(trade_intents.get("actions", [])),
        "selected": selected_universe, "engine": res.engine,
        "append": append_report, "publish": publish_report,
        "post_pipeline": {"canary_ok": canary.get("ok"),
                          "line_health": "deferred_to_post_replay_healthcheck",
                          "config_canary_ok": config_canary.get("ok")},
    }, default=str)}
