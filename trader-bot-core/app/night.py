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

Portfolio-load + settled-mark seam: the first-class clean-core portfolio loader is
a P9 item; until it lands this uses the shared ``src.steps.paper_trader`` loader
(a first-class shared util, NOT a ``runs/`` prototype import).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

EXPIRES_AFTER_DAYS = 3


def _load_universe_df(config_dir):
    import pandas as pd  # pipeline glue (universe DataFrame), not compute
    return pd.read_csv(config_dir / "universe.csv")


def run_night(event: dict, bucket: str, region: str) -> Dict[str, Any]:
    """Run one forward night. Returns a Lambda-shaped response dict."""
    from src.utils.s3_client import S3Client
    from src.utils.sns_alerts import send_alert
    from src.steps import paper_trader
    from decide.cutover import (run_cutover, load_brain_config,
                                production_forecaster, _regime_compat_path)
    from decide.freshness_gate import _latest_settled_trading_day
    from lines.append import append_settled_point_for_publish
    from publish.dashboard import build_line_surface, publish_line
    from monitors.canary_gate import run_post_pipeline_canaries
    from monitors.watchdog import run_daily_health_check, emit_run_heartbeat

    s3 = S3Client(bucket, region)
    settled = event.get("run_date") or _latest_settled_trading_day()

    config = dict(load_brain_config())
    config["mode"] = "live"
    universe_df = _load_universe_df(_regime_compat_path().parent)
    portfolio_state = paper_trader.load_portfolio_state(s3)

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

    # ---- persist intents (EXACT schema the morning executor consumes) ----
    trade_intents = dict(res.trade_intents)
    trade_intents.setdefault("generated_timestamp", datetime.now().isoformat())
    trade_intents.setdefault("expires_after_days", EXPIRES_AFTER_DAYS)
    s3.write_json(trade_intents, f"daily/{settled}/trade_intents.json")

    # ---- append the settled leaf, then publish non-destructively ----
    append_report = append_settled_point_for_publish(s3.s3, settled, portfolio_state,
                                                      bucket=bucket)
    dashboard_data = build_line_surface(s3.s3)
    publish_report = publish_line(dashboard_data, s3.s3, phase="night",
                                  run_date=settled)

    # ---- post-pipeline watchdogs (the check missing during the silent freeze) ----
    canary = run_post_pipeline_canaries(tier="live", alert=True)
    health = run_daily_health_check(s3)
    emit_run_heartbeat(region=region, ok=True)

    return {"statusCode": 200, "body": json.dumps({
        "status": "success", "phase": "night", "date": settled,
        "intents_count": len(trade_intents.get("actions", [])),
        "selected": res.selected_universe, "engine": res.engine,
        "append": append_report, "publish": publish_report,
        "post_pipeline": {"canary_ok": canary.get("ok"),
                          "health_ok": health.get("ok")},
    }, default=str)}
