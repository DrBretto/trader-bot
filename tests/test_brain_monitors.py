from src.brain import monitors


class FakeS3:
    def __init__(self, docs):
        self.docs = docs

    def read_json(self, key):
        return self.docs.get(key)

    def read_jsonl(self, key):
        return self.docs.get(key, [])


def test_stale_publish_uses_challenger_line_date_not_file_as_of():
    alerts = []
    s3 = FakeS3({
        "dashboard/shadow_timeseries.json": {
            "as_of": "2026-06-30T03:30:27+00:00",
            "shadow_A": [["2026-06-26", 114035.88]],
        },
        "daily/latest.json": {"intents_date": "2026-06-30"},
    })

    status = monitors.check_stale_publish(s3, alert=lambda s, b: alerts.append((s, b)))

    assert status["stale"] is True
    assert status["brain_as_of"] == "2026-06-30"
    assert status["challenger_last_date"] == "2026-06-26"
    assert status["lag_trading_days"] == 2
    assert alerts


def test_daily_health_reports_challenger_plotted_date():
    emails = []
    s3 = FakeS3({
        "dashboard/shadow_timeseries.json": {
            "as_of": "2026-06-30T03:30:27+00:00",
            "shadow_A": [["2026-06-30", 117025.29]],
        },
        "daily/latest.json": {"intents_date": "2026-06-30"},
        "canon/equity_ledger/equity_history.jsonl": [{"date": "2026-06-30"}],
    })

    status = monitors.run_daily_health_check(
        s3,
        alert=lambda subject, body: emails.append((subject, body)),
        today="2026-06-30",
    )

    challenger = status["lines"]["challenger (dotted)"]
    assert challenger["stale"] is False
    assert challenger["at"] == "2026-06-30"
    assert status["ok"] is True
    assert emails
