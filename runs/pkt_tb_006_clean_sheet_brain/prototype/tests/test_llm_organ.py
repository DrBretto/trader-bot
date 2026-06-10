"""Tests for the LLM sentiment organ (BUILD_SPEC §3): funnel determinism, seen-cache
decay math, neutral artifact shape, spend-cap refusal, feature emission masks."""
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm_organ as lo  # noqa: E402
import features_llm as fl  # noqa: E402


def make_day_df(n_extra: int = 0) -> pd.DataFrame:
    """Synthetic finance-relevant records frame matching the records parquet schema."""
    rows = [
        # syndicated story: same slug words on 3 domains -> one cluster, n_sources=3
        dict(url="https://a.com/news/fed-holds-rates-steady-as-inflation-cools.html",
             source_domain="a.com", tone=-1.0,
             themes="ECON_INTEREST_RATE|EPU_POLICY_INTEREST", locations="US",
             orgs="federal reserve", quotations="we need greater confidence"),
        dict(url="https://b.com/2026/02/fed-holds-rates-steady-as-inflation-cools",
             source_domain="b.com", tone=-0.5,
             themes="ECON_INTEREST_RATE", locations="US", orgs="", quotations=""),
        dict(url="https://c.com/markets/inflation-cools-as-fed-holds-rates-steady/99",
             source_domain="c.com", tone=-0.8,
             themes="ECON_INFLATION", locations="US", orgs="", quotations=""),
        # separate story, wordy
        dict(url="https://d.com/china-export-curbs-hit-global-chipmakers-hard",
             source_domain="d.com", tone=-3.2,
             themes="ECON_TRADE_DISPUTE|ECON_STOCKMARKET", locations="CH",
             orgs="nvidia|tsmc", quotations="a serious blow to the industry | more"),
        # non-wordy slug (numeric id only) -> clusters but never promptable
        dict(url="https://e.com/article/123456", source_domain="e.com", tone=2.0,
             themes="ECON_STOCKMARKET", locations="US", orgs="", quotations=""),
    ]
    for i in range(n_extra):
        rows.append(dict(
            url=f"https://x{i}.com/unique-story-number-{'x' * (i % 3 + 1)}-alpha-beta-gamma-delta-{chr(97 + i % 26)}",
            source_domain=f"x{i}.com", tone=float(i % 5 - 2),
            themes="ECON_STOCKMARKET", locations="US", orgs="", quotations=""))
    df = pd.DataFrame(rows)
    df["ts"] = pd.Timestamp("2026-02-05 06:00:00", tz="UTC")
    for c in ["pos", "neg", "polarity"]:
        df[c] = 1.0
    return df


# ------------------------------------------------------------------ funnel

def test_funnel_determinism():
    df = make_day_df(n_extra=30)
    c1 = lo.build_clusters(df)
    c2 = lo.build_clusters(df.sample(frac=1.0, random_state=7))  # shuffled rows
    assert [c["cluster_id"] for c in c1] == [c["cluster_id"] for c in c2]
    assert c1 == c2


def test_syndication_dedup_and_ranking():
    df = make_day_df()
    clusters = lo.build_clusters(df)
    fed = [c for c in clusters if "fed" in c["headline"]]
    assert len(fed) == 1, "word-order variants of the same slug must collapse"
    assert fed[0]["n_sources"] == 3
    assert fed[0]["wordy"]
    # bucket-mapped cluster gets theme_priority 2.0
    assert lo.theme_priority(fed[0]) == 2.0
    # rank: n_sources * (1+|tone|/5) * priority
    exp = 3 * (1 + abs(fed[0]["tone"]) / 5.0) * 2.0
    assert lo.rank_score(fed[0]) == pytest.approx(exp)
    # non-wordy cluster exists but is not promptable
    nonwordy = [c for c in clusters if not c["wordy"]]
    assert nonwordy and all(len(c["headline"].split()) < 4 for c in nonwordy)


def test_quote_truncation():
    df = make_day_df()
    clusters = lo.build_clusters(df)
    chip = [c for c in clusters if "chipmakers" in c["headline"]][0]
    assert chip["quote"] == "a serious blow to the industry"  # first quote only
    assert len(chip["quote"].split()) <= lo.QUOTE_MAX_WORDS


# ------------------------------------------------------------------ F3 seen-cache decay

def test_seen_cache_decay_math():
    day0 = dt.date(2026, 2, 5)
    c = dict(cluster_id="abc", n_sources=8, buckets=["us_broad"], tone=-1.0,
             headline="h", wordy=True, themes=[], actors=[], quote="")
    seen = lo.update_seen({}, [c], [], day0)
    assert seen["abc"]["weight"] == 8.0

    # day+1: suppressed, decayed mass = 8 * 0.5
    novel, mass, supp = lo.split_seen([c], seen, day0 + dt.timedelta(days=1))
    assert novel == []
    assert mass["us_broad"] == pytest.approx(4.0)
    assert supp == pytest.approx(4.0)

    # day+3: 8 * 0.5^3
    _, mass3, _ = lo.split_seen([c], seen, day0 + dt.timedelta(days=3))
    assert mass3["us_broad"] == pytest.approx(1.0)

    # salience prior normalization: decayed 4.0 vs novel mass 12 -> 4/16
    prior = lo.salience_prior({"us_broad": 4.0}, novel_mass=12.0, supp_total=4.0)
    assert prior["us_broad"] == pytest.approx(0.25)

    # window pruning: with no reappearance, entry expires after 5 calendar days
    pruned = lo.update_seen(seen, [], [], day0 + dt.timedelta(days=6))
    assert "abc" not in pruned
    novel2, _, _ = lo.split_seen([c], pruned, day0 + dt.timedelta(days=6))
    assert novel2 == [c], "expired story is novel (billable) again"

    # reappearance refreshes last_seen, keeping it suppressed past day 5
    seen2 = lo.update_seen(dict(seen), [], ["abc"], day0 + dt.timedelta(days=4))
    assert "abc" in lo.update_seen(seen2, [], [], day0 + dt.timedelta(days=8))


# ------------------------------------------------------------------ neutral artifact

def test_neutral_artifact_shape():
    day = dt.date(2026, 2, 5)
    payload = lo.neutral_payload(day)
    assert set(payload["buckets"]) == set(lo.BUCKETS) and len(lo.BUCKETS) == 27
    assert all(v == {"sent": 0.0, "conf": 0.0, "sal": 0.0}
               for v in payload["buckets"].values())
    assert payload["global"] == {"risk_appetite": 0.0, "rates_pressure": 0.0,
                                 "geopolitical_risk": 0.0}
    assert payload["event_flags"] == []
    # the neutral payload itself satisfies the frozen schema
    lo.validate_response(json.dumps(payload))

    art = lo._base_artifact(day, "thin_input", 3, {})
    assert art["visible_from"] == "2026-02-06"
    assert art["llm_status"] == "thin_input"
    assert art["input_tokens"] == 0 and art["model_used"] is None


def test_clamp_to_canon():
    obj = lo.neutral_payload(dt.date(2026, 2, 5))
    obj["buckets"]["tech"]["sent"] = 0.9
    obj["buckets"]["made_up_bucket"] = {"sent": 1, "conf": 1, "sal": 1}
    del obj["buckets"]["usd"]  # missing bucket -> zeros
    obj["event_flags"] = [{"flag": "cb_decision", "buckets": ["us_broad", "nope"],
                           "severity": 7}]
    out = lo.clamp_to_canon(obj)
    assert set(out["buckets"]) == set(lo.BUCKETS)
    assert "made_up_bucket" not in out["buckets"]
    assert out["buckets"]["usd"] == {"sent": 0.0, "conf": 0.0, "sal": 0.0}
    assert out["buckets"]["tech"]["sent"] == 0.9
    assert out["event_flags"][0]["buckets"] == ["us_broad"]
    assert out["event_flags"][0]["severity"] == 1.0  # clamped


# ------------------------------------------------------------------ spend cap

def test_spend_cap_refusal(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(lo, "LEDGER_PATH", ledger)
    # near-cap ledger
    lo.ledger_append(None, 0, 0, "seed")
    with open(ledger) as f:
        entry = json.loads(f.readline())
    entry["cumulative_usd"] = lo.HARD_CAP_USD - 0.001
    ledger.write_text(json.dumps(entry) + "\n")

    def boom(*a, **k):  # the cap must refuse BEFORE any network call
        raise AssertionError("bedrock was invoked past the cap")

    monkeypatch.setattr(lo, "_invoke_bedrock", boom)
    with pytest.raises(lo.SpendCapError):
        lo.check_cap_or_refuse(est_input_tokens=10_000)
    # under-cap estimate passes
    entry["cumulative_usd"] = 0.50
    ledger.write_text(json.dumps(entry) + "\n")
    lo.check_cap_or_refuse(est_input_tokens=10_000)  # no raise


def test_model_cutoff_assertion():
    with pytest.raises(lo.ModelCutoffError):
        lo.assert_model_cutoff(dt.date(2023, 8, 31))  # not strictly after cutoff
    with pytest.raises(lo.ModelCutoffError):
        lo.assert_model_cutoff(dt.date(2022, 1, 1))
    lo.assert_model_cutoff(dt.date(2024, 1, 2))  # ok


# ------------------------------------------------------------------ feature emission

def _write_artifact(store: Path, day: dt.date, status: str, sent: float = 0.3,
                    sal_prior: dict | None = None, flags: list | None = None):
    payload = lo.neutral_payload(day)
    if status == "ok":
        payload["buckets"]["us_broad"] = {"sent": sent, "conf": 0.6, "sal": 0.4}
        payload["global"]["risk_appetite"] = 0.2
        payload["event_flags"] = flags or []
    art = lo._base_artifact(day, status, 50, sal_prior or {})
    art["llm"] = payload
    if status == "ok":
        art["input_tokens"] = 9000
        art["output_tokens"] = 1200
        art["model_used"] = lo.MODEL_ID
    store.mkdir(parents=True, exist_ok=True)
    with open(store / f"{day.isoformat()}.json", "w") as f:
        json.dump(art, f)


def test_feature_emission_masks(tmp_path):
    store = tmp_path / "llm"
    d0 = dt.date(2026, 2, 5)
    _write_artifact(store, d0, "ok", sent=0.4,
                    flags=[{"flag": "cb_decision", "buckets": ["us_broad"],
                            "severity": 0.8}])
    _write_artifact(store, d0 + dt.timedelta(days=1), "ok", sent=0.2,
                    sal_prior={"us_broad": 0.9})
    # day +2 missing entirely; day +3 thin_input neutral
    _write_artifact(store, d0 + dt.timedelta(days=3), "thin_input")

    df = fl.build_features(store_dir=store, grid_start=dt.date(2026, 2, 3),
                           grid_end=d0 + dt.timedelta(days=3))
    assert len(df) == 6  # 2026-02-03 .. 2026-02-08 inclusive
    g = lambda d: df.loc[pd.Timestamp(d)]  # noqa: E731

    # pre-coverage + gap rows: zeros, unavailable
    for d in ["2026-02-03", "2026-02-04", "2026-02-07"]:
        row = g(d)
        assert row["llm_available"] == 0 and row["llm_status_ok"] == 0
        assert row["llm_sent_us_broad"] == 0 and row["llm_sent_us_broad_ema3"] == 0

    assert g("2026-02-05")["llm_available"] == 1
    assert g("2026-02-05")["llm_status_ok"] == 1
    assert g("2026-02-05")["llm_sent_us_broad"] == pytest.approx(0.4)
    assert g("2026-02-05")["llm_event_cb_decision"] == 1
    assert g("2026-02-06")["llm_event_cb_decision"] == 0
    # sal merge: 0.4 LLM sal + 0.9 prior clamped to 1.0
    assert g("2026-02-06")["llm_sal_us_broad"] == pytest.approx(1.0)
    # d1 on available days: 0.2 - 0.4
    assert g("2026-02-06")["llm_sent_us_broad_d1"] == pytest.approx(-0.2)
    # thin_input day: available but not ok, neutral values
    assert g("2026-02-08")["llm_available"] == 1
    assert g("2026-02-08")["llm_status_ok"] == 0
    assert g("2026-02-08")["llm_sent_us_broad"] == 0
    # visible_from = d+1
    assert g("2026-02-05")["visible_from"] == pd.Timestamp("2026-02-06")
    # full feature width: 27*5 + 3 axes + 12 flags + 2 masks + visible_from
    assert df.shape[1] == 27 * 5 + 3 + 12 + 2 + 1 + 1  # +n_clusters (visible_from in the +2+1 base)
