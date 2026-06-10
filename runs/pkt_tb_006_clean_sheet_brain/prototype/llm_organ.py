"""PKT-TB-006 SYN-1 prototype — LLM sentiment organ (BUILD_SPEC §3; PROPOSAL_LLM_SENTIMENT).

Implements the F0-F4 funnel over the per-day finance-relevant GKG records cached at
gdelt_cache/records/<YYYYMMDD>.parquet, assembles ONE Bedrock Haiku call per GDELT day,
validates the response against the frozen schema, and writes per-day artifacts to
store/llm/<YYYY-MM-DD>.json. Every call is appended to bedrock_spend.jsonl under a HARD
$3.10 cap — the driver refuses any call that could exceed it.

Funnel (proposal §1, BUILD_SPEC §3):
  F0  read records/<d>.parquet (the 2-4 sampled GKG files of UTC day d, pre-parsed)
  F1  relevance: theme prefix in THEMES_FIN OR location fips in ACTOR_MAP (non-US) —
      identical rule to the parse-time pre-filter (re-applied, harmless)
  F2  slug-shingle clustering: sorted stopword-stripped slug tokens, first-6-word shingle
      hash = cluster_id; n_sources = distinct source domains in cluster
  F3  5-calendar-day seen-cache, chronologically threaded: a billed story is never
      re-billed; its weight decays x0.5/day into per-bucket salience priors
  F4  rank = n_sources * (1 + |tone|/5) * theme_priority; top 120 clusters;
      <10 clusters => no call, neutral artifact with llm_status="thin_input"

Model-cutoff rule (TOURNAMENT §4.6.1): hard assertion model_cutoff < window_start.
Replays NEVER re-invoke the LLM — they read the stored artifacts only.

Forced gap-fills (no spec definition existed; surfaced in validation_looks.jsonl):
  - theme_priority: 2.0 if any cluster theme maps to a bucket via theme_to_sector.json
    (frozen dict), else 1.0.
  - slug segment: path segment with the most alpha tokens (proposal says "last path
    segment", but GDELT URLs frequently end in a numeric article id; last-segment-only
    would silently drop those headlines).
  - salience-prior merge: feature-side sal = clamp(llm_sal + sal_prior, 0, 1), where
    sal_prior[b] = decayed suppressed mass for b / total (novel + decayed) mass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

PROTO = Path(__file__).resolve().parent
RECORDS_DIR = PROTO / "gdelt_cache" / "records"
DAILY_DIR = PROTO / "gdelt_cache" / "daily"
DICTS_DIR = PROTO / "dicts"
STORE_DIR = PROTO / "store" / "llm"
SEEN_STATE_PATH = PROTO / "store" / "llm_seen_state.json"
LEDGER_PATH = PROTO / "bedrock_spend.jsonl"
PROMPT_PATH = PROTO / "prompts" / "llm_sentiment_system.txt"
SCHEMA_PATH = PROTO / "schemas" / "llm_sentiment.json"
LOOKS_PATH = PROTO / "validation_looks.jsonl"

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_CUTOFF = dt.date(2023, 8, 31)  # public cutoff 2023-08 (TOURNAMENT §4.6.1)
PRICE_IN = 0.25 / 1e6   # $/input token
PRICE_OUT = 1.25 / 1e6  # $/output token
HARD_CAP_USD = 3.10     # Phase C Bedrock hard cap (BUILD_SPEC §3)
MAX_TOKENS = 1500
TEMPERATURE = 0.0
TOP_N_CLUSTERS = 120    # registered constant (BUILD_SPEC §16)
MIN_CLUSTERS = 10
SEEN_WINDOW_DAYS = 5
DECAY_PER_DAY = 0.5
QUOTE_MAX_WORDS = 25
AWS_PROFILE = os.environ.get("AWS_PROFILE", "personal")
REGION = "us-east-1"

BUCKETS_LINE = (
    "us_broad, tech, semis, financials_banks, energy_oil, natgas, real_estate, intl_dev,\n"
    "europe, japan, em_broad, china, india, brazil, healthcare_biotech, industrials_defense,\n"
    "consumer_disc, consumer_staples, utilities, materials, rates_duration, credit, gold_pm,\n"
    "commodity_broad, usd, eur, volatility"
)
EVENT_FLAGS = [
    "cb_decision", "cb_speech", "inflation_print", "jobs_print", "earnings_megacap",
    "geopolitical_escalation", "sanctions_tariffs", "credit_event", "energy_supply_shock",
    "election_political", "regulatory_action", "natural_disaster",
]
FLAGS_LINE = (
    "cb_decision, cb_speech, inflation_print, jobs_print, earnings_megacap,\n"
    "geopolitical_escalation, sanctions_tariffs, credit_event, energy_supply_shock, "
    "election_political,\nregulatory_action, natural_disaster"
)

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the to was "
    "were will with after amid over under into out up down new says said say his her their "
    "this these those you your what when why how who which".split()
)

# ------------------------------------------------------------------ dictionaries (frozen)

with open(DICTS_DIR / "bucket_map.json") as _f:
    BUCKET_MAP: Dict[str, List[str]] = json.load(_f)
BUCKETS: List[str] = list(BUCKET_MAP.keys())  # 27, insertion order
with open(DICTS_DIR / "THEMES_FIN.json") as _f:
    THEMES_FIN = tuple(json.load(_f))
with open(DICTS_DIR / "ACTOR_MAP.json") as _f:
    ACTOR_MAP = json.load(_f)
with open(DICTS_DIR / "theme_to_sector.json") as _f:
    THEME_TO_SECTOR = json.load(_f)
_T2S_PREFIXES = sorted(THEME_TO_SECTOR.items(), key=lambda kv: -len(kv[0]))
_t2s_memo: Dict[str, Optional[str]] = {}

# fips -> bucket (all ACTOR_MAP countries, incl. US for salience attribution)
FIPS_TO_BUCKET: Dict[str, str] = {}
for _cc, _spec in ACTOR_MAP.items():
    for _fips in _spec["fips"]:
        FIPS_TO_BUCKET[_fips] = _spec["bucket"]
# finance-relevance fips: non-US only (mirrors gdelt_backfill.py parse-time rule)
FIN_FIPS = {f for cc, s in ACTOR_MAP.items() if cc != "US" for f in s["fips"]}

SYSTEM_PROMPT = PROMPT_PATH.read_text()
with open(SCHEMA_PATH) as _f:
    OUTPUT_SCHEMA = json.load(_f)

# Forced gap-fill (logged in validation_looks.jsonl): the frozen system prompt says
# "emit a single JSON object" but never communicates the wrapper structure; in pilot
# run 1 the pinned Haiku emitted bucket keys at TOP level (plus an invented "event_risk"
# key) and failed schema validation on 14/14 days, repair retry included. The user
# message therefore carries an explicit OUTPUT FORMAT block (the system prompt stays
# frozen verbatim; the user template is the call-assembly surface).
OUTPUT_FORMAT_BLOCK = (
    "OUTPUT FORMAT — a single compact JSON object with top-level keys exactly "
    '["as_of","buckets","global","event_flags"]. Under "buckets", include an entry for '
    "every BUCKET listed above that has relevant stories (others may be omitted), each "
    'shaped {"sent":number,"conf":number,"sal":number}. Example SHAPE only — the numbers '
    "below are placeholders, score from tonight's stories:\n"
    '{"as_of":"<date>","buckets":{"us_broad":{"sent":-0.3,"conf":0.6,"sal":0.5},'
    '"energy_oil":{"sent":0.4,"conf":0.5,"sal":0.2}},'
    '"global":{"risk_appetite":-0.2,"rates_pressure":0.1,"geopolitical_risk":0.3},'
    '"event_flags":[{"flag":"<one of the ALLOWED EVENT FLAGS, only if such an event is '
    'actually in the stories>","buckets":["<bucket>"],"severity":0.6}]}'
)


def theme_bucket(theme: str) -> Optional[str]:
    """Theme code -> bucket key (exact, then longest prefix), memoized."""
    if theme in _t2s_memo:
        return _t2s_memo[theme]
    bucket = THEME_TO_SECTOR.get(theme)
    if bucket is None:
        for prefix, b in _T2S_PREFIXES:
            if theme.startswith(prefix):
                bucket = b
                break
    _t2s_memo[theme] = bucket
    return bucket


# ------------------------------------------------------------------ errors

class SpendCapError(RuntimeError):
    """Raised instead of making any Bedrock call that could exceed HARD_CAP_USD."""


class ModelCutoffError(RuntimeError):
    """Raised when a backfill window starts at/before the model's training cutoff."""


# ------------------------------------------------------------------ funnel F1-F2

_EXT_RE = re.compile(r"\.(html?|php|aspx?|cfm|shtml|htm|amp)$", re.I)


def slug_tokens(url: str) -> List[str]:
    """Headline-equivalent tokens from the URL path segment with the most alpha tokens."""
    try:
        path = urlsplit(url).path
    except ValueError:
        return []
    best: List[str] = []
    for seg in path.split("/"):
        seg = _EXT_RE.sub("", seg)
        toks = [t.lower() for t in re.split(r"[-_+]", seg)
                if t.isalpha() and 2 <= len(t) <= 24]
        if len(toks) > len(best):
            best = toks
    return best


def record_relevant(themes: List[str], locations: List[str]) -> bool:
    """F1 — identical to the parse-time pre-filter."""
    for t in themes:
        if t.startswith(THEMES_FIN):
            return True
    return any(c in FIN_FIPS for c in locations)


def record_buckets(themes: List[str], locations: List[str]) -> List[str]:
    """Buckets a record touches (for salience attribution + theme_priority)."""
    out = []
    for t in themes:
        b = theme_bucket(t)
        if b and b not in out:
            out.append(b)
    for c in locations:
        b = FIPS_TO_BUCKET.get(c)
        if b and b not in out:
            out.append(b)
    return out


def build_clusters(df) -> List[Dict[str, Any]]:
    """F1+F2 over a day's records frame -> deterministic list of cluster dicts."""
    groups: Dict[str, Dict[str, Any]] = {}
    # deterministic iteration: sort by (url, ts) so cluster aggregates never depend on
    # parquet row order
    rows = df.to_dict("records")
    rows.sort(key=lambda r: (str(r.get("url", "")), str(r.get("ts", ""))))
    for r in rows:
        themes = [t for t in str(r.get("themes") or "").split("|") if t]
        locations = [c for c in str(r.get("locations") or "").split("|") if c]
        if not record_relevant(themes, locations):
            continue  # F1 (records are pre-filtered; re-applied for self-containment)
        toks = slug_tokens(str(r.get("url") or ""))
        norm = sorted({t for t in toks if t not in STOPWORDS})
        if not norm:
            continue  # nothing to cluster on
        shingle = "|".join(norm[:6])
        cid = hashlib.sha1(shingle.encode()).hexdigest()[:16]
        g = groups.get(cid)
        if g is None:
            g = groups[cid] = {
                "cluster_id": cid, "domains": set(), "tones": [], "members": 0,
                "rep_toks": [], "rep_themes": [], "rep_orgs": [], "quote": "",
                "buckets": [],
            }
        g["domains"].add(str(r.get("source_domain") or ""))
        g["tones"].append(float(r.get("tone") or 0.0))
        g["members"] += 1
        for b in record_buckets(themes, locations):
            if b not in g["buckets"]:
                g["buckets"].append(b)
        if len(toks) > len(g["rep_toks"]):
            g["rep_toks"] = toks
            g["rep_themes"] = themes[:4]
            orgs = [o for o in str(r.get("orgs") or "").split("|") if o]
            g["rep_orgs"] = orgs[:3]
        if not g["quote"]:
            q = str(r.get("quotations") or "").split(" | ")[0].strip()
            if q:
                g["quote"] = " ".join(q.split()[:QUOTE_MAX_WORDS])

    clusters = []
    for g in groups.values():
        clusters.append({
            "cluster_id": g["cluster_id"],
            "headline": " ".join(g["rep_toks"]),
            "wordy": len(g["rep_toks"]) >= 4,
            "themes": g["rep_themes"],
            "actors": g["rep_orgs"],
            "tone": sum(g["tones"]) / len(g["tones"]),
            "n_sources": len(g["domains"]),
            "quote": g["quote"],
            "buckets": g["buckets"],
        })
    clusters.sort(key=lambda c: c["cluster_id"])  # deterministic base order
    return clusters


def theme_priority(cluster: Dict[str, Any]) -> float:
    """Forced gap-fill (logged): 2.0 if the cluster maps to any bucket, else 1.0."""
    return 2.0 if cluster["buckets"] else 1.0


def rank_score(cluster: Dict[str, Any]) -> float:
    return cluster["n_sources"] * (1.0 + abs(cluster["tone"]) / 5.0) * theme_priority(cluster)


# ------------------------------------------------------------------ F3 seen-cache

def split_seen(clusters: List[Dict[str, Any]], seen: Dict[str, Any], day: dt.date
               ) -> Tuple[List[Dict[str, Any]], Dict[str, float], float]:
    """Partition into (novel, decayed-bucket-mass-of-suppressed, suppressed_total_mass).

    seen: cluster_id -> {"billed": "YYYY-MM-DD", "last_seen": "YYYY-MM-DD",
                         "weight": float, "buckets": [...]}
    Mutates nothing; pruning + insertion happen in update_seen().
    """
    novel, decayed_mass, supp_total = [], {}, 0.0
    for c in clusters:
        e = seen.get(c["cluster_id"])
        if e is None:
            novel.append(c)
            continue
        age = (day - dt.date.fromisoformat(e["billed"])).days
        w = e["weight"] * (DECAY_PER_DAY ** max(age, 0))
        supp_total += w
        for b in e["buckets"]:
            decayed_mass[b] = decayed_mass.get(b, 0.0) + w
    return novel, decayed_mass, supp_total


def update_seen(seen: Dict[str, Any], billed: List[Dict[str, Any]],
                reappeared_ids: List[str], day: dt.date) -> Dict[str, Any]:
    """Add billed clusters, refresh last_seen on reappearance, prune the 5-day window."""
    d = day.isoformat()
    for c in billed:
        seen[c["cluster_id"]] = {
            "billed": d, "last_seen": d,
            "weight": float(c["n_sources"]), "buckets": list(c["buckets"]),
        }
    for cid in reappeared_ids:
        if cid in seen:
            seen[cid]["last_seen"] = d
    return {
        cid: e for cid, e in seen.items()
        if (day - dt.date.fromisoformat(e["last_seen"])).days <= SEEN_WINDOW_DAYS
    }


def salience_prior(decayed_mass: Dict[str, float], novel_mass: float,
                   supp_total: float) -> Dict[str, float]:
    denom = novel_mass + supp_total
    if denom <= 0:
        return {}
    return {b: round(min(1.0, m / denom), 4) for b, m in sorted(decayed_mass.items())}


# ------------------------------------------------------------------ prompt assembly

def _san(s: str) -> str:
    return s.replace("|", "/").replace("\n", " ").strip()


def build_user_message(day: dt.date, window: Tuple[str, str],
                       lines: List[Dict[str, Any]]) -> str:
    story_lines = []
    for i, c in enumerate(lines, 1):
        parts = [
            str(i),
            _san(c["headline"]),
            _san(",".join(c["themes"])),
            _san(";".join(c["actors"])),
            f"{c['tone']:.1f}",
            str(c["n_sources"]),
        ]
        if c["quote"]:
            parts.append('"' + _san(c["quote"]) + '"')
        story_lines.append(" | ".join(parts))
    return (
        f"DATE: {day.isoformat()}  WINDOW: {window[0]} -> {window[1]}\n"
        f"BUCKETS: {BUCKETS_LINE}\n"
        f"ALLOWED EVENT FLAGS: {FLAGS_LINE}\n\n"
        f"STORIES ({len(lines)} clusters, fields: id | headline | themes | actors | tone "
        f"| n_sources | quote?):\n"
        + "\n".join(story_lines)
        + "\n\n" + OUTPUT_FORMAT_BLOCK
        + "\n\nRespond with the JSON object now.\n"
    )


# ------------------------------------------------------------------ spend ledger

def _ledger_lines() -> List[Dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    out = []
    with open(LEDGER_PATH) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def cumulative_spend() -> float:
    lines = _ledger_lines()
    return lines[-1]["cumulative_usd"] if lines else 0.0


def ledger_append(date_scored: Optional[str], input_tokens: int, output_tokens: int,
                  status: str, model_used: str = MODEL_ID) -> float:
    cost = input_tokens * PRICE_IN + output_tokens * PRICE_OUT
    cum = cumulative_spend() + cost
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "date_scored": date_scored,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 8),
        "cumulative_usd": round(cum, 8),
        "model_used": model_used,
        "status": status,
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return cum


def seed_ledger_phase_a() -> None:
    """Seed the Phase A verification call (9 in / 4 out) exactly once."""
    for ln in _ledger_lines():
        if ln.get("status") == "phase_a_verification":
            return
    ledger_append(None, 9, 4, "phase_a_verification")


def check_cap_or_refuse(est_input_tokens: int) -> None:
    """Refuse any call whose WORST-CASE cost would push cumulative spend past the cap."""
    worst = est_input_tokens * PRICE_IN + MAX_TOKENS * PRICE_OUT
    cum = cumulative_spend()
    if cum + worst > HARD_CAP_USD:
        raise SpendCapError(
            f"refusing call: cumulative ${cum:.4f} + worst-case ${worst:.4f} "
            f"> hard cap ${HARD_CAP_USD:.2f}"
        )


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5) + 16  # conservative chars/token for English+JSON


# ------------------------------------------------------------------ Bedrock

_brt = None


def _client():
    global _brt
    if _brt is None:
        import boto3
        from botocore.config import Config
        _brt = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION).client(
            "bedrock-runtime",
            config=Config(read_timeout=30, connect_timeout=10,
                          retries={"max_attempts": 0}),
        )
    return _brt


def _invoke_bedrock(system: str, user: str) -> Dict[str, Any]:
    """One invocation, 2 transport retries with exp backoff. Returns
    {text, input_tokens, output_tokens, stop_reason}."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    last_err = None
    for attempt in range(3):  # 1 try + 2 retries
        if attempt:
            time.sleep(2 ** attempt)
        try:
            resp = _client().invoke_model(modelId=MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            return {
                "text": "".join(blk.get("text", "") for blk in payload.get("content", [])),
                "input_tokens": int(payload["usage"]["input_tokens"]),
                "output_tokens": int(payload["usage"]["output_tokens"]),
                "stop_reason": payload.get("stop_reason"),
            }
        except Exception as e:  # noqa: BLE001 — retried, then surfaced
            last_err = e
    raise RuntimeError(f"bedrock dark after 3 attempts: {last_err}")


# ------------------------------------------------------------------ schema / clamp

def _preclamp(obj: Any) -> Any:
    """Clamp-on-ingest (proposal §2): numeric range overshoots in the expected slots are
    clamped BEFORE validation; structural breakage is left for the validator."""
    if not isinstance(obj, dict):
        return obj
    b = obj.get("buckets")
    if isinstance(b, dict):
        for e in b.values():
            if isinstance(e, dict):
                for k, lo_, hi in (("sent", -1, 1), ("conf", 0, 1), ("sal", 0, 1)):
                    if isinstance(e.get(k), (int, float)):
                        e[k] = max(lo_, min(hi, e[k]))
    g = obj.get("global")
    if isinstance(g, dict):
        for k, lo_, hi in (("risk_appetite", -1, 1), ("rates_pressure", -1, 1),
                           ("geopolitical_risk", 0, 1)):
            if isinstance(g.get(k), (int, float)):
                g[k] = max(lo_, min(hi, g[k]))
    fl = obj.get("event_flags")
    if isinstance(fl, list):
        kept = []
        for e in fl:
            if isinstance(e, dict):
                if isinstance(e.get("severity"), (int, float)):
                    e["severity"] = max(0, min(1, e["severity"]))
                # out-of-enum flags are DROPPED on ingest (conservative: the organ
                # discards invented flags rather than failing the whole surface;
                # pilot 2026-02-07 emitted "terror")
                if isinstance(e.get("flag"), str) and e["flag"] not in EVENT_FLAGS:
                    continue
            kept.append(e)
        obj["event_flags"] = kept
    return obj


def validate_response(text: str) -> Dict[str, Any]:
    """Strict parse + clamp-on-ingest + jsonschema validation. Raises on any failure."""
    import jsonschema
    obj = _preclamp(json.loads(text))
    jsonschema.validate(obj, OUTPUT_SCHEMA)
    return obj


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0


def clamp_to_canon(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Project a schema-valid object onto the canonical 27 buckets / 12 flags."""
    buckets = {}
    raw_b = obj.get("buckets", {})
    for b in BUCKETS:
        e = raw_b.get(b, {})
        buckets[b] = {
            "sent": _clamp(e.get("sent", 0), -1, 1),
            "conf": _clamp(e.get("conf", 0), 0, 1),
            "sal": _clamp(e.get("sal", 0), 0, 1),
        }
    g = obj.get("global", {})
    flags = []
    for fl in obj.get("event_flags", []):
        if fl.get("flag") in EVENT_FLAGS:
            flags.append({
                "flag": fl["flag"],
                "buckets": [b for b in fl.get("buckets", []) if b in BUCKETS],
                "severity": _clamp(fl.get("severity", 0.5), 0, 1),
            })
    return {
        "as_of": str(obj.get("as_of", "")),
        "buckets": buckets,
        "global": {
            "risk_appetite": _clamp(g.get("risk_appetite", 0), -1, 1),
            "rates_pressure": _clamp(g.get("rates_pressure", 0), -1, 1),
            "geopolitical_risk": _clamp(g.get("geopolitical_risk", 0), 0, 1),
        },
        "event_flags": flags,
    }


def neutral_payload(day: dt.date) -> Dict[str, Any]:
    return {
        "as_of": day.isoformat(),
        "buckets": {b: {"sent": 0.0, "conf": 0.0, "sal": 0.0} for b in BUCKETS},
        "global": {"risk_appetite": 0.0, "rates_pressure": 0.0, "geopolitical_risk": 0.0},
        "event_flags": [],
    }


# ------------------------------------------------------------------ per-day scoring

def artifact_path(day: dt.date) -> Path:
    return STORE_DIR / f"{day.isoformat()}.json"


def _write_artifact(day: dt.date, art: Dict[str, Any]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = artifact_path(day).with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(art, f, indent=1)
    os.replace(tmp, artifact_path(day))


def _base_artifact(day: dt.date, status: str, n_clusters: int,
                   sal_prior_map: Dict[str, float]) -> Dict[str, Any]:
    return {
        "date": day.isoformat(),
        "visible_from": (day + dt.timedelta(days=1)).isoformat(),
        "llm_status": status,
        "model_used": None,
        "prompt_sha256": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "n_clusters": n_clusters,
        "truncated": False,
        "sal_prior": sal_prior_map,
        "llm": neutral_payload(day),
        "raw_response": None,
    }


def score_day(day: dt.date, seen: Dict[str, Any], dry_run: bool = False
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the funnel + (maybe) the call for GDELT day `day`. Returns (artifact, seen')."""
    import pandas as pd
    rec_path = RECORDS_DIR / f"{day.strftime('%Y%m%d')}.parquet"
    if not rec_path.exists():
        art = _base_artifact(day, "dark", 0, {})
        art["note"] = "no records file"
        _write_artifact(day, art)
        return art, seen

    df = pd.read_parquet(rec_path)
    clusters = build_clusters(df)
    novel, decayed_mass, supp_total = split_seen(clusters, seen, day)
    novel_mass = float(sum(c["n_sources"] for c in novel))
    prior = salience_prior(decayed_mass, novel_mass, supp_total)
    reappeared = [c["cluster_id"] for c in clusters if c["cluster_id"] in seen]

    promptable = [c for c in novel if c["wordy"]]
    promptable.sort(key=lambda c: (-rank_score(c), c["cluster_id"]))
    lines = promptable[:TOP_N_CLUSTERS]

    if len(lines) < MIN_CLUSTERS:
        art = _base_artifact(day, "thin_input", len(lines), prior)
        _write_artifact(day, art)
        seen = update_seen(seen, lines, reappeared, day)
        return art, seen

    if len(df):
        ts = pd.to_datetime(df["ts"])
        window = (str(ts.min()), str(ts.max()))
    else:  # pragma: no cover
        window = (day.isoformat(), day.isoformat())
    user_msg = build_user_message(day, window, lines)
    prompt_sha = hashlib.sha256((SYSTEM_PROMPT + "\x00" + user_msg).encode()).hexdigest()

    art = _base_artifact(day, "dark", len(lines), prior)
    art["prompt_sha256"] = prompt_sha

    if dry_run:
        art["llm_status"] = "dry_run"
        art["est_input_tokens"] = estimate_tokens(SYSTEM_PROMPT + user_msg)
        seen = update_seen(seen, lines, reappeared, day)
        return art, seen

    in_tok = out_tok = 0
    truncated = False
    parsed = None
    status = "dark"
    raw_text = None
    try:
        check_cap_or_refuse(estimate_tokens(SYSTEM_PROMPT + user_msg))
        resp = _invoke_bedrock(SYSTEM_PROMPT, user_msg)
        in_tok += resp["input_tokens"]
        out_tok += resp["output_tokens"]
        raw_text = resp["text"]
        truncated = resp["stop_reason"] == "max_tokens"
        ledger_append(day.isoformat(), resp["input_tokens"], resp["output_tokens"], "call")
        try:
            parsed = validate_response(resp["text"])
            status = "ok"
        except Exception as ve:  # noqa: BLE001 — one schema-repair retry
            repair_msg = (
                user_msg
                + f"\n\nYour previous response failed validation: {type(ve).__name__}: "
                + str(ve)[:500]
                + "\nRespond again with ONLY the corrected JSON object."
            )
            check_cap_or_refuse(estimate_tokens(SYSTEM_PROMPT + repair_msg))
            resp2 = _invoke_bedrock(SYSTEM_PROMPT, repair_msg)
            in_tok += resp2["input_tokens"]
            out_tok += resp2["output_tokens"]
            raw_text = resp2["text"]
            truncated = truncated or resp2["stop_reason"] == "max_tokens"
            ledger_append(day.isoformat(), resp2["input_tokens"],
                          resp2["output_tokens"], "schema_repair")
            try:
                parsed = validate_response(resp2["text"])
                status = "ok"
            except Exception:  # noqa: BLE001 — fail-soft, never fabricate
                status = "dark"
    except SpendCapError:
        raise
    except Exception as e:  # noqa: BLE001 — bedrock dark => neutral artifact
        art["note"] = f"{type(e).__name__}: {str(e)[:300]}"
        status = "dark"

    art["llm_status"] = status
    art["model_used"] = MODEL_ID if (in_tok or out_tok) else None
    art["input_tokens"] = in_tok
    art["output_tokens"] = out_tok
    art["truncated"] = bool(truncated)
    art["raw_response"] = raw_text
    if parsed is not None:
        art["llm"] = clamp_to_canon(parsed)
    _write_artifact(day, art)
    seen = update_seen(seen, lines, reappeared, day)
    return art, seen


# ------------------------------------------------------------------ backfill driver

def _save_seen(seen: Dict[str, Any], as_of: dt.date) -> None:
    SEEN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SEEN_STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"as_of": as_of.isoformat(), "seen": seen}, f)
    os.replace(tmp, SEEN_STATE_PATH)


def _load_seen(expect_prev: dt.date) -> Optional[Dict[str, Any]]:
    if not SEEN_STATE_PATH.exists():
        return None
    with open(SEEN_STATE_PATH) as f:
        st = json.load(f)
    if st.get("as_of") == expect_prev.isoformat():
        return st["seen"]
    return None


def assert_model_cutoff(window_start: dt.date) -> None:
    if not MODEL_CUTOFF < window_start:
        raise ModelCutoffError(
            f"TOURNAMENT §4.6.1 violation: model cutoff {MODEL_CUTOFF} is not strictly "
            f"before window start {window_start} — scoring NOTHING."
        )


def wait_for_records(day: dt.date, max_wait_s: int = 900, poll_s: int = 60) -> bool:
    path = RECORDS_DIR / f"{day.strftime('%Y%m%d')}.parquet"
    waited = 0
    while not path.exists() and waited < max_wait_s:
        print(f"  waiting for {path.name} ({waited}s)...", flush=True)
        time.sleep(poll_s)
        waited += poll_s
    return path.exists()


def run_window(start: dt.date, end: dt.date, label: str, fresh_cache: bool = True,
               wait_missing: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """Chronological scoring of [start, end] inclusive (calendar GDELT days)."""
    assert_model_cutoff(start)
    seed_ledger_phase_a()
    seen: Dict[str, Any] = {}
    if not fresh_cache:
        resumed = _load_seen(start - dt.timedelta(days=1))
        if resumed is None:
            raise RuntimeError(
                f"--continue-cache requested but seen-state is not as_of {start - dt.timedelta(days=1)}"
            )
        seen = resumed
    missing_days, scored, t0 = [], 0, time.time()
    day = start
    try:
        while day <= end:
            rec = RECORDS_DIR / f"{day.strftime('%Y%m%d')}.parquet"
            if not rec.exists() and wait_missing:
                wait_for_records(day)
            if not rec.exists():
                missing_days.append(day.isoformat())
            art, seen = score_day(day, seen, dry_run=dry_run)
            _save_seen(seen, day)
            scored += 1
            if scored % 25 == 0 or day == end:
                print(f"[{label}] {day} status={art['llm_status']} "
                      f"n_clusters={art['n_clusters']} cum=${cumulative_spend():.4f} "
                      f"({scored} days, {time.time()-t0:.0f}s)", flush=True)
            day += dt.timedelta(days=1)
    except SpendCapError as e:
        print(f"[{label}] HARD CAP STOP at {day}: {e}", flush=True)
        return {"label": label, "stopped_at": day.isoformat(), "cap_hit": True,
                "scored": scored, "missing_days": missing_days,
                "cumulative_usd": cumulative_spend()}
    return {"label": label, "scored": scored, "missing_days": missing_days,
            "cap_hit": False, "cumulative_usd": cumulative_spend()}


# ------------------------------------------------------------------ reporting helpers

def window_stats(start: dt.date, end: dt.date) -> Dict[str, Any]:
    days, day = [], start
    while day <= end:
        p = artifact_path(day)
        if p.exists():
            with open(p) as f:
                days.append(json.load(f))
        day += dt.timedelta(days=1)
    called = [a for a in days if a["input_tokens"] > 0]
    ok = [a for a in days if a["llm_status"] == "ok"]
    trunc = [a for a in called if a.get("truncated")]
    tot_cost = sum(a["input_tokens"] * PRICE_IN + a["output_tokens"] * PRICE_OUT
                   for a in days)
    return {
        "days": len(days),
        "called": len(called),
        "ok": len(ok),
        "ok_rate": len(ok) / len(called) if called else None,
        "truncation_rate": len(trunc) / len(called) if called else None,
        "mean_input_tokens": (sum(a["input_tokens"] for a in called) / len(called))
        if called else None,
        "mean_output_tokens": (sum(a["output_tokens"] for a in called) / len(called))
        if called else None,
        "cost_window_usd": round(tot_cost, 4),
        "cost_per_day_usd": round(tot_cost / len(called), 6) if called else None,
        "statuses": {s: sum(1 for a in days if a["llm_status"] == s)
                     for s in sorted({a["llm_status"] for a in days})},
    }


def log_look(component: str, decision: str) -> None:
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "component": component, "decision": decision, "phase": "C-build"}
    with open(LOOKS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------ stage-1 kill checks

# Published Federal Reserve 2026 FOMC meeting calendar (decision = second day).
FOMC_DECISION_DAYS = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]


def stage1_checks(start: dt.date, end: dt.date) -> Dict[str, Any]:
    import numpy as np

    arts, day = [], start
    while day <= end:
        p = artifact_path(day)
        if p.exists():
            with open(p) as f:
                arts.append(json.load(f))
        day += dt.timedelta(days=1)
    ok = [a for a in arts if a["llm_status"] == "ok"]
    sent = np.array([a["llm"]["buckets"]["us_broad"]["sent"] for a in ok])

    # 1. non-degenerate variance + sign balance
    var = float(sent.var())
    nz = sent[sent != 0]
    sign_share = float(max((nz > 0).mean(), (nz < 0).mean())) if len(nz) else 1.0
    check1 = var > 0 and sign_share <= 0.95

    # 2. tone-proxy: corr vs raw daily V2TONE global tone (gdelt_cache/daily tone_mean)
    tones, sents = [], []
    for a in ok:
        dpath = DAILY_DIR / f"{a['date'].replace('-', '')}.json"
        if dpath.exists():
            with open(dpath) as f:
                dd = json.load(f)
            if dd.get("tone_mean") is not None:
                tones.append(dd["tone_mean"])
                sents.append(a["llm"]["buckets"]["us_broad"]["sent"])
    corr = float(np.corrcoef(sents, tones)[0, 1]) if len(tones) >= 3 else float("nan")
    check2 = bool(corr < 0.8)

    # 3. event flags: <20% of ordinary days; fire on >=3 known scheduled events
    flag_days = [a for a in ok if a["llm"]["event_flags"]]
    known = {dt.date.fromisoformat(s) for s in FOMC_DECISION_DAYS
             if start <= dt.date.fromisoformat(s) <= end}
    scheduled_hits = []
    for a in ok:
        d = dt.date.fromisoformat(a["date"])
        flags = {f["flag"] for f in a["llm"]["event_flags"]}
        if "cb_decision" in flags and any(abs((d - k).days) <= 1 for k in known):
            scheduled_hits.append((a["date"], "cb_decision/FOMC"))
        if "inflation_print" in flags and 9 <= d.day <= 15:
            scheduled_hits.append((a["date"], "inflation_print/CPI-window"))
        if "jobs_print" in flags and d.weekday() == 4 and d.day <= 7:
            scheduled_hits.append((a["date"], "jobs_print/first-Friday"))
    ordinary = [a for a in ok
                if dt.date.fromisoformat(a["date"]) not in known
                and not (9 <= dt.date.fromisoformat(a["date"]).day <= 15)]
    ordinary_fire = (sum(1 for a in ordinary if a["llm"]["event_flags"]) / len(ordinary)
                     if ordinary else float("nan"))
    check3 = len(scheduled_hits) >= 3 and ordinary_fire < 0.20

    # 4. truncation < 2%
    called = [a for a in arts if a["input_tokens"] > 0]
    trunc_rate = (sum(1 for a in called if a.get("truncated")) / len(called)
                  if called else 0.0)
    check4 = trunc_rate < 0.02

    return {
        "window": [start.isoformat(), end.isoformat()],
        "n_artifacts": len(arts), "n_ok": len(ok),
        "1_variance": {"var": var, "max_sign_share": sign_share, "pass": bool(check1)},
        "2_tone_proxy": {"corr_us_broad_vs_v2tone": corr, "n_pairs": len(tones),
                         "bar": 0.8, "pass": bool(check2)},
        "3_event_flags": {
            "scheduled_hits": scheduled_hits[:40],
            "n_scheduled_hits": len(scheduled_hits),
            "any_flag_day_share": len(flag_days) / len(ok) if ok else None,
            "ordinary_day_fire_rate": ordinary_fire, "pass": bool(check3)},
        "4_truncation": {"rate": trunc_rate, "n_called": len(called),
                         "pass": bool(check4)},
        "all_pass": bool(check1 and check2 and check3 and check4),
    }


# ------------------------------------------------------------------ CLI

PILOT = (dt.date(2026, 2, 5), dt.date(2026, 2, 18))
TIER1 = (dt.date(2026, 1, 29), dt.date(2026, 6, 9))
TIER2 = (dt.date(2024, 1, 2), dt.date(2026, 1, 28))


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM sentiment organ backfill driver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, (s, e) in [("pilot", PILOT), ("tier1", TIER1), ("tier2", TIER2)]:
        p = sub.add_parser(name)
        p.add_argument("--start", default=s.isoformat())
        p.add_argument("--end", default=e.isoformat())
        p.add_argument("--continue-cache", action="store_true")
        p.add_argument("--wait-missing", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    st = sub.add_parser("stats")
    st.add_argument("start")
    st.add_argument("end")
    s1 = sub.add_parser("stage1")
    s1.add_argument("--start", default=TIER1[0].isoformat())
    s1.add_argument("--end", default=TIER1[1].isoformat())
    args = ap.parse_args()

    if args.cmd in ("pilot", "tier1", "tier2"):
        start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
        res = run_window(start, end, args.cmd, fresh_cache=not args.continue_cache,
                         wait_missing=args.wait_missing, dry_run=args.dry_run)
        print(json.dumps(res, indent=1))
        print(json.dumps(window_stats(start, end), indent=1))
    elif args.cmd == "stats":
        print(json.dumps(window_stats(dt.date.fromisoformat(args.start),
                                      dt.date.fromisoformat(args.end)), indent=1))
    elif args.cmd == "stage1":
        print(json.dumps(stage1_checks(dt.date.fromisoformat(args.start),
                                       dt.date.fromisoformat(args.end)), indent=1))


if __name__ == "__main__":
    main()
