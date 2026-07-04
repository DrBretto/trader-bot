"""
PKT-TB-006 GDELT backfill: restartable downloader + parser for GKG 2.0 + Events 2.0.

Per UTC day d, at density N files/day (4 = 00/06/12/18 UTC, 2 = 00/12 UTC), downloads
GKG + events zips from http://data.gdeltproject.org/gdeltv2/ and writes:

  gdelt_cache/records/<YYYYMMDD>.parquet
      Slim per-record extract of FINANCE-RELEVANT GKG records only (a record is relevant
      if any V1 theme prefix-matches THEMES_FIN.json, or any V1Location country code is
      in ACTOR_MAP.json fips sets). Columns:
        ts, source_domain, url, themes (V2 codes stripped of offsets, pipe-joined),
        tone, pos, neg, polarity, locations (country codes pipe-joined),
        orgs (top 10 pipe-joined), quotations (concatenated, truncated 2000 chars).
      This serves the LLM funnel (PROPOSAL_LLM_SENTIMENT F0-F4): URL slugs, quotes,
      themes, tone, source domains — no re-download needed.

  gdelt_cache/daily/<YYYYMMDD>.json
      Day aggregates over ALL records (not just finance-relevant): n_records, global
      tone mean/std/neg_share/polarity_mean; theme_counts (top 500); per-bucket theme
      mention counts + per-bucket tone mean/std (via dicts/theme_to_sector.json, values
      are bucket_map.json keys); location country counts (top 50); org counts (top 100);
      us_share; PLUS Events 2.0 aggregates: per-country (US,CN,JP,BR,IN,RU,EU aggregate)
      mention-weighted mean Goldstein + conflict share (QuadClass 3/4 mention share);
      n_events.

  gdelt_cache/manifest.jsonl  — one line per completed day (density break is visible here).

NO-LOOK-AHEAD (downstream join rule, documented here, enforced downstream):
  rows for UTC day d become visible_from = d+1. All record `ts` values carry the true
  GKG file timestamp so the join is always possible. This script does not enforce it.

Restart safety: a day is skipped iff BOTH outputs already exist (written atomically via
tmp+rename), so crashes never lose work and never re-download. Missing/failed files are
tolerated and LOGGED (never silent). Parsing is deterministic — no sampling.

Column indices VERIFIED EMPIRICALLY on 20260608120000 files (2026-06-10):
  GKG (27 tab-sep fields, no header): 1=DATE(YYYYMMDDHHMMSS), 3=SourceCommonName,
    4=DocumentIdentifier(URL), 7=V1THEMES(;-sep), 8=V2ENHANCEDTHEMES(code,offset;...),
    9=V1LOCATIONS(type#name#FIPScc#...;...), 13=V1ORGS(;-sep),
    15=V1TONE(tone,pos,neg,polarity,activity,selfref,wordcount),
    22=Quotations(offset|len|verb|quote#...).
  Events (61 tab-sep cols): 7=Actor1CountryCode(ISO3/CAMEO e.g. USA,CHN),
    17=Actor2CountryCode, 28=EventRootCode, 29=QuadClass, 30=GoldsteinScale,
    31=NumMentions, 34=AvgTone. NOTE: ActionGeo_CountryCode is col 53 (col 51 is
    ActionGeo_Type) — verified on real data; we key G2 off actor codes 7/17.

Usage:
  .venv/bin/python gdelt_backfill.py --start 2026-02-05 --end 2026-06-10 --density 4
  .venv/bin/python gdelt_backfill.py --plan deep
      (= 2/day 2015-02-18..2022-12-31 then 4/day 2023-01-01..2026-02-04)
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICTS_DIR = os.path.join(BASE_DIR, "dicts")
CACHE_DIR = os.path.join(BASE_DIR, "gdelt_cache")
RECORDS_DIR = os.path.join(CACHE_DIR, "records")
DAILY_DIR = os.path.join(CACHE_DIR, "daily")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.jsonl")

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
HOURS_BY_DENSITY = {4: ["000000", "060000", "120000", "180000"], 2: ["000000", "120000"]}
POLITENESS_DELAY = 0.3
TIMEOUT = 60
RETRIES = 2

# ---------------------------------------------------------------- dictionaries

with open(os.path.join(DICTS_DIR, "theme_to_sector.json")) as f:
    THEME_TO_SECTOR = json.load(f)
with open(os.path.join(DICTS_DIR, "ACTOR_MAP.json")) as f:
    ACTOR_MAP = json.load(f)
with open(os.path.join(DICTS_DIR, "THEMES_FIN.json")) as f:
    THEMES_FIN = tuple(json.load(f))

# prefix list for theme->bucket, longest first so the most specific prefix wins
_T2S_PREFIXES = sorted(THEME_TO_SECTOR.items(), key=lambda kv: -len(kv[0]))
_t2s_memo = {}
_t2s_lock = threading.Lock()

# FIPS country codes considered "ACTOR_MAP country" for finance relevance.
# US is EXCLUDED here on purpose (PROPOSAL_LLM_SENTIMENT F1(c): location qualifies a
# record only if it names a country with a *country ETF* — China/Japan/Brazil/India/
# Russia/Eurozone). US records must qualify via finance themes, otherwise nearly every
# US-dateline article becomes "finance-relevant" and the funnel/cache bloats (~88%
# observed vs the proposal's expected ~15-40%).
FIN_FIPS = set()
for _cc, _spec in ACTOR_MAP.items():
    if _cc != "US":
        FIN_FIPS.update(_spec["fips"])

# events: country key -> set of ISO3/CAMEO actor codes
EVENT_COUNTRY_SETS = {cc: set(spec["iso3"]) for cc, spec in ACTOR_MAP.items()}

_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        _thread_local.session.headers["User-Agent"] = "trader-bot-pkt-tb-006-backfill"
    return _thread_local.session


def theme_bucket(theme):
    """Map a V1 theme code to a bucket_map key (or None). Exact, then longest prefix."""
    try:
        return _t2s_memo[theme]
    except KeyError:
        pass
    bucket = THEME_TO_SECTOR.get(theme)
    if bucket is None:
        for prefix, b in _T2S_PREFIXES:
            if theme.startswith(prefix):
                bucket = b
                break
    with _t2s_lock:
        _t2s_memo[theme] = bucket
    return bucket


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- download

def fetch_zip_lines(url):
    """Download a GDELT zip, return list of decoded lines, or None (reason logged by caller)."""
    last_err = None
    for attempt in range(RETRIES + 1):
        time.sleep(POLITENESS_DELAY)
        try:
            r = get_session().get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                return None, "404"
            if r.status_code != 200:
                last_err = f"http {r.status_code}"
                continue
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                names = z.namelist()
                if not names:
                    return None, "empty zip"
                data = z.read(names[0])
            return data.decode("utf-8", errors="replace").split("\n"), None
        except zipfile.BadZipFile:
            return None, "bad zip"
        except Exception as e:  # noqa: BLE001 - log and retry, never silent
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err or "unknown"


# ---------------------------------------------------------------- parsing

def parse_quotations(raw):
    """Quotations field: offset|len|verb|quote blocks separated by '#'."""
    if not raw:
        return ""
    quotes = []
    for block in raw.split("#"):
        parts = block.split("|", 3)
        if len(parts) >= 4 and parts[3].strip():
            quotes.append(parts[3].strip())
    return " | ".join(quotes)[:2000]


def parse_gkg_lines(lines, day_acc, fin_rows):
    """Accumulate one GKG file into the day accumulator + finance-relevant rows."""
    n = 0
    for ln in lines:
        f = ln.split("\t")
        if len(f) != 27:
            continue
        # tone: tone,pos,neg,polarity,activity,selfref,wordcount
        tparts = f[15].split(",")
        if len(tparts) < 4:
            continue
        try:
            tone = float(tparts[0])
            pos = float(tparts[1])
            neg = float(tparts[2])
            polarity = float(tparts[3])
        except ValueError:
            continue
        n += 1
        day_acc["tone_sum"] += tone
        day_acc["tone_sumsq"] += tone * tone
        day_acc["polarity_sum"] += polarity
        if tone < 0:
            day_acc["n_neg"] += 1

        # V1 themes -> theme counts + buckets
        v1_themes = [t for t in f[7].split(";") if t] if f[7] else []
        record_buckets = set()
        fin_relevant = False
        for t in v1_themes:
            day_acc["theme_counts"][t] += 1
            b = theme_bucket(t)
            if b is not None:
                day_acc["bucket_mentions"][b] += 1
                record_buckets.add(b)
            if not fin_relevant and t.startswith(THEMES_FIN):
                fin_relevant = True
        for b in record_buckets:
            bt = day_acc["bucket_tone"][b]
            bt[0] += tone
            bt[1] += tone * tone
            bt[2] += 1

        # V1 locations -> country codes
        loc_codes = []
        if f[9]:
            for loc in f[9].split(";"):
                parts = loc.split("#")
                if len(parts) > 2 and parts[2]:
                    loc_codes.append(parts[2])
        for c in loc_codes:
            day_acc["loc_counts"][c] += 1
        if not fin_relevant and any(c in FIN_FIPS for c in loc_codes):
            fin_relevant = True

        # V1 orgs
        orgs = [o for o in f[13].split(";") if o] if f[13] else []
        for o in orgs:
            day_acc["org_counts"][o] += 1

        if fin_relevant:
            # V2 enhanced themes stripped of offsets; fallback V1
            themes_out = []
            seen = set()
            if f[8]:
                for item in f[8].split(";"):
                    code = item.rsplit(",", 1)[0] if "," in item else item
                    if code and code not in seen:
                        seen.add(code)
                        themes_out.append(code)
            if not themes_out:
                themes_out = list(dict.fromkeys(v1_themes))
            # ts from GKG DATE field (YYYYMMDDHHMMSS)
            try:
                ts = datetime.strptime(f[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                ts = None
            loc_dedup = list(dict.fromkeys(loc_codes))
            fin_rows.append({
                "ts": ts,
                "source_domain": f[3],
                "url": f[4],
                "themes": "|".join(themes_out),
                "tone": tone,
                "pos": pos,
                "neg": neg,
                "polarity": polarity,
                "locations": "|".join(loc_dedup),
                "orgs": "|".join(orgs[:10]),
                "quotations": parse_quotations(f[22]),
            })
    return n


def parse_events_lines(lines, ev_acc):
    """Accumulate one Events 2.0 export file into per-country aggregates."""
    n = 0
    for ln in lines:
        f = ln.split("\t")
        if len(f) != 61:
            continue
        try:
            goldstein = float(f[30]) if f[30] else 0.0
            mentions = int(f[31]) if f[31] else 0
            quad = int(f[29]) if f[29] else 0
        except ValueError:
            continue
        n += 1
        if mentions <= 0:
            continue
        a1, a2 = f[7], f[17]
        conflict = 1 if quad in (3, 4) else 0
        for cc, iso3set in EVENT_COUNTRY_SETS.items():
            if a1 in iso3set or a2 in iso3set:
                c = ev_acc[cc]
                c["mentions"] += mentions
                c["goldstein_mw_sum"] += goldstein * mentions
                if conflict:
                    c["conflict_mentions"] += mentions
    return n


# ---------------------------------------------------------------- per-day driver

def new_day_acc():
    return {
        "tone_sum": 0.0, "tone_sumsq": 0.0, "polarity_sum": 0.0, "n_neg": 0,
        "theme_counts": Counter(), "bucket_mentions": Counter(),
        "bucket_tone": {b: [0.0, 0.0, 0] for b in set(THEME_TO_SECTOR.values())},
        "loc_counts": Counter(), "org_counts": Counter(),
    }


def mean_std(s, ssq, n):
    if n == 0:
        return None, None
    m = s / n
    var = max(ssq / n - m * m, 0.0)
    return m, math.sqrt(var)


def process_day(date_str, density):
    """date_str: YYYYMMDD. Returns dict for summary; raises nothing (errors logged)."""
    rec_path = os.path.join(RECORDS_DIR, f"{date_str}.parquet")
    daily_path = os.path.join(DAILY_DIR, f"{date_str}.json")
    if os.path.exists(rec_path) and os.path.exists(daily_path):
        return {"date": date_str, "status": "skip"}

    day_acc = new_day_acc()
    fin_rows = []
    ev_acc = {cc: {"mentions": 0, "goldstein_mw_sum": 0.0, "conflict_mentions": 0}
              for cc in EVENT_COUNTRY_SETS}
    n_records = 0
    n_events = 0
    gkg_ok, gkg_missing, ev_ok, ev_missing = [], [], [], []

    for hh in HOURS_BY_DENSITY[density]:
        stamp = f"{date_str}{hh}"
        gkg_url = f"{GDELT_BASE}/{stamp}.gkg.csv.zip"
        lines, err = fetch_zip_lines(gkg_url)
        if lines is None:
            log(f"MISS gkg {stamp}: {err}")
            gkg_missing.append(hh)
        else:
            n_records += parse_gkg_lines(lines, day_acc, fin_rows)
            gkg_ok.append(hh)

        ev_url = f"{GDELT_BASE}/{stamp}.export.CSV.zip"
        lines, err = fetch_zip_lines(ev_url)
        if lines is None:
            log(f"MISS events {stamp}: {err}")
            ev_missing.append(hh)
        else:
            n_events += parse_events_lines(lines, ev_acc)
            ev_ok.append(hh)

    # ---- write records parquet (always, even if empty, so skip-existing works)
    cols = ["ts", "source_domain", "url", "themes", "tone", "pos", "neg",
            "polarity", "locations", "orgs", "quotations"]
    df = pd.DataFrame(fin_rows, columns=cols)
    tmp = rec_path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, rec_path)

    # ---- daily aggregates json
    tone_mean, tone_std = mean_std(day_acc["tone_sum"], day_acc["tone_sumsq"], n_records)
    buckets = {}
    for b in sorted(day_acc["bucket_tone"]):
        s, ssq, n = day_acc["bucket_tone"][b]
        bm, bs = mean_std(s, ssq, n)
        buckets[b] = {
            "mention_count": int(day_acc["bucket_mentions"].get(b, 0)),
            "record_count": n,
            "tone_mean": bm,
            "tone_std": bs,
        }
    total_loc = sum(day_acc["loc_counts"].values())
    per_country = {}
    for cc, c in ev_acc.items():
        m = c["mentions"]
        per_country[cc] = {
            "mentions": m,
            "goldstein_mw_mean": (c["goldstein_mw_sum"] / m) if m else None,
            "conflict_share": (c["conflict_mentions"] / m) if m else None,
        }
    daily = {
        "date": date_str,
        "density": density,
        "files": {"gkg_ok": gkg_ok, "gkg_missing": gkg_missing,
                  "events_ok": ev_ok, "events_missing": ev_missing},
        "n_records": n_records,
        "n_fin_records": len(fin_rows),
        "tone_mean": tone_mean,
        "tone_std": tone_std,
        "neg_share": (day_acc["n_neg"] / n_records) if n_records else None,
        "polarity_mean": (day_acc["polarity_sum"] / n_records) if n_records else None,
        "theme_counts": dict(day_acc["theme_counts"].most_common(500)),
        "buckets": buckets,
        "location_country_counts": dict(day_acc["loc_counts"].most_common(50)),
        "org_counts": dict(day_acc["org_counts"].most_common(100)),
        "us_share": (day_acc["loc_counts"].get("US", 0) / total_loc) if total_loc else None,
        "events": {"n_events": n_events, "per_country": per_country},
        "visible_from": (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    tmp = daily_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(daily, f)
    os.replace(tmp, daily_path)

    # ---- manifest (O_APPEND single line)
    line = json.dumps({
        "date": date_str, "density": density,
        "gkg_ok": len(gkg_ok), "gkg_missing": gkg_missing,
        "events_ok": len(ev_ok), "events_missing": ev_missing,
        "n_records": n_records, "n_fin_records": len(fin_rows), "n_events": n_events,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    with open(MANIFEST_PATH, "a") as f:
        f.write(line + "\n")

    return {"date": date_str, "status": "done", "n_records": n_records,
            "n_fin": len(fin_rows), "n_events": n_events,
            "missing": len(gkg_missing) + len(ev_missing)}


# ---------------------------------------------------------------- main

def daterange(start, end):
    d = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    while d <= e:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def run_phase(start, end, density, workers):
    days = daterange(start, end)
    pending = [d for d in days
               if not (os.path.exists(os.path.join(RECORDS_DIR, f"{d}.parquet"))
                       and os.path.exists(os.path.join(DAILY_DIR, f"{d}.json")))]
    log(f"PHASE {start}..{end} density={density}: {len(days)} days, "
        f"{len(days) - len(pending)} already cached, {len(pending)} to fetch")
    t0 = time.time()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_day, d, density): d for d in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            try:
                r = fut.result()
                done += 1
                if i % 25 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1e-9)
                    eta_h = (len(futs) - i) / rate / 3600 if rate > 0 else float("inf")
                    log(f"PROGRESS {i}/{len(futs)} (last {d}: n={r.get('n_records')}, "
                        f"fin={r.get('n_fin')}, ev={r.get('n_events')}) "
                        f"rate={rate * 3600:.0f} d/h eta={eta_h:.2f} h")
            except Exception as e:  # noqa: BLE001
                failed += 1
                log(f"DAY FAIL {d}: {type(e).__name__}: {e}")
    log(f"PHASE DONE {start}..{end}: done={done} failed={failed} "
        f"elapsed={(time.time() - t0) / 60:.1f} min")
    return failed


def main():
    ap = argparse.ArgumentParser(description="PKT-TB-006 GDELT backfill (restartable)")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--density", type=int, choices=(2, 4), default=4)
    ap.add_argument("--plan", choices=("deep",),
                    help="deep = 2/day 2015-02-18..2022-12-31 then 4/day 2023-01-01..2026-02-04")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(RECORDS_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    if args.plan == "deep":
        phases = [("2015-02-18", "2022-12-31", 2), ("2023-01-01", "2026-02-04", 4)]
    else:
        if not (args.start and args.end):
            ap.error("--start/--end required unless --plan is given")
        phases = [(args.start, args.end, args.density)]

    log(f"gdelt_backfill start {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"phases={phases} workers={args.workers}")
    total_failed = 0
    for start, end, density in phases:
        total_failed += run_phase(start, end, density, args.workers)
    log(f"ALL PHASES COMPLETE failed_days={total_failed}")
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
