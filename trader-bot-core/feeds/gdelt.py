"""feeds/gdelt.py — the rebuilt GDELT feed with a WRITABLE store (RO-FS fix).

Why this is a REBUILD, not a relocation
---------------------------------------
The brain GDELT organ (``runs/pkt_tb_006…/prototype/gdelt_backfill.py`` +
``features_gdelt.py``) works locally but is **frozen at 2026-06-10 in production**.
The nightly forward-fetch crashes every night with::

    OSError: [Errno 30] Read-only file system:
        '/var/task/.../gdelt_cache/manifest.jsonl'

because it wrote its manifest + new daily aggregates back into the *baked* code
path (``/var/task`` on Lambda is read-only). So the GDELT signal can never
advance past the seed's last day. The chair correction (categorization committee,
``05_CLEAN_REBUILD_ARCHITECTURE.md``) is explicit: the gate is that the **brain
gdelt features** (``gdelt_features.parquet``) are nonzero AND advance past the
settled date — and the fix is a real **writable store**, not just a move.

The store model (the fix)
-------------------------
Two roots, cleanly separated:

  * ``SEED_DIR``  (read-only, baked with the code): the frozen backfill
    ``daily/<YYYYMMDD>.json`` aggregates 2015-02-18 .. 2026-06-10. On Lambda this
    lives under ``/var/task`` and MUST only ever be READ.
  * ``STORE_DIR`` (writable, resolved at runtime): where the nightly forward-fetch
    writes new ``daily/<YYYYMMDD>.json`` + ``manifest.jsonl``. On Lambda this
    resolves to ``/tmp/gdelt_cache`` (the only writable FS); locally to a
    gitignored working dir. **Never ``/var/task``.**

``forward_fetch`` reads BOTH roots to know what already exists (seed ∪ store),
fetches only the genuinely-missing settled days, and writes exclusively to
``STORE_DIR``. ``build_features`` merges seed ∪ store daily aggregates (store
wins on collision) into one ``gdelt_features.parquet``.

Honest failure, never silent zeros
----------------------------------
A day that cannot be fetched at all (every GDELT file 404s / errors) is NOT
written as an all-zero daily aggregate — that is exactly the "silent zeros" the
dead ``src/steps/ingest_gdelt.py`` produced. Instead the day is skipped and
surfaced in ``errors[]`` as a named honest fallback; ``build_features`` already
skips + reports missing days (never interpolates). A *partial* day (some hourly
files present) IS written, with the missing hours recorded in ``files`` and an
``errors[]`` note. There is no bare-except that swallows a fetch failure into a
zero row.

Column indices VERIFIED EMPIRICALLY on 20260608120000 files (carried verbatim
from the PKT-TB-006 backfill — do not "fix" without re-verifying on real data):
  GKG (27 tab-sep fields): 1=DATE 3=SourceCommonName 4=URL 7=V1THEMES 8=V2THEMES
    9=V1LOCATIONS 13=V1ORGS 15=V1TONE 22=Quotations.
  Events (61 cols): 7=Actor1CountryCode 17=Actor2CountryCode 29=QuadClass
    30=GoldsteinScale 31=NumMentions 34=AvgTone.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
import time
import zipfile
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Paths & store resolution                                                     #
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve().parent                     # trader-bot-core/feeds
_CORE = _HERE.parent                                        # trader-bot-core
DICTS_DIR = _HERE / "gdelt_dicts"                           # tracked config
SEED_DIR = _CORE / "store" / "seeds" / "gdelt_cache"        # read-only baked seed
SEED_DAILY = SEED_DIR / "daily"

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
HOURS_BY_DENSITY = {4: ["000000", "060000", "120000", "180000"], 2: ["000000", "120000"]}
POLITENESS_DELAY = 0.3
TIMEOUT = 60
RETRIES = 2


def _fs_is_writable(path: Path) -> bool:
    """Probe whether ``path`` (or its nearest existing parent) is writable — the
    honest test for the read-only ``/var/task`` FS, done by trying, not guessing."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def resolve_store_dir(explicit: Optional[str] = None) -> Path:
    """Resolve the WRITABLE store root. Precedence:

      1. ``explicit`` arg (diag / test override),
      2. ``$GDELT_STORE_DIR`` env,
      3. Lambda (``$AWS_LAMBDA_FUNCTION_NAME`` set, or seed under ``/var/task``)
         -> ``/tmp/gdelt_cache`` (Lambda's only writable FS),
      4. local default ``trader-bot-core/store/gdelt_cache`` (gitignored),
         BUT if that turns out non-writable (baked read-only deploy that is not
         flagged as Lambda) fall back to ``/tmp/gdelt_cache`` rather than crash.

    This is the crux of the RO-FS fix: the store NEVER resolves to the baked
    seed/code path.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("GDELT_STORE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    on_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME")) or str(SEED_DIR).startswith("/var/task")
    if on_lambda:
        return Path("/tmp/gdelt_cache")
    local = _CORE / "store" / "gdelt_cache"
    if _fs_is_writable(local):
        return local
    return Path("/tmp/gdelt_cache")


# --------------------------------------------------------------------------- #
# Dictionaries (finance-relevance + theme->bucket mapping)                     #
# --------------------------------------------------------------------------- #

with open(DICTS_DIR / "theme_to_sector.json") as _f:
    THEME_TO_SECTOR = json.load(_f)
with open(DICTS_DIR / "ACTOR_MAP.json") as _f:
    ACTOR_MAP = json.load(_f)
with open(DICTS_DIR / "THEMES_FIN.json") as _f:
    THEMES_FIN = tuple(json.load(_f))

_T2S_PREFIXES = sorted(THEME_TO_SECTOR.items(), key=lambda kv: -len(kv[0]))
_t2s_memo: Dict[str, Optional[str]] = {}
_t2s_lock = threading.Lock()

FIN_FIPS: set = set()
for _cc, _spec in ACTOR_MAP.items():
    if _cc != "US":
        FIN_FIPS.update(_spec["fips"])
EVENT_COUNTRY_SETS = {cc: set(spec["iso3"]) for cc, spec in ACTOR_MAP.items()}

_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers["User-Agent"] = "trader-bot-core-gdelt-forward"
        _thread_local.session = s
    return _thread_local.session


def _theme_bucket(theme: str) -> Optional[str]:
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


# --------------------------------------------------------------------------- #
# Download + parse (verified column indices — carried verbatim)               #
# --------------------------------------------------------------------------- #

def _fetch_zip_lines(url: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Download a GDELT zip -> (lines, None) or (None, reason). Never raises to the
    caller; the reason is always named (logged by the caller into errors[])."""
    last_err = None
    for _ in range(RETRIES + 1):
        time.sleep(POLITENESS_DELAY)
        try:
            r = _get_session().get(url, timeout=TIMEOUT)
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
        except Exception as e:  # noqa: BLE001 — transport-layer; named + retried, never silent
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err or "unknown"


def _parse_quotations(raw: str) -> str:
    if not raw:
        return ""
    quotes = []
    for block in raw.split("#"):
        parts = block.split("|", 3)
        if len(parts) >= 4 and parts[3].strip():
            quotes.append(parts[3].strip())
    return " | ".join(quotes)[:2000]


def _parse_gkg_lines(lines: List[str], day_acc: dict, fin_rows: list) -> int:
    n = 0
    for ln in lines:
        f = ln.split("\t")
        if len(f) != 27:
            continue
        tparts = f[15].split(",")
        if len(tparts) < 4:
            continue
        try:
            tone = float(tparts[0]); pos = float(tparts[1])
            neg = float(tparts[2]); polarity = float(tparts[3])
        except ValueError:
            continue
        n += 1
        day_acc["tone_sum"] += tone
        day_acc["tone_sumsq"] += tone * tone
        day_acc["polarity_sum"] += polarity
        if tone < 0:
            day_acc["n_neg"] += 1

        v1_themes = [t for t in f[7].split(";") if t] if f[7] else []
        record_buckets = set()
        fin_relevant = False
        for t in v1_themes:
            day_acc["theme_counts"][t] += 1
            b = _theme_bucket(t)
            if b is not None:
                day_acc["bucket_mentions"][b] += 1
                record_buckets.add(b)
            if not fin_relevant and t.startswith(THEMES_FIN):
                fin_relevant = True
        for b in record_buckets:
            bt = day_acc["bucket_tone"][b]
            bt[0] += tone; bt[1] += tone * tone; bt[2] += 1

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

        orgs = [o for o in f[13].split(";") if o] if f[13] else []
        for o in orgs:
            day_acc["org_counts"][o] += 1

        if fin_relevant:
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
            try:
                ts = datetime.strptime(f[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                ts = None
            loc_dedup = list(dict.fromkeys(loc_codes))
            fin_rows.append({
                "ts": ts, "source_domain": f[3], "url": f[4],
                "themes": "|".join(themes_out), "tone": tone, "pos": pos,
                "neg": neg, "polarity": polarity, "locations": "|".join(loc_dedup),
                "orgs": "|".join(orgs[:10]), "quotations": _parse_quotations(f[22]),
            })
    return n


def _parse_events_lines(lines: List[str], ev_acc: dict) -> int:
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


def _new_day_acc() -> dict:
    return {
        "tone_sum": 0.0, "tone_sumsq": 0.0, "polarity_sum": 0.0, "n_neg": 0,
        "theme_counts": Counter(), "bucket_mentions": Counter(),
        "bucket_tone": {b: [0.0, 0.0, 0] for b in set(THEME_TO_SECTOR.values())},
        "loc_counts": Counter(), "org_counts": Counter(),
    }


def _mean_std(s: float, ssq: float, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    m = s / n
    var = max(ssq / n - m * m, 0.0)
    return m, math.sqrt(var)


# --------------------------------------------------------------------------- #
# Forward fetch — writes ONLY to the writable store                           #
# --------------------------------------------------------------------------- #

@dataclass
class ForwardReport:
    """Per-run accounting so a diag can SEE what advanced and where fetch failed —
    the honest 'named fallback, never silent zeros' evidence."""
    store_dir: str = ""
    seed_max_date: Optional[str] = None
    requested: List[str] = field(default_factory=list)
    written: List[str] = field(default_factory=list)         # days written to store
    skipped_existing: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)  # named honest fallbacks
    read_only_fs_oserror: bool = False


def _existing_daily_dates(store_daily: Path) -> set:
    """Union of dates already present across seed ∪ store daily dirs."""
    have = set()
    for d in (SEED_DAILY, store_daily):
        if d.exists():
            for p in d.glob("*.json"):
                stem = p.stem
                if len(stem) == 8 and stem.isdigit():
                    have.add(stem)
    return have


def _process_day(date_str: str, density: int, store: Path,
                 report: ForwardReport, write_records: bool) -> None:
    """Fetch one UTC day and write its daily aggregate to the WRITABLE store.

    A total miss (no hourly file retrievable) is recorded in ``report.errors`` and
    NOT written as an all-zero daily (no silent zeros). A partial day is written
    with the missing hours recorded. Any ``OSError`` from a read-only FS is caught,
    flagged on the report, and re-raised as a named error — never swallowed into a
    zero row."""
    daily_dir = store / "daily"
    records_dir = store / "records"
    manifest_path = store / "manifest.jsonl"
    daily_path = daily_dir / f"{date_str}.json"

    day_acc = _new_day_acc()
    fin_rows: list = []
    ev_acc = {cc: {"mentions": 0, "goldstein_mw_sum": 0.0, "conflict_mentions": 0}
              for cc in EVENT_COUNTRY_SETS}
    n_records = n_events = 0
    gkg_ok, gkg_missing, ev_ok, ev_missing = [], [], [], []

    for hh in HOURS_BY_DENSITY[density]:
        stamp = f"{date_str}{hh}"
        lines, err = _fetch_zip_lines(f"{GDELT_BASE}/{stamp}.gkg.csv.zip")
        if lines is None:
            gkg_missing.append(hh)
            report.errors.append({"date": date_str, "file": "gkg", "hour": hh,
                                  "kind": "fetch_miss", "reason": err})
        else:
            n_records += _parse_gkg_lines(lines, day_acc, fin_rows)
            gkg_ok.append(hh)

        lines, err = _fetch_zip_lines(f"{GDELT_BASE}/{stamp}.export.CSV.zip")
        if lines is None:
            ev_missing.append(hh)
            report.errors.append({"date": date_str, "file": "events", "hour": hh,
                                  "kind": "fetch_miss", "reason": err})
        else:
            n_events += _parse_events_lines(lines, ev_acc)
            ev_ok.append(hh)

    # Total miss -> honest skip, NOT a zero row.
    if not gkg_ok and not ev_ok:
        report.errors.append({"date": date_str, "kind": "day_total_miss",
                              "reason": "no GKG/events file retrievable; day skipped (no silent zeros)"})
        return

    tone_mean, tone_std = _mean_std(day_acc["tone_sum"], day_acc["tone_sumsq"], n_records)
    buckets = {}
    for b in sorted(day_acc["bucket_tone"]):
        s, ssq, n = day_acc["bucket_tone"][b]
        bm, bs = _mean_std(s, ssq, n)
        buckets[b] = {"mention_count": int(day_acc["bucket_mentions"].get(b, 0)),
                      "record_count": n, "tone_mean": bm, "tone_std": bs}
    total_loc = sum(day_acc["loc_counts"].values())
    per_country = {}
    for cc, c in ev_acc.items():
        m = c["mentions"]
        per_country[cc] = {"mentions": m,
                           "goldstein_mw_mean": (c["goldstein_mw_sum"] / m) if m else None,
                           "conflict_share": (c["conflict_mentions"] / m) if m else None}
    daily = {
        "date": date_str, "density": density,
        "files": {"gkg_ok": gkg_ok, "gkg_missing": gkg_missing,
                  "events_ok": ev_ok, "events_missing": ev_missing},
        "n_records": n_records, "n_fin_records": len(fin_rows),
        "tone_mean": tone_mean, "tone_std": tone_std,
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

    # --- WRITE to the writable store (atomic tmp+rename). A read-only FS here is
    #     the exact production crash — catch it, flag it, and re-raise NAMED.
    try:
        daily_dir.mkdir(parents=True, exist_ok=True)
        if write_records:
            records_dir.mkdir(parents=True, exist_ok=True)
            cols = ["ts", "source_domain", "url", "themes", "tone", "pos", "neg",
                    "polarity", "locations", "orgs", "quotations"]
            rec_path = records_dir / f"{date_str}.parquet"
            rtmp = rec_path.with_suffix(".parquet.tmp")
            pd.DataFrame(fin_rows, columns=cols).to_parquet(rtmp, index=False)
            os.replace(rtmp, rec_path)
        tmp = daily_path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(daily, fh)
        os.replace(tmp, daily_path)
        with open(manifest_path, "a") as fh:
            fh.write(json.dumps({
                "date": date_str, "density": density,
                "gkg_ok": len(gkg_ok), "gkg_missing": gkg_missing,
                "events_ok": len(ev_ok), "events_missing": ev_missing,
                "n_records": n_records, "n_fin_records": len(fin_rows), "n_events": n_events,
                "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }) + "\n")
    except OSError as e:
        report.read_only_fs_oserror = report.read_only_fs_oserror or (e.errno == 30)
        report.errors.append({"date": date_str, "kind": "store_write_error",
                              "errno": e.errno, "reason": f"{type(e).__name__}: {e}",
                              "store_dir": str(store)})
        raise

    report.written.append(date_str)
    if gkg_missing or ev_missing:
        report.errors.append({"date": date_str, "kind": "partial_day",
                              "gkg_missing": gkg_missing, "events_missing": ev_missing,
                              "reason": "day written with missing hourly files noted"})


def latest_complete_utc_day(now: Optional[datetime] = None) -> str:
    """The last GDELT UTC day whose files are all published. GDELT posts 15-min
    files continuously; a day is 'complete enough' for the 4x density once its
    18:00 UTC slice is up. We conservatively take yesterday (UTC) as the last day
    guaranteed complete, unless we are already past 19:00 UTC today."""
    now = now or datetime.now(timezone.utc)
    day = now.date()
    if now.hour < 19:
        day = day - timedelta(days=1)
    return day.strftime("%Y%m%d")


def _seed_max_date() -> Optional[str]:
    if not SEED_DAILY.exists():
        return None
    dates = sorted(p.stem for p in SEED_DAILY.glob("*.json")
                   if len(p.stem) == 8 and p.stem.isdigit())
    return dates[-1] if dates else None


def forward_fetch(
    end: Optional[str] = None,
    start: Optional[str] = None,
    density: int = 4,
    workers: int = 6,
    store_dir: Optional[str] = None,
    write_records: bool = False,
) -> ForwardReport:
    """Fetch settled GDELT days from just-after the seed's last day up to ``end``,
    writing ONLY to the resolved writable store. Returns a ``ForwardReport``.

    ``start`` defaults to seed_max_date + 1 (so we only fetch the genuinely-missing
    forward days). ``end`` defaults to the last complete UTC day.
    ``write_records`` defaults False: the G1-G5 feature path needs only the daily
    aggregates; the per-record parquet (LLM funnel) is out of scope for the
    feature-currency fix and would bloat the store.
    """
    store = resolve_store_dir(store_dir)
    daily_store = store / "daily"
    report = ForwardReport(store_dir=str(store), seed_max_date=_seed_max_date())

    seed_max = report.seed_max_date
    if start is None:
        if seed_max is None:
            raise RuntimeError("no seed daily aggregates found; cannot infer forward start")
        start = (datetime.strptime(seed_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    if end is None:
        end = latest_complete_utc_day()

    have = _existing_daily_dates(daily_store)
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    pending, d = [], d0
    while d <= d1:
        ds = d.strftime("%Y%m%d")
        report.requested.append(ds)
        if ds in have:
            report.skipped_existing.append(ds)
        else:
            pending.append(ds)
        d += timedelta(days=1)

    if not pending:
        return report

    # Fail fast + honestly if the store root is not writable (the RO-FS class).
    if not _fs_is_writable(daily_store):
        report.read_only_fs_oserror = True
        report.errors.append({"kind": "store_not_writable",
                              "reason": f"resolved store {store} is not writable",
                              "store_dir": str(store)})
        return report

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_day, ds, density, store, report, write_records): ds
                for ds in pending}
        for fut in as_completed(futs):
            ds = futs[fut]
            try:
                fut.result()
            except OSError:
                # already recorded (store_write_error / read_only_fs); keep going so
                # the report is complete rather than aborting on the first bad day.
                continue
            except Exception as e:  # noqa: BLE001 — named into errors[], never silent
                report.errors.append({"date": ds, "kind": "process_error",
                                      "reason": f"{type(e).__name__}: {e}"})
    report.written.sort()
    return report


# --------------------------------------------------------------------------- #
# Feature build (G1-G5 panel) — merges seed ∪ store daily aggregates          #
# --------------------------------------------------------------------------- #

Z_WINDOW = 90
Z_MIN_PERIODS = 30
JSD_MIN_DAYS = 30
NOVELTY_SMOOTH = 21
COUNTRIES = ["US", "CN", "JP", "BR", "IN", "RU", "EU"]


def _gdelt_buckets() -> List[str]:
    t2s = json.loads((DICTS_DIR / "theme_to_sector.json").read_text())
    return sorted(set(t2s.values()))


def _hhi(counts: Dict[str, float]) -> float:
    tot = float(sum(counts.values()))
    if tot <= 0:
        return np.nan
    return float(sum((v / tot) ** 2 for v in counts.values()))


def _merged_daily_paths(store_dir: Optional[str] = None) -> List[Path]:
    """date -> path across seed ∪ store; store wins on collision (fresh over seed)."""
    store = resolve_store_dir(store_dir)
    by_date: Dict[str, Path] = {}
    if SEED_DAILY.exists():
        for p in SEED_DAILY.glob("*.json"):
            if len(p.stem) == 8 and p.stem.isdigit():
                by_date[p.stem] = p
    sd = store / "daily"
    if sd.exists():
        for p in sd.glob("*.json"):
            if len(p.stem) == 8 and p.stem.isdigit():
                by_date[p.stem] = p       # store overrides seed
    return [by_date[k] for k in sorted(by_date)]


def _load_daily_raw(paths: List[Path]) -> Tuple[pd.DataFrame, List[Dict[str, float]], List[str]]:
    buckets = _gdelt_buckets()
    rows: List[Dict[str, Any]] = []
    theme_dists: List[Dict[str, float]] = []
    for f in paths:
        d = json.loads(f.read_text())
        day = pd.Timestamp(dt.datetime.strptime(d["date"], "%Y%m%d").date())
        n_gkg = len(d.get("files", {}).get("gkg_ok", [])) or np.nan
        row: Dict[str, Any] = {
            "gdelt_date": day, "visible_from": day + pd.Timedelta(days=1),
            "density": d.get("density"), "n_files_gkg": n_gkg,
            "n_records": d.get("n_records"), "n_fin_records": d.get("n_fin_records"),
            "g3_tone_mean": d.get("tone_mean"), "g3_tone_std": d.get("tone_std"),
            "g3_neg_share": d.get("neg_share"), "g3_polarity": d.get("polarity_mean"),
            "g5_hhi_loc": _hhi(d.get("location_country_counts", {})),
            "g5_hhi_org": _hhi(d.get("org_counts", {})),
            "g5_us_share": d.get("us_share"),
        }
        bk = d.get("buckets", {})
        for b in buckets:
            info = bk.get(b, {})
            row[f"_count_{b}"] = info.get("mention_count", 0) or 0
            row[f"g3_tone_{b}"] = info.get("tone_mean", np.nan)
        ev = (d.get("events") or {}).get("per_country", {})
        for cc in COUNTRIES:
            info = ev.get(cc, {})
            row[f"g2_goldstein_{cc.lower()}"] = info.get("goldstein_mw_mean", np.nan)
            row[f"g2_conflict_{cc.lower()}"] = info.get("conflict_share", np.nan)
        rows.append(row)
        tc = d.get("theme_counts", {}) or {}
        tot = float(sum(tc.values()))
        theme_dists.append({k: v / tot for k, v in tc.items()} if tot > 0 else {})

    raw = pd.DataFrame(rows).sort_values("gdelt_date").reset_index(drop=True)
    if len(raw):
        full = pd.date_range(raw["gdelt_date"].min(), raw["gdelt_date"].max(), freq="D")
        missing = sorted(set(full) - set(raw["gdelt_date"]))
    else:
        missing = []
    return raw, theme_dists, [str(m.date()) for m in missing]


def _trailing_z(series: pd.Series, dates: pd.Series,
                window: int = Z_WINDOW, min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    s = pd.Series(series.values, index=pd.DatetimeIndex(dates.values))
    cal = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    shifted = cal.shift(1)
    mean = shifted.rolling(window, min_periods=min_periods).mean()
    std = shifted.rolling(window, min_periods=min_periods).std()
    z = (cal - mean) / std.replace(0.0, np.nan)
    return pd.Series(z.reindex(pd.DatetimeIndex(dates.values)).values, index=series.index)


def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    if not p or not q:
        return np.nan
    keys = set(p) | set(q)
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    pv = pv / pv.sum(); qv = qv / qv.sum()
    m = 0.5 * (pv + qv)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(pv > 0, pv * np.log(pv / m), 0.0).sum()
        kl_qm = np.where(qv > 0, qv * np.log(qv / m), 0.0).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _theme_novelty(raw: pd.DataFrame, theme_dists: List[Dict[str, float]],
                   window: int = Z_WINDOW, min_days: int = JSD_MIN_DAYS) -> pd.Series:
    out = np.full(len(raw), np.nan)
    dq: deque = deque()
    ref_sum: Dict[str, float] = {}
    for i, (day, dist) in enumerate(zip(raw["gdelt_date"], theme_dists)):
        while dq and (day - dq[0][0]).days > window:
            _, old = dq.popleft()
            for k, v in old.items():
                nv = ref_sum.get(k, 0.0) - v
                if nv <= 1e-12:
                    ref_sum.pop(k, None)
                else:
                    ref_sum[k] = nv
        if len(dq) >= min_days and dist:
            n = len(dq)
            ref = {k: v / n for k, v in ref_sum.items()}
            out[i] = _jsd(dist, ref)
        if dist:
            dq.append((day, dist))
            for k, v in dist.items():
                ref_sum[k] = ref_sum.get(k, 0.0) + v
    return pd.Series(out, index=raw.index)


def build_features(store_dir: Optional[str] = None,
                   out_path: Optional[Path] = None) -> Tuple[pd.DataFrame, dict]:
    """Build the G1-G5 gdelt feature panel from seed ∪ store daily aggregates.

    Returns ``(df, report)``. Writes ``gdelt_features.parquet`` +
    ``gdelt_features_report.json`` into the writable store (never the seed)."""
    store = resolve_store_dir(store_dir)
    if out_path is None:
        out_path = store / "gdelt_features.parquet"
    paths = _merged_daily_paths(store_dir)
    raw, theme_dists, missing = _load_daily_raw(paths)
    buckets = _gdelt_buckets()
    df = raw.copy()

    count_cols = [f"_count_{b}" for b in buckets]
    counts = df[count_cols].to_numpy(dtype=float)
    total = counts.sum(axis=1)
    total[total <= 0] = np.nan
    shares = counts / total[:, None]
    for j, b in enumerate(buckets):
        df[f"g1_share_{b}"] = shares[:, j]
    for b in buckets:
        df[f"g1_z_{b}"] = _trailing_z(df[f"g1_share_{b}"], df["gdelt_date"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(shares > 0, shares * np.log(shares), 0.0), axis=1)
    df["g1_share_entropy"] = np.where(np.isnan(total), np.nan, ent)

    tone_cols = [f"g3_tone_{b}" for b in buckets]
    df["g3_tone_dispersion"] = df[tone_cols].std(axis=1)

    nf = df["n_files_gkg"].to_numpy(dtype=float)
    for j, b in enumerate(buckets):
        per_file = counts[:, j] / nf
        df[f"g4_burst_z_{b}"] = _trailing_z(pd.Series(per_file, index=df.index), df["gdelt_date"])
    df["g4_doc_surprise"] = _trailing_z(df["n_records"] / df["n_files_gkg"], df["gdelt_date"])
    df["g4_theme_novelty"] = _theme_novelty(df, theme_dists)
    nov = pd.Series(df["g4_theme_novelty"].values, index=pd.DatetimeIndex(df["gdelt_date"].values))
    cal = nov.reindex(pd.date_range(nov.index.min(), nov.index.max(), freq="D"))
    sm = cal.rolling(NOVELTY_SMOOTH, min_periods=7).mean()
    df["g4_theme_novelty_21"] = sm.reindex(pd.DatetimeIndex(df["gdelt_date"].values)).values

    df["gdelt_available"] = 1
    df = df.drop(columns=count_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    report = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "first": str(df["gdelt_date"].min().date()) if len(df) else None,
        "last": str(df["gdelt_date"].max().date()) if len(df) else None,
        "n_buckets": len(buckets), "n_columns": int(df.shape[1]),
        "z_window_days": Z_WINDOW, "z_min_periods": Z_MIN_PERIODS,
        "n_missing_days": len(missing), "missing_days_skipped": missing[-30:],
        "out_path": str(out_path),
    }
    (out_path.parent / "gdelt_features_report.json").write_text(json.dumps(report, indent=1))
    return df, report


# --------------------------------------------------------------------------- #
# diag — the reality test (nonzero + advancing, RO-FS proof)                   #
# --------------------------------------------------------------------------- #

def diag(store_dir: Optional[str] = None, end: Optional[str] = None,
         write_records: bool = False) -> dict:
    """Governed reality test: forward-fetch the missing settled days into the
    writable store, rebuild the feature panel, and return an honest receipt
    proving (nonzero AND advancing past the seed's 2026-06-10, no RO-FS OSError,
    fetch failure surfaces as a named fallback)."""
    fwd = forward_fetch(end=end, store_dir=store_dir, write_records=write_records)
    df, feat = build_features(store_dir=store_dir)

    # Nonzero check over the ADVANCED tail (days strictly after the seed max).
    seed_max = fwd.seed_max_date
    nonzero = {}
    tail = df
    if seed_max:
        cut = pd.Timestamp(dt.datetime.strptime(seed_max, "%Y%m%d").date())
        tail = df[df["gdelt_date"] > cut]
    check_cols = ["n_records", "n_fin_records", "g3_tone_mean", "g3_tone_std",
                  "g3_neg_share", "g3_polarity", "g2_goldstein_us", "g2_goldstein_cn"]
    for c in check_cols:
        if c in tail.columns and len(tail):
            col = pd.to_numeric(tail[c], errors="coerce")
            nonzero[c] = bool((col.fillna(0) != 0).any())
    advanced_days = int((tail["gdelt_date"] > pd.Timestamp(
        dt.datetime.strptime(seed_max, "%Y%m%d").date())).sum()) if seed_max and len(tail) else 0

    return {
        "store_dir": fwd.store_dir,
        "seed_max_date": seed_max,
        "forward": {
            "requested_days": len(fwd.requested),
            "written_days": fwd.written,
            "n_written": len(fwd.written),
            "skipped_existing": len(fwd.skipped_existing),
            "errors": fwd.errors,
            "read_only_fs_oserror": fwd.read_only_fs_oserror,
        },
        "features": {
            "built": bool(feat["rows"] > 0),
            "rows": feat["rows"],
            "first": feat["first"],
            "last": feat["last"],
            "n_columns": feat["n_columns"],
            "out_path": feat["out_path"],
        },
        "acceptance": {
            "advances_past_seed": bool(feat["last"] and feat["last"] > (
                str(dt.datetime.strptime(seed_max, "%Y%m%d").date()) if seed_max else "")),
            "advanced_day_count": advanced_days,
            "tail_nonzero": nonzero,
            "all_checked_nonzero": bool(nonzero) and all(nonzero.values()),
            "no_read_only_fs_oserror": not fwd.read_only_fs_oserror,
            "fetch_failures_named": [e for e in fwd.errors if e.get("kind") in
                                     ("fetch_miss", "day_total_miss", "partial_day",
                                      "store_write_error", "store_not_writable")][:20],
        },
    }


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="trader-bot-core GDELT feed (writable store)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("forward", help="forward-fetch missing settled days -> writable store")
    pf.add_argument("--end"); pf.add_argument("--start")
    pf.add_argument("--store-dir"); pf.add_argument("--write-records", action="store_true")
    pb = sub.add_parser("build", help="rebuild gdelt_features.parquet from seed ∪ store")
    pb.add_argument("--store-dir")
    pd_ = sub.add_parser("diag", help="reality test: forward + build + honest receipt")
    pd_.add_argument("--end"); pd_.add_argument("--store-dir")
    pd_.add_argument("--write-records", action="store_true")
    args = ap.parse_args()

    if args.cmd == "forward":
        rep = forward_fetch(end=args.end, start=args.start, store_dir=args.store_dir,
                            write_records=args.write_records)
        print(json.dumps(rep.__dict__, indent=1, default=str))
    elif args.cmd == "build":
        _df, rep = build_features(store_dir=args.store_dir)
        print(json.dumps(rep, indent=1, default=str))
    elif args.cmd == "diag":
        print(json.dumps(diag(store_dir=args.store_dir, end=args.end,
                              write_records=args.write_records), indent=1, default=str))


if __name__ == "__main__":
    _main()
