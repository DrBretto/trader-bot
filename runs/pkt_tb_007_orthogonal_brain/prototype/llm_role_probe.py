"""PKT-TB-007 role 6 (LLM/Event Role-Finder) — empirical decomposition of the LLM organ.

HONESTY RAIL: pre-holdout only. Analysis decision dates D satisfy D+5 trading days
< 2026-03-11 (no forward window touches the holdout). LLM coverage used: the
contiguous 2024-08-15 -> 2026-03-10 run (orphan week 2024-01-02..08 dropped).

Outputs: llm_role_probe.json (machine record) + printed summary.
Multiple-comparisons: every screen is ledgered; BH-FDR computed over the full ledger.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(4242)
B_BOOT = 2000
BLOCK = 10  # >= 2x the 5d overlap

HERE = os.path.dirname(os.path.abspath(__file__))
P6 = "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype"
HOLDOUT = pd.Timestamp("2026-03-11")

# ---------------------------------------------------------------- panel + LLM
z = np.load(os.path.join(P6, "store/panel.npz"), allow_pickle=True)
dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
buckets = [str(b) for b in z["buckets"]]
symbols = [str(s) for s in z["symbols"]]
y5b = z["y5_bucket"]            # [N,27] bucket 5d abnormal (open D -> open D+5, minus xs mean)
y5_raw = z["y5_raw"]            # [N,64]
open_px = z["open_px"]; close_px = z["close_px"]
fwd_book_vol5 = z["fwd_book_vol5"]
sym_mask = z["symbol_mask"]

N = len(dates)
i_hold = int(np.searchsorted(dates, HOLDOUT))
LAST_D = i_hold - 6            # D such that open(D+5) <= 2026-03-10

llm = pd.read_parquet(os.path.join(P6, "store/llm_features.parquet"))
llm = llm[llm["llm_available"] == 1.0].copy()
llm.index = pd.to_datetime(llm.index)
llm["visible_from"] = pd.to_datetime(llm["visible_from"])
llm = llm[llm.index >= "2024-08-15"]          # contiguous run only
llm = llm.sort_values("visible_from")

# join artifact -> decision date D: latest artifact visible_from <= D, gap <= 4 cal days
pdates = pd.DataFrame({"D": dates})
joined = pd.merge_asof(pdates, llm.reset_index().rename(columns={"date": "art_date"}),
                       left_on="D", right_on="visible_from",
                       tolerance=pd.Timedelta(days=4), direction="backward")
joined = joined.set_index("D")
have = joined["llm_available"].fillna(0).to_numpy() == 1.0

# analysis mask
mask = have & (np.arange(N) <= LAST_D) & np.asarray(dates >= pd.Timestamp("2024-08-15"))
idx = np.where(mask)[0]
print(f"analysis decision dates: {len(idx)}  {dates[idx[0]].date()} -> {dates[idx[-1]].date()}")

SEGS = {  # sign-consistency segments (fold-respecting: F5 tail / F6 / post-F6 pre-holdout)
    "S1_F5tail": (pd.Timestamp("2024-08-15"), pd.Timestamp("2025-02-01")),
    "S2_F6":     (pd.Timestamp("2025-02-01"), pd.Timestamp("2026-02-01")),
    "S3_tail":   (pd.Timestamp("2026-02-01"), HOLDOUT),
}

def seg_of(i):
    d = dates[i]
    for k, (a, b) in SEGS.items():
        if a <= d < b:
            return k
    return None

# ---------------------------------------------------------------- helpers
def block_boot_mean(x, B=B_BOOT, block=BLOCK):
    """circular moving-block bootstrap CI for the mean of a daily series."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 30: return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    means = np.empty(B)
    for b in range(B):
        starts = RNG.integers(0, n, nb)
        sel = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[sel[:n]].mean()
    return (float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95)))

def overlap_t(x, h=5):
    """t-stat for mean of daily series with h-day overlapping targets (n_eff = n/h)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 30: return np.nan, np.nan, n
    ne = n / h
    t = x.mean() / (x.std(ddof=1) / np.sqrt(ne))
    p = 2 * stats.t.sf(abs(t), df=ne - 1)
    return float(t), float(p), n

def spearman_boot(a, b, h=5):
    """Spearman + block-bootstrap p (days resampled jointly)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 30: return np.nan, np.nan, n
    r = stats.spearmanr(a, b).statistic
    nb = int(np.ceil(n / BLOCK)); rs = np.empty(B_BOOT)
    for k in range(B_BOOT):
        starts = RNG.integers(0, n, nb)
        sel = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
        sel = sel[:n]
        rs[k] = stats.spearmanr(a[sel], b[sel]).statistic
    lo, hi = np.quantile(rs, [0.05, 0.95])
    p = 2 * min((rs <= 0).mean(), (rs >= 0).mean())  # bootstrap sign test on r
    p = max(p, 1.0 / B_BOOT)
    return float(r), float(p), n, (float(lo), float(hi))

LEDGER = []
def screen(test_id, family, target, stat_name, stat, p, n, extra=None):
    LEDGER.append(dict(id=test_id, family=family, target=target,
                       stat=stat_name, value=None if stat is None or (isinstance(stat, float) and np.isnan(stat)) else round(float(stat), 4),
                       p=None if p is None or (isinstance(p, float) and np.isnan(p)) else round(float(p), 5),
                       n=int(n), **(extra or {})))

def seg_signs(daily, idx_local):
    out = {}
    for k in SEGS:
        v = [daily[j] for j, i in enumerate(idx_local) if seg_of(i) == k and not np.isnan(daily[j])]
        out[k] = None if len(v) < 15 else round(float(np.mean(v)), 4)
    return out

# ---------------------------------------------------------------- bucket targets
# bucket 1d abnormal return (open D -> open D+1, minus xs mean), same convention as y5
r1_sym = np.full((N, len(symbols)), np.nan)
with np.errstate(invalid="ignore", divide="ignore"):
    r1_sym[:-1] = open_px[1:] / open_px[:-1] - 1.0
r1_sym[~sym_mask] = np.nan
xs1 = np.nanmean(r1_sym, axis=1, keepdims=True)
r1_ab = r1_sym - xs1
BUCKET_SYMS = json.load(open(os.path.join(P6, "bucket_map.json"))) if os.path.exists(os.path.join(P6, "bucket_map.json")) else None
# reconstruct bucket membership from B-panel convention: use y5_bucket builder's map via store
# fallback: derive from proposal table
if BUCKET_SYMS is None:
    BUCKET_SYMS = {
        "us_broad": "SPY QQQ IWM DIA RSP VTI VOO IVV VT", "tech": "XLK XLC ARKK",
        "semis": "SOXX SMH", "financials_banks": "XLF KRE", "energy_oil": "XLE USO",
        "natgas": "UNG", "real_estate": "VNQ IYR", "intl_dev": "VEA EFA", "europe": "VGK",
        "japan": "EWJ", "em_broad": "VWO EEM", "china": "FXI", "india": "INDA",
        "brazil": "EWZ", "healthcare_biotech": "XLV XBI IBB",
        "industrials_defense": "XLI ITA IYT", "consumer_disc": "XLY XRT",
        "consumer_staples": "XLP", "utilities": "XLU", "materials": "XLB",
        "rates_duration": "TLT IEF SHY TIP AGG BND MUB", "credit": "LQD HYG",
        "gold_pm": "GLD SLV", "commodity_broad": "DBC", "usd": "UUP", "eur": "FXE",
        "volatility": "VIXY"}
    BUCKET_SYMS = {k: v.split() for k, v in BUCKET_SYMS.items()}
sym_ix = {s: i for i, s in enumerate(symbols)}
y1b = np.full((N, len(buckets)), np.nan)
for kb, b in enumerate(buckets):
    js = [sym_ix[s] for s in BUCKET_SYMS.get(b, []) if s in sym_ix]
    if js:
        y1b[:, kb] = np.nanmean(r1_ab[:, js], axis=1)

# LLM matrices aligned to panel dates
def col(name): return joined[name].to_numpy(float)
sent  = np.stack([col(f"llm_sent_{b}") for b in buckets], 1)
sent3 = np.stack([col(f"llm_sent_{b}_ema3") for b in buckets], 1)
sentd = np.stack([col(f"llm_sent_{b}_d1") for b in buckets], 1)
conf  = np.stack([col(f"llm_conf_{b}") for b in buckets], 1)
sal   = np.stack([col(f"llm_sal_{b}") for b in buckets], 1)
risk_app = col("llm_risk_appetite"); rates_p = col("llm_rates_pressure"); geopol = col("llm_geopol_risk")
EVENTS = [c for c in joined.columns if c.startswith("llm_event_")]
ev = np.stack([col(c) for c in EVENTS], 1)
n_clusters = col("n_clusters")

# ---------------------------------------------------------------- (a) DIRECTION
def daily_xb_ic(sig, tgt, min_active=6, active=None):
    """daily cross-bucket Spearman; active = bool [N,27] of buckets to include."""
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        s, t = sig[i].copy(), tgt[i]
        ok = ~(np.isnan(s) | np.isnan(t))
        if active is not None: ok &= active[i]
        if ok.sum() < min_active or np.nanstd(s[ok]) == 0: continue
        out[j] = stats.spearmanr(s[ok], t[ok]).statistic
    return out

act = sal > 0.05  # buckets the LLM actually saw news for
for fam, sig in [("sent_level", sent), ("sent_ema3", sent3), ("sent_d1", sentd),
                 ("sent_x_conf", sent * conf)]:
    for tname, tgt, h in [("y5_bucket", y5b, 5), ("y1_bucket", y1b, 1)]:
        ics = daily_xb_ic(sig, tgt, active=act)
        t, p, n = overlap_t(ics, h)
        lo, hi = block_boot_mean(ics)
        screen(f"DIR_{fam}_{tname}", fam, tname, "mean_daily_xbucket_IC",
               np.nanmean(ics), p, n,
               dict(t=round(t, 2) if not np.isnan(t) else None, ci90=[round(lo, 4), round(hi, 4)],
                    segs=seg_signs(ics, idx)))

# per-bucket time-series IC for sent (exploratory map, single family screen on max |IC| noted)
per_bucket = {}
for kb, b in enumerate(buckets):
    s = sent[idx, kb]; t5 = y5b[idx, kb]
    ok = ~(np.isnan(s) | np.isnan(t5)) & (np.abs(s) > 0)
    if ok.sum() >= 40:
        r = stats.spearmanr(s[ok], t5[ok]).statistic
        per_bucket[b] = dict(ic5=round(float(r), 3), n=int(ok.sum()))

# global axes vs their natural books (time-series)
book_r5 = np.nanmean(open_px[5:] / open_px[:-5] - 1.0, axis=1)   # rough equal-w book 5d
book_r5 = np.concatenate([book_r5, np.full(5, np.nan)])
kb_rates = buckets.index("rates_duration"); kb_gold = buckets.index("gold_pm")
kb_vol = buckets.index("volatility")
r5_bucket_abs = y5b  # abnormal; for axes also use raw bucket return where sensible
for tid, sig, tgt, note in [
    ("AX_riskapp_book5", risk_app[idx], book_r5[idx], "risk_appetite vs book fwd 5d"),
    ("AX_ratesp_bonds5", -rates_p[idx], y5b[idx, kb_rates], "-rates_pressure vs rates_duration y5b"),
    ("AX_geopol_gold5", geopol[idx], y5b[idx, kb_gold], "geopol vs gold_pm y5b"),
    ("AX_geopol_vix5", geopol[idx], y5b[idx, kb_vol], "geopol vs volatility-bucket y5b")]:
    r, p, n, ci = spearman_boot(sig, tgt)
    screen(tid, "global_axes", note, "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))

# ---------------------------------------------------------------- (b) DISPERSION
# disagreement measures
salw = np.where(np.isnan(sal), 0, sal)
def wstd(x, w):
    out = np.full(x.shape[0], np.nan)
    for i in range(x.shape[0]):
        xi, wi = x[i], w[i]
        ok = ~np.isnan(xi)
        if ok.sum() < 5 or wi[ok].sum() <= 0: continue
        m = np.average(xi[ok], weights=wi[ok])
        out[i] = np.sqrt(np.average((xi[ok] - m) ** 2, weights=wi[ok]))
    return out
disag_u = np.nanstd(sent, axis=1)            # unweighted cross-bucket std
disag_w = wstd(sent, salw)                   # salience-weighted

# realized cross-sectional dispersion of r5 (forward), and trailing control
r5_sym = np.full((N, len(symbols)), np.nan)
r5_sym[:-5] = open_px[5:] / open_px[:-5] - 1.0
r5_sym[~sym_mask] = np.nan
disp_fwd = np.nanstd(r5_sym - np.nanmean(r5_sym, axis=1, keepdims=True), axis=1)
disp_trail = np.full(N, np.nan); disp_trail[5:] = disp_fwd[:-5]  # r5 ending at D

def resid_rank(y, x):
    """rank-residualize y on x (both 1d, aligned)."""
    ok = ~(np.isnan(y) | np.isnan(x))
    ry = np.full_like(y, np.nan); rx = stats.rankdata(x[ok]); ryy = stats.rankdata(y[ok])
    b = np.polyfit(rx, ryy, 1)
    res = ryy - np.polyval(b, rx)
    ry[ok] = res
    return ry

# CAST / GBM daily OOF IC (folds 5,6)
def member_daily_ic(member):
    rows = {}
    for f in (5, 6):
        zz = np.load(os.path.join(P6, f"oof/fold_{f}_{member}.npz"), allow_pickle=True)
        dts = pd.to_datetime(zz["dates"]); mu = zz["mu"]
        for k, d in enumerate(dts):
            i = int(np.searchsorted(dates, d))
            if i >= N or dates[i] != d: continue
            m, t = mu[k], y5_raw[i]
            ok = ~(np.isnan(m) | np.isnan(t)) & sym_mask[i]
            if ok.sum() >= 20:
                rows[i] = stats.spearmanr(m[ok], t[ok]).statistic
    return rows
cast_ic = member_daily_ic("cast"); gbm_ic = member_daily_ic("gbm_cond")
print(f"CAST OOF IC days in LLM window: {sum(1 for i in cast_ic if mask[i])}, "
      f"GBM: {sum(1 for i in gbm_ic if mask[i])}")

disp_fwd_res = resid_rank(disp_fwd, disp_trail)
for sid, sig in [("disag_unw", disag_u), ("disag_salw", disag_w)]:
    # vs realized fwd dispersion (raw + trailing-controlled)
    r, p, n, ci = spearman_boot(sig[idx], disp_fwd[idx])
    screen(f"DISP_{sid}_raw", "dispersion", "fwd_xs_dispersion_r5", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))
    r, p, n, ci = spearman_boot(sig[idx], disp_fwd_res[idx])
    screen(f"DISP_{sid}_ctl", "dispersion", "fwd_xs_dispersion_r5|trail", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))
    # vs member IC
    for mname, mic in [("cast", cast_ic), ("gbm", gbm_ic)]:
        ii = [i for i in idx if i in mic]
        a = np.array([sig[i] for i in ii]); b = np.array([mic[i] for i in ii])
        r, p, n, ci = spearman_boot(a, b)
        screen(f"DISP_{sid}_{mname}IC", "dispersion", f"{mname}_daily_IC", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))

# ---------------------------------------------------------------- (c) VOL EXPANSION
book_dr = np.nanmean(np.diff(close_px, axis=0) / close_px[:-1], axis=1)
book_dr = np.concatenate([[np.nan], book_dr])
trail21 = pd.Series(book_dr).rolling(21).std().to_numpy()
vol_expand = fwd_book_vol5 / trail21
vol_expand_res = resid_rank(vol_expand, trail21)
sal_total = np.nansum(salw, axis=1)
ev_count = np.nansum(ev, axis=1)
conf_mean = np.nanmean(conf, axis=1)
for sid, sig in [("geopol", geopol), ("sal_total", sal_total), ("ev_count", ev_count),
                 ("sal_volatility", sal[:, kb_vol]), ("n_clusters", n_clusters)]:
    r, p, n, ci = spearman_boot(sig[idx], vol_expand[idx])
    screen(f"VOL_{sid}_raw", "vol_expansion", "fwd_book_vol5/trail21", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))
    r, p, n, ci = spearman_boot(sig[idx], vol_expand_res[idx])
    screen(f"VOL_{sid}_ctl", "vol_expansion", "fwd_vol_expand|trail", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))

# ---------------------------------------------------------------- (e) CONDITIONAL IC
def cond_split(cond_vec, mic, label):
    ii = [i for i in idx if i in mic and not np.isnan(cond_vec[i])]
    c = np.array([cond_vec[i] for i in ii]); v = np.array([mic[i] for i in ii])
    if len(ii) < 60: return
    med = np.median(c)
    hi, lo = v[c > med], v[c <= med]
    d = hi.mean() - lo.mean()
    # block bootstrap on the paired daily series of (indicator, ic)
    n = len(v); nb = int(np.ceil(n / BLOCK)); ds = np.empty(B_BOOT)
    for k in range(B_BOOT):
        starts = RNG.integers(0, n, nb)
        sel = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
        sel = sel[:n]
        cc, vv = c[sel], v[sel]
        m2 = np.median(cc)
        h2, l2 = vv[cc > m2], vv[cc <= m2]
        ds[k] = h2.mean() - l2.mean() if len(h2) > 5 and len(l2) > 5 else np.nan
    ds = ds[~np.isnan(ds)]
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean()); p = max(p, 1.0 / B_BOOT)
    lo_, hi_ = np.quantile(ds, [0.05, 0.95])
    r, pr, _, _ = spearman_boot(c, v)
    screen(f"COND_{label}", "conditional_IC", label.split("__")[1] + "_IC", "IC_hi_minus_lo",
           d, p, len(v), dict(ci90=[round(float(lo_), 4), round(float(hi_), 4)],
                              ic_hi=round(float(hi.mean()), 4), ic_lo=round(float(lo.mean()), 4),
                              spearman=round(r, 3) if not np.isnan(r) else None,
                              spearman_p=round(pr, 4) if not np.isnan(pr) else None))

for cname, cvec in [("sal_total", sal_total), ("conf_mean", conf_mean),
                    ("ev_count", ev_count), ("geopol", geopol), ("disag_salw", disag_w)]:
    for mname, mic in [("cast", cast_ic), ("gbm", gbm_ic)]:
        cond_split(cvec, mic, f"{cname}__{mname}")

# ---------------------------------------------------------------- event flags + chattiness
flag_stats = {}
onsets_any = np.zeros(N)
abs_book_r1 = np.abs(np.concatenate([np.diff(np.log(np.nanmean(close_px, axis=1)))[0:], [np.nan]]))
book_close = np.nanmean(close_px / close_px[0], axis=1)
book_lr1 = np.concatenate([[np.nan], np.diff(np.log(book_close))])
fwd_abs1 = np.concatenate([np.abs(book_lr1[1:]), [np.nan]])  # |book ret| on D->D+1 close
for k, c in enumerate(EVENTS):
    f = np.where(np.isnan(ev[:, k]), 0, ev[:, k])
    fa = f[idx]
    fire = float(np.mean(fa > 0))
    prev = np.concatenate([[0], f[:-1]])
    onset = ((f > 0) & (prev == 0)).astype(float)
    onsets_any += onset
    runs = []
    run = 0
    for v in fa:
        if v > 0: run += 1
        elif run: runs.append(run); run = 0
    if run: runs.append(run)
    flag_stats[c.replace("llm_event_", "")] = dict(
        fire_rate=round(fire, 3), onset_rate=round(float(np.mean(onset[idx])), 3),
        mean_run=round(float(np.mean(runs)), 1) if runs else 0.0)
# level vs onset: predict next-day |book return| and vol expansion
for sid, sig in [("ev_level_count", ev_count), ("ev_onset_count", onsets_any)]:
    r, p, n, ci = spearman_boot(sig[idx], fwd_abs1[idx], h=1)
    screen(f"EV_{sid}_absr1", "event_flags", "next_day_abs_book_ret", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))
    r, p, n, ci = spearman_boot(sig[idx], vol_expand_res[idx])
    screen(f"EV_{sid}_volexp", "event_flags", "fwd_vol_expand|trail", "spearman", r, p, n, dict(ci90=list(np.round(ci, 4))))

# severity from raw artifacts (de-chattering design substrate)
sev_rows = []
for i in idx:
    d = dates[i]
    # artifact actually joined to D
    ad = joined["art_date"].iloc[i]
    if pd.isna(ad): continue
    p = os.path.join(P6, "store/llm", f"{pd.Timestamp(ad).date()}.json")
    if not os.path.exists(p): continue
    j = json.load(open(p))
    for fl in (j.get("llm") or {}).get("event_flags", []):
        sev_rows.append(dict(D=str(d.date()), flag=fl.get("flag"),
                             sev=fl.get("severity", np.nan),
                             nb=len(fl.get("buckets", []))))
sev = pd.DataFrame(sev_rows)
sev_summary = {}
if len(sev):
    g = sev.groupby("flag")["sev"]
    sev_summary = {k: dict(n=int(v.size), mean=round(float(v.mean()), 2),
                           q50=round(float(v.quantile(.5)), 2), q80=round(float(v.quantile(.8)), 2),
                           pct_ge_05=round(float((v >= 0.5).mean()), 2),
                           pct_ge_07=round(float((v >= 0.7).mean()), 2))
                   for k, v in g}
    # days with at least one severity>=0.7 flag
    day_max = sev.groupby("D")["sev"].max()
    hi_days = set(day_max[day_max >= 0.7].index)
    hi_vec = np.array([1.0 if str(dates[i].date()) in hi_days else 0.0 for i in idx])
    r, p, n, ci = spearman_boot(hi_vec, vol_expand_res[idx])
    screen("EV_sev07_volexp", "event_flags", "fwd_vol_expand|trail", "spearman", r, p, n,
           dict(ci90=list(np.round(ci, 4)), note="day has any flag severity>=0.7"))
    r, p, n, ci = spearman_boot(hi_vec, fwd_abs1[idx], h=1)
    screen("EV_sev07_absr1", "event_flags", "next_day_abs_book_ret", "spearman", r, p, n,
           dict(ci90=list(np.round(ci, 4))))

# ---------------------------------------------------------------- FDR + write
ps = [(i, t["p"]) for i, t in enumerate(LEDGER) if t["p"] is not None]
m = len(ps)
bh = sorted(ps, key=lambda x: x[1])
surv = set()
for rank, (i, p) in enumerate(bh, 1):
    if p <= 0.10 * rank / m:
        surv = {j for j, _ in bh[:rank]}
for i, t in enumerate(LEDGER):
    t["bh_fdr10_survivor"] = i in surv

out = dict(generated=pd.Timestamp.now().isoformat(), n_days=len(idx),
           window=[str(dates[idx[0]].date()), str(dates[idx[-1]].date())],
           n_screens=m, n_bh_survivors=len(surv),
           ledger=LEDGER, per_bucket_sent_ic5=per_bucket,
           flag_chattiness=flag_stats, flag_severity=sev_summary)
with open(os.path.join(HERE, "llm_role_probe.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)

print(f"\nscreens: {m}  BH-FDR(10%) survivors: {len(surv)}")
for t in LEDGER:
    star = " *" if t["bh_fdr10_survivor"] else ""
    print(f"{t['id']:34s} {t['stat']:24s} v={t['value']!s:9s} p={t['p']!s:9s} n={t['n']}{star}")
print("\nper-bucket sent IC5 (|IC|>=0.08):",
      {k: v for k, v in per_bucket.items() if abs(v['ic5']) >= 0.08})
print("\nchattiness:", json.dumps(flag_stats, indent=0))
print("\nseverity:", json.dumps(sev_summary, indent=0))
