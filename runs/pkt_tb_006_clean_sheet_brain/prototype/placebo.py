"""PKT-TB-006 — permuted-dictionary placebo (TOURNAMENT §4.6.3; Skeptic G-K4).

50 seeded random permutations of the theme→sleeve dictionary (same cardinality:
the multiset of sleeve labels is preserved, themes are re-assigned). For each
dictionary (real + 50 placebos), G1 per-bucket share + trailing-90d z features
are recomputed CHEAPLY from gdelt_cache/daily theme counts via the
features_gdelt machinery (_trailing_z), and a cheap proxy fit (elastic net on
the G1 block) produces a training-fold rank-IC. The real dictionary's IC must
exceed the 95th percentile of the placebo distribution, or the G1 dictionary is
reported `0 (measured)` regardless of the block arm.

TRAINING FOLDS ONLY: every fit trains on the per-fold expanding embargoed
training mask (folds.train_mask_for_fold) and scores rank-IC on F1–F6 fold days
(all <= 2026-02-06 by fold_date_mask). No replay; no holdout (holdout-guarded
by folds.assert_no_holdout). Because F5/F6 days are read, the gate state is a
validation-fold decision and is appended to validation_looks.jsonl.

FORCED CHOICES (apples-to-apples; logged in the result + ledger):
  - Both real AND placebo G1 features are recomputed from the daily
    ``theme_counts`` (top-500) aggregates — NOT the backfill's full-record
    bucket counts — so the real/placebo contrast is computed by the identical
    path. (The shipped panel's g1 features use the full-record counts; this
    placebo isolates the DICTIONARY's information, which is what §4.6.3 tests.)
  - Proxy target: per-date cross-sectional rank of the panel's forward-5d
    bucket return (y5_bucket) over the 20 themed sleeves; rank-IC = per-date
    Spearman of prediction vs target, pooled over F1–F6 fold days.
  - Proxy model: ElasticNet(alpha=1e-3, l1_ratio=0.5) on [g1_share, g1_z] per
    (date, sleeve) row, standardized on the training mask.
  - Alignment to trading dates reuses the panel's ``gdelt_day_joined`` column
    (identical visible_from join as production).

Seeds: permutation k uses int(sha256("PKT-TB-006-PLACEBO::perm<k>")[:8], 16)
(the §4.4 component-seed recipe).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import folds as fd
import features_gdelt as fg

PROTO = Path(__file__).resolve().parent
RESULT_PATH = PROTO / "placebo_result.json"
VALIDATION_LEDGER = PROTO / "validation_looks.jsonl"
N_PERMUTATIONS = 50


def perm_seed(k: int) -> int:
    h = hashlib.sha256(f"PKT-TB-006-PLACEBO::perm{k}".encode())
    return int(h.hexdigest()[:8], 16)


# ============================ data assembly ====================================
def load_theme_counts(daily_dir: Path = fg.GDELT_DAILY,
                      themes: list[str] | None = None):
    """[n_days x n_themes] count matrix over the dictionary's themes."""
    t2s = json.loads((fg.DICTS / "theme_to_sector.json").read_text())
    themes = themes or sorted(t2s)
    t_idx = {t: i for i, t in enumerate(themes)}
    files = sorted(Path(daily_dir).glob("*.json"))
    dates, rows = [], []
    for f in files:
        d = json.loads(f.read_text())
        tc = d.get("theme_counts", {}) or {}
        row = np.zeros(len(themes))
        for t, v in tc.items():
            j = t_idx.get(t)
            if j is not None:
                row[j] = float(v)
        dates.append(pd.Timestamp(dt.datetime.strptime(d["date"], "%Y%m%d").date()))
        rows.append(row)
    return (pd.Series(dates), np.asarray(rows), themes,
            [t2s[t] for t in themes])


def g1_features_for_assignment(gdates: pd.Series, counts: np.ndarray,
                               assignment: list[str], sleeves: list[str]):
    """(share[n_days x n_sleeves], z[n_days x n_sleeves]) for one theme→sleeve
    assignment, via the features_gdelt trailing-z machinery (days < d only)."""
    A = np.zeros((counts.shape[1], len(sleeves)))
    s_idx = {s: j for j, s in enumerate(sleeves)}
    for i, s in enumerate(assignment):
        A[i, s_idx[s]] = 1.0
    bucket_counts = counts @ A
    total = bucket_counts.sum(axis=1)
    total[total <= 0] = np.nan
    share = bucket_counts / total[:, None]
    z = np.empty_like(share)
    for j in range(len(sleeves)):
        z[:, j] = fg._trailing_z(pd.Series(share[:, j]), gdates).to_numpy()
    return share, z


def join_to_trading_dates(panel, gdates: pd.Series, share, z, sleeves):
    """Align gdelt-day features to panel trading dates via gdelt_day_joined."""
    g_lut = {str(d.date()): i for i, d in enumerate(gdates)}
    tdates = np.asarray(panel["dates"]).astype(str)
    joined = np.asarray(panel["gdelt_day_joined"]).astype(str)
    n = len(tdates)
    share_t = np.full((n, len(sleeves)), np.nan)
    z_t = np.full((n, len(sleeves)), np.nan)
    for i, gj in enumerate(joined):
        k = g_lut.get(gj)
        if k is not None:
            share_t[i] = share[k]
            z_t[i] = z[k]
    return share_t, z_t


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa <= 1e-12 or sb <= 1e-12:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def training_fold_rank_ic(share_t, z_t, y5_sleeves, tdates) -> dict:
    """Pooled F1–F6 fold-day rank-IC of the elastic-net G1 proxy.

    Per fold f: fit on train_mask_for_fold rows (pooled date×sleeve), predict
    fold days, per-date Spearman over sleeves; statistic = mean over all pooled
    fold days."""
    from sklearn.linear_model import ElasticNet
    n, m = share_t.shape
    ics = []
    per_fold = {}
    for f in range(1, 7):
        tm = fd.train_mask_for_fold(tdates, f)
        vm = fd.fold_date_mask(tdates, f)
        fd.assert_no_holdout(tdates, tm)
        fd.assert_no_holdout(tdates, vm)
        def pool(mask):
            rows_x, rows_y, day_of = [], [], []
            idx = np.where(mask)[0]
            for i in idx:
                x = np.stack([share_t[i], z_t[i]], axis=1)       # [m,2]
                y = y5_sleeves[i]
                ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
                if ok.sum() < max(5, m // 2):
                    continue
                yr = _rank(y[ok]) / max(ok.sum() - 1, 1)
                rows_x.append(x[ok])
                rows_y.append(yr)
                day_of.append(np.full(int(ok.sum()), i))
            if not rows_x:
                return None, None, None
            return (np.concatenate(rows_x), np.concatenate(rows_y),
                    np.concatenate(day_of))
        Xtr, ytr, _ = pool(tm)
        Xva, yva, day_va = pool(vm)
        if Xtr is None or Xva is None:
            per_fold[str(f)] = None
            continue
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd <= 1e-12] = 1.0
        net = ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000)
        net.fit((Xtr - mu) / sd, ytr)
        pred = net.predict((Xva - mu) / sd)
        fold_ics = []
        for i in np.unique(day_va):
            sel = day_va == i
            if sel.sum() >= 5:
                ic = _spearman(pred[sel], yva[sel])
                if np.isfinite(ic):
                    fold_ics.append(ic)
        per_fold[str(f)] = float(np.mean(fold_ics)) if fold_ics else None
        ics.extend(fold_ics)
    return {"mean_ic": float(np.mean(ics)) if ics else float("nan"),
            "n_fold_days": len(ics), "per_fold": per_fold}


# ============================ the gate =========================================
def run_placebo(n_permutations: int = N_PERMUTATIONS,
                panel_path: Path = PROTO / "store" / "panel.npz",
                daily_dir: Path = fg.GDELT_DAILY,
                result_path: Path | None = RESULT_PATH,
                ledger_path: Path | None = VALIDATION_LEDGER,
                verbose: bool = True) -> dict:
    panel = np.load(panel_path, allow_pickle=True)
    tdates = np.asarray(panel["dates"]).astype(str)
    buckets_panel = [str(b) for b in panel["buckets"]]
    gdates, counts, themes, real_assignment = load_theme_counts(daily_dir)
    sleeves = sorted(set(real_assignment))
    sleeve_panel_idx = [buckets_panel.index(s) for s in sleeves]
    y5_sleeves = np.asarray(panel["y5_bucket"], dtype=np.float64)[:, sleeve_panel_idx]

    def ic_for(assignment) -> dict:
        share, z = g1_features_for_assignment(gdates, counts, assignment, sleeves)
        share_t, z_t = join_to_trading_dates(panel, gdates, share, z, sleeves)
        return training_fold_rank_ic(share_t, z_t, y5_sleeves, tdates)

    if verbose:
        print(f"placebo: {len(gdates)} gdelt days, {len(themes)} themes, "
              f"{len(sleeves)} sleeves, {n_permutations} permutations")
    real = ic_for(real_assignment)
    if verbose:
        print(f"  REAL dictionary: mean rank-IC {real['mean_ic']:+.4f} "
              f"on {real['n_fold_days']} pooled fold-days; per-fold {real['per_fold']}")
    placebo_ics = []
    arr = np.asarray(real_assignment, dtype=object)
    for k in range(n_permutations):
        rng = np.random.default_rng(perm_seed(k))
        perm = arr[rng.permutation(len(arr))].tolist()
        r = ic_for(perm)
        placebo_ics.append(r["mean_ic"])
        if verbose and (k + 1) % 10 == 0:
            print(f"  placebo {k + 1}/{n_permutations}: latest {r['mean_ic']:+.4f}")
    pl = np.asarray(placebo_ics, dtype=np.float64)
    pl_f = pl[np.isfinite(pl)]
    percentile = float((real["mean_ic"] > pl_f).mean() * 100) if len(pl_f) else float("nan")
    p95 = float(np.percentile(pl_f, 95)) if len(pl_f) else float("nan")
    pass_95 = bool(np.isfinite(real["mean_ic"]) and real["mean_ic"] > p95)
    result = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "spec": "TOURNAMENT §4.6.3 — permuted-dictionary placebo, training folds "
                "only (no replay, no holdout)",
        "n_permutations": int(n_permutations),
        "n_themes": len(themes), "n_sleeves": len(sleeves),
        "n_gdelt_days": int(len(gdates)),
        "real_ic": real["mean_ic"], "real_per_fold": real["per_fold"],
        "n_fold_days": real["n_fold_days"],
        "placebo_ics": [float(x) for x in pl],
        "placebo_mean": float(pl_f.mean()) if len(pl_f) else None,
        "placebo_sd": float(pl_f.std(ddof=1)) if len(pl_f) > 1 else None,
        "placebo_p95": p95,
        "percentile": percentile,
        "pass_95": pass_95,
        "verdict": ("PASS — real G1 dictionary beats the 95th placebo percentile"
                    if pass_95 else
                    "FAIL — G1 dictionary reported `0 (measured)` regardless of "
                    "the block arm (§4.6.3)"),
        "forced_choices": [
            "real + placebo G1 recomputed from top-500 theme_counts (identical path)",
            "target = cross-sectional rank of y5_bucket over the 20 themed sleeves",
            "proxy = ElasticNet(alpha=1e-3, l1_ratio=0.5) on [g1_share, g1_z]",
            "join via panel gdelt_day_joined (production visible_from convention)",
        ],
    }
    if result_path is not None:
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=1)
    if ledger_path is not None:
        rec = {"ts": result["generated"], "component": "placebo_g1_dictionary",
               "kind": "gate_state_read",
               "decision": f"§4.6.3 placebo gate: real IC {real['mean_ic']:+.4f}, "
                           f"percentile {percentile:.1f}, p95 {p95:+.4f} -> "
                           f"{'PASS' if pass_95 else 'FAIL'}",
               "reads_folds": "F1-F6 fold days (includes F5/F6 — ledgered)",
               "provenance": "placebo.py; placebo_result.json"}
        with open(ledger_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    if verbose:
        print(f"  placebo distribution: mean {result['placebo_mean']:+.4f} "
              f"sd {result['placebo_sd']:.4f} p95 {p95:+.4f}")
        print(f"  REAL percentile: {percentile:.1f} -> "
              f"{'PASS' if pass_95 else 'FAIL'}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    run_placebo(n_permutations=args.n,
                ledger_path=None if args.no_ledger else VALIDATION_LEDGER)
