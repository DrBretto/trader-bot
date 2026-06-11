"""PKT-TB-006 fold geometry — BUILD_SPEC §10.1 (single source of truth).

NOTE FOR ORCHESTRATOR: this module was written by the brain-build lane because
``folds.py`` did not exist at build time (member-training lane builds in parallel).
If a parallel copy lands, reconcile to ONE module; the API below is minimal:

    folds()                          -> list of 6 FoldWindow
    fold_date_mask(dates, fold)      -> bool mask of decision dates inside fold f
    train_mask_for_fold(dates, fold) -> head-training mask (expanding, embargoed)
    EMBARGO_TD, FITNESS_END, HOLDOUT_START, FINE_TUNE_CUTOFF

All boundaries are calendar strings; masks are computed against a sorted
``dates`` array of 'YYYY-MM-DD' strings (the panel convention).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMBARGO_TD = 22                      # h_max + 1 (TR S1)
FITNESS_END = "2026-02-06"           # fitness/selection data ends here (§8, §10.1)
HOLDOUT_START = "2026-03-11"         # never touched by training/EA
FINE_TUNE_CUTOFF = "2026-03-03"      # §7.6 (= holdout_start - h - 1)
FINE_TUNE_START = "2025-08-04"
HEAD_TRAIN_START_DEEP = "2014-08-29"     # CAST / RiskNet
HEAD_TRAIN_START_PANEL = "2015-02-18"    # GBM / EventHead (GDELT-rich panel)

_FOLD_BOUNDS = [
    ("F1", "2020-02-01", "2021-02-01"),
    ("F2", "2021-02-01", "2022-02-01"),
    ("F3", "2022-02-01", "2023-02-01"),
    ("F4", "2023-02-01", "2024-02-01"),
    ("F5", "2024-02-01", "2025-02-01"),
    ("F6", "2025-02-01", "2026-02-01"),
]


@dataclass(frozen=True)
class FoldWindow:
    fold: int            # 1..6
    name: str
    start: str           # inclusive calendar bound
    end: str             # exclusive calendar bound


def folds() -> list[FoldWindow]:
    return [FoldWindow(i + 1, n, s, e) for i, (n, s, e) in enumerate(_FOLD_BOUNDS)]


def fold_date_mask(dates: np.ndarray, fold: int, clip_fitness_end: bool = True) -> np.ndarray:
    """Boolean mask of decision dates inside fold ``fold`` (1-based)."""
    fw = folds()[fold - 1]
    d = np.asarray(dates)
    m = (d >= fw.start) & (d < fw.end)
    if clip_fitness_end:
        m &= d <= FITNESS_END
    return m


def train_mask_for_fold(dates: np.ndarray, fold: int, panel_start: str = HEAD_TRAIN_START_PANEL) -> np.ndarray:
    """Expanding head-training window: panel_start -> fold.start − EMBARGO_TD trading days."""
    fw = folds()[fold - 1]
    d = np.asarray(dates)
    idx_start = int(np.searchsorted(d, fw.start, side="left"))
    cut = max(0, idx_start - EMBARGO_TD)
    m = np.zeros(len(d), dtype=bool)
    m[:cut] = True
    m &= d >= panel_start
    return m


def embargoed_split(n: int, val_frac: float = 0.15, embargo: int = EMBARGO_TD):
    """Chronological train/val split with an embargo gap (executive early-stop)."""
    n_val = max(1, int(round(n * val_frac)))
    cut = n - n_val
    train_idx = np.arange(0, max(0, cut - embargo))
    val_idx = np.arange(cut, n)
    return train_idx, val_idx


# --------------------------------------------------------------------------
# Member-training extensions (PKT-TB-006 member lane; same single module)
# --------------------------------------------------------------------------

OOF_SEEDS = [101, 102, 103]            # TOURNAMENT §4.4 (fixed list)
DEPLOY_SEEDS = [11, 13, 17, 19, 23]    # TOURNAMENT §4.4 (fixed list)


def component_master_seed(component: str, train_window_end: str) -> int:
    """TOURNAMENT §4.4: int of first 8 hex digits of
    sha256('PKT-TB-006-SYN1::' + component + '::' + train_window_end)."""
    import hashlib
    h = hashlib.sha256(
        f"PKT-TB-006-SYN1::{component}::{train_window_end}".encode())
    return int(h.hexdigest()[:8], 16)


def deploy_train_mask(dates: np.ndarray,
                      panel_start: str = HEAD_TRAIN_START_PANEL) -> np.ndarray:
    """Deploy ('full pre-holdout window') training mask: panel start →
    FINE_TUNE_CUTOFF inclusive (= holdout_start − h − 1 td, so no 5-day target
    window crosses into the holdout)."""
    d = np.asarray(dates)
    m = (d >= panel_start) & (d <= FINE_TUNE_CUTOFF)
    assert_no_holdout(d, m)
    return m


def assert_no_holdout(dates: np.ndarray, mask: np.ndarray) -> None:
    """Hard guard: no training/validation decision date >= HOLDOUT_START."""
    d = np.asarray(dates)
    if bool(np.any(mask & (d >= HOLDOUT_START))):
        raise AssertionError(
            f"fold/deploy mask touches the holdout (>= {HOLDOUT_START})")


def fold_train_val(dates: np.ndarray, fold: int,
                   panel_start: str = HEAD_TRAIN_START_PANEL):
    """(train_mask, val_mask) for fold 1..6, holdout-guarded."""
    tm = train_mask_for_fold(dates, fold, panel_start)
    vm = fold_date_mask(dates, fold)
    assert_no_holdout(dates, tm)
    assert_no_holdout(dates, vm)
    if bool(np.any(tm & vm)):
        raise AssertionError("train/val overlap")
    return tm, vm


# ==========================================================================
# PKT-TB-007 extensions (BUILD_SPEC_007 §1/§5.1) — appended to the verbatim
# TB-006 copy. Fold geometry F1-F6 unchanged; what changes:
#   * per-member embargo: 21 td default, M2 = 26 td (21 + 5, TOURNAMENT G1)
#   * horizon-aware deploy cutoffs: the LAST training decision date D must
#     have its full target window realized strictly before HOLDOUT_START
#     (h = last target bar offset in trading days: M1/M3/M4 -> 5+1,
#      M2 -> 21+1, M6 -> 1+1; we use the TB-006 convention cutoff index =
#      idx(HOLDOUT_START) - h - 1).
#   * 007 seeds (TOURNAMENT §4.7): master 4242; CAST OOF ensemble {4242, 4243}.
# ==========================================================================

EMBARGO_TD_007 = 22          # = TB-006 EMBARGO_TD (21-td embargo + 1, h_max+1)
EMBARGO_TD_007_M2 = 27       # 26-td embargo + 1 (M2 target ends at D+21)
MASTER_SEED_007 = 4242
CAST_OOF_SEEDS_007 = [4242, 4243]


def train_mask_for_fold_007(dates: np.ndarray, fold: int, embargo: int,
                            panel_start: str = HEAD_TRAIN_START_PANEL) -> np.ndarray:
    """Expanding head-training window with a member-specific embargo."""
    fw = folds()[fold - 1]
    d = np.asarray(dates)
    idx_start = int(np.searchsorted(d, fw.start, side="left"))
    cut = max(0, idx_start - embargo)
    m = np.zeros(len(d), dtype=bool)
    m[:cut] = True
    m &= d >= panel_start
    assert_no_holdout(d, m)
    return m


def fold_train_val_007(dates: np.ndarray, fold: int, embargo: int = EMBARGO_TD_007,
                       panel_start: str = HEAD_TRAIN_START_PANEL):
    tm = train_mask_for_fold_007(dates, fold, embargo, panel_start)
    vm = fold_date_mask(dates, fold)
    assert_no_holdout(dates, tm)
    assert_no_holdout(dates, vm)
    if bool(np.any(tm & vm)):
        raise AssertionError("train/val overlap")
    return tm, vm


def deploy_train_mask_007(dates: np.ndarray, horizon_td: int,
                          panel_start: str = HEAD_TRAIN_START_PANEL) -> np.ndarray:
    """Deploy mask: panel start -> last decision date whose target window
    (last bar at D + horizon_td) is fully realized before HOLDOUT_START.
    cutoff index = idx(HOLDOUT_START) - horizon_td - 1 (TB-006 convention:
    FINE_TUNE_CUTOFF = holdout_start - h - 1 with h = 5)."""
    d = np.asarray(dates)
    ih = int(np.searchsorted(d, HOLDOUT_START, side="left"))
    cut = ih - horizon_td - 1          # last allowed index (inclusive)
    m = np.zeros(len(d), dtype=bool)
    m[:cut + 1] = True
    m &= d >= panel_start
    assert_no_holdout(d, m)
    return m
