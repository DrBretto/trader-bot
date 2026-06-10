"""Unit tests for folds.py boundaries (BUILD_SPEC §10.1 / deliverable 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))

import folds  # noqa: E402


@pytest.fixture(scope="module")
def dates():
    z = np.load(PROTO / "store" / "panel.npz", allow_pickle=False)
    return z["dates"]


def test_six_folds_yearly_feb_boundaries():
    fw = folds.folds()
    assert [f.name for f in fw] == [f"F{i}" for i in range(1, 7)]
    for i, f in enumerate(fw):
        assert f.start == f"{2020 + i}-02-01"
        assert f.end == f"{2021 + i}-02-01"


def test_fold_sizes_about_250_td(dates):
    for f in range(1, 7):
        n = int(folds.fold_date_mask(dates, f).sum())
        assert 245 <= n <= 255, (f, n)


def test_embargo_gap_exactly_22_td(dates):
    for f in range(1, 7):
        tm, vm = folds.fold_train_val(dates, f)
        last_train = int(np.nonzero(tm)[0][-1])
        first_val = int(np.nonzero(vm)[0][0])
        # gap of exactly EMBARGO_TD trading days strictly between
        assert first_val - last_train - 1 == folds.EMBARGO_TD


def test_no_train_val_overlap_and_expanding(dates):
    prev_n = 0
    for f in range(1, 7):
        tm, vm = folds.fold_train_val(dates, f)
        assert not np.any(tm & vm)
        n = int(tm.sum())
        assert n > prev_n            # expanding head-training windows
        prev_n = n


def test_holdout_never_touched(dates):
    for f in range(1, 7):
        tm, vm = folds.fold_train_val(dates, f)
        for m in (tm, vm):
            assert not np.any(np.asarray(dates)[m] >= folds.HOLDOUT_START)
    dm = folds.deploy_train_mask(dates)
    assert not np.any(np.asarray(dates)[dm] >= folds.HOLDOUT_START)


def test_holdout_guard_raises(dates):
    bad = np.zeros(len(dates), dtype=bool)
    bad[-1] = True                   # 2026-06-10 ≥ holdout start
    with pytest.raises(AssertionError):
        folds.assert_no_holdout(dates, bad)


def test_deploy_train_end_is_fine_tune_cutoff(dates):
    dm = folds.deploy_train_mask(dates)
    last = np.asarray(dates)[dm][-1]
    assert last == folds.FINE_TUNE_CUTOFF == "2026-03-03"
    # last 5-td target window ends before the holdout boundary
    d = np.asarray(dates)
    i = int(np.nonzero(dm)[0][-1])
    assert d[i + 5] < folds.HOLDOUT_START


def test_fitness_end_constant():
    assert folds.FITNESS_END == "2026-02-06"
    assert folds.HOLDOUT_START == "2026-03-11"


def test_f6_validation_pre_fitness_end(dates):
    vm = folds.fold_date_mask(dates, 6)
    v = np.asarray(dates)[vm]
    assert v[-1] <= folds.FITNESS_END


def test_seed_formula_deterministic():
    a = folds.component_master_seed("cast", "2026-03-03")
    b = folds.component_master_seed("cast", "2026-03-03")
    c = folds.component_master_seed("gbm_cond", "2026-03-03")
    assert a == b and a != c and 0 <= a < 2**32
    assert folds.OOF_SEEDS == [101, 102, 103]
    assert folds.DEPLOY_SEEDS == [11, 13, 17, 19, 23]
