"""The most important tests in the repo: prove no subject leaks across folds."""

import numpy as np
import pytest

from eegpipe.data.splits import (
    assert_no_subject_leakage,
    leave_one_subject_out,
    subject_kfold,
)


def test_subject_kfold_no_leakage(synthetic_eeg):
    X, y, groups = synthetic_eeg
    for tr, te in subject_kfold(X, y, groups, n_splits=4):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_loso_holds_out_one_subject(synthetic_eeg):
    X, y, groups = synthetic_eeg
    folds = list(leave_one_subject_out(X, y, groups))
    assert len(folds) == len(np.unique(groups))
    for _, te in folds:
        assert len(np.unique(groups[te])) == 1


def test_assert_raises_on_leakage():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_subject_leakage(np.array([1, 2]), np.array([2, 3]))


def test_kfold_caps_splits_to_n_subjects(synthetic_eeg):
    X, y, groups = synthetic_eeg
    with pytest.warns(UserWarning):
        folds = list(subject_kfold(X, y, groups, n_splits=10))  # only 4 subjects
    assert len(folds) == 4
