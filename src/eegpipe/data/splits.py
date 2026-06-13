"""Subject-aware cross-validation.

The single most common methodological error in EEG ML is letting epochs from the same
subject appear in both train and test folds. Because EEG is dominated by between-subject
variance, this inflates accuracy by 10-30 points and the model fails to generalise.

Every splitter here is *group-aware by construction*. The default training entry point
uses these — never plain ``KFold`` — and :func:`assert_no_subject_leakage` guards it.
"""
from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold


def subject_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K-fold where each subject is wholly in train OR test.

    ``stratified=True`` additionally balances class frequencies across folds
    (``StratifiedGroupKFold``), which matters for imbalanced paradigms like sleep staging.
    """
    n_subjects = len(np.unique(groups))
    if n_subjects < n_splits:
        warnings.warn(
            f"n_splits={n_splits} but only {n_subjects} subjects; "
            f"reducing n_splits to {n_subjects}.",
            stacklevel=2,
        )
        n_splits = n_subjects
    splitter = StratifiedGroupKFold(n_splits=n_splits) if stratified else GroupKFold(n_splits=n_splits)
    yield from splitter.split(X, y, groups)


def leave_one_subject_out(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """LOSO — the gold standard for reporting cross-subject BCI generalisation."""
    yield from LeaveOneGroupOut().split(X, y, groups)


def nested_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_outer: int = 5,
    n_inner: int = 3,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Nested CV for unbiased hyper-parameter selection.

    Yields ``(train_idx, val_idx, test_idx)`` where the inner split (train/val) is carved
    out of the outer training set, all respecting subject groups. Use ``val`` for HPO /
    early stopping and ``test`` only for the final unbiased estimate.
    """
    for outer_train, test_idx in subject_kfold(X, y, groups, n_splits=n_outer):
        inner_groups = groups[outer_train]
        inner = subject_kfold(
            X[outer_train], y[outer_train], inner_groups, n_splits=n_inner
        )
        tr_rel, val_rel = next(inner)  # first inner fold; loop externally for full nesting
        yield outer_train[tr_rel], outer_train[val_rel], test_idx


def assert_no_subject_leakage(
    train_groups: np.ndarray, test_groups: np.ndarray
) -> None:
    """Raise if any subject appears in both partitions. Call inside every fold."""
    overlap = set(np.unique(train_groups)) & set(np.unique(test_groups))
    if overlap:
        raise ValueError(f"Subject leakage detected: subjects {sorted(overlap)} in train AND test.")
