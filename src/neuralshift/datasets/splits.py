"""Stratified train/validation splits; official test indices are a separate namespace."""

import numpy as np
from sklearn.model_selection import train_test_split


def make_splits(labels, validation_fraction=0.1, seed=2026):
    indices = np.arange(len(labels))
    train, val = train_test_split(indices, test_size=validation_fraction, random_state=seed, stratify=labels)
    return np.sort(train), np.sort(val)


def subset_indices(indices, limit, seed):
    if limit is None or limit >= len(indices):
        return np.asarray(indices)
    return np.sort(np.random.default_rng(seed).choice(indices, size=limit, replace=False))
