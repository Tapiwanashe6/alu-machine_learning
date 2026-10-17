#!/usr/bin/env python3
"""Calculate the total intra-cluster variance of a dataset."""

import numpy as np


def variance(X, C):
    """
    Calculate the total squared distance to the nearest centroid.

    Args:
        X: Data array of shape (n, d).
        C: Centroid array of shape (k, d).

    Returns:
        Total intra-cluster variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    n, d = X.shape
    k, centroid_d = C.shape
    if n == 0 or d == 0 or k == 0 or centroid_d != d:
        return None

    squared_distances = np.sum(
        (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
        axis=2
    )
    return np.sum(np.min(squared_distances, axis=1))
