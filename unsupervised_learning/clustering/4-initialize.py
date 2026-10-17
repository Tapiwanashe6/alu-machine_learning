#!/usr/bin/env python3
"""Initialize the parameters of a Gaussian Mixture Model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize equal priors, K-means centroids, and identity covariances.

    Args:
        X: Dataset of shape (n, d).
        k: Positive integer number of clusters.

    Returns:
        Tuple (pi, m, S), or (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    n, d = X.shape
    if n < 1 or d < 1:
        return None, None, None

    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > n:
        return None, None, None

    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
