#!/usr/bin/env python3
"""Test K-means cluster counts using intra-cluster variance."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Calculate variance improvements across different cluster counts.

    Args:
        X: Dataset with shape (n, d).
        kmin: Smallest number of clusters to test.
        kmax: Largest number of clusters to test, or None.
        iterations: Maximum K-means iterations.

    Returns:
        K-means results and variance differences, or (None, None)
        on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n, d = X.shape
    if n < 2 or d < 1:
        return None, None

    if not isinstance(kmin, int) or isinstance(kmin, bool) or kmin < 1:
        return None, None

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or isinstance(kmax, bool)
            or kmax > n or kmax <= kmin):
        return None, None

    if (not isinstance(iterations, int) or isinstance(iterations, bool)
            or iterations < 1):
        return None, None

    results = []
    d_vars = []
    initial_variance = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        current_variance = variance(X, C)
        if current_variance is None:
            return None, None

        results.append((C, clss))
        if initial_variance is None:
            initial_variance = current_variance
        d_vars.append(initial_variance - current_variance)

    return results, d_vars
