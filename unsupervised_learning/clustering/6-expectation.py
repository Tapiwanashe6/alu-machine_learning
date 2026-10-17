
#!/usr/bin/env python3
"""Calculates the expectation step for a Gaussian Mixture Model."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X: Dataset with shape (n, d).
        pi: Cluster priors with shape (k,).
        m: Cluster means with shape (k, d).
        S: Covariance matrices with shape (k, d, d).

    Returns:
        g: Posterior probabilities with shape (k, n).
        l: Total log-likelihood.
        (None, None) if the inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if n < 1 or d < 1 or k < 1:
        return None, None

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        p = pdf(X, m[i], S[i])

        if p is None:
            return None, None

        g[i] = pi[i] * p

    total = np.sum(g, axis=0)

    if np.any(total <= 0):
        return None, None

    l = np.sum(np.log(total))
    g = g / total

    return g, l
