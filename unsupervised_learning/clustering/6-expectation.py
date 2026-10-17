
#!/usr/bin/env python3
"""Calculates the expectation step in the EM algorithm for a GMM."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        pi: numpy.ndarray of shape (k,) containing cluster priors.
        m: numpy.ndarray of shape (k, d) containing cluster means.
        S: numpy.ndarray of shape (k, d, d) containing covariances.

    Returns:
        tuple: Posterior probabilities of shape (k, n) and the
        total log-likelihood, or (None, None) on failure.
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

    if not np.all(np.isfinite(X)):
        return None, None

    if not np.all(np.isfinite(pi)):
        return None, None

    if not np.all(np.isfinite(m)):
        return None, None

    if not np.all(np.isfinite(S)):
        return None, None

    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        p = pdf(X, m[i], S[i])

        if p is None or not isinstance(p, np.ndarray):
            return None, None

        if p.shape != (n,):
            return None, None

        g[i] = pi[i] * p

    total = np.sum(g, axis=0)

    if np.any(total <= 0):
        return None, None

    if not np.all(np.isfinite(total)):
        return None, None

    log_likelihood = np.sum(np.log(total))
    g = g / total

    return g, log_likelihood
