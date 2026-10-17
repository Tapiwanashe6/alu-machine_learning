#!/usr/bin/env python3
"""Find the best number of GMM clusters using the BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Select the best GMM cluster count using Bayesian Information Criterion.

    Args:
        X: Dataset with shape (n, d).
        kmin: Minimum number of clusters to test.
        kmax: Maximum number of clusters to test, or None.
        iterations: Maximum EM iterations.
        tol: Non-negative convergence tolerance.
        verbose: Whether EM should print its progress.

    Returns:
        Tuple of best cluster count, best (pi, m, S), log-likelihood
        array, and BIC array; four None values on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape
    if n < 1 or d < 1:
        return None, None, None, None

    if not isinstance(kmin, int) or isinstance(kmin, bool) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or isinstance(kmax, bool)
            or kmax < kmin or kmax > n):
        return None, None, None, None

    if (not isinstance(iterations, int) or isinstance(iterations, bool)
            or iterations < 1):
        return None, None, None, None

    if not isinstance(tol, float) or not np.isfinite(tol) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    likelihoods = []
    bics = []
    best_k = None
    best_result = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if (pi is None or m is None or S is None
                or log_likelihood is None
                or not np.isfinite(log_likelihood)):
            return None, None, None, None

        parameters = k - 1 + k * d + k * d * (d + 1) / 2
        bic = parameters * np.log(n) - 2 * log_likelihood

        likelihoods.append(log_likelihood)
        bics.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(likelihoods), np.array(bics)
