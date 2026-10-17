#!/usr/bin/env python3
"""Perform expectation maximization for a Gaussian Mixture Model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Fit a Gaussian Mixture Model using expectation maximization.

    Args:
        X: Dataset of shape (n, d).
        k: Positive number of clusters.
        iterations: Maximum number of EM iterations.
        tol: Non-negative log-likelihood convergence tolerance.
        verbose: Whether to print log-likelihood progress.

    Returns:
        Cluster priors, means, covariances, responsibilities, and
        log-likelihood; five None values on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    n, d = X.shape
    if n < 1 or d < 1 or not np.all(np.isfinite(X)):
        return None, None, None, None, None
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        return None, None, None, None, None
    if (not isinstance(iterations, int) or isinstance(iterations, bool)
            or iterations < 1):
        return None, None, None, None, None
    if not isinstance(tol, float) or not np.isfinite(tol) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    message = "Log Likelihood after {} iterations: {}"
    if verbose:
        print(message.format(0, np.round(log_likelihood, 5)))

    for i in range(iterations):
        previous = log_likelihood
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        if verbose and (i + 1) % 10 == 0:
            print(message.format(i + 1, np.round(log_likelihood, 5)))

        if abs(log_likelihood - previous) <= tol:
            break

    if verbose and (i + 1) % 10 != 0:
        print(message.format(i + 1, np.round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
