
#!/usr/bin/env python3
"""
This module contains a function that calculates
the expectation step in the EM algorithm for a GMM.
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step for a Gaussian Mixture Model.

    X: numpy.ndarray (n, d) containing the dataset
        - n is the number of data points
        - d is the number of dimensions
    pi: numpy.ndarray (k,) containing cluster priors
    m: numpy.ndarray (k, d) containing centroid means
    S: numpy.ndarray (k, d, d) containing covariance matrices

    Returns:
        g: numpy.ndarray (k, n) containing posterior probabilities
        l: total log-likelihood of the model
        (None, None) on failure
    """
    if (not isinstance(X, np.ndarray)
            or not isinstance(pi, np.ndarray)
            or not isinstance(m, np.ndarray)
            or not isinstance(S, np.ndarray)):
        return None, None

    if (len(X.shape) != 2
            or len(S.shape) != 3
            or len(pi.shape) != 1
            or len(m.shape) != 2):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (n < 1 or d < 1 or k < 1
            or m.shape != (k, d)
            or S.shape != (k, d, d)
            or np.any(pi < 0)
            or not np.isclose(np.sum(pi), 1)):
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
