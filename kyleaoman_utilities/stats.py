import numpy as np


def weighted_median(a, weights=None, axis=None):
    # Could be generalized parallel to np.quantile (and np.percentile)
    # however this requires a lot of attention to broadcasting of weight,
    # quantile array support, etc. A job for later, if needed.

    if weights is None:
        return np.median(a, axis=axis)
    elif axis is not None:
        raise NotImplementedError("'axis' kwarg not implemented.")
    elif a.size == 0:
        return np.nan
    else:
        a = a.flatten()
        weights = weights.flatten()
        isort = np.argsort(a)
        a = a[isort]
        weights = np.cumsum(weights[isort]) / np.sum(weights)
        return np.interp(.5, weights, a)


def weighted_nanmedian(a, weights=None, axis=None):
    # Could be generalized parallel to np.quantile (and np.percentile)
    # however this requires a lot of attention to broadcasting of weight,
    # quantile array support, etc. A job for later, if needed.

    if weights is None:
        return np.nanmedian(a, axis=axis)
    elif axis is not None:
        raise NotImplementedError("'axis' kwarg not implemented.")
    else:
        a = a.flatten()
        weights = weights.flatten()
        mask = np.logical_not(np.logical_or(np.isnan(a), np.isnan(weights)))
        return weighted_median(a[mask], weights=weights[mask], axis=axis)


def binomial_CI(f, N, CL=0.8413, S=1):
    """
    Determines the 1-sided binomial confidence limit for confidence
    level given by CL (default is for gaussian 1-sigma). S is the
    number of Gaussian sigma's corresponding to CL (could calculate this
    internally, but laziness prevails).
    f is a measured fraction, N is the total count. This CL is useful for
    cases when I want an error on a fraction from finite counts. For instance,
    100 measurements, 55 are positive, then f=.55, N=100, and the result is
    .55(-.055)(+.054) for a 1-sigma CL.

    See 1986ApJ...303..336G.
    """

    scl = all([not hasattr(arg, '__iter__') for arg in (f, N, CL, S)])

    def _ns(f, N):
        n1 = f * N
        n2 = N - n1
        return n1, n2

    n1, n2 = np.vectorize(_ns)(f, N)

    def _p1u(n1, n2, CL, S):
        if (n1 == 0) or (n2 == 1):
            if n1 == 0:
                return 1 - np.power(1 - CL, 1 / n2)
            if n2 == 1:
                return np.power(CL, 1 / (n1 + n2))
        else:
            eps = 0.64 * (1 - S) * np.exp(-n2)
            lam = (np.power(S, 2) - 3) / 6
            h = 2 * np.power(1 / (2 * n2 - 1) + 1 / (2 * n1 + 1), -1)
            w = S * np.sqrt(h + lam) / h \
                + (1 / (2 * n2 - 1) - 1 / (2 * n1 + 1)) \
                * (lam + 5 / 6 - 2 / (3 * h))
            return ((n1 + 1) * np.exp(2 * w) + eps * n2) \
                / ((n1 + 1) * np.exp(2 * w) + n2)

    p1u = np.vectorize(_p1u)(n1, n2, CL, S)
    p1l = 1 - np.vectorize(_p1u)(n2, n1, CL, S)
    if isinstance(p1u, np.ndarray):
        p1u[np.isnan(p1u)] = 1
    elif np.isnan(p1l):
        p1u = np.array(1)
    if isinstance(p1l, np.ndarray):
        p1l[np.isnan(p1l)] = 0
    elif np.isnan(p1l):
        p1l = np.array(0)

    if scl:
        return float(p1l), float(p1u)
    else:
        return p1l, p1u
