import numpy as np
from scipy.stats.morestats import Anderson_ksampResult
import math
import warnings


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


def binomial_CI(f, N, CL=None, S=None, twosided=True):
    """
    Determines the 1-sided or 2-sided binomial confidence limits.

    Determines the 1-sided or 2-sided binomial confidence limits for
    confidence level given by CL (default is for gaussian 1-sigma).
    f is a measured fraction, N is the total count; since these arise from
    finite counts the product should be an integer! This CL is useful for
    cases when an error on a fraction from finite counts is needed. For
    instance, 100 measurements, 55 are positive, then f=.55, N=100, and the
    result is .55(-.055)(+.054) for a 1-sigma two-sided CL. All numerical
    arguments may be arrays and will use normal numpy array broadcasting.

    Parameters
    ----------
    f : float
        Fraction of events which are of type 1.
    N : int
        Total event count of types 1 and 2.
    CL : float
        Confidence level, e.g. .95 for 95%. Provide CL or S, not both.
    S : float
        Confidence level, in number of Gaussian sigmas, e.g. S=1 is .68 (two-
        sided) or .84 (1-sided). Provide CL or S, not both.
    twosided : bool
        If True, calculate a confidence interval. If False, calculate upper
        and lower limits instead (usually use one or the other as appropriate
        in this case).

    Returns
    -------
    out : tuple
        A 2-tuple containing the (lower, upper) bounds of the confidence
        interval, or the (lower, upper) limits.

    Notes
    -----
    See NASA ADS reference 1986ApJ...303..336G.
    """

    from scipy.special import erf, erfinv

    if ((CL is not None) and (S is not None)) or \
       ((CL is None) and (S is None)):
        raise ValueError('Provide CL or S (and not both).')

    if CL is not None:
        if twosided:
            CL = (1 + CL) / 2  # get equivalent one-sided value
        S = np.abs(np.sqrt(2) * erfinv(2 * CL - 1))
    else:  # S is not None
        CL = erf(S / np.sqrt(2)) / 2 + .5  # one-sided

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


def _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N, weights, W, Wstar):
    """Compute A2akN equation 7 of Scholz and Stephens.
    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.
    weights : sequence of 1-D array_like
        Array of sample weight arrays.
    W : array_like
        Array of all weights sorted as all samples.
    Wstar: array_like
        Array of weights associated to unique observations, sorted as unique
        samples.
    Returns
    -------
    A2aKN : float
        The A2aKN statistics of Scholz and Stephens 1987,
        modified to allow sample weights.
    """
    from itertools import combinations
    for s1, s2 in combinations(samples, 2):
        if np.in1d(s1, s2).any():
            raise NotImplementedError(
                'Common elements across samples not supported.'
            )
    A2akN = 0.
    # Z_ssorted_left = Z.searchsorted(Zstar, 'left')
    if N == Zstar.size:
        lj = W
        Bj = .5 * (np.r_[0, np.cumsum(W)[:-1]] + np.cumsum(W))
    else:
        lj = Wstar
        Bj = .5 * (np.r_[0, np.cumsum(Wstar)[:-1]] + np.cumsum(Wstar))
    for i in np.arange(0, k):
        # s = np.sort(samples[i])
        ssort = np.argsort(samples[i])
        s = samples[i][ssort]
        # w = weights[i][ssort]
        s_ssorted_right = s.searchsorted(Zstar, side='right')
        s_ssorted_left = s.searchsorted(Zstar, side='left')
        Mij = np.r_[0, np.cumsum(Wstar)][s_ssorted_right]
        fij = np.cumsum(Wstar)[s_ssorted_right] \
            - np.cumsum(Wstar)[s_ssorted_left]
        Mij -= fij / 2.
        inner = lj / float(N) * (N*Mij - Bj*n[i])**2 / (Bj*(N - Bj) - N*lj/4.)
        A2akN += inner.sum() / n[i]
    A2akN *= (N - 1.) / N
    return A2akN


def _anderson_ksamp_right(samples, Z, Zstar, k, n, N, weights, W, Wstar):
    """Compute A2akN equation 6 of Scholz & Stephens.
    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.
    Returns
    -------
    A2KN : float
        The A2KN statistics of Scholz and Stephens 1987.
    """
    raise NotImplementedError(
        '_anderson_ksamp_right not updated for weights'
    )

    A2kN = 0.
    lj = Z.searchsorted(Zstar[:-1], 'right') - Z.searchsorted(Zstar[:-1],
                                                              'left')
    Bj = lj.cumsum()
    for i in np.arange(0, k):
        s = np.sort(samples[i])
        Mij = s.searchsorted(Zstar[:-1], side='right')
        inner = lj / float(N) * (N * Mij - Bj * n[i])**2 / (Bj * (N - Bj))
        A2kN += inner.sum() / n[i]
    return A2kN


def anderson_ksamp_weighted(samples, weights, midrank=True):
    """The Anderson-Darling test for k-samples.
    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples. This version is
    adapted to allow sample weights. The implementation is somewhat
    incomplete, as evidenced by the NotImmplementedErrors.
    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample data in arrays.
    weights : sequence of 1-D array_like
        Array of sample weight arrays.
    midrank : bool, optional
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.
    Returns
    -------
    statistic : float
        Normalized k-sample Anderson-Darling test statistic.
    critical_values : array
        The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
        0.5%, 0.1%.
    significance_level : float
        An approximate significance level at which the null hypothesis for the
        provided samples can be rejected. The value is floored / capped at
        0.1% / 25%.
    Raises
    ------
    ValueError
        If less than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.
    See Also
    --------
    ks_2samp : 2 sample Kolmogorov-Smirnov test
    anderson : 1 sample Anderson-Darling test
    Notes
    -----
    [1]_ defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.
    The critical values corresponding to the significance levels from 0.01
    to 0.25 are taken from [1]_. p-values are floored / capped
    at 0.1% / 25%. Since the range of critical values might be extended in
    future releases, it is recommended not to test ``p == 0.25``, but rather
    ``p >= 0.25`` (analogously for the lower bound).
    .. versionadded:: 0.14.0
    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.
    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    The null hypothesis that the two random samples come from the same
    distribution can be rejected at the 5% level because the returned
    test value is greater than the critical value for 5% (1.961) but
    not at the 2.5% level. The interpolation gives an approximate
    significance level of 3.2%:
    >>> stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(loc=0.5, size=30)])
    (1.974403288713695,
      array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546]),
      0.04991293614572478)
    The null hypothesis cannot be rejected for three samples from an
    identical distribution. The reported p-value (25%) has been capped and
    may not be very accurate (since it corresponds to the value 0.449
    whereas the statistic is -0.731):
    >>> stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(size=30), rng.normal(size=20)])
    (-0.29103725200789504,
      array([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856,
      4.07210043, 5.56419101]),
      0.25)
    """
    k = len(samples)
    if (k < 2):
        raise ValueError("anderson_ksamp needs at least two samples")

    samples = list(map(np.asarray, samples))
    weights = list(map(np.asarray, weights))
    weights = [w / np.sum(w) * len(w) for w in weights]
    Zsort = np.argsort(np.hstack(samples))
    Z = np.hstack(samples)[Zsort]
    W = np.hstack(weights)[Zsort]
    N = Z.size
    Zstar, unique_inv = np.unique(Z, return_inverse=True)
    Wstar = np.bincount(unique_inv, W.reshape(-1))
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct "
                         "observation")

    n = np.array([sample.size for sample in samples])
    if np.any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without "
                         "observations")

    if midrank:
        A2kN = _anderson_ksamp_midrank(
            samples, Z, Zstar, k, n, N, weights, W, Wstar)
    else:
        A2kN = _anderson_ksamp_right(
            samples, Z, Zstar, k, n, N, weights, W, Wstar)

    H = (1. / n).sum()
    hs_cs = (1. / np.arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / np.arange(2, N)).sum()

    a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k**2 - 4*h*k
    sigmasq = (a*N**3 + b*N**2 + c*N + d) / ((N - 1.) * (N - 2.) * (N - 3.))
    m = k - 1
    A2 = (A2kN - m) / math.sqrt(sigmasq)

    # The b_i values are the interpolation coefficients from Table 2
    # of Scholz and Stephens 1987
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    critical = b0 + b1 / math.sqrt(m) + b2 / m

    sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    if A2 < critical.min():
        p = sig.max()
        warnings.warn("p-value capped: true value larger than {}".format(p),
                      stacklevel=2)
    elif A2 > critical.max():
        p = sig.min()
        warnings.warn("p-value floored: true value smaller than {}".format(p),
                      stacklevel=2)
    else:
        # interpolation of probit of significance level
        pf = np.polyfit(critical, np.log(sig), 2)
        p = math.exp(np.polyval(pf, A2))

    return Anderson_ksampResult(A2, critical, p)
