from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from scipy._lib.six import callable
from collections import namedtuple

BinnedReduceResult = namedtuple('BinnedReduceResult',
                                ('reduction', 'bin_edges', 'binnumber'))


def binned_reduce(x, values, function=None, bins=10, range=None):
    """
    Compute a binned reduction for a set of data.
    This is a generalization of a histogram function. A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin. This function allows the application of an arbitrary function
    (which returns a scalar) to multiple data arrays, in bins. This mostly
    functions like scipy.stats.binned_statistic, with changes to values
    (passed as a tuple of arrays) and function (which must be provided
    explicitly and take a number of arguments equal to the number of
    arrays in values).
    Parameters
    ----------
    x : array_like
        A sequence of values to be binned.
    values : tuple of array_like
        The values on which the reduction will be computed.  This must be
        a tuple of array(s) with the same shapes as `x`.
    function : callable
        The function to apply to the values; the arguments will be passed in
        the order in values. For empty bins, function is called on a set of
        empty arrays, if this raises an error, the result is np.nan.
    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10 by default).  If `bins` is a sequence, it defines the
        bin edges, including the rightmost edge, allowing for non-uniform bin
        widths.  Values in `x` that are smaller than lowest bin edge are
        assigned to bin number 0, values beyond the highest bin are assigned to
        ``bins[-1]``.
    range : (float, float) or [(float, float)], optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(x.min(), x.max())``.  Values outside the range are
        ignored.
    Returns
    -------
    reduction : array
        The values of the selected statistic in each bin.
    bin_edges : array of dtype float
        Return the bin edges ``(length(statistic)+1)``.
    binnumber : 1-D ndarray of ints
        This assigns to each observation an integer that represents the bin
        in which this observation falls. Array has the same length as values.
    See Also
    --------
    numpy.histogram, scipy.stats.binned_statistic,
    binned_reduction, binned_reduction_2d, binned_reduction_dd

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
    ``[3, 4]``, which *includes* 4.

    """

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1:
        bins = [np.asarray(bins, float)]

    if range is not None:
        if len(range) == 2:
            range = [range]

    reduction, edges, xy = binned_reduce_dd([x], values, function, bins, range)

    return BinnedReduceResult(reduction, edges[0], xy)


BinnedReduce2dResult = namedtuple('BinnedReduce2dResult',
                                  ('reduction', 'x_edge', 'y_edge',
                                   'binnumber'))


def binned_reduce_2d(x, y, values, function=None, bins=10, range=None):
    """
    Compute a bidimensional binned reduction for a set of data.
    This is a generalization of a histogram2d function. A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin. This function allows the application of an arbitrary function
    (which returns a scalar) to multiple data arrays, in bins. This mostly
    functions like scipy.stats.binned_statistic_2d, with changes to values
    (passed as a tuple of arrays) and function (which must be provided
    explicitly and take a number of arguments equal to the number of
    arrays in values).
    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned along the first dimension.
    y : (N,) array_like
        A sequence of values to be binned along the second dimension.
    values : tuple of (N,) array_like
        The values on which the reduction will be computed. This must be
        a tuple of array(s) with the same shapes as `x`.
    function : callable
        The function to apply to the values; the arguments will be passed in
        the order in values. For empty bins, function is called on a set of
        empty arrays, if this raises an error, the result is np.nan.
    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:
          * the number of bins for the two dimensions (nx=ny=bins),
          * the number of bins in each dimension (nx, ny = bins),
          * the bin edges for the two dimensions (x_edges = y_edges = bins),
          * the bin edges in each dimension (x_edges, y_edges = bins).
    range : (2,2) array_like, optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
        considered outliers and not tallied in the histogram.
    Returns
    -------
    reduction : (nx, ny) ndarray
        The values returned by the passed function in each two-dimensional bin.
    x_edges : (nx + 1) ndarray
        The bin edges along the first dimension.
    y_edges : (ny + 1) ndarray
        The bin edges along the second dimension.
    binnumber : 1-D ndarray of ints
        This assigns to each observation an integer that represents the bin
        in which this observation falls. Array has the same length as `values`.
    See Also
    --------
    numpy.histogram2d, scipy.stats.binned_statistic_2d,
    binned_reduction, binned_reduction_dd
    """

    # This code is based on np.histogram2d
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        xedges = yedges = np.asarray(bins, float)
        bins = [xedges, yedges]

    reduction, edges, xy = binned_reduce_dd([x, y], values, function, bins,
                                            range)

    return BinnedReduce2dResult(reduction, edges[0], edges[1], xy)


BinnedReduceddResult = namedtuple('BinnedReduceddResult',
                                  ('reduction', 'bin_edges', 'binnumber'))


def binned_reduce_dd(sample, values, function=None, bins=10, range=None):
    """
    Compute a multidimensional binned reduction for a set of data.
    This is a generalization of a histogramdd function. A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin. This function allows the application of an arbitrary function
    (which returns a scalar) to multiple data arrays, in bins. This mostly
    functions like scipy.stats.binned_statistic_dd, with changes to values
    (passed as a tuple of arrays) and function (which must be provided
    explicitly and take a number of arguments equal to the number of
    arrays in values).

    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of D arrays of length N, or
        as an (N,D) array.
    values : tuple of array_like
        The values on which the statistic will be computed. This must be a
        tuple, each entry must be an array of shape (N,).
    function : callable
        The function to apply to the values; the arguments will be passed in
        the order in values. For empty bins, function is called on a set of
        empty arrays, if this raises an error, the result is np.nan.
    bins : sequence or int, optional
        The bin specification:
          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (nx, ny, ... =bins)
          * The number of bins for all dimensions (nx=ny=...=bins).
    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitely in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    Returns
    -------
    reduction : ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin
    bin_edges : list of ndarrays
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension
    binnumber : 1-D ndarray of ints
        This assigns to each observation an integer that represents the bin
        in which this observation falls. Array has the same length as values.
    See Also
    --------
    np.histogramdd, scipy.stats.binned_statistic_dd,
    binned_reduction, binned_reduction_2d
    """

    if not callable(function):
        raise ValueError('function not callable')

    # This code is based on np.histogramdd
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = D * [bins]

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(0), float))
        smax = np.atleast_1d(np.array(sample.max(0), float))
    else:
        smin = np.zeros(D)
        smax = np.zeros(D)
        for i in np.arange(D):
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in np.arange(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in np.arange(D):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
        else:
            edges[i] = np.asarray(bins[i], float)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    # Compute the bin number each sample falls into.
    Ncount = {}
    for i in np.arange(D):
        Ncount[i] = np.digitize(sample[:, i], edges[i])

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in np.arange(D):
        # Rounding precision
        decimal = int(-np.log10(dedges[i].min())) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where(np.around(sample[:, i], decimal)
                           == np.around(edges[i][-1], decimal))[0]
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened statistic matrix.
    ni = nbin.argsort()
    xy = np.zeros(N, int)
    for i in np.arange(0, D - 1):
        xy += Ncount[ni[i]] * nbin[ni[i + 1:]].prod()
    xy += Ncount[ni[-1]]

    result = np.empty(nbin.prod(), float)

    with warnings.catch_warnings():
        # Numpy generates a warnings for mean/std/... with empty list
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        old = np.seterr(invalid='ignore')
        try:
            empty = tuple([[]] * len(values))
            null = function(empty)
        except (ValueError, RuntimeError):
            null = np.nan
        np.seterr(**old)
    result.fill(null)
    for i in np.unique(xy):
        result[i] = function(*tuple([v[xy == i] for v in values]))

    # Shape into a proper matrix
    result = result.reshape(np.sort(nbin))
    for i in np.arange(nbin.size):
        j = ni.argsort()[i]
        result = result.swapaxes(i, j)
        ni[i], ni[j] = ni[j], ni[i]

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D * [slice(1, -1)]
    result = result[core]

    if (result.shape != nbin - 2).any():
        raise RuntimeError('Internal Shape Error')

    return BinnedReduceddResult(result, edges, xy)
