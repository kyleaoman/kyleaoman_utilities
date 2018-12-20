import numpy as np

# Could be generalized parallel to np.quantile (and np.percentile)
# however this requires a lot of attention to broadcasting of weight,
# quantile array support, etc. A job for later, if needed.


def weighted_median(a, weights=None, axis=None):
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
    if weights is None:
        return np.nanmedian(a, axis=axis)
    elif axis is not None:
        raise NotImplementedError("'axis' kwarg not implemented.")
    else:
        a = a.flatten()
        weights = weights.flatten()
        mask = np.logical_not(np.logical_or(np.isnan(a), np.isnan(weights)))
        return weighted_median(a[mask], weights=weights[mask], axis=axis)
