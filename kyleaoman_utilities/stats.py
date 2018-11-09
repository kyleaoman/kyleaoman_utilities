import numpy as np

# Could be generalized parallel to np.quantile (and np.percentile)
# however this requires a lot of attention to broadcasting of weight,
# quantile array support, etc. A job for later, if needed.


def weighted_median(a, weights=None, axis=None):
    if weights is None:
        return np.median(a, axis=axis)
    elif axis is not None:
        raise NotImplementedError("'axis' kwarg not implemented.")
    else:
        a = a.flatten()
        weights = weights.flatten()
        isort = np.argsort(a)
        a = a[isort]
        weights = weights[isort]
        invweights = np.cumsum(weights[::-1])[::-1] / np.sum(weights)
        weights = np.cumsum(weights[isort]) / np.sum(weights)
        if np.logical_or((weights == 0.5).any(), (invweights == 0.5).any()):
            lwm = a[np.logical_and(weights >= .5, invweights == .5)][0]
            uwm = a[np.logical_and(weights == .5, invweights >= .5)][-1]
            return .5 * (lwm + uwm)
        else:
            return a[np.logical_and(weights >= .5, invweights >= .5)]


def weighted_nanmedian(a, weights=None, axis=None):
    if weights is None:
        return np.nanmedian(a, **kwargs)
    elif axis is not None:
        raise NotImplementedError("'axis' kwarg not implemented.")
    else:
        a = a.flatten()
        weights = weights.flatten()
        mask = np.logical_not(np.logical_or(np.isnan(a), np.isnan(weights)))
        return weighted_median(a[mask], weights[mask], **kwargs)
