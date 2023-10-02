import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Callable, Union


def compute_quantile_robust(values:np.ndarray, cdf:np.ndarray, alpha:Union[str, float]='all', isotonic:bool=True, rescaling:bool=True):

    # correction of the cdf using isotonic regression
    if isotonic:
        for i in range(cdf.shape[0]):
            cdf[i] = IsotonicRegression(y_min=0, y_max=cdf[i].max()).fit_transform(range(cdf.shape[1]), cdf[i])
    if rescaling:
        max_cdf = np.outer(cdf.max(axis=-1), np.ones(cdf.shape[1]))
        min_cdf = np.outer(cdf.min(axis=-1), np.ones(cdf.shape[1]))
        cdf = (cdf - min_cdf)/(max_cdf - min_cdf)

    # if alpha = all, return the entire cdf
    if alpha=='all':
        return values, cdf

    # otherwise, search for the quantile at level alpha
    quantiles = np.zeros(cdf.shape[0])
    for j in range(cdf.shape[0]):
        for i, level in enumerate(cdf[j]):
            if level >= alpha:
                if i == 0:
                    quantiles[j] = -np.inf
                quantiles[j] = values[i-1]
                break
            
        # special case where we exceeded the maximum observed value
        if i == cdf.shape[0] - 1:
            quantiles[j] = np.inf

    return quantiles