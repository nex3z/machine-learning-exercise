import numpy as np


def feature_normalize(x):
    """
    Normalizes the features in x.

    Parameters
    ----------
    x : ndarray
        Features to be normalized.

    Returns
    -------
    x_norm : ndarray
        A normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    mu : ndarray
        The mean value.
    sigma : ndarray
        The standard deviation.
    """
    m = x.shape[0]
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    x_norm = (x - np.ones((m, 1)) * mu) / (np.ones((m, 1)) * sigma)

    return x_norm, mu, sigma
