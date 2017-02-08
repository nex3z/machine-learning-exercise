import numpy as np


def feature_normalize(x, mu=None, sigma=None):
    """
    Normalizes the features in x.

    Parameters
    ----------
    x : ndarray
        Features to be normalized.
    mu : float
        Mean value for normalization. If not provided, it will be calculated from x.
    sigma : float
        Standard deviation for normalization. If not provided, it will be calculated from x.

    Returns
    -------
    x_norm : ndarray
        The normalized features.
    mu : float
        Mean value of x.
    sigma : float
        Standard deviation of x.
    """
    if mu is None:
        mu = np.mean(x, axis=0)

    if sigma is None:
        sigma = np.std(x, ddof=1, axis=0)

    x_norm = (x - mu) / sigma

    return x_norm, mu, sigma
