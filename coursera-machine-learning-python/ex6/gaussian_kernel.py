import numpy as np


def gaussian_kernel(x1, x2, sigma):
    """
    Returns the similarity between x1 and x2 using a Gaussian kernel.

    Parameters
    ----------
    x1 : ndarray
        Sample 1.
    x2 : ndarray
        Sample 2.
    sigma : float
        Bandwidth for Gaussian kernel.

    Returns
    -------
    float
        The similarity between x1 and x2 with bandwidth sigma.
    """
    return np.exp(-np.sum(np.square(x1 - x2)) / (2.0 * sigma ** 2))
