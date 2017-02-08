import numpy as np


def poly_features(x, p):
    """
    Maps X (1D vector) into the p-th power.

    Parameters
    ----------
    x : ndarray
        Features to be mapped to polynomial ones.
    p : int
        Power of polynomial features.

    Returns
    -------
    ndarray
        Polynomial features.
    """
    x_poly = np.zeros((x.size, p))

    for i in range(p):
        x_poly[:, i] = np.power(x, i + 1).flatten()

    return x_poly
