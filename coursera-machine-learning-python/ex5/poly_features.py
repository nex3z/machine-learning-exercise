import numpy as np


def poly_features(X, p):
    """
    Maps X (1D vector) into the p-th power.

    Parameters
    ----------
    X : ndarray, shape (n_samples, 1)
        Features to be mapped to polynomial ones.
    p : int
        Power of polynomial features.

    Returns
    -------
    ndarray, shape (n_samples, p)
        Polynomial features.
    """
    X_poly = np.zeros((len(X), p))

    for i in range(p):
        X_poly[:, i] = np.power(X, i + 1).ravel()

    return X_poly
