import numpy as np


def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features. Inputs X1, X2 must be the same size.

    Parameters
    ----------
    X1 : ndarray, shape (n_samples,)
        Input feature.
    X2 : ndarray, shape (n_samples,)
        Input feature.

    Returns
    -------
    Out : ndarray, shape (n_samples, 28)
        A new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    degree = 6
    Out = np.ones(len(X1))

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            tmp = np.power(X1, i - j) * np.power(X2, j)
            Out = np.vstack((Out, tmp))
    return Out.T
