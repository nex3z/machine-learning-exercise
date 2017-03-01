import numpy as np


def compute_cost_multi(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Linear regression parameter.

    Returns
    -------
    j : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in X and y.
    """
    m = len(y)
    tmp = X.dot(theta) - y
    j = 1.0 / (2 * m) * tmp.T.dot(tmp)
    return j
