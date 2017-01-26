import numpy as np


def compute_cost_multi(x, y, theta):
    """
    Compute cost for linear regression with multiple variables.

    Parameters
    ----------
    x : ndarray
        Training data. It's a m by n matrix, where m is the number of data samples and n is the number of features.
    y : ndarray
        Labels, m by 1 matrix.
    theta : ndarray
        Linear regression parameter, n by 1 matrix.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in x and y.
    """
    m = len(y)
    tmp = x.dot(theta) - y
    j = 1.0 / (2 * m) * tmp.T.dot(tmp)
    return j.flatten()[0]
