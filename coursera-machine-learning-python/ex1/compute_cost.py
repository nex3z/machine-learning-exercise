import numpy as np


def compute_cost(x, y, theta):
    """
    Compute cost for linear regression.

    Parameters
    ----------
    x : ndarray
        Training data. It's a n by m matrix, where n is the number of data samples and m is the number of features.
    y : ndarray
        Labels, n by 1 matrix.
    theta : ndarray
        Linear regression parameter, n by 1 matrix.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in x and y.
    """
    m = len(y)
    j = np.sum(np.square(x.dot(theta) - y)) / (2 * m)

    return j
