import numpy as np


def normal_eqn(x, y):
    """
    Computes the closed-form solution to linear regression.

    Parameters
    ----------
    x : ndarray
        Training data. It's a m by n matrix, where m is the number of data samples and n is the number of features.
    y : ndarray
        Labels, m by 1 matrix.

    Returns
    -------
    theta : ndarray
        The closed-form solution to linear regression using the normal equations, which is an n by 1 matrix.
    """
    theta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    return theta
