import numpy as np


def linear_reg_cost_function(theta, x, y, l):
    """
    Compute cost and gradient for regularized linear regression with multiple variables.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter.
    x : ndarray
        Training data.
    y : ndarray
        Labels.
    l : float
        Regularization parameter.

    Returns
    -------
    j : numpy.float64
        The cost of using theta as the parameter for linear regression.
    grad : ndarray
        The gradient of using theta as the parameter for linear regression.
    """
    m, n = x.shape
    theta = theta.reshape(n, 1)

    j = 1.0 / (2 * m) * np.sum(np.square(x.dot(theta) - y)) + 1.0 * l / (2 * m) * np.sum(np.square(theta[1:, ]))

    mask = np.eye(len(theta))
    mask[0, 0] = 0
    grad = 1.0 / m * x.T.dot(x.dot(theta) - y) + 1.0 * l / m * (mask.dot(theta))

    return j, grad
