import numpy as np

from sigmoid import sigmoid


def cost_function(theta, x, y):
    """
    Compute cost and gradient for logistic regression.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in x and y.
    grad: ndarray
        The gradient of the cost w.r.t. the parameters, n by 1 matrix.
    """
    m, n = x.shape
    theta = theta.reshape(n, 1)
    x_dot_theta = x.dot(theta)

    j = 1.0 / m * (np.dot(-y.T, np.log(sigmoid(x_dot_theta))) - np.dot((1 - y).T, np.log(1 - sigmoid(x_dot_theta))))
    j = j.flatten()

    grad = 1.0 / m * np.dot((sigmoid(x_dot_theta) - y).T, x).T

    return j[0], grad
