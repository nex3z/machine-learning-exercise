import numpy as np

from sigmoid import sigmoid


def cost_function_reg(theta, x, y, l):
    """
    Compute cost and gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.
    l : float
        Regularization parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for regularized logistic regression w.r.t. the parameters.
    grad: ndarray
        Partial derivatives of the cost w.r.t. each parameter in theta, n by 1 matrix.
    """
    (m, n) = x.shape
    theta = theta.reshape(n, 1)

    x_dot_theta = x.dot(theta)
    mask = np.eye(theta.shape[0])
    # Skip the theta[0, 0] parameter when performing regularization
    mask[0, 0] = 0

    j = 1.0 / m * (np.dot(-y.T, np.log(sigmoid(x_dot_theta))) - np.dot((1 - y).T, np.log(1 - sigmoid(x_dot_theta)))) \
        + 1.0 * l / (2 * m) * np.sum(np.power((mask.dot(theta)), 2))
    grad = 1.0 / m * np.dot((sigmoid(x_dot_theta) - y).T, x).T + 1.0 * l / m * (mask.dot(theta))

    return j.ravel()[0], grad
