import numpy as np

from sigmoid import sigmoid


def cost_function_reg(theta, x, y, lam):
    """
    Compute cost and gradient for logistic regression
    :param theta: logistic regression parameter
    :param x: training data
    :param y: labels
    :return:
    """
    (m, n) = x.shape

    theta = theta.reshape(n, 1)
    tmp = np.dot(x, theta)

    mask = np.eye(theta.shape[0])
    mask[0, 0] = 0

    j = 1.0 / m * (np.dot(-y.T, np.log(sigmoid(tmp))) - np.dot((1 - y).T, np.log(1 - sigmoid(tmp)))) \
        + lam / (2 * m) * np.sum(np.power((mask.dot(theta)), 2))
    grad = 1.0 / m * np.dot((sigmoid(tmp) - y).T, x).T + lam / m * (mask.dot(theta))

    return j, grad
