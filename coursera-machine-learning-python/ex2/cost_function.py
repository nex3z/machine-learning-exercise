import numpy as np

from sigmoid import sigmoid


def cost_function(theta, x, y):
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

    j = 1.0 / m * (np.dot(-y.T, np.log(sigmoid(tmp))) - np.dot((1 - y).T, np.log(1 - sigmoid(tmp))))
    grad = 1.0 / m * np.dot((sigmoid(tmp) - y).T, x).T

    return j, grad
