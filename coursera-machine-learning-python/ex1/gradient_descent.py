import numpy as np

from compute_cost import compute_cost


def gradient_descent(x, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    :param x: training data
    :param y: labels
    :param theta: initial linear regression parameter
    :param alpha: learning rate
    :param num_iters: number of iteration
    :return: linear regression parameter and cost history
    """
    m = len(y)
    j_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta -= alpha / m * (np.dot((np.dot(x, theta) - y).T, x)).T
        j_history[i] = compute_cost(x, y, theta)

    return [theta, j_history]