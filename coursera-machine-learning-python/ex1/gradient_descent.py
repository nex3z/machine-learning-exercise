import numpy as np

from compute_cost import compute_cost


def gradient_descent(mat_x, mat_y, mat_theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    :param mat_x: training data
    :param mat_y: labels
    :param mat_theta: initial linear regression parameter
    :param alpha: learning rate
    :param num_iters: number of iteration
    :return: linear regression parameter and cost history
    """
    m = len(mat_y)
    j_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        mat_theta -= alpha / m * ((mat_x * mat_theta - mat_y).T * mat_x).T
        j_history[i] = compute_cost(mat_x, mat_y, mat_theta)

    return [mat_theta, j_history]