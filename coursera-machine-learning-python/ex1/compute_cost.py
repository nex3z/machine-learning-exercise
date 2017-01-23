import numpy as np


def compute_cost(x, y, theta):
    """
    Compute cost for linear regression.
    :param x: training data
    :param y: labels
    :param theta: linear regression parameter
    :return: cost of using mat_theta as the parameter for linear regression to fit the data points in mat_x and mat_y
    """
    m = len(y)
    j = np.sum(np.square(np.dot(x, theta) - y)) / (2 * m)

    return j
