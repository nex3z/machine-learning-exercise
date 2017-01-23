import numpy as np


def compute_cost(mat_x, mat_y, mat_theta):
    """
    Compute cost for linear regression.
    :param mat_x: training data
    :param mat_y: labels
    :param mat_theta: linear regression parameter
    :return: cost of using mat_theta as the parameter for linear regression to fit the data points in mat_x and mat_y
    """
    m = len(mat_y)
    j = np.sum(np.square(mat_x * mat_theta - mat_y)) / (2 * m)

    return j
