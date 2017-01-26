import numpy as np


def sigmoid(z):
    """
    Compute sigmoid function.
    :param z: variable for sigmoid function
    :return: the sigmoid of each value of z
    """
    g = 1 / (1 + np.exp(-z))
    return g
