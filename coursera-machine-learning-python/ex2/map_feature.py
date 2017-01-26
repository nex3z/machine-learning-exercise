import numpy as np


def map_feature(x1, x2):
    """
    Feature mapping function to polynomial features. Inputs X1, X2 must be the same size.
    :param x1: input feature
    :param x2: input feature
    :return: a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    degree = 6
    out = np.ones(x1[:, 0:1].shape)

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            tmp = np.power(x1, i - j) * np.power(x2, j)
            out = np.concatenate((out, tmp), axis=1)
    return out
