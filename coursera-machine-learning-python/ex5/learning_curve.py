import numpy as np
import matplotlib.pyplot as plt

from train_linear_reg import train_linear_reg


def learning_curve(x, y, x_val, y_val, l):
    """
    Generates the train and cross validation set errors needed to plot a learning curve.

    Parameters
    ----------
    x : ndarray
        Training data of train set.
    y : ndarray
        Labels of train set.
    x_val : ndarray
        Training data of cross validation set.
    y_val : ndarray
        Labels of cross validation set.
    l : float
        Regularization parameter.

    Returns
    -------
    error_train : ndarray
        Train set error.
    error_val : ndarray
        Cross validation set error.
    """
    m = x.shape[0]
    m_val = x_val.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m + 1):
        theta = train_linear_reg(x[:i, ], y[:i, ], l)
        error_train[i - 1] = 1.0 / (2 * i) * np.sum(np.square(x[:i, ].dot(theta) - y[:i, ]))
        error_val[i - 1] = 1.0 / (2 * m_val) * np.sum(np.square(x_val.dot(theta) - y_val))

    return error_train, error_val
