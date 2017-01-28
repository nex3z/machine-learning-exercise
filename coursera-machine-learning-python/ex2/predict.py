import numpy as np

from sigmoid import sigmoid


def predict(theta, x):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.

    Returns
    -------
    ndarray
        The predictions for X using a threshold at 0.5, m by 1 matrix.
    """
    p = sigmoid(x.dot(theta)) >= 0.5
    p.resize(len(p), 1)
    return p.astype(int)
