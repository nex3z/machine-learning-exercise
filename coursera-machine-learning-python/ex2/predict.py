import numpy as np

from sigmoid import sigmoid


def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta.

    Parameters
    ----------
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    ndarray, shape (n_samples,)
        The predictions for X using a threshold at 0.5.
    """
    p = sigmoid(X.dot(theta)) >= 0.5
    return p.astype(int)
