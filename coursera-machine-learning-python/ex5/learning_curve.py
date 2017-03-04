import numpy as np
import matplotlib.pyplot as plt

from train_linear_reg import train_linear_reg


def learning_curve(X, y, X_val, y_val, l):
    """
    Generates the train and cross validation set errors needed to plot a learning curve.

    Parameters
    ----------
    X : ndarray, shape (n_train_samples, n_features)
        Samples of training set, where n_train_samples is the number of samples in training set and n_features is the
        number of features.
    y : ndarray, shape (n_train_samples,)
        Labels of training set.
    X_val : ndarray, shape (n_val_samples, n_features)
        Samples of cross validation set, where n_val_samples is the number of samples in cross validation set.
    y_val : ndarray, shape (n_val_samples,)
        Labels of cross validation set.
    l : float
        Regularization parameter.

    Returns
    -------
    error_train : ndarray, shape (n_train_samples,)
        Train set error, error_train[i] contains the training error for the model trained by the first (i + 1) training
        samples.
    error_val : ndarray, shape (n_train_samples,)
        Cross validation set error, error_val[i] contains the cross validation error for the model trained by the first
        (i + 1) training samples.
    """
    m = X.shape[0]
    m_val = X_val.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m + 1):
        theta = train_linear_reg(X[:i, ], y[:i, ], l)
        error_train[i - 1] = 1.0 / (2 * i) * np.sum(np.square(X[:i, ].dot(theta) - y[:i, ]))
        error_val[i - 1] = 1.0 / (2 * m_val) * np.sum(np.square(X_val.dot(theta) - y_val))

    return error_train, error_val
