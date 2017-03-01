import numpy as np


def normal_eqn(X, y):
    """
    Computes the closed-form solution to linear regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.

    Returns
    -------
    theta : ndarray, shape (n_features,)
        The closed-form solution to linear regression using the normal equations.
    """
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
