import numpy as np


def linear_reg_cost_function(theta, X, y, l):
    """
    Compute cost and gradient for regularized linear regression with multiple variables.

    Parameters
    ----------
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    l : float
        Regularization parameter.

    Returns
    -------
    j : numpy.float64
        The cost of using theta as the parameter for linear regression.
    grad : ndarray, shape (n_features,)
        The gradient of using theta as the parameter for linear regression.
    """
    m = X.shape[0]

    j = 1.0 / (2 * m) * np.sum(np.square(X.dot(theta) - y)) + 1.0 * l / (2 * m) * np.sum(np.square(theta[1:]))

    mask = np.eye(len(theta))
    mask[0, 0] = 0
    grad = 1.0 / m * X.T.dot(X.dot(theta) - y) + 1.0 * l / m * (mask.dot(theta))

    return j, grad
