import numpy as np

from sigmoid import sigmoid


def lr_cost_function(theta, X, y, l):
    """
    Compute cost and gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    l : float
        Regularization parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for regularized logistic regression w.r.t. the parameters.
    grad : ndarray, shape (n_features,)
        Partial derivatives of the cost w.r.t. each parameter in theta.
    """
    m, n = X.shape

    mask = np.eye(len(theta))
    # Skip the theta[0, 0] parameter when performing regularization
    mask[0, 0] = 0

    X_dot_theta = X.dot(theta)

    J = 1.0 / m * (-y.T.dot(np.log(sigmoid(X_dot_theta))) - (1 - y).T.dot(np.log(1 - sigmoid(X_dot_theta)))) \
        + l / (2.0 * m) * np.sum(np.square(mask.dot(theta)))

    grad = 1.0 / m * (sigmoid(X_dot_theta) - y).T.dot(X).T + 1.0 * l / m * (mask.dot(theta))

    return J, grad
