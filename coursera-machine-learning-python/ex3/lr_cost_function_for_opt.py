from lr_cost_function import lr_cost_function


def lr_cost_opt(theta, x, y, l):
    """
    Compute cost for logistic regression with regularization.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.
    l : float
        Regularization parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in x and y.
    """
    j, g = lr_cost_function(theta, x, y, l)
    return j


def lr_gradient_opt(theta, x, y, l):
    """
    Compute gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.
    l : float
        Regularization parameter.

    Returns
    -------
    grad: ndarray
        Partial derivatives of the cost w.r.t. each parameter in theta, n by 1 matrix.
    """
    j, g = lr_cost_function(theta, x, y, l)
    return g.flatten()
