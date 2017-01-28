from cost_function_reg import cost_function_reg


def cost_reg_opt(theta, x, y, l):
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
    j, g = cost_function_reg(theta, x, y, l)
    return j


def gradient_reg_opt(theta, x, y, l):
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
    j, g = cost_function_reg(theta, x, y, l)
    return g
