from cost_function import cost_function


def cost_opt(theta, x, y):
    """
    Compute cost for logistic regression.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.


    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in x and y.
    """
    j, g = cost_function(theta, x, y)
    return j


def gradient_opt(theta, x, y):
    """
    Compute gradient for logistic regression.

    Parameters
    ----------
    theta : ndarray
        Linear regression parameter, n by 1 matrix where n is the number of features.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.

    Returns
    -------
    grad: ndarray
        The gradient of the cost w.r.t. the parameters, n by 1 matrix.
    """
    j, g = cost_function(theta, x, y)
    return g
