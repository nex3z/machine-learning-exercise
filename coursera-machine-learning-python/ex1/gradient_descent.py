import numpy as np

from compute_cost import compute_cost


def gradient_descent(x, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.

    Parameters
    ----------
    x : ndarray
        Training data. It's a n by m matrix, where n is the number of data samples and m is the number of features.
    y : ndarray
        Labels, n by 1 matrix.
    theta : ndarray
        Linear regression parameter, n by 1 matrix.
    alpha : float
        Learning rate.
    num_iters: int
        Number of iteration. num_iters by 1 matrix.

    Returns
    -------
    theta : ndarray
        Linear regression parameter, n by 1 matrix.
    j_history: ndarray
        Cost history.
    """
    m = len(y)
    j_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta -= alpha / m * (np.dot((np.dot(x, theta) - y).T, x)).T
        j_history[i] = compute_cost(x, y, theta)

    return [theta, j_history]
