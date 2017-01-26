import numpy as np

from compute_cost import compute_cost


def gradient_descent(x, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.

    Parameters
    ----------
    x : ndarray
        Training data. It's a m by n matrix, where m is the number of data samples and n is the number of features.
    y : ndarray
        Labels, m by 1 matrix.
    theta : ndarray
        Initial linear regression parameter, n by 1 matrix.
    alpha : float
        Learning rate.
    num_iters: int
        Number of iteration.

    Returns
    -------
    theta : ndarray
        Linear regression parameter, n by 1 matrix.
    j_history: ndarray
        Cost history, num_iters by 1 matrix.
    """
    m = len(y)
    j_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta -= alpha / m * (np.dot((np.dot(x, theta) - y).T, x)).T
        j_history[i] = compute_cost(x, y, theta)

    return theta, j_history
