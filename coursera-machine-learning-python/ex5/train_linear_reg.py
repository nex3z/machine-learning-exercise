import numpy as np
import scipy.optimize as opt

from linear_reg_cost_function import linear_reg_cost_function


def train_linear_reg(x, y, l, iteration=200):
    """
    Trains linear regression given a dataset (X, y) and a regularization parameter lambda.

    Parameters
    ----------
    x : ndarray
        Training data.
    y : ndarray
        Labels.
    l : float
        Regularization parameter.
    iteration : int
        Max number of iteration.

    Returns
    -------
    ndarray
        Trained linear regression parameters.
    """
    m, n = x.shape
    initial_theta = np.zeros((n, 1))

    result = opt.minimize(fun=linear_reg_cost_function, x0=initial_theta, args=(x, y, l), method='TNC', jac=True,
                          options={'maxiter': iteration})
    # print 'train_linear_reg:', result.success
    return result.x.reshape(n, 1)
