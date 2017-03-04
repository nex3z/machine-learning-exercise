import numpy as np
import scipy.optimize as opt

from linear_reg_cost_function import linear_reg_cost_function


def train_linear_reg(X, y, l, iteration=200):
    """
    Trains linear regression given a dataset (X, y) and a regularization parameter lambda.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    l : float
        Regularization parameter.
    iteration : int
        Max number of iteration.

    Returns
    -------
    ndarray, shape (n_features,)
        Trained linear regression parameters.
    """
    m, n = X.shape
    initial_theta = np.zeros((n, 1))

    result = opt.minimize(fun=linear_reg_cost_function, x0=initial_theta, args=(X, y, l), method='TNC', jac=True,
                          options={'maxiter': iteration})
    # print 'train_linear_reg:', result.success

    return result.x
