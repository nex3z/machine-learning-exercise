import numpy as np
import scipy.optimize as opt

from lr_cost_function import lr_cost_function


def one_vs_all(x, y, num_labels, l):
    """
    Trains multiple logistic regression classifiers and returns all the classifiers in a matrix all_theta, where the
    i-th row of all_theta corresponds to the classifier for label i.

    Parameters
    ----------
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.
    y : ndarray
        Labels, m by 1 matrix.
    num_labels : int
        Number of labels.
    l : float
        Regularization parameter.

    Returns
    -------
    all_theta : ndarray
         All classifiers where the i-th row of all_theta corresponds to the classifier for label i, num_labels by
         (n + 1) matrix.
    """
    m, n = x.shape
    all_theta = np.zeros((num_labels, n + 1))
    x = np.hstack((np.ones((m, 1)), x))
    initial_theta = np.zeros((n + 1, 1)).flatten()

    for i in range(0, 10):
        label = 10 if i == 0 else i
        result = opt.minimize(fun=lr_cost_function, x0=initial_theta, args=(x, (y == label).astype(int), l),
                              method='TNC', jac=True)
        print 'one_vs_all(): label =', label, ', success =', result.success
        all_theta[i, :] = result.x

    return all_theta
