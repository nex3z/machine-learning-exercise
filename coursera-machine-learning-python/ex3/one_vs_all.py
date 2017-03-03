import numpy as np
import scipy.optimize as opt

from lr_cost_function import lr_cost_function


def one_vs_all(X, y, num_labels, l):
    """
    Trains multiple logistic regression classifiers and returns all the classifiers in a matrix all_theta, where the
    i-th row of all_theta corresponds to the classifier for label i.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    num_labels : int
        Number of labels.
    l : float
        Regularization parameter.

    Returns
    -------
    all_theta : ndarray, shape (num_labels, n_features + 1)
         All classifiers where the i-th row of all_theta corresponds to the classifier for label i.
    """
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))
    initial_theta = np.zeros(n + 1)

    for i in range(0, 10):
        label = 10 if i == 0 else i
        result = opt.minimize(fun=lr_cost_function, x0=initial_theta, args=(X, (y==label).astype(int), l),
                              method='TNC', jac=True)
        print 'one_vs_all(): label =', label, ', success =', result.success
        all_theta[i, :] = result.x

    return all_theta
