import numpy as np


def predict_one_vs_all(all_theta, x):
    """
    Predict the label for a trained one-vs-all classifier.

    Parameters
    ----------
    all_theta : ndarray
         A matrix where the i-th row is a trained logistic regression theta vector for the i-th class.
    x : ndarray
        Training data, m by n matrix where m is the number of data samples.

    Returns
    -------
    p : ndarray
         The prediction for x, m by 1 array.
    """
    m, n = x.shape
    p = np.zeros((m, 1))
    x = np.hstack((np.ones((m, 1)), x))
    p = np.argmax(x.dot(all_theta.T), axis=1)
    p[p == 0] = 10
    return p.reshape(m, 1)
