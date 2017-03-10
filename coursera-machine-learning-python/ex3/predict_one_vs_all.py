import numpy as np


def predict_one_vs_all(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier.

    Parameters
    ----------
    all_theta : ndarray, shape (num_labels, n_features + 1)
         A matrix where the i-th row is a trained logistic regression theta vector for the i-th class, where num_labels
         is the number of labels and n_features is the number of features.
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples.

    Returns
    -------
    p : ndarray, shape (n_samples,)
         The prediction for X.
    """
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    p = np.argmax(X.dot(all_theta.T), axis=1)
    p[p == 0] = 10

    return p
