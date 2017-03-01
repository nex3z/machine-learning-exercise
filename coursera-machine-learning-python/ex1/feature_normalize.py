import numpy as np


def feature_normalize(X):
    """
    Normalizes the features in x.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Features to be normalized.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        A normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    mu : ndarray, shape (n_features,)
        The mean value.
    sigma : ndarray, shape (n_features,)
        The standard deviation.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
