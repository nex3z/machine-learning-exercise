import numpy as np


def estimate_gaussian(X):
    """
    Estimates the parameters of a Gaussian distribution using the data in X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    mu : ndarray, shape (n_feature,)
        The mean of each feature.
    sigma2 : ndarray, shape (n_feature,)
        The variance of each feature.
    """
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2
