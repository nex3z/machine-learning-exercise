import numpy as np


def feature_normalize(X, mu=None, sigma=None):
    """
    Normalizes the features in x.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples to be normalized, where n_samples is the number of samples and n_features is the number of features.
    mu : ndarray, shape (n_features,)
        Mean value for normalization. If not provided, it will be calculated from X.
    sigma : ndarray, shape (n_features,)
        Standard deviation for normalization. If not provided, it will be calculated from X.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        The normalized features.
    mu : ndarray, shape (n_features,)
        Mean value of X.
    sigma : ndarray, shape (n_features,)
        Standard deviation of X.
    """
    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, ddof=1, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
