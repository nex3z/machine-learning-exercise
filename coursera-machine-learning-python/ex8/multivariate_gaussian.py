import numpy as np


def multivariate_gaussian(X, mu, Sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    mu : ndarray, shape (n_feature,)
        The mean of each feature.
    sigma2 : ndarray, shape (n_feature,)
        The variance of each feature.

    Returns
    -------
    p : ndarray, shape (n_samples,)
        The probability density function of the examples X under the multivariate gaussian distribution with parameters
        mu and Sigma2.
    """
    k = len(mu)

    if len(Sigma2.shape) == 1:
        Sigma2 = np.diag(Sigma2)

    X_mu = X - mu
    p = (2 * np.pi) ** (-k / 2.0) * np.linalg.det(Sigma2) ** (-0.5) \
        * np.exp(-0.5 * np.sum(X_mu.dot(np.linalg.pinv(Sigma2)) * X_mu, axis=1))

    return p
