import numpy as np
import matplotlib.pyplot as plt

from multivariate_gaussian import multivariate_gaussian


def visualize_fit(X, mu, sigma2):
    """
    Visualize the dataset and its estimated distribution.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    mu : ndarray, shape (n_feature,)
        The mean of each feature.
    sigma2 : ndarray, shape (n_feature,)
        The variance of each feature.
    """
    l = np.arange(0, 35.5, 0.5)
    X1, X2 = np.meshgrid(l, l)

    X_tmp = np.vstack((X1.ravel(), X2.ravel())).T
    Z = multivariate_gaussian(X_tmp, mu, sigma2)
    Z.resize(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 0, 3))
